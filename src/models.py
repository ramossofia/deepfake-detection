import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_model(model_name: str, device: str, dropout: float = 0.3, num_classes: int = 1):
    """Crea un modelo preentrenado de timm con la capa final adaptada.
    Usado para 'xception' y 'efficientnet_b4' (03a, 03b, 04, 05).
    """
    model = timm.create_model(model_name, pretrained=True,
                               num_classes=num_classes, drop_rate=dropout)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model_name} — parámetros entrenables: {total_params:,}')
    return model


def build_optimizer(model, lr: float, lr_head: float, weight_decay: float):
    """AdamW con lr diferenciada: mayor para la capa final (head/fc/classifier),
    menor para el resto del backbone.
    """
    head_params = [p for n, p in model.named_parameters()
                   if 'head' in n or 'fc' in n or 'classifier' in n]
    backbone_params = [p for n, p in model.named_parameters()
                        if 'head' not in n and 'fc' not in n and 'classifier' not in n]

    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=weight_decay)


def load_model(model_obj: nn.Module, ckpt_path, device: str):
    """Carga un checkpoint .pth y devuelve (model, val_auc, epoch) en eval mode.
    Usado en 09, 10, 11 para cargar los distintos checkpoints entrenados.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_obj.load_state_dict(ckpt['model_state'])
    model_obj = model_obj.to(device)
    model_obj.eval()
    return model_obj, ckpt.get('val_auc', 'N/A'), ckpt.get('epoch', 'N/A')


class SpectralConv2d(nn.Module):
    """Convolución espectral 2D: núcleo del Fourier Neural Operator.

    Aprende pesos directamente en el espacio de Fourier sobre los
    primeros modes1 x modes2 modos de frecuencia; los modos altos se
    truncan (filtro pasa-bajos aprendible). Se mantienen dos conjuntos
    de pesos para capturar la simetría conjugada del espectro real
    (rfft2 solo devuelve la mitad).
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        return torch.einsum('bixy,ioxy->boxy', inp, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    """Bloque residual FNO: SpectralConv2d + bypass Conv1x1 + BatchNorm + GELU."""
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm     = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class XceptionFNO(nn.Module):
    def __init__(self, dropout=0.3, fno_channels=32, modes=16, num_blocks=4):
        super().__init__()
        self.backbone = timm.create_model(
            'xception', pretrained=True, num_classes=0, global_pool='avg'
        )
        spatial_dim = self.backbone.num_features

        self.lift = nn.Conv2d(1, fno_channels, kernel_size=1)
        self.fno_blocks = nn.Sequential(*[
            FNOBlock(fno_channels, modes, modes) for _ in range(num_blocks)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(fno_channels, 128, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
        )

        self.spatial_norm = nn.BatchNorm1d(spatial_dim)
        self.fno_norm     = nn.BatchNorm1d(256)

        # Gate: decide cuánto pesar la rama espectral, arranca casi cerrado
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim + 256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12 al inicio

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        spatial_features = self.backbone(x)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        fno_features = self.project(self.fno_blocks(self.lift(gray)))

        spatial_n = self.spatial_norm(spatial_features)
        fno_n     = self.fno_norm(fno_features)

        gate_input = torch.cat([spatial_n, fno_n], dim=1)
        g = torch.sigmoid(self.gate(gate_input))

        fused = torch.cat([spatial_n, g * fno_n], dim=1)
        return self.classifier(fused)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise (grupos=in_ch) + Pointwise (1x1) + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BiFPNLayer(nn.Module):
    """Una capa BiFPN sobre 3 niveles de features (P2, P3, P4).
    Fusión ponderada rápida: w_i >= 0 (ReLU), normalizado con eps=1e-4.
    """
    def __init__(self, channels):
        super().__init__()
        C, self.eps = channels, 1e-4

        self.w_td_p3 = nn.Parameter(torch.ones(2))
        self.w_td_p2 = nn.Parameter(torch.ones(2))
        self.w_bu_p3 = nn.Parameter(torch.ones(3))
        self.w_bu_p4 = nn.Parameter(torch.ones(2))

        self.conv_td_p3 = DepthwiseSeparableConv(C, C)
        self.conv_td_p2 = DepthwiseSeparableConv(C, C)
        self.conv_bu_p3 = DepthwiseSeparableConv(C, C)
        self.conv_bu_p4 = DepthwiseSeparableConv(C, C)

    def fuse(self, weights, *tensors):
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * t for i, t in enumerate(tensors))

    def forward(self, p2, p3, p4):
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_td = self.conv_td_p3(self.fuse(self.w_td_p3, p3, p4_up))

        p3_up = F.interpolate(p3_td, size=p2.shape[-2:], mode='nearest')
        p2_td = self.conv_td_p2(self.fuse(self.w_td_p2, p2, p3_up))

        p2_dn  = F.adaptive_avg_pool2d(p2_td, output_size=p3.shape[-2:])
        p3_out = self.conv_bu_p3(self.fuse(self.w_bu_p3, p3, p3_td, p2_dn))

        p3_dn  = F.adaptive_avg_pool2d(p3_out, output_size=p4.shape[-2:])
        p4_out = self.conv_bu_p4(self.fuse(self.w_bu_p4, p4, p3_dn))

        return p2_td, p3_out, p4_out


class XceptionBiFPN(nn.Module):
    def __init__(self, dropout=0.3, bifpn_channels=128, num_layers=3):
        super().__init__()
        C = bifpn_channels

        self.backbone = timm.create_model(
            'xception', pretrained=True, features_only=True, out_indices=(2, 3, 4))
        feat_chs = [256, 728, 2048]

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, C, kernel_size=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ) for ch in feat_chs
        ])

        self.bifpn = nn.ModuleList([BiFPNLayer(C) for _ in range(num_layers)])

        self.level_weights = nn.Parameter(torch.ones(3))
        self.classifier = nn.Sequential(
            nn.Linear(C, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        p2, p3, p4 = self.backbone(x)
        p2, p3, p4 = self.proj[0](p2), self.proj[1](p3), self.proj[2](p4)

        for layer in self.bifpn:
            p2, p3, p4 = layer(p2, p3, p4)

        pool = nn.AdaptiveAvgPool2d(1)
        w = F.softmax(self.level_weights, dim=0)
        feat = (w[0] * pool(p2).flatten(1) +
                w[1] * pool(p3).flatten(1) +
                w[2] * pool(p4).flatten(1))
        return self.classifier(feat)