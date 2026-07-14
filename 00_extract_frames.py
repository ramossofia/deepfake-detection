import cv2
import os
from pathlib import Path
from tqdm import tqdm


DATASET_PATH = "/DEEPFAKES/FaceForensics++_C23"
OUTPUT_PATH  = "./frames"
FRAMES_PER_VIDEO = 10
IMG_SIZE = (299, 299)

METHODS = [
    "original",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
]

def extract_frames(video_path, output_dir, n_frames, size):
    """Extrae n_frames distribuidos uniformemente a lo largo del video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return 0

    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, size)
        out_path = output_dir / f"frame_{idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def process_dataset(dataset_path, output_path, frames_per_video, img_size):
    dataset_path = Path(dataset_path)
    output_path  = Path(output_path)

    total_videos  = 0
    total_frames  = 0

    for method in METHODS:
        method_dir = dataset_path / method

        label = "real" if method == "original" else method

        if not method_dir.exists():
            print(f"No se encontró la carpeta: {method_dir}")
            continue

        videos = list(method_dir.rglob("*.mp4"))
        print(f"\n{method} — {len(videos)} videos encontrados")

        for video_path in tqdm(videos, desc=method):
            # Estructura de salida: frames/<label>/<nombre_video>/frame_XXXXX.jpg
            video_name   = video_path.stem
            output_dir   = output_path / label / video_name
            output_dir.mkdir(parents=True, exist_ok=True)

            saved = extract_frames(video_path, output_dir, frames_per_video, img_size)
            total_frames += saved
            total_videos += 1

    print(f"\n{total_videos} videos procesados, {total_frames} frames guardados")
    print(f"Frames en: {output_path.resolve()}")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, OUTPUT_PATH, FRAMES_PER_VIDEO, IMG_SIZE)