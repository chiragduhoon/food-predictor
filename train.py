from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT_DIR / "dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IndianFoodAI YOLO model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to YOLO data.yaml file.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--model",
        default="yolo11s.yaml",
        help="YOLO architecture/config checkpoint to initialize from.",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Directory where training runs are saved.",
    )
    parser.add_argument("--name", default="scratch_v1", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = args.data.resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if torch.cuda.is_available():
        device = 0
        amp = True
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        amp = False
        print("GPU not detected. Falling back to CPU training.")

    model = YOLO(args.model)
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        amp=amp,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
