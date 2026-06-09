import argparse
import tarfile
import tempfile
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minicpmv46_helper import cast_vision_input  # noqa: E402
from minicpmv46_helper import load_minicpmv46_meta  # noqa: E402
from minicpmv46_helper import preprocess_fixed_image  # noqa: E402


def parse_input_shape(shape_text: str) -> tuple[int, int]:
    text = shape_text.strip().lower().replace("*", "x")
    parts = text.split("x")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise argparse.ArgumentTypeError(
            f"invalid --input-shape {shape_text!r}, expected forms like 448x448 or 392x392"
        )
    return int(parts[0]), int(parts[1])


def collect_image_paths(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in exts)


def default_tar_name(height: int, width: int) -> str:
    if height == width:
        return f"minicpmv4_6_vision_{height}_calibration.tar"
    return f"minicpmv4_6_vision_h{height}_w{width}_calibration.tar"


def prepare_one_image(image_path: Path, model_path: str, height: int, width: int) -> np.ndarray:
    meta = load_minicpmv46_meta(model_path)
    raw_patches = preprocess_fixed_image(
        image_path=str(image_path),
        tgt_height=height,
        tgt_width=width,
        patch_size=meta.vision_patch_size,
    )
    return cast_vision_input(
        raw_patches=raw_patches,
        input_shape=raw_patches.shape,
        input_dtype=np.float32,
        mean=meta.image_mean,
        std=meta.image_std,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare MiniCPM-V-4.6 fixed-shape vision calibration inputs")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../python/MiniCPM-V-4.6",
        help="Path to the original MiniCPM-V-4.6 tokenizer/config directory",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to a calibration image directory or a .tar image archive",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/minicpmv4_6_vision_448_calibration",
        help="Directory used for intermediate .npy files",
    )
    parser.add_argument(
        "--input-shape",
        type=parse_input_shape,
        default=(448, 448),
        help="Fixed vision input shape, for example 448x448 or 392x392",
    )
    parser.add_argument(
        "--tar_name",
        type=str,
        default="",
        help="Optional output tar file name",
    )
    args = parser.parse_args()

    height, width = args.input_shape
    dataset_path = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_from_paths(image_paths):
        saved_files = []
        for image_path in image_paths:
            pixel_values = prepare_one_image(image_path, args.model_path, height, width)
            output_path = output_dir / f"{image_path.stem}.pixel_values.npy"
            np.save(output_path, pixel_values.astype(np.float32))
            saved_files.append(output_path)
            print(f"saved {output_path}")
        return saved_files

    if dataset_path.is_dir():
        image_paths = collect_image_paths(dataset_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found under {dataset_path}")
        saved_files = _prepare_from_paths(image_paths)
    elif dataset_path.is_file() and tarfile.is_tarfile(dataset_path):
        with tempfile.TemporaryDirectory(prefix="minicpm46_calib_") as tmp_dir:
            extract_dir = Path(tmp_dir)
            with tarfile.open(dataset_path, "r") as tar:
                tar.extractall(extract_dir)
            image_paths = collect_image_paths(extract_dir)
            if not image_paths:
                raise FileNotFoundError(f"No images found inside tar file {dataset_path}")
            saved_files = _prepare_from_paths(image_paths)
    else:
        raise FileNotFoundError(
            f"`dataset_dir` must be an image directory or a tar archive, got: {dataset_path}"
        )

    tar_name = args.tar_name or default_tar_name(height, width)
    tar_path = output_dir.parent / tar_name
    with tarfile.open(tar_path, "w") as tar:
        for npy_path in saved_files:
            tar.add(npy_path, arcname=npy_path.name)

    print(f"packed {len(saved_files)} files -> {tar_path}")


if __name__ == "__main__":
    main()
