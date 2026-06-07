import argparse
import json
import re
from pathlib import Path
import sys

import torch
from transformers import AutoModelForImageTextToText
from transformers import __version__ as transformers_version

THIS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = THIS_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minicpmv46_helper import fixed_target_sizes
from minicpmv46_helper import image_token_count_for_fixed_shape


MIN_TRANSFORMERS_VERSION = (5, 7, 0)
MIN_TRANSFORMERS_VERSION_STR = "5.7.0"


def parse_version(version: str):
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class MiniCPMV46VisionWrapper(torch.nn.Module):
    def __init__(self, model, target_sizes: torch.Tensor, downsample_mode: str):
        super().__init__()
        self.model = model
        self.register_buffer("target_sizes", target_sizes, persistent=False)
        self.downsample_mode = downsample_mode

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            target_sizes=self.target_sizes,
            downsample_mode=self.downsample_mode,
        )
        return torch.cat(outputs.pooler_output, dim=0).unsqueeze(0)


def parse_input_shape(shape_text: str) -> tuple[int, int]:
    text = shape_text.strip().lower()
    match = re.fullmatch(r"(\d+)\s*[x\*,]\s*(\d+)", text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid --input-shape {shape_text!r}, expected forms like 448x448 or 392*392"
        )
    return int(match.group(1)), int(match.group(2))


def resolve_hw(args) -> tuple[int, int]:
    height = args.height
    width = args.width
    if args.input_shape is not None:
        height, width = args.input_shape
    return int(height), int(width)


def validate_shape(height: int, width: int, patch_size: int):
    align = patch_size * 4
    if height <= 0 or width <= 0:
        raise ValueError(f"input shape must be positive, got {height}x{width}")
    if height % align != 0 or width % align != 0:
        raise ValueError(
            f"input shape {height}x{width} must be divisible by patch_size*4={align} "
            "to match MiniCPM-V-4.6 fixed-shape patch merge requirements"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export MiniCPM-V-4.6 fixed-shape vision encoder ONNX")
    parser.add_argument("--model", required=True, help="Path to original openbmb/MiniCPM-V-4.6 model")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--input-shape",
        type=parse_input_shape,
        default=None,
        help="Fixed vision input shape, for example 448x448 or 392*392",
    )
    parser.add_argument("--height", type=int, default=448, help="Fixed input height")
    parser.add_argument("--width", type=int, default=448, help="Fixed input width")
    parser.add_argument("--patch-size", type=int, default=14, help="Vision patch size")
    parser.add_argument("--downsample-mode", default="16x", choices=["4x", "16x"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention backend used during export. Default eager avoids the current ONNX sdpa+GQA export limitation.",
    )
    return parser.parse_args()


def require_supported_transformers():
    current = parse_version(transformers_version)
    if current < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"MiniCPM-V-4.6 ONNX export requires transformers>={MIN_TRANSFORMERS_VERSION_STR}, "
            f"but current version is {current}."
        )


def main():
    args = parse_args()
    require_supported_transformers()
    height, width = resolve_hw(args)
    validate_shape(height, width, args.patch_size)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation=args.attn_implementation,
    ).eval()

    target_sizes = torch.from_numpy(fixed_target_sizes(height, width, args.patch_size))
    seq_len = args.patch_size * (height // args.patch_size) * (width // args.patch_size)
    dummy_pixel_values = torch.zeros((1, 3, args.patch_size, seq_len), dtype=torch.float32)

    wrapper = MiniCPMV46VisionWrapper(model, target_sizes=target_sizes, downsample_mode=args.downsample_mode).eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_pixel_values,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=args.opset,
            dynamic_axes=None,
        )

    meta = {
        "input_shape": list(dummy_pixel_values.shape),
        "image_size": [height, width],
        "target_sizes": target_sizes.tolist(),
        "downsample_mode": args.downsample_mode,
        "output_shape": [1, image_token_count_for_fixed_shape(height, width, args.patch_size, args.downsample_mode), model.config.text_config.hidden_size],
        "transformers_version": transformers_version,
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("exported_onnx:", output_path)
    print("metadata_json:", output_path.with_suffix(".json"))
    print("attn_implementation:", args.attn_implementation)
    print("note: large ONNX exports may emit external data files next to the .onnx file")


if __name__ == "__main__":
    main()
