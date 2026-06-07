import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class MiniCPMV46Meta:
    image_token: str
    video_token: str
    image_start_token: str
    image_end_token: str
    image_id_start_token: str
    image_id_end_token: str
    vision_width: int
    vision_height: int
    vision_patch_size: int
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_minicpmv46_meta(hf_model_dir: str) -> MiniCPMV46Meta:
    cfg = _read_json(os.path.join(hf_model_dir, "config.json"))
    tok_cfg = _read_json(os.path.join(hf_model_dir, "tokenizer_config.json"))
    prep_cfg = _read_json(os.path.join(hf_model_dir, "preprocessor_config.json"))
    extra = tok_cfg.get("extra_special_tokens", {})
    return MiniCPMV46Meta(
        image_token=tok_cfg.get("image_token", extra.get("image_token", "<|image_pad|>")),
        video_token=tok_cfg.get("video_token", extra.get("video_token", "<|video_pad|>")),
        image_start_token=extra.get("image_start_token", "<image>"),
        image_end_token=extra.get("image_end_token", "</image>"),
        image_id_start_token=extra.get("image_id_start_token", "<image_id>"),
        image_id_end_token=extra.get("image_id_end_token", "</image_id>"),
        vision_width=int(prep_cfg.get("scale_resolution", 448)),
        vision_height=int(prep_cfg.get("scale_resolution", 448)),
        vision_patch_size=int(prep_cfg.get("patch_size", cfg["vision_config"]["patch_size"])),
        image_mean=tuple(float(x) for x in prep_cfg.get("image_mean", [0.5, 0.5, 0.5])),
        image_std=tuple(float(x) for x in prep_cfg.get("image_std", [0.5, 0.5, 0.5])),
    )


def render_text_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_single_image_prompt(prompt: str, meta: MiniCPMV46Meta, image_token_count: int) -> str:
    image_block = (
        f"{meta.image_id_start_token}0{meta.image_id_end_token}"
        f"{meta.image_start_token}"
        f"{meta.image_token * image_token_count}"
        f"{meta.image_end_token}"
    )
    return f"<|im_start|>user\n{image_block}\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def image_token_count_for_fixed_shape(height: int, width: int, patch_size: int, downsample_mode: str = "16x") -> int:
    grid_h = height // patch_size
    grid_w = width // patch_size
    divisor = 4 if downsample_mode == "4x" else 16
    return (grid_h * grid_w) // divisor


def reshape_by_patch(image_chw: np.ndarray, patch_size: int) -> np.ndarray:
    channels, height, width = image_chw.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    reshaped = image_chw.reshape(channels, grid_h, patch_size, grid_w, patch_size)
    reshaped = reshaped.transpose(0, 2, 1, 3, 4)
    return reshaped.reshape(channels, patch_size, grid_h * grid_w * patch_size)


def preprocess_fixed_image(
    image_path: str,
    tgt_height: int,
    tgt_width: int,
    patch_size: int,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((tgt_width, tgt_height), Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.uint8)
    chw = rgb.transpose(2, 0, 1)
    patches = reshape_by_patch(chw, patch_size)
    return patches[np.newaxis, ...]


def cast_vision_input(raw_patches: np.ndarray, input_shape, input_dtype, mean, std) -> np.ndarray:
    arr = np.asarray(raw_patches)
    if tuple(arr.shape) != tuple(input_shape):
        arr = arr.reshape(tuple(input_shape))
    name = str(input_dtype).lower()
    if "uint8" in name or "u8" in name:
        return arr.astype(np.uint8, copy=False)

    arr = arr.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    arr = (arr - mean_arr) / std_arr
    if "float16" in name or "fp16" in name:
        return arr.astype(np.float16)
    return arr.astype(np.float32)


def fixed_target_sizes(height: int, width: int, patch_size: int) -> np.ndarray:
    return np.asarray([[height // patch_size, width // patch_size]], dtype=np.int32)


def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(
    image_size: Tuple[int, int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    best_width = ensure_divide(width, patch_size * 4)
    best_height = ensure_divide(height, patch_size * 4)
    return best_height, best_width
