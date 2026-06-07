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


def fixed_position_embeddings(embedding: torch.nn.Embedding, grid_h: int, grid_w: int, num_patches_per_side: int):
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / grid_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / grid_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
    return embedding(pos_ids).unsqueeze(0)


def fixed_window_index(grid_h: int, grid_w: int, window_h: int, window_w: int):
    index = torch.arange(grid_h * grid_w).reshape(grid_h, grid_w)
    num_windows_h = grid_h // window_h
    num_windows_w = grid_w // window_w
    num_windows = num_windows_h * num_windows_w
    index = index.reshape(num_windows_h, window_h, num_windows_w, window_w)
    index = index.permute(0, 2, 1, 3).reshape(num_windows, window_h * window_w)
    window_index = index.reshape(-1)
    restore_index = torch.argsort(window_index)
    return window_index, restore_index


class FixedMiniCPMV46VisionWrapper(torch.nn.Module):
    def __init__(self, model, height: int, width: int, patch_size: int, downsample_mode: str):
        super().__init__()
        self.model = model
        self.downsample_mode = downsample_mode

        grid_h = height // patch_size
        grid_w = width // patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.embed_dim = model.model.vision_tower.config.hidden_size
        self.num_heads = model.model.vision_tower.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.insert_layer_id = model.model.vision_tower.config.insert_layer_id

        self.register_buffer(
            "position_embeddings",
            fixed_position_embeddings(
                model.model.vision_tower.embeddings.position_embedding,
                grid_h,
                grid_w,
                model.model.vision_tower.embeddings.num_patches_per_side,
            ).detach(),
            persistent=False,
        )
        self.register_buffer("cu_seqlens", torch.tensor([0, grid_h * grid_w], dtype=torch.int32), persistent=False)
        self.max_seqlen = grid_h * grid_w

        self.window_h, self.window_w = tuple(model.model.vision_tower.vit_merger.window_kernel_size)
        window_index, restore_index = fixed_window_index(grid_h, grid_w, self.window_h, self.window_w)
        self.register_buffer("window_index", window_index, persistent=False)
        self.register_buffer("restore_index", restore_index, persistent=False)

        ds_h = grid_h // self.window_h
        ds_w = grid_w // self.window_w
        self.register_buffer("downsampled_target_sizes", torch.tensor([[ds_h, ds_w]], dtype=torch.int32), persistent=False)
        self.register_buffer("downsampled_cu_seqlens", torch.tensor([0, ds_h * ds_w], dtype=torch.int32), persistent=False)
        self.downsampled_max_seqlen = ds_h * ds_w

        if downsample_mode == "16x":
            final_target_sizes = self.downsampled_target_sizes
        else:
            final_target_sizes = torch.tensor([[grid_h, grid_w]], dtype=torch.int32)
        self.register_buffer("final_target_sizes", final_target_sizes, persistent=False)

    def _run_encoder_layer(self, layer: torch.nn.Module, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
        residual = hidden_states
        hidden_states = layer.layer_norm1(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=None,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.layer_norm2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _run_window_attention(self, hidden_states: torch.Tensor):
        attn = self.model.model.vision_tower.vit_merger.self_attn
        num_windows, seq_len, _ = hidden_states.shape

        query = attn.q_proj(hidden_states).view(num_windows, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = attn.k_proj(hidden_states).view(num_windows, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = attn.v_proj(hidden_states).view(num_windows, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * attn.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(num_windows, seq_len, self.embed_dim).contiguous()
        return attn.out_proj(attn_output)

    def _run_vit_merger(self, hidden_states: torch.Tensor):
        merger = self.model.model.vision_tower.vit_merger

        residual = hidden_states
        hidden_states = merger.layer_norm1(hidden_states)
        hidden_states = hidden_states[:, self.window_index, :]

        num_windows = (self.grid_h // self.window_h) * (self.grid_w // self.window_w)
        hidden_states = hidden_states.view(num_windows, self.window_h * self.window_w, self.embed_dim)
        hidden_states = self._run_window_attention(hidden_states)
        hidden_states = hidden_states.reshape(1, num_windows * self.window_h * self.window_w, self.embed_dim)
        hidden_states = hidden_states[:, self.restore_index, :]
        hidden_states = residual + hidden_states

        patch = hidden_states[0].view(
            self.grid_h // self.window_h,
            self.window_h,
            self.grid_w // self.window_w,
            self.window_w,
            self.embed_dim,
        ).permute(0, 2, 1, 3, 4)
        merged = patch.reshape(-1, self.window_h * self.window_w * self.embed_dim)
        residual = patch.reshape(-1, self.window_h * self.window_w, self.embed_dim).mean(dim=1)

        merged = merger.pre_norm(merged)
        merged = merger.linear_1(merged)
        merged = merger.act(merged)
        merged = merger.linear_2(merged)
        return (merged + residual).unsqueeze(0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vt = self.model.model.vision_tower
        hidden_states = vt.embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embeddings

        if self.downsample_mode == "16x":
            for layer_index, layer in enumerate(vt.encoder.layers):
                hidden_states = self._run_encoder_layer(layer, hidden_states, self.cu_seqlens, self.max_seqlen)
                if layer_index == self.insert_layer_id:
                    hidden_states = self._run_vit_merger(hidden_states)
                    break

            for layer in vt.encoder.layers[self.insert_layer_id + 1 :]:
                hidden_states = self._run_encoder_layer(
                    layer,
                    hidden_states,
                    self.downsampled_cu_seqlens,
                    self.downsampled_max_seqlen,
                )
        else:
            for layer in vt.encoder.layers:
                hidden_states = self._run_encoder_layer(layer, hidden_states, self.cu_seqlens, self.max_seqlen)

        hidden_states = vt.post_layernorm(hidden_states)
        outputs = self.model.model.merger(hidden_states, self.final_target_sizes)
        return torch.cat(outputs, dim=0).unsqueeze(0)


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

    seq_len = args.patch_size * (height // args.patch_size) * (width // args.patch_size)
    dummy_pixel_values = torch.zeros((1, 3, args.patch_size, seq_len), dtype=torch.float32)

    wrapper = FixedMiniCPMV46VisionWrapper(
        model=model,
        height=height,
        width=width,
        patch_size=args.patch_size,
        downsample_mode=args.downsample_mode,
    ).eval()
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
