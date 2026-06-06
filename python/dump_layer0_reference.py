import argparse
import hashlib
import json

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_device(name: str):
    if name != "auto":
        return name
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def dump_stats(name: str, value: torch.Tensor):
    tensor = value.detach().float().cpu()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    payload = {
        "shape": list(tensor.shape),
        "dtype": str(value.dtype),
        "finite_ratio": float(finite.float().mean().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean_abs": float(finite_values.abs().mean().item()) if finite_values.numel() else None,
        "sha256_16_f32": hashlib.sha256(tensor.numpy().astype("float32").tobytes()).hexdigest()[:16],
    }
    print(f"{name}: {json.dumps(payload, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description="Dump MiniCPM-V-4.6 layer0 linear attention reference tensors")
    parser.add_argument("--model-path", required=True, help="Path to the original openbmb/MiniCPM-V-4.6 Hugging Face model")
    parser.add_argument("--prompt", default="你好，请做一个简短自我介绍。")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model_path)
    load_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": device,
    }
    if dtype != "auto":
        load_kwargs["dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load_kwargs).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    layer = model.model.language_model.layers[args.layer]
    linear_attn = layer.linear_attn
    captures = {}

    for name, module in [
        ("in_proj_qkv", linear_attn.in_proj_qkv),
        ("in_proj_z", linear_attn.in_proj_z),
        ("in_proj_a", linear_attn.in_proj_a),
        ("in_proj_b", linear_attn.in_proj_b),
        ("out_proj", linear_attn.out_proj),
    ]:
        module.register_forward_hook(
            lambda module, module_inputs, module_output, name=name: captures.setdefault(
                name, module_output[0] if isinstance(module_output, tuple) else module_output
            )
        )

    conv_fn = linear_attn.causal_conv1d_fn
    if conv_fn is not None:
        def wrapped_conv_fn(**kwargs):
            output = conv_fn(**kwargs)
            captures["conv1d_fn"] = output
            return output

        linear_attn.causal_conv1d_fn = wrapped_conv_fn

    with torch.inference_mode():
        outputs = model.model.language_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    linear_attn.causal_conv1d_fn = conv_fn

    print("model_path:", args.model_path)
    print("device:", model.device)
    print("layer:", args.layer)
    print("input_ids:", inputs.input_ids[0].tolist())
    dump_stats("layer_hidden", outputs.hidden_states[args.layer + 1])

    for name in ["in_proj_qkv", "conv1d_fn", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]:
        if name in captures:
            dump_stats(name, captures[name])


if __name__ == "__main__":
    main()
