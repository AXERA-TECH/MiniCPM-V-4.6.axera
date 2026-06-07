import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import __version__ as transformers_version


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


def require_supported_transformers():
    current = parse_version(transformers_version)
    if current < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"MiniCPM-V-4.6 multimodal torch inference requires transformers>={MIN_TRANSFORMERS_VERSION_STR}, "
            f"but current version is {current}."
        )


def build_messages(args):
    content = []
    if args.image:
        content.append({"type": "image", "image": Image.open(args.image).convert("RGB")})
    if args.video:
        content.append({"type": "video", "url": str(Path(args.video).expanduser().resolve())})
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.6 official torch reference inference")
    parser.add_argument("--model-path", required=True, help="Path to the original openbmb/MiniCPM-V-4.6 Hugging Face model")
    parser.add_argument("--prompt", default="你好，请做一个简短自我介绍。")
    parser.add_argument("--image", default=None, help="Optional local image path")
    parser.add_argument("--video", default=None, help="Optional local video path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--downsample-mode", default="16x", choices=["4x", "16x"])
    parser.add_argument("--max-slice-nums", type=int, default=None)
    parser.add_argument("--max-num-frames", type=int, default=128)
    parser.add_argument("--stack-frames", type=int, default=1)
    args = parser.parse_args()

    require_supported_transformers()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    messages = build_messages(args)

    processor = AutoProcessor.from_pretrained(args.model_path)

    load_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": device,
    }
    if dtype != "auto":
        load_kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load_kwargs).eval()

    apply_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "downsample_mode": args.downsample_mode,
    }
    if args.image:
        apply_kwargs["max_slice_nums"] = args.max_slice_nums if args.max_slice_nums is not None else 36
    if args.video:
        apply_kwargs["max_num_frames"] = args.max_num_frames
        apply_kwargs["stack_frames"] = args.stack_frames
        apply_kwargs["max_slice_nums"] = args.max_slice_nums if args.max_slice_nums is not None else 1
        apply_kwargs["use_image_id"] = False

    inputs = processor.apply_chat_template(messages, **apply_kwargs).to(model.device)

    print("transformers_version:", transformers_version)
    print("model_path:", args.model_path)
    print("device:", model.device)
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"{key}.shape:", tuple(value.shape), str(value.dtype))
    print("prompt_token_count:", int(inputs.input_ids.shape[-1]))
    print("prompt_template_repr:", processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False).encode("unicode_escape").decode())

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            downsample_mode=args.downsample_mode,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

    trimmed_ids = generated_ids[0, inputs.input_ids.shape[-1] :]
    output_text = processor.batch_decode(
        [trimmed_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("generated_ids:", trimmed_ids.tolist())
    print("output_text:", output_text)


if __name__ == "__main__":
    main()
