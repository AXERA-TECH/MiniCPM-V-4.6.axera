import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_PATH = (
    "/data/tmp/yongqiang/nfs/auto_model_deployment/"
    "Minicpm-V-4.6-hf-original/MiniCPM-V-4.6"
)


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


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.6 official torch text-only inference")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="你好，请做一个简短自我介绍。")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
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

    prompt_text = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    print("model_path:", args.model_path)
    print("device:", model.device)
    print("input_ids.shape:", tuple(inputs.input_ids.shape))
    print("input_ids:", inputs.input_ids[0].tolist())
    print("prompt_template_repr:", prompt_text.encode("unicode_escape").decode())

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

    trimmed_ids = generated_ids[0, inputs.input_ids.shape[-1] :]
    output_text = processor.decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("generated_ids:", trimmed_ids.tolist())
    print("output_text:", output_text)


if __name__ == "__main__":
    main()
