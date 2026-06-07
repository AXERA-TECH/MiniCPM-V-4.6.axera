import argparse
import atexit
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ml_dtypes import bfloat16
from transformers import AutoTokenizer

from minicpmv46_helper import cast_vision_input
from minicpmv46_helper import image_token_count_for_fixed_shape
from minicpmv46_helper import load_minicpmv46_meta
from minicpmv46_helper import preprocess_fixed_image
from minicpmv46_helper import render_single_image_prompt
from minicpmv46_helper import render_text_prompt


def resolve_default_hf_model() -> str:
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "MiniCPM-V-4.6"),
        os.path.join(base_dir, "MiniCPM-V-4.6-GPTQ"),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    return candidates[0]


DEFAULT_HF_MODEL = resolve_default_hf_model()


def load_inference_session():
    from axengine import InferenceSession

    return InferenceSession


def release_ax_inference_session(session):
    inner = getattr(session, "_sess", None)
    unload = getattr(inner, "_unload", None)
    if not callable(unload):
        return

    try:
        unload()
    except Exception as exc:
        print(f"[WARN] Failed to unload axengine session cleanly: {exc}")
    finally:
        try:
            inner._unload = lambda: None
        except Exception:
            pass


def bf16_zeros(shape: Sequence[int]) -> np.ndarray:
    return np.zeros(tuple(shape), dtype=bfloat16)


def dtype_from_axengine(dtype) -> np.dtype:
    name = str(dtype).lower()
    if "bfloat16" in name or "bf16" in name:
        return bfloat16
    if "float32" in name or "fp32" in name:
        return np.float32
    if "float16" in name or "fp16" in name:
        return np.float16
    if "uint32" in name or "u32" in name:
        return np.uint32
    raise ValueError(f"Unsupported axengine dtype: {dtype}")


def tensor_digest(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype == bfloat16:
        raw = arr.view(np.uint16).tobytes()
    else:
        raw = arr.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def tensor_stats(arr: np.ndarray) -> str:
    arr32 = np.asarray(arr, dtype=np.float32)
    finite = bool(np.isfinite(arr32).all())
    if finite:
        return (
            f"finite=True hash={tensor_digest(np.asarray(arr))} "
            f"sum={float(arr32.sum()):.6f} max={float(arr32.max()):.6f} min={float(arr32.min()):.6f}"
        )
    return (
        f"finite=False hash={tensor_digest(np.asarray(arr))} "
        f"nans={int(np.isnan(arr32).sum())} infs={int(np.isinf(arr32).sum())}"
    )


def ensure_finite(name: str, arr: np.ndarray):
    arr32 = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr32).all():
        raise RuntimeError(f"{name} is non-finite: {tensor_stats(np.asarray(arr))}")


def load_text_config(hf_model: str) -> Tuple[dict, int]:
    with open(os.path.join(hf_model, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    text_cfg = cfg["text_config"]
    eos_token_id = cfg.get("eos_token_id")
    if eos_token_id is None:
        eos_token_id = text_cfg.get("eos_token_id")
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    return text_cfg, int(eos_token_id if eos_token_id is not None else 248044)


@dataclass
class LayerFiles:
    layer_paths: List[str]
    post_path: str


def detect_layer_files(model_dir: str, max_layers: Optional[int] = None) -> LayerFiles:
    names = os.listdir(model_dir)
    layer_pattern = re.compile(r"^(?P<prefix>.*)_p(?P<prefill>\d+)_l(?P<idx>\d+)_together\.axmodel$")
    decode_layer_pattern = re.compile(r"^(?P<prefix>.*)_l(?P<idx>\d+)\.axmodel$")
    post_pattern = re.compile(r"^(?P<prefix>.*)_post\.axmodel$")

    prefix_map = {}
    for name in names:
        m = layer_pattern.match(name)
        if not m:
            m = decode_layer_pattern.match(name)
        if not m:
            continue
        prefix = m.group("prefix")
        idx = int(m.group("idx"))
        prefix_map.setdefault(prefix, []).append((idx, name))

    if not prefix_map:
        raise FileNotFoundError(f"No layer axmodel found under {model_dir}")

    prefix = max(prefix_map.items(), key=lambda kv: len(kv[1]))[0]
    layer_items = sorted(prefix_map[prefix], key=lambda it: it[0])
    if max_layers is not None:
        layer_items = layer_items[: max_layers]
    layer_paths = [os.path.join(model_dir, name) for _, name in layer_items]

    post_name = None
    for name in names:
        m = post_pattern.match(name)
        if m and m.group("prefix") == prefix:
            post_name = name
            break
    if post_name is None:
        raise FileNotFoundError(f"No post axmodel found for prefix {prefix} under {model_dir}")

    return LayerFiles(layer_paths=layer_paths, post_path=os.path.join(model_dir, post_name))


class MiniCPMTextAxModelRunner:
    def __init__(
        self,
        hf_model: str,
        axmodel_dir: str,
        embed_bin: Optional[str],
        max_layers: Optional[int],
        vision_axmodel: Optional[str] = None,
        kv_cache_len: int = 255,
    ):
        self.hf_model = hf_model
        self.axmodel_dir = axmodel_dir
        self.meta = load_minicpmv46_meta(hf_model)
        self.text_cfg, self.eos_token_id = load_text_config(hf_model)
        self.hidden_size = int(self.text_cfg["hidden_size"])
        self.vocab_size = int(self.text_cfg["vocab_size"])
        self.kv_cache_len = int(kv_cache_len)
        self.layer_types = list(self.text_cfg.get("layer_types") or [])
        self.num_attention_heads = int(self.text_cfg["num_attention_heads"])
        self.num_key_value_heads = int(self.text_cfg["num_key_value_heads"])
        self.head_dim = int(self.text_cfg.get("head_dim") or (self.hidden_size // self.num_attention_heads))
        self.full_attn_kv_dim = self.num_key_value_heads * self.head_dim

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        self.layer_files = detect_layer_files(axmodel_dir, max_layers=max_layers)
        self.vision_axmodel = self.resolve_vision_axmodel(vision_axmodel)
        self.vision_session = None
        self.InferenceSession = load_inference_session()

        if embed_bin is None:
            candidate = os.path.join(axmodel_dir, "model.embed_tokens.weight.bfloat16.bin")
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    "Embedding bin not found under axmodel_dir, please pass --embed-bin explicitly"
                )
            embed_bin = candidate
        self.embed_bin = embed_bin
        self.embed_matrix = np.memmap(embed_bin, mode="r", dtype=np.uint16).view(bfloat16).reshape(
            self.vocab_size, self.hidden_size
        )
        self.image_token_id = int(self.tokenizer.encode(self.meta.image_token, add_special_tokens=False)[0])

        self.decoder_sessions = [self.InferenceSession(path) for path in self.layer_files.layer_paths]
        self.post_session = None
        self._closed = False
        atexit.register(self.close)

        self.layer_decode_input_shapes = []
        self.layer_decode_input_dtypes = []
        self.layer_prefill_input_shapes = []
        self.layer_prefill_input_dtypes = []
        self.layer_decode_output_names = []
        self.layer_prefill_output_names = []
        for layer_idx, session in enumerate(self.decoder_sessions):
            decode_input_shapes = {x.name: tuple(x.shape) for x in session.get_inputs(shape_group=0)}
            decode_input_dtypes = {x.name: dtype_from_axengine(x.dtype) for x in session.get_inputs(shape_group=0)}
            # Some decode-only AX650 exports hide `indices` / `mask` from get_inputs(),
            # but axengine still validates them as required runtime inputs.
            decode_input_shapes.setdefault("indices", (1, 1))
            decode_input_dtypes.setdefault("indices", np.uint32)
            decode_input_shapes.setdefault("mask", (1, 1))
            decode_input_dtypes.setdefault("mask", bfloat16)
            if layer_idx < len(self.layer_types) and self.layer_types[layer_idx] == "full_attention":
                decode_input_shapes["K_cache"] = (1, self.kv_cache_len, self.full_attn_kv_dim)
                decode_input_shapes["V_cache"] = (1, self.kv_cache_len, self.full_attn_kv_dim)
            self.layer_decode_input_shapes.append(decode_input_shapes)
            self.layer_decode_input_dtypes.append(decode_input_dtypes)
            self.layer_decode_output_names.append([x.name for x in session.get_outputs(shape_group=0)])
            prefill_shape_groups = []
            prefill_dtype_groups = []
            prefill_output_groups = []
            for shape_group in range(1, 64):
                try:
                    prefill_inputs = session.get_inputs(shape_group=shape_group)
                    prefill_shape_groups.append({x.name: tuple(x.shape) for x in prefill_inputs})
                    prefill_dtype_groups.append({x.name: dtype_from_axengine(x.dtype) for x in prefill_inputs})
                    prefill_output_groups.append([x.name for x in session.get_outputs(shape_group=shape_group)])
                except Exception:
                    break
            self.layer_prefill_input_shapes.append(prefill_shape_groups)
            self.layer_prefill_input_dtypes.append(prefill_dtype_groups)
            self.layer_prefill_output_names.append(prefill_output_groups)
        self.decode_input_shapes = self.layer_decode_input_shapes[0]
        self.decode_input_dtypes = self.layer_decode_input_dtypes[0]
        self.prefill_input_shapes = (
            self.layer_prefill_input_shapes[0][0]
            if self.layer_prefill_input_shapes and self.layer_prefill_input_shapes[0]
            else {}
        )
        self.prefill_input_dtypes = (
            self.layer_prefill_input_dtypes[0][0]
            if self.layer_prefill_input_dtypes and self.layer_prefill_input_dtypes[0]
            else {}
        )
        self.prefill_len = int(self.prefill_input_shapes["input"][1]) if "input" in self.prefill_input_shapes else 0
        self.decode_output_names = self.layer_decode_output_names[0]
        self.prefill_output_names = (
            self.layer_prefill_output_names[0][0]
            if self.layer_prefill_output_names and self.layer_prefill_output_names[0]
            else []
        )
        self.hidden_dtype = self.decode_input_dtypes["input"]

    def close(self):
        if self._closed:
            return
        for session in getattr(self, "decoder_sessions", []):
            release_ax_inference_session(session)
        if getattr(self, "post_session", None) is not None:
            release_ax_inference_session(self.post_session)
        if getattr(self, "vision_session", None) is not None:
            release_ax_inference_session(self.vision_session)
        self.decoder_sessions = []
        self.post_session = None
        self.vision_session = None
        self._closed = True

    def resolve_vision_axmodel(self, explicit_path: Optional[str]) -> Optional[str]:
        if explicit_path:
            if not os.path.exists(explicit_path):
                raise FileNotFoundError(f"vision axmodel not found: {explicit_path}")
            return explicit_path
        candidates = [
            os.path.join(self.axmodel_dir, "minicpmv4_6_vision_448.axmodel"),
            os.path.join(os.path.dirname(self.axmodel_dir), "minicpmv4_6_vision_448.axmodel"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def get_vision_session(self):
        if self.vision_axmodel is None:
            raise RuntimeError(
                "vision axmodel is required for image inference; pass --vision-axmodel or place "
                "minicpmv4_6_vision_448.axmodel next to --axmodel-dir"
            )
        if self.vision_session is None:
            self.vision_session = self.InferenceSession(self.vision_axmodel)
        return self.vision_session

    def tokenize_prompt(self, prompt: str) -> List[int]:
        prompt_text = render_text_prompt(prompt)
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def embed_token(self, token_id: int) -> np.ndarray:
        return np.asarray(self.embed_matrix[int(token_id)], dtype=self.hidden_dtype).reshape(1, 1, self.hidden_size)

    def embed_tokens(self, token_ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(list(token_ids), dtype=np.int64)
        return np.asarray(self.embed_matrix[ids], dtype=self.hidden_dtype)

    def build_prompt_inputs(self, prompt: str, image_path: Optional[str] = None) -> Tuple[List[int], str, np.ndarray]:
        if image_path is None:
            prompt_text = render_text_prompt(prompt)
            token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            return token_ids, prompt_text, self.embed_tokens(token_ids)

        image_token_count = image_token_count_for_fixed_shape(
            self.meta.vision_height,
            self.meta.vision_width,
            self.meta.vision_patch_size,
            downsample_mode="16x",
        )
        prompt_text = render_single_image_prompt(prompt, self.meta, image_token_count)
        token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_embeds = self.embed_tokens(token_ids)
        placeholder_positions = [idx for idx, token_id in enumerate(token_ids) if int(token_id) == self.image_token_id]
        vision_features = self.run_vision_image(image_path)
        if len(placeholder_positions) != vision_features.shape[0]:
            raise RuntimeError(
                f"image placeholder count mismatch: placeholders={len(placeholder_positions)} "
                f"vision_tokens={vision_features.shape[0]}"
            )
        prompt_embeds[np.asarray(placeholder_positions, dtype=np.int64)] = np.asarray(
            vision_features, dtype=self.hidden_dtype
        )
        return token_ids, prompt_text, prompt_embeds

    def run_vision_image(self, image_path: str) -> np.ndarray:
        session = self.get_vision_session()
        input_meta = session.get_inputs()[0]
        input_shape = tuple(int(x) for x in input_meta.shape)
        input_dtype = dtype_from_axengine(input_meta.dtype)
        raw_patches = preprocess_fixed_image(
            image_path,
            self.meta.vision_height,
            self.meta.vision_width,
            self.meta.vision_patch_size,
        )
        pixel_values = cast_vision_input(
            raw_patches,
            input_shape,
            input_dtype,
            self.meta.image_mean,
            self.meta.image_std,
        )
        outputs = session.run(None, {input_meta.name: pixel_values})
        image_features = np.asarray(outputs[0])
        image_features = image_features.reshape(-1, self.hidden_size)
        ensure_finite("vision image features", image_features)
        return image_features

    def alloc_layer_states(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        k_states = [
            np.zeros(shapes["K_cache"], dtype=dtypes["K_cache"])
            for shapes, dtypes in zip(self.layer_decode_input_shapes, self.layer_decode_input_dtypes)
        ]
        v_states = [
            np.zeros(shapes["V_cache"], dtype=dtypes["V_cache"])
            for shapes, dtypes in zip(self.layer_decode_input_shapes, self.layer_decode_input_dtypes)
        ]
        return k_states, v_states

    @staticmethod
    def make_feed(shapes: dict, values: dict) -> dict:
        return {name: value for name, value in values.items() if name in shapes}

    def is_linear_layer(self, layer_idx: int) -> bool:
        return layer_idx >= len(self.layer_types) or self.layer_types[layer_idx] != "full_attention"

    def prefill_history_capacity(self, shapes: dict) -> int:
        if not shapes:
            return -1
        input_len = int(shapes.get("input", (1, self.prefill_len))[1])
        mask_shape = shapes.get("mask")
        if mask_shape is not None and len(mask_shape) == 3:
            return max(0, int(mask_shape[-1]) - input_len)
        k_shape = shapes.get("K_cache")
        if k_shape is not None and len(k_shape) >= 2:
            return int(k_shape[1])
        return 0

    def select_prefill_shape_group(self, layer_idx: int, history_len: int) -> int:
        groups = self.layer_prefill_input_shapes[layer_idx]
        if not groups:
            raise RuntimeError(f"layer {layer_idx} has no prefill shape_group")
        if history_len <= 0:
            return 1

        candidates = []
        for offset, shapes in enumerate(groups):
            cap = self.prefill_history_capacity(shapes)
            if cap >= history_len:
                candidates.append((cap, offset + 1))
        if candidates:
            return min(candidates)[1]

        # Linear-attention layers may prune reusable warm groups. Reuse the
        # largest available group when its state shape is independent of history.
        return max(range(1, len(groups) + 1), key=lambda gid: self.prefill_history_capacity(groups[gid - 1]))

    def inspect(self):
        print("hf_model:", self.hf_model)
        print("axmodel_dir:", self.axmodel_dir)
        print("embed_bin:", self.embed_bin)
        print("vision_axmodel:", self.vision_axmodel)
        print("num_layers:", len(self.decoder_sessions))
        print("hidden_size:", self.hidden_size)
        print("vocab_size:", self.vocab_size)
        print("eos_token_id:", self.eos_token_id)
        print("kv_cache_len:", self.kv_cache_len)
        print("prefill_len:", self.prefill_len)
        print("decode_inputs:", sorted((k, v, str(self.decode_input_dtypes[k])) for k, v in self.decode_input_shapes.items()))
        print("decode_outputs:", [(x.name, tuple(x.shape), str(x.dtype)) for x in self.decoder_sessions[0].get_outputs(0)])
        unique_k_shapes = sorted({tuple(spec["K_cache"]) for spec in self.layer_decode_input_shapes})
        unique_v_shapes = sorted({tuple(spec["V_cache"]) for spec in self.layer_decode_input_shapes})
        print("decode_k_cache_shapes:", unique_k_shapes)
        print("decode_v_cache_shapes:", unique_v_shapes)
        if self.prefill_input_shapes:
            print("prefill_group_count_layer0:", len(self.layer_prefill_input_shapes[0]))
            unique_prefill_k_shapes = sorted(
                {
                    tuple(spec["K_cache"])
                    for groups in self.layer_prefill_input_shapes
                    for spec in groups
                    if "K_cache" in spec
                }
            )
            unique_prefill_v_shapes = sorted(
                {
                    tuple(spec["V_cache"])
                    for groups in self.layer_prefill_input_shapes
                    for spec in groups
                    if "V_cache" in spec
                }
            )
            unique_prefill_mask_shapes = sorted(
                {
                    tuple(spec["mask"])
                    for groups in self.layer_prefill_input_shapes
                    for spec in groups
                    if "mask" in spec
                }
            )
            print("prefill_inputs_layer0:", sorted((k, v, str(self.prefill_input_dtypes[k])) for k, v in self.prefill_input_shapes.items()))
            print("prefill_k_cache_shapes:", unique_prefill_k_shapes)
            print("prefill_v_cache_shapes:", unique_prefill_v_shapes)
            print("prefill_mask_shapes:", unique_prefill_mask_shapes)
            print("prefill_outputs:", [(x.name, tuple(x.shape), str(x.dtype)) for x in self.decoder_sessions[0].get_outputs(1)])
        else:
            print("prefill_inputs: []")
            print("prefill_outputs: []")
        post_session = self.get_post_session()
        print("post_inputs:", [(x.name, tuple(x.shape), str(x.dtype)) for x in post_session.get_inputs()])
        print("post_outputs:", [(x.name, tuple(x.shape), str(x.dtype)) for x in post_session.get_outputs()])

    def get_post_session(self):
        if self.post_session is None:
            self.post_session = self.InferenceSession(self.layer_files.post_path)
        return self.post_session

    def run_prefill(self, token_ids: Sequence[int], verbose: bool = False, return_states: bool = False):
        return self.run_prefill_embeds(self.embed_tokens(token_ids), verbose=verbose, return_states=return_states)

    def run_prefill_embeds(self, prompt_embeds: np.ndarray, verbose: bool = False, return_states: bool = False):
        k_states, v_states = self.alloc_layer_states()
        last_hidden = None
        for start in range(0, len(prompt_embeds), self.prefill_len):
            chunk = np.asarray(prompt_embeds[start : start + self.prefill_len], dtype=self.hidden_dtype)
            chunk_len = len(chunk)
            data = np.zeros((1, self.prefill_len, self.hidden_size), dtype=self.hidden_dtype)
            data[0, :chunk_len, :] = chunk

            for layer_idx, session in enumerate(self.decoder_sessions):
                if self.is_linear_layer(layer_idx):
                    out = np.zeros_like(data)
                    for j in range(chunk_len):
                        hidden = data[:, j : j + 1, :]
                        hidden = self.run_single_layer_decode_step(
                            layer_idx,
                            hidden,
                            start + j,
                            k_states,
                            v_states,
                            verbose=False,
                        )
                        out[:, j : j + 1, :] = hidden
                    data = out
                    if verbose:
                        print(
                            f"prefill chunk={start // self.prefill_len} layer={layer_idx} "
                            f"linear_decode_replay tokens={chunk_len} {tensor_stats(data)}"
                        )
                    continue

                shape_group = self.select_prefill_shape_group(layer_idx, start)
                layer_shapes = self.layer_prefill_input_shapes[layer_idx][shape_group - 1]
                layer_dtypes = self.layer_prefill_input_dtypes[layer_idx][shape_group - 1]
                if not layer_shapes:
                    raise RuntimeError(f"layer {layer_idx} has no prefill shape_group={shape_group}")
                history_cap = self.prefill_history_capacity(layer_shapes)
                history_len = min(start, history_cap)
                indices = None
                if "indices" in layer_shapes:
                    indices = np.zeros(layer_shapes["indices"], dtype=layer_dtypes["indices"])
                    indices.reshape(-1)[:chunk_len] = np.arange(start, start + chunk_len, dtype=np.uint32)
                mask = None
                if "mask" in layer_shapes:
                    if self.layer_types[layer_idx] == "full_attention" and len(layer_shapes["mask"]) == 3:
                        mask = np.full(layer_shapes["mask"], -65536.0, dtype=np.float32)
                        for q in range(chunk_len):
                            mask[:, q, : history_len + q + 1] = 0.0
                    else:
                        mask = np.zeros(layer_shapes["mask"], dtype=np.float32)
                        mask.reshape(-1)[:chunk_len] = 1.0
                    mask = mask.astype(layer_dtypes["mask"])

                k_feed = k_states[layer_idx]
                v_feed = v_states[layer_idx]
                if self.layer_types[layer_idx] == "full_attention":
                    k_feed = np.zeros(layer_shapes["K_cache"], dtype=layer_dtypes["K_cache"])
                    v_feed = np.zeros(layer_shapes["V_cache"], dtype=layer_dtypes["V_cache"])
                    if history_len > 0:
                        k_feed[:, :history_len, :] = k_states[layer_idx][:, :history_len, :]
                        v_feed[:, :history_len, :] = v_states[layer_idx][:, :history_len, :]
                outputs = session.run(
                    None,
                    self.make_feed(
                        layer_shapes,
                        {
                            "K_cache": k_feed,
                            "V_cache": v_feed,
                            **({"indices": indices} if indices is not None else {}),
                            "input": data.astype(layer_dtypes["input"], copy=False),
                            **({"mask": mask} if mask is not None else {}),
                        },
                    ),
                    shape_group=shape_group,
                )
                output_map = dict(zip(self.layer_prefill_output_names[layer_idx][shape_group - 1], outputs))
                k_out = output_map.get("K_cache_out")
                if k_out is not None:
                    if self.layer_types[layer_idx] == "full_attention":
                        k_states[layer_idx][:, start : start + chunk_len, :] = k_out[:, :chunk_len, :]
                    else:
                        k_states[layer_idx] = k_out
                v_out = output_map.get("V_cache_out")
                if v_out is not None:
                    if self.layer_types[layer_idx] == "full_attention":
                        v_states[layer_idx][:, start : start + chunk_len, :] = v_out[:, :chunk_len, :]
                    else:
                        v_states[layer_idx] = v_out
                data = output_map["output"]
                ensure_finite(f"prefill layer {layer_idx} output", data)
                if verbose:
                    print(
                        f"prefill chunk={start // self.prefill_len} layer={layer_idx} "
                        f"shape_group={shape_group} history_len={history_len} {tensor_stats(data)}"
                    )
            last_hidden = data[:, chunk_len - 1 : chunk_len, :]
        if return_states:
            return k_states, v_states, last_hidden
        return last_hidden

    def run_single_layer_decode_step(
        self,
        layer_idx: int,
        hidden: np.ndarray,
        position: int,
        k_states: List[np.ndarray],
        v_states: List[np.ndarray],
        verbose: bool = False,
    ) -> np.ndarray:
        session = self.decoder_sessions[layer_idx]
        layer_shapes = self.layer_decode_input_shapes[layer_idx]
        layer_dtypes = self.layer_decode_input_dtypes[layer_idx]
        indices = None
        if "indices" in layer_shapes:
            indices = np.zeros(layer_shapes["indices"], dtype=layer_dtypes["indices"])
            indices.reshape(-1)[0] = position
        mask = None
        if "mask" in layer_shapes:
            if self.layer_types[layer_idx] == "full_attention" and len(layer_shapes["mask"]) == 3:
                mask = np.full(layer_shapes["mask"], -65536.0, dtype=np.float32)
                valid_past = min(position, layer_shapes["mask"][-1] - 1)
                if valid_past > 0:
                    mask[:, :, :valid_past] = 0.0
                mask[:, :, -1:] = 0.0
            else:
                mask = np.ones(layer_shapes["mask"], dtype=np.float32)
            mask = mask.astype(layer_dtypes["mask"])
        outputs = session.run(
            None,
            self.make_feed(
                layer_shapes,
                {
                    "K_cache": k_states[layer_idx],
                    "V_cache": v_states[layer_idx],
                    **({"indices": indices} if indices is not None else {}),
                    "input": hidden,
                    **({"mask": mask} if mask is not None else {}),
                },
            ),
            shape_group=0,
        )
        output_map = dict(zip(self.layer_decode_output_names[layer_idx], outputs))
        k_out = output_map.get("K_cache_out")
        if k_out is not None:
            if self.layer_types[layer_idx] == "full_attention" and k_states[layer_idx].shape != k_out.shape:
                pos_end = position + k_out.shape[1]
                k_states[layer_idx][:, position:pos_end, :] = k_out
            else:
                k_states[layer_idx] = k_out
        v_out = output_map.get("V_cache_out")
        if v_out is not None:
            if self.layer_types[layer_idx] == "full_attention" and v_states[layer_idx].shape != v_out.shape:
                pos_end = position + v_out.shape[1]
                v_states[layer_idx][:, position:pos_end, :] = v_out
            else:
                v_states[layer_idx] = v_out
        out = output_map["output"]
        ensure_finite(f"decode step={position} layer={layer_idx} output", out)
        if verbose:
            print(f"decode step={position} layer={layer_idx} {tensor_stats(out)}")
        return out

    def run_decode_step(
        self,
        hidden: np.ndarray,
        position: int,
        k_states: List[np.ndarray],
        v_states: List[np.ndarray],
        verbose: bool = False,
    ) -> np.ndarray:
        data = hidden
        for layer_idx in range(len(self.decoder_sessions)):
            data = self.run_single_layer_decode_step(layer_idx, data, position, k_states, v_states, verbose=verbose)
        return data

    def decode_replay_prompt(
        self,
        token_ids: Sequence[int],
        limit_prompt_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        return self.decode_replay_prompt_embeds(
            self.embed_tokens(token_ids),
            limit_prompt_tokens=limit_prompt_tokens,
            verbose=verbose,
        )

    def decode_replay_prompt_embeds(
        self,
        prompt_embeds: np.ndarray,
        limit_prompt_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        if limit_prompt_tokens is not None:
            prompt_embeds = prompt_embeds[:limit_prompt_tokens]
        k_states, v_states = self.alloc_layer_states()
        last_hidden = None
        for pos, hidden in enumerate(prompt_embeds):
            hidden = np.asarray(hidden, dtype=self.hidden_dtype).reshape(1, 1, self.hidden_size)
            last_hidden = self.run_decode_step(hidden, pos, k_states, v_states, verbose=verbose)
        return k_states, v_states, last_hidden

    def run_post(self, hidden: np.ndarray) -> np.ndarray:
        logits = self.get_post_session().run(None, {"input": hidden})[0]
        ensure_finite("post logits", logits)
        return logits

    def greedy_next_token(self, hidden: np.ndarray) -> int:
        logits = self.run_post(hidden)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        return int(np.argmax(logits))

    def generate(self, token_ids: Sequence[int], max_new_tokens: int, prompt_mode: str, verbose: bool = False):
        return self.generate_from_embeds(
            token_ids,
            self.embed_tokens(token_ids),
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
            verbose=verbose,
        )

    def generate_from_embeds(
        self,
        token_ids: Sequence[int],
        prompt_embeds: np.ndarray,
        max_new_tokens: int,
        prompt_mode: str,
        verbose: bool = False,
    ):
        if prompt_mode == "prefill":
            k_states, v_states, last_hidden = self.run_prefill_embeds(prompt_embeds, verbose=verbose, return_states=True)
        else:
            k_states, v_states, last_hidden = self.decode_replay_prompt_embeds(prompt_embeds, verbose=verbose)
        prompt_len = len(token_ids)
        generated = []
        for step in range(max_new_tokens):
            next_token = self.greedy_next_token(last_hidden)
            generated.append(next_token)
            print(f"gen step={step} token_id={next_token} piece={self.decode_tokens([next_token])!r}")
            if next_token == self.eos_token_id:
                break
            last_hidden = self.run_decode_step(
                self.embed_token(next_token),
                prompt_len + step,
                k_states,
                v_states,
                verbose=verbose,
            )

        print("generated_ids:", generated)
        print("generated_text:", self.decode_tokens(generated))


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiniCPM-V-4.6 AX650 axmodel Python runner/debugger"
    )
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="Tokenizer/config path")
    parser.add_argument("--axmodel-dir", required=True, help="Compiled axmodel directory")
    parser.add_argument("--embed-bin", default=None, help="Embedding bf16 bin path")
    parser.add_argument("--vision-axmodel", default=None, help="Optional fixed-shape vision axmodel path for image inference")
    parser.add_argument(
        "--mode",
        default="inspect",
        choices=["inspect", "prefill", "decode_replay", "generate"],
        help="Execution mode",
    )
    parser.add_argument("--prompt", default="你好，请做一个简短自我介绍。", help="User prompt")
    parser.add_argument("--prompt-file", default=None, help="Read user prompt from a UTF-8 text file")
    parser.add_argument("--image", default=None, help="Optional local image path; enables single-image prompt injection")
    parser.add_argument("--max-layers", type=int, default=None, help="Only load the first N decoder layers")
    parser.add_argument(
        "--limit-prompt-tokens",
        type=int,
        default=None,
        help="Only consume the first N prompt tokens in decode_replay mode",
    )
    parser.add_argument(
        "--prompt-mode",
        default="decode_replay",
        choices=["decode_replay", "prefill"],
        help="How to consume the prompt before generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Generation length for --mode generate")
    parser.add_argument("--kv-cache-len", type=int, default=255, help="Decode KV cache length used at compile time")
    parser.add_argument("--verbose", action="store_true", help="Print per-layer tensor stats")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompt = f.read()
    runner = MiniCPMTextAxModelRunner(
        hf_model=args.hf_model,
        axmodel_dir=args.axmodel_dir,
        embed_bin=args.embed_bin,
        max_layers=args.max_layers,
        vision_axmodel=args.vision_axmodel,
        kv_cache_len=args.kv_cache_len,
    )
    try:
        token_ids, prompt_text, prompt_embeds = runner.build_prompt_inputs(args.prompt, image_path=args.image)
        print("prompt_token_count:", len(token_ids))
        print("prompt_token_ids:", token_ids)
        print("prompt_template_repr:", prompt_text.encode("unicode_escape").decode())

        if args.mode == "inspect":
            runner.inspect()
            return

        if args.mode == "prefill":
            hidden = runner.run_prefill_embeds(prompt_embeds, verbose=args.verbose)
            print("prefill_last_hidden:", tensor_stats(hidden))
            logits = runner.run_post(hidden)
            print("post_logits:", tensor_stats(logits))
            print("greedy_next_token:", runner.greedy_next_token(hidden))
            return

        if args.mode == "decode_replay":
            _, _, hidden = runner.decode_replay_prompt_embeds(
                prompt_embeds,
                limit_prompt_tokens=args.limit_prompt_tokens,
                verbose=args.verbose,
            )
            print("decode_replay_last_hidden:", tensor_stats(hidden))
            logits = runner.run_post(hidden)
            print("post_logits:", tensor_stats(logits))
            print("greedy_next_token:", runner.greedy_next_token(hidden))
            return

        runner.generate_from_embeds(
            token_ids,
            prompt_embeds,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
            verbose=args.verbose,
        )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
