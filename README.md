# MiniCPM-V-4.6.axera

> `openbmb/MiniCPM-V-4.6` 在 `AX650 / NPU3` 上的复现工程。

本仓库的目标是帮助用户完成两类工作：

- 复现板端运行与精度验证
- 重新导出固定 shape Vision ONNX，并重新编译 `MiniCPM-V-4.6` 的 LLM 主干产物

本仓库面向需要完整复现实验过程、重新编译模型或核对精度的用户。

> 当前仓库只保存复现所需的脚本、tokenizer/config 和示例素材，不提交 `.axmodel`、embedding、ONNX、`safetensors` 等编译或推理产物。如果你希望直接体验面向用户的实际 Demo，请参考 Hugging Face 发布页：<https://huggingface.co/AXERA-TECH/MiniCPM-V-4.6>。

## 适用范围

- 平台：`AX650 / NPU3`
- 支持的板端能力：
  - 文本对话
  - 长 prompt 多 chunk prefill 调试
  - 单图图文调试
  - 发布包中的视频理解验证
- `python/infer_axmodel.py` 负责板端文本链路调试，以及在提供 `minicpmv4_6_vision_448.axmodel` 时做单图 prompt 注入调试
- `python/infer_torch.py` 为 `x86 + HuggingFace/Torch` 官方多模态参考脚本，依赖 `transformers>=5.7.0`
- `model_convert/export_onnx.py` 提供固定 shape Vision ONNX 导出脚本，发布包默认使用 `448x448 / 16x`
- `model_convert/prepare_calibration.py` 与 `model_convert/pulsar2_configs/config_vision_448_npu3.json` 提供固定 shape Vision calibration 与 `pulsar2 build` 所需的仓库内入口
- 当前 Hugging Face 发布包中的 `minicpmv4_6_vision_448.axmodel` 已验证为 `NPU3` 版本
- 当前仓库可以重新生成一个可用的 Vision `axmodel`，但如果你的目标是严格复现 Hugging Face 发布包中的同名产物，请优先使用发布包里的已验证 Vision `axmodel`

## 仓库职责

```text
.
├── python/         # tokenizer/config、板端调试脚本、Torch 参考脚本
├── model_convert/  # Vision ONNX 导出、LLM 主干编译与开发侧说明
└── assets/         # Demo 或 smoke test 使用的示例素材
```

根目录 `README.md` 负责说明“如何准备运行目录并在板端复现结果”。
如果你需要重新导出 Vision ONNX 或重新编译 LLM 主干 axmodel，请阅读 [model_convert/README.md](./model_convert/README.md)。

## 运行前准备

### 1. 准备运行目录

在执行板端命令前，请确认以下文件已经准备好：

```text
python/
├── MiniCPM-V-4.6/                   # tokenizer / config / processor 相关文件
├── MiniCPM-V-4.6_axmodel/           # 用户本地编译得到的 LLM 运行目录，不提交仓库
├── infer_axmodel.py
└── minicpmv4_6_vision_448.axmodel   # 可选，单图调试时需要；通常直接使用发布包中的已验证文件
```

如果你只希望直接运行完整的文本、图像和视频能力，请使用 Hugging Face 发布包：

```text
AXERA-TECH/MiniCPM-V-4.6
```

发布包已经包含 `bin/axllm`、LLM axmodel、VIT axmodel、embedding 和 runtime config，可以直接执行 `axllm serve .`。

### 2. 安装板端依赖

板端运行 `python/infer_axmodel.py` 需要以下 Python 依赖：

- `pyaxengine`
- `numpy`
- `ml_dtypes`
- `transformers`
- `pillow`

如果板端无法直接联网安装，可以先把依赖准备到某个目录，再通过 `PYTHONPATH` 注入：

```bash
export PYDEPS_DIR=/path/to/python_deps
export PYTHONPATH="${PYDEPS_DIR:+$PYDEPS_DIR:}$PYTHONPATH"
```

上面的 `PYDEPS_DIR` 由用户自行决定；README 不假设任何固定私有路径。
如果依赖已经直接安装到当前 Python 环境，可以跳过这一步。

### 3. 多模态运行说明

本仓库的 Python 调试脚本不追求完整替代 `axllm serve`。
当前板端 `python/infer_axmodel.py` 主要覆盖：

- 文本 LLM 主干逐层调试
- 单图 prompt 占位符注入与 Vision `axmodel` 输出检查

图像和视频的完整端到端能力仍建议通过 Hugging Face 发布包中的 `axllm serve` 验证。

## 板端复现

以下命令默认在仓库根目录执行，并且板端已经可以访问本仓库文件。
`python/infer_axmodel.py` 首次启动时会预加载 LLM 子模型；在 AX650 板端通常需要先等待一段时间，随后才会开始生成。

### 文本 LLM 调试

```bash
cd python

python3 infer_axmodel.py \
  --hf-model ./MiniCPM-V-4.6 \
  --axmodel-dir ./MiniCPM-V-4.6_axmodel \
  --mode generate \
  --prompt "1+1等于几？请直接回答。" \
  --prompt-mode prefill \
  --max-new-tokens 16 \
  --kv-cache-len 2047
```

说明：

- `MiniCPM-V-4.6_axmodel/` 需要由用户本地编译生成，或从已验证产物复制到该目录
- 文本调试脚本不加载 VIT，不支持图片或视频输入
- 如果你本地使用的是其他 LLM 编译输出目录，请通过 `--axmodel-dir` 显式传入

### 板端单图 `.axmodel` 调试

```bash
cd python

python3 infer_axmodel.py \
  --hf-model ./MiniCPM-V-4.6 \
  --axmodel-dir ./MiniCPM-V-4.6_axmodel \
  --vision-axmodel ./minicpmv4_6_vision_448.axmodel \
  --image ../assets/smoke_image.png \
  --mode generate \
  --prompt "请简要描述这张图片。" \
  --prompt-mode prefill \
  --max-new-tokens 64 \
  --kv-cache-len 2047
```

说明：

- 该模式默认覆盖单图固定 `448x448 / 16x` prompt 注入调试
- 如果不传 `--vision-axmodel`，脚本只执行纯文本链路
- 视频理解仍建议使用发布包 `axllm serve`

### x86 Torch 多模态参考

```bash
cd python

python3 infer_torch.py \
  --model-path /path/to/openbmb/MiniCPM-V-4.6 \
  --prompt "1+1等于几？请直接回答。"
```

说明：

- 该脚本依赖 `transformers>=5.7.0`
- 文本、单图和视频都通过官方 Hugging Face `processor/model.generate()` 路径执行
- 该命令不属于板端部署流程

单图示例：

```bash
python3 infer_torch.py \
  --model-path /path/to/openbmb/MiniCPM-V-4.6 \
  --image ../assets/smoke_image.png \
  --prompt "请简要描述这张图片。"
```

### 发布包 `axllm serve` 验证

推荐在 Hugging Face 发布包目录执行：

```bash
cd /path/to/MiniCPM-V-4.6
chmod +x ./bin/axllm
./bin/axllm serve . --port 18080
```

文本请求：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AXERA-TECH/MiniCPM-V-4.6-AX650-C128-P1152-CTX2047",
    "messages": [
      {"role": "user", "content": "1+1等于几？只输出数字。"}
    ],
    "max_tokens": 32,
    "temperature": 0
  }'
```

图像和视频请求请参考 Hugging Face 发布包 README。

## 模型转换入口

本仓库的重新导出与重新编译流程统一放在 [model_convert/README.md](./model_convert/README.md)：

- 固定 shape Vision ONNX 导出
- BF16 / GPTQ 原始权重准备
- `pulsar2 llm_build` 编译
- 编译输出目录与板端加载检查

根目录 `README.md` 不重复展开这些开发侧命令。
