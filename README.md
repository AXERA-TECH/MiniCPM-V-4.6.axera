# MiniCPM-V-4.6.axera

> `openbmb/MiniCPM-V-4.6` 在 `AX650 / AX650N` 上的复现工程。

本仓库的目标是帮助用户完成两类工作：

- 复现板端运行与精度验证
- 重新编译 `MiniCPM-V-4.6` 的 LLM 主干产物

本仓库面向需要完整复现实验过程、重新编译模型或核对精度的用户。

> 当前仓库只保存复现所需的脚本、tokenizer/config 和示例素材，不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。如果你希望直接体验面向用户的实际 Demo，请参考 Hugging Face 发布页：<https://huggingface.co/AXERA-TECH/MiniCPM-V-4.6>。

## 适用范围

- 平台：`AX650 / AX650N`
- 支持的板端能力：
  - 文本对话
  - 长 prompt 多 chunk prefill
  - 单图图文对话
  - 视频理解
- 当前 `python/infer_axmodel.py` 仅用于 LLM 文本链路调试，不提供图像或视频端到端推理
- 当前仓库不提供 VIT 重新导出、校准和编译脚本；视觉产物请使用 Hugging Face 发布包中的已验证 axmodel
- `python/infer_torch.py` 脚本仅用于 `x86 + HuggingFace/Torch` 文本参考验证，不属于板端复现主流程

## 仓库职责

```text
.
├── python/         # tokenizer/config、文本调试脚本和 Torch 参考脚本
├── model_convert/  # LLM 主干编译脚本和转换说明
└── assets/         # Demo 或 smoke test 使用的示例素材
```

根目录 `README.md` 负责说明“如何准备运行目录并在板端复现结果”。  
如果你需要重新编译 LLM 主干 axmodel，请阅读 [model_convert/README.md](./model_convert/README.md)。

## 运行前准备

### 1. 准备运行目录

在执行板端命令前，请确认以下文件已经准备好：

```text
python/
├── MiniCPM-V-4.6/              # tokenizer / config / processor 相关文件
├── MiniCPM-V-4.6_axmodel/      # 用户本地编译得到的 LLM 运行目录，不提交仓库
└── infer_axmodel.py
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

本仓库的 Python 调试脚本不负责图像或视频端到端推理。  
图像和视频能力通过 `axllm serve` 在 Hugging Face 发布包中验证；对应 VIT axmodel 和 runtime config 也只随发布包提供。

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

### x86 Torch 文本参考

```bash
cd python

python3 infer_torch.py \
  --model-path /path/to/openbmb/MiniCPM-V-4.6 \
  --prompt "1+1等于几？请直接回答。"
```

说明：

- 该命令用于对齐 tokenizer、chat template 和文本输出
- 该命令不属于板端部署流程

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

## 模型转换

本仓库提供 LLM 主干编译脚本：

```bash
cd model_convert
export INPUT_PATH=/path/to/original/MiniCPM-V-4.6
./llm_build_ax650.sh
```

脚本默认输出到：

```text
python/MiniCPM-V-4.6_axmodel/
```

`*_axmodel/`、`.axmodel`、embedding `.bin`、ONNX 和 safetensors 均已被 `.gitignore` 忽略，不应提交到本仓库。

如果你需要重新执行编译，请阅读 [model_convert/README.md](./model_convert/README.md)。
