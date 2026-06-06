# MiniCPM-V-4.6.axera

> `openbmb/MiniCPM-V-4.6` 在 `AX650 / AX650N` 上的复现工程。

本仓库面向需要重新编译模型、核对精度或理解 AXERA 适配流程的开发者。  
如果你只需要直接在板端运行，请优先使用 Hugging Face 发布包：<https://huggingface.co/AXERA-TECH/MiniCPM-V-4.6>。

## 适用范围

- 平台：`AX650 / AX650N`
- Runtime：`axllm serve` / `axllm run`
- LLM 主干：`Qwen3.5 dense`，非 GPTQ 权重
- 视觉编码器：固定 shape `448x448`
- 上下文配置：`prefill_len=128`，`kv_cache_len=2047`，`prefill_max_token_num=1280`

当前发布包已经验证：

- 文本对话
- 长 prompt 多 chunk prefill
- 单图理解
- 视频理解

当前仓库提供：

- LLM 主干的 `pulsar2 llm_build` 编译脚本
- 板端 Python 文本 LLM 调试脚本
- x86 HuggingFace/Torch 文本参考脚本
- tokenizer/config 等运行所需的小型元数据

当前仓库不提交任何 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。  
当前仓库没有整理完整的视觉导出与重新编译脚本；视觉部分请直接使用 Hugging Face 发布包中的已验证产物。

## 仓库职责

```text
.
├── assets/          # 示例素材
├── model_convert/   # LLM 编译脚本和转换说明
├── python/          # 板端调试脚本、Torch 参考脚本、tokenizer/config 元数据
└── README.md
```

根目录 `README.md` 说明如何理解仓库内容和复现板端验证。  
重新执行 `pulsar2 llm_build` 请阅读 [model_convert/README.md](./model_convert/README.md)。

## 与发布包的关系

本仓库是开发侧复现仓库，包含转换脚本、调试脚本和中间验证产物。  
面向最终用户的可直接运行包是 Hugging Face 仓库：

```text
AXERA-TECH/MiniCPM-V-4.6
```

发布包采用根目录直接 `axllm serve .` 的布局；本 `.axera` 仓库则保留 `python/` 和 `model_convert/` 结构，便于复现与调试。

## 环境准备

### 编译环境

`pulsar2 llm_build` 需要 AXERA NPU 开发环境。示例：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODEBASE_ROOT"
source script/npu_dev
```

本仓库的脚本默认使用内部验证路径；外部用户应通过环境变量覆盖：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export INPUT_PATH=/path/to/openbmb/MiniCPM-V-4.6
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
```

### 板端运行环境

`.axmodel` 只能在 AX650 板端运行，不要在 x86 服务器上执行。

Python 调试脚本需要：

- `pyaxengine`
- `numpy`
- `ml_dtypes`
- `transformers`
- `pillow`

如果只是使用最终发布包的 `axllm serve`，不需要 Python 调试依赖。

## 重新编译 LLM

在仓库根目录执行：

```bash
cd model_convert
./llm_build_ax650.sh
```

默认输出到本地工作区：

```text
python/MiniCPM-V-4.6_axmodel/
```

`*_axmodel/` 已被 `.gitignore` 忽略，不应提交到本仓库。

也可以显式指定输出目录：

```bash
./llm_build_ax650.sh /path/to/output_axmodel
```

脚本默认启用：

```bash
FLOAT_MATMUL_USE_CONV_EU=1
```

该选项在 AX650 上可以明显改善 TTFT。

## 板端调试

### Python 文本 LLM 调试

以下命令用于调试 LLM `.axmodel`，不覆盖最终发布包的多模态能力：

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

### 发布包 `axllm serve` 验证

推荐在最终发布包目录执行：

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

## 本仓库内容

本仓库只保留复现所需的脚本和小型元数据：

```text
python/
├── MiniCPM-V-4.6/              # tokenizer / config / processor 相关文件，不包含原始权重
├── infer_axmodel.py
├── infer_torch.py
└── dump_layer0_reference.py
```

说明：

- `python/MiniCPM-V-4.6_axmodel/` 是本地编译输出目录，不提交仓库
- VIT axmodel、LLM axmodel、embedding 等产物只存在于 Hugging Face 发布包或用户本地编译输出中
- 原始 Hugging Face 权重不随本仓库发布
- 最终用户部署请使用 Hugging Face 发布包，而不是直接把本仓库当作运行包
