# MiniCPM-V-4.6 模型转换与编译

本文档描述 `openbmb/MiniCPM-V-4.6` 及其 GPTQ 版本在 AXERA 平台上的开发侧工作流，覆盖以下内容：

- LLM 主干 `pulsar2 llm_build` 编译
- BF16 / GPTQ 输入权重来源
- 编译输出目录约定
- 编译产物的板端加载检查
- 与 Hugging Face 发布包的产物同步关系

本文档默认面向开发者使用，所有命令默认在 `model_convert/` 目录下执行。

> 当前仓库不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。编译输出目录已通过 `.gitignore` 忽略。

## 目录说明

```text
model_convert/
├── README.md
└── llm_build_ax650.sh      # AX650 LLM 主干编译脚本
```

当前仓库只提供 LLM 主干编译脚本。  
`MiniCPM-V-4.6` 的视觉编码器产物已经在 Hugging Face 发布包中提供，但本目录不包含 VIT ONNX 导出、校准或重新编译脚本。

## 环境准备

`pulsar2 llm_build` 相关命令需要 AXERA NPU 编译环境。请先准备：

- 可直接执行 `pulsar2 llm_build` 的 shell 环境
- 原始 Hugging Face 模型目录：`openbmb/MiniCPM-V-4.6` 或 `openbmb/MiniCPM-V-4.6-GPTQ`
- 已安装 `pulsar2 llm_build` 依赖的 Python/conda 环境

脚本通过环境变量接收路径：

```bash
export INPUT_PATH=/path/to/openbmb/MiniCPM-V-4.6

# 如果当前 shell 还没有进入编译环境，可以额外设置：
# export CONDA_SH=/path/to/conda.sh
# export CONDA_ENV=npu
```

其中：

- `INPUT_PATH`：原始 Hugging Face 模型目录；可以是 BF16 版本，也可以是 GPTQ 版本
- `CONDA_SH` / `CONDA_ENV`：可选，用于激活编译环境；如果当前 shell 已经在正确环境中，可以不设置

下文所有脚本默认以 `model_convert/` 为当前工作目录。

## 推荐顺序

如果你的目标是复现当前 LLM 编译流程，建议按下面顺序执行：

1. 准备原始 Hugging Face 权重目录
2. 确认当前 shell 可以执行 `pulsar2 llm_build`
3. 设置 `INPUT_PATH`，必要时设置 `CONDA_SH`、`CONDA_ENV`
4. 执行 `./llm_build_ax650.sh`
5. 在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载
6. 将编译输出整理到 Hugging Face 发布包布局
7. 使用发布包中的 `axllm serve` 做端到端验证

## 已验证配置

当前已验证的 LLM 编译配置：

| Item | Value |
|---|---|
| 输入权重 | `openbmb/MiniCPM-V-4.6` / `openbmb/MiniCPM-V-4.6-GPTQ` |
| GPTQ 量化格式 | W4A16 GPTQModel upstream checkpoint |
| `model_type` | `qwen3_5_text` |
| `hidden_state_type` | `bf16` |
| `prefill_len` | `128` |
| `kv_cache_len` | `2047` |
| `last_kv_cache_len` | `128, 256, 384, 512, 640, 768, 896, 1024, 1152` |
| `chip` | `AX650` |
| `parallel` | `32` |
| MatMul 优化 | `FLOAT_MATMUL_USE_CONV_EU=1` |

`FLOAT_MATMUL_USE_CONV_EU=1` 是当前 AX650 验证中使用的配置，可明显改善 TTFT。

## 准备输入模型

### BF16 权重

从 Hugging Face 下载 BF16 原始模型权重，并用 `$INPUT_PATH` 指向该目录：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6 /path/to/original/MiniCPM-V-4.6
export INPUT_PATH=/path/to/original/MiniCPM-V-4.6
```

### GPTQ 权重

GPTQ 输入权重从 `openbmb/MiniCPM-V-4.6-GPTQ` 获取。该仓库是官方提供的 W4A16 GPTQModel 量化版本：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ /path/to/original/MiniCPM-V-4.6-GPTQ
export INPUT_PATH=/path/to/original/MiniCPM-V-4.6-GPTQ
```

BF16 和 GPTQ 原始权重都不提交到本仓库。

## 编译 LLM axmodel

在 `model_convert/` 目录执行：

```bash
./llm_build_ax650.sh
```

默认输出到：

```text
../python/MiniCPM-V-4.6_axmodel
```

如果当前 `INPUT_PATH` 指向 GPTQ 权重，建议显式指定 GPTQ 输出目录，避免和 BF16 产物混淆：

```bash
./llm_build_ax650.sh ../python/MiniCPM-V-4.6-GPTQ_axmodel
```

也可以显式指定任意输出目录：

```bash
./llm_build_ax650.sh /path/to/output_axmodel
```

等价核心命令如下：

```bash
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_type qwen3_5_text \
  --hidden_state_type bf16 \
  --prefill_len 128 \
  --kv_cache_len 2047 \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --last_kv_cache_len 768 \
  --last_kv_cache_len 896 \
  --last_kv_cache_len 1024 \
  --last_kv_cache_len 1152 \
  --chip AX650 \
  -c 0 \
  --parallel 32
```

## 输出目录说明

编译完成后，输出目录通常包含：

```text
MiniCPM-V-4.6_axmodel/
├── qwen3_5_text_p128_l0_together.axmodel
├── ...
├── qwen3_5_text_p128_l23_together.axmodel
├── qwen3_5_text_post.axmodel
├── model.embed_tokens.weight.bfloat16.bin
├── minicpm_v46_tokenizer.txt
├── config.json
└── post_config.json
```

GPTQ 编译输出的文件名结构相同，建议使用独立目录名，例如：

```text
MiniCPM-V-4.6-GPTQ_axmodel/
```

这些文件属于编译产物，不提交到 `.axera` 仓库。  
如果需要发布，请整理到 Hugging Face 发布包的最终布局中。

## 板端加载检查

`.axmodel` 只能在 AX650 板端执行。可以先检查单个子图能否加载：

```bash
cd /path/to/MiniCPM-V-4.6_axmodel
/opt/bin/ax_run_model -m qwen3_5_text_p128_l0_together.axmodel -g 0 --skip-running
```

端到端验证建议使用 Hugging Face 发布包：

```bash
cd /path/to/MiniCPM-V-4.6
./bin/axllm serve . --port 18080
```

图像和视频能力需要发布包中的 VIT axmodel 与 runtime config；本 `model_convert/` 目录不生成这些文件。
