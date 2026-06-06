# MiniCPM-V-4.6 模型转换与编译

本文档描述 `openbmb/MiniCPM-V-4.6` 在 AXERA 平台上的开发侧工作流，覆盖以下内容：

- LLM 主干 `pulsar2 llm_build` 编译
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

`pulsar2 llm_build` 相关命令需要 AXERA NPU 开发环境。请先准备：

- npu-codebase：包含 `script/npu_dev` 和 `pulsar2 llm_build` 集成代码
- 原始 Hugging Face 模型目录：`openbmb/MiniCPM-V-4.6`
- 可用的 `npu` conda 环境或等价 Python 环境

脚本通过环境变量接收路径：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export INPUT_PATH=/path/to/openbmb/MiniCPM-V-4.6
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
```

其中：

- `CODEBASE_ROOT`：npu-codebase 根目录
- `DEPLOY_ROOT`：模型部署工作区根目录
- `INPUT_PATH`：原始 Hugging Face 模型目录
- `CONDA_SH` / `CONDA_ENV`：用于激活编译环境

下文所有脚本默认以 `model_convert/` 为当前工作目录。

## 推荐顺序

如果你的目标是复现当前 LLM 编译流程，建议按下面顺序执行：

1. 准备原始 Hugging Face 权重目录
2. 设置 `CODEBASE_ROOT`、`DEPLOY_ROOT`、`INPUT_PATH`、`CONDA_SH`、`CONDA_ENV`
3. 执行 `./llm_build_ax650.sh`
4. 在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载
5. 将编译输出整理到 Hugging Face 发布包布局
6. 使用发布包中的 `axllm serve` 做端到端验证

## 已验证配置

当前已验证的 LLM 编译配置：

| Item | Value |
|---|---|
| `model_type` | `qwen3_5_text` |
| `hidden_state_type` | `bf16` |
| `prefill_len` | `128` |
| `kv_cache_len` | `2047` |
| `last_kv_cache_len` | `128, 256, 384, 512, 640, 768, 896, 1024, 1152` |
| `chip` | `AX650` |
| `parallel` | `32` |
| MatMul 优化 | `FLOAT_MATMUL_USE_CONV_EU=1` |

`FLOAT_MATMUL_USE_CONV_EU=1` 是当前 AX650 验证中使用的配置，可明显改善 TTFT。

## 准备原始模型

从 Hugging Face 下载原始模型权重。以下示例假设克隆到仓库同级工作区，并用 `$INPUT_PATH` 引用：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6 ../../Minicpm-V-4.6-hf-original/MiniCPM-V-4.6
export INPUT_PATH=../../Minicpm-V-4.6-hf-original/MiniCPM-V-4.6
```

原始权重不提交到本仓库。

## 编译 LLM axmodel

在 `model_convert/` 目录执行：

```bash
./llm_build_ax650.sh
```

默认输出到：

```text
../python/MiniCPM-V-4.6_axmodel
```

也可以显式指定输出目录：

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
