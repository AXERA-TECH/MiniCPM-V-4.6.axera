# MiniCPM-V-4.6 模型转换与编译

本文档描述 `openbmb/MiniCPM-V-4.6` 及其 GPTQ 版本在 AXERA 平台上的开发侧工作流，覆盖以下内容：

- LLM 主干 `pulsar2 llm_build` 编译
- 固定 shape Vision ONNX 导出（发布包默认 `448x448 / 16x`）
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
├── requirements.txt
├── export_onnx.py          # 固定 shape Vision ONNX 导出
└── llm_build_ax650.sh      # AX650 LLM 主干编译脚本
```

当前仓库提供：

- LLM 主干 `pulsar2 llm_build` 编译脚本
- 支持固定 shape Vision ONNX 导出，并已验证可编译到 AX650 的导出脚本

当前仓库暂不提供视觉 calibration 与 pulsar2 编译脚本。  
如果你的目标只是复现板端运行，请优先直接使用 Hugging Face 发布包中的已验证 `minicpmv4_6_vision_448.axmodel`。

## 环境准备

Vision ONNX 导出与 `pulsar2 llm_build` 分别依赖不同环境。

Vision ONNX 导出相关命令在以下依赖版本下验证：

- `transformers==5.7.0`
- `torch==2.8.0`
- `torchvision==0.23.0`
- `av==17.0.1`
- `onnx==1.21.0`

在 `model_convert/` 目录执行：

```bash
python -m pip install -r requirements.txt
```

`pulsar2 llm_build` 相关命令需要 AXERA NPU 编译环境。请另外准备：

- 可直接执行 `pulsar2 llm_build` 的 shell 环境
- 原始 Hugging Face 模型目录：`openbmb/MiniCPM-V-4.6` 或 `openbmb/MiniCPM-V-4.6-GPTQ`
- 已安装 `pulsar2 llm_build` 依赖的 Python/conda 环境

脚本通过环境变量接收路径：

```bash
export INPUT_PATH=/path/to/openbmb/MiniCPM-V-4.6

# 如果当前 shell 还没有进入编译环境，可以额外设置：
# export CONDA_SH=/path/to/conda.sh
# export CONDA_ENV=<your_build_env>
```

其中：

- `INPUT_PATH`：原始 Hugging Face 模型目录；可以是 BF16 版本，也可以是 GPTQ 版本
- `CONDA_SH` / `CONDA_ENV`：可选，用于激活编译环境；如果当前 shell 已经在正确环境中，可以不设置

下文所有脚本默认以 `model_convert/` 为当前工作目录。

## 推荐顺序

如果你的目标是复现当前 LLM 编译流程，建议按下面顺序执行：

1. 准备原始 Hugging Face 权重目录
2. 如需重新导出视觉 ONNX，先执行 `python export_onnx.py`
3. 确认当前 shell 可以执行 `pulsar2 llm_build`
4. 设置 `INPUT_PATH`，必要时设置 `CONDA_SH`、`CONDA_ENV`
5. 执行 `./llm_build_ax650.sh`
6. 在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载
7. 将编译输出整理到 Hugging Face 发布包布局
8. 使用发布包中的 `axllm serve` 做端到端验证

## 导出固定 Vision ONNX

当前发布包使用固定 `448x448` 输入、`patch_size=14`、`downsample_mode=16x` 的 Vision 编码器。  
`export_onnx.py` 默认导出这一固定 profile，也支持导出其他满足约束的 fixed-shape Vision ONNX。

从 Hugging Face 下载原始模型权重，以下示例假设克隆到仓库同级目录 `hf-models/MiniCPM-V-4.6`，并用 `$HF_MODEL` 引用：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6 ../../hf-models/MiniCPM-V-4.6
export HF_MODEL=../../hf-models/MiniCPM-V-4.6
```

在 `model_convert/` 目录执行默认 `448x448` 导出命令：

```bash
python export_onnx.py \
  --model "$HF_MODEL" \
  --output ./vit-models/minicpmv4_6_vision_448.onnx
```

如果需要导出其他 fixed shape，例如 `392x392`，在 `model_convert/` 目录执行：

```bash
python export_onnx.py \
  --model "$HF_MODEL" \
  --output ./vit-models/minicpmv4_6_vision_392.onnx \
  --input-shape 392x392
```

导出完成后会同时生成：

- `vit-models/minicpmv4_6_vision_448.onnx`
- `vit-models/minicpmv4_6_vision_448.json`
- 与 `.onnx` 同目录的一组 external data 权重文件

其中 `.json` 记录本次固定 profile 的输入输出形状：

- 输入 `pixel_values`: `[1, 3, 14, 14336]`
- 固定 `image_size`: `[448, 448]`
- 固定 `downsample_mode`: `"16x"`
- 输出 `image_features`: `[1, 64, 1024]`

说明：

- 该脚本依赖 `transformers>=5.7.0`
- 由于模型较大，PyTorch ONNX 导出通常会使用 external data 格式；请整体保留输出目录，不要只拷贝单个 `.onnx`
- 当前导出路径会把 MiniCPM-V-4.6 的 fixed-shape 位置编码与 `vit_merger` 窗口注意力展开为静态图，避免生成 `Range` 节点和大扇出 `Concat`，以便后续 `pulsar2 build`
- 当前只导出单图 fixed-shape Vision 编码器，不覆盖视频专用重新打包逻辑
- 当前仓库不提供 Vision calibration 与 axmodel 编译脚本；如需直接运行，请继续使用已验证发布包

shape 约束：

- 当前默认 `patch_size=14`
- 输入高宽需要是 `56` 的整数倍
- 常用 fixed shape 示例：`448x448`、`392x392`、`560x560`

当前开发侧已经额外验证过以下闭环：

- 使用 `python export_onnx.py --model "$HF_MODEL" --output ./vit-models/minicpmv4_6_vision_448.onnx` 导出的 `448x448 / 16x` Vision ONNX 可以在标准 `pulsar2 build` 流程下成功编译为 AX650 `axmodel`
- 编译得到的 Vision `axmodel` 已在 AX650 板端完成 `ax_run_model --skip-running` 加载检查
- 结合已发布的 MiniCPM-V-4.6 文本主干产物，在板端 `python/infer_axmodel.py` 单图 smoke test 中可以正常生成图片描述
- 新编译 Vision `axmodel` 与 Hugging Face 发布包中的已验证 `minicpmv4_6_vision_448.axmodel` 特征输出接近，但不是逐 bit 相同；如果你的目标是严格复现发布包结果，请直接使用发布包中的 Vision `axmodel`

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
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6 ../../hf-models/MiniCPM-V-4.6
export INPUT_PATH=../../hf-models/MiniCPM-V-4.6
```

### GPTQ 权重

GPTQ 输入权重从 `openbmb/MiniCPM-V-4.6-GPTQ` 获取。该仓库是官方提供的 W4A16 GPTQModel 量化版本：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ ../../hf-models/MiniCPM-V-4.6-GPTQ
export INPUT_PATH=../../hf-models/MiniCPM-V-4.6-GPTQ
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
