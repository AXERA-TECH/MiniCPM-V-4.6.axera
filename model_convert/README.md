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
├── prepare_calibration.py  # 固定 shape Vision calibration 数据生成
├── download_dataset.sh     # calibration 数据入口脚本
├── llm_build_ax650.sh      # AX650 LLM 主干编译脚本
├── datasets/               # calibration 输入与 tar 包目录
├── pulsar2_configs/        # Vision pulsar2 build 配置
└── vit-models/             # Vision ONNX 与元数据输出目录
```

当前仓库提供：

- LLM 主干 `pulsar2 llm_build` 编译脚本
- 支持固定 shape Vision ONNX 导出，并已验证可编译到 AX650 的导出脚本
- 固定 shape Vision calibration 数据生成脚本
- 固定 `448x448 / 16x` Vision `pulsar2 build` 参考配置

当前仓库仍未提供一个公开发布的 calibration 数据集包。
如果你的目标只是复现板端运行，请优先直接使用 Hugging Face 发布包中的已验证 `minicpmv4_6_vision_448.axmodel`。

## 环境准备

Vision ONNX 导出与 `pulsar2 llm_build` 分别依赖不同环境。

以下命令默认在 `conda activate hf` 之后执行。

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

如果你的目标是复现当前开发流程，建议按下面顺序执行：

1. 准备原始 Hugging Face 权重目录
2. 如需重新导出 Vision ONNX，先执行 `python export_onnx.py`
3. 确认当前 shell 可以执行 `pulsar2 llm_build`
4. 设置 `INPUT_PATH`，必要时设置 `CONDA_SH`、`CONDA_ENV`
5. 执行 `./llm_build_ax650.sh`
6. 在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载
7. 将编译输出整理到 Hugging Face 发布包布局
8. 使用发布包中的 `axllm serve` 做端到端验证

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

当前已验证的 Vision 导出闭环：

| Item | Value |
|---|---|
| 发布包默认 profile | `448x448 / 16x` |
| `patch_size` | `14` |
| ONNX 输出 | `vit-models/minicpmv4_6_vision_448.onnx` |
| calibration tar | `datasets/minicpmv4_6_vision_448_calibration.tar` |
| `pulsar2 build` 配置 | `pulsar2_configs/config_vision_448_npu3.json` |
| 元数据输出 | `vit-models/minicpmv4_6_vision_448.json` |
| 继续编译时关键条件 | `pulsar2 build` 使用 `npu_mode = NPU3` |

当前仓库可以重新生成一个可用的 `448x448 / 16x` Vision `axmodel`，默认输出为：

```text
compiled_output_448/compiled.axmodel
```

但当前重新编译结果与 Hugging Face 发布包中的 `minicpmv4_6_vision_448.axmodel` 不是逐 bit 相同的二进制文件。
如果你的目标是严格复现发布包中的同名产物，请直接使用 Hugging Face 发布包里已经验证过的 Vision `axmodel`。

## 最短复现路径

如果你只想从当前仓库重新生成一个可用的 `448x448 / 16x` Vision `axmodel`，推荐按下面顺序执行。

1. 准备原始 Hugging Face 权重目录。

如果你已经把完整原始权重放到了仓库内的 `../python/MiniCPM-V-4.6/`，可以直接复用：

```bash
cd model_convert
export HF_MODEL=../python/MiniCPM-V-4.6
```

如果当前仓库里还没有完整原始权重，则下载到当前仓库内：

```bash
cd model_convert
hf download openbmb/MiniCPM-V-4.6 --local-dir ./hf-models/MiniCPM-V-4.6
export HF_MODEL=./hf-models/MiniCPM-V-4.6
```

2. 准备 calibration 数据。

如果当前仓库里已经有自己的图片目录或图片 tar，可以直接生成 calibration：

```bash
cd model_convert
python prepare_calibration.py \
  --model_path "$HF_MODEL" \
  --dataset_dir /path/to/image_dir_or_tar \
  --output_dir ./datasets/minicpmv4_6_vision_448_calibration \
  --input-shape 448x448
```

如果你不想自己准备 calibration 数据，可以先尝试：

```bash
cd model_convert
bash download_dataset.sh
```

3. 在同一环境下导出 ONNX：

```bash
cd model_convert
python export_onnx.py \
  --model "$HF_MODEL" \
  --output ./vit-models/minicpmv4_6_vision_448.onnx
```

4. 在 AXERA `pulsar2` 编译环境下继续编译：

```bash
cd model_convert
pulsar2 build \
  --output_dir ./compiled_output_448 \
  --config pulsar2_configs/config_vision_448_npu3.json \
  --npu_mode NPU3 \
  --input vit-models/minicpmv4_6_vision_448.onnx \
  --compiler.check 0 \
  --target_hardware AX650
```

5. 编译成功后检查最终 Vision `axmodel`：

```bash
ls -lh ./compiled_output_448/compiled.axmodel
```

## 导出固定 Vision ONNX

当前发布包使用固定 `448x448` 输入、`patch_size=14`、`downsample_mode=16x` 的 Vision 编码器。
`export_onnx.py` 默认导出这一固定 profile，也支持导出其他满足约束的 fixed-shape Vision ONNX。

如果你已经把完整原始权重放到了当前仓库内的 `../python/MiniCPM-V-4.6/`，可以直接将它作为导出输入：

```bash
export HF_MODEL=../python/MiniCPM-V-4.6
```

这是一种本地开发侧快捷路径；公开仓库本身不会提交原始权重。

其中：

- `config.json`、`preprocessor_config.json`、`tokenizer*.json` 等文件保留在当前目录
- `model.safetensors` 可以是普通文件，也可以是指向原始权重的软连接

如果当前仓库内还没有完整原始权重，再从 Hugging Face 下载到 `model_convert/hf-models/` 目录，并用 `$HF_MODEL` 引用：

```bash
hf download openbmb/MiniCPM-V-4.6 --local-dir ./hf-models/MiniCPM-V-4.6
export HF_MODEL=./hf-models/MiniCPM-V-4.6
```

如果你改用 `git clone`，请确认本机已安装 `git-lfs`，否则模型权重不会完整下载。

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

shape 约束：

- 当前默认 `patch_size=14`
- 输入高宽需要是 `56` 的整数倍
- 常用 fixed shape 示例：`448x448`、`392x392`、`560x560`

当前开发侧已经额外验证过以下闭环：

- 使用 `python export_onnx.py --model "$HF_MODEL" --output ./vit-models/minicpmv4_6_vision_448.onnx` 导出的 `448x448 / 16x` Vision ONNX 可以在标准 `pulsar2 build` 流程下成功编译为 AX650 `axmodel`
- 上述 Vision `axmodel` 在对齐当前发布包时使用 `npu_mode = NPU3`
- 当前发布包中的 `minicpmv4_6_vision_448.axmodel` 在 AX650 板端用 `/opt/bin/ax_run_model -g 0 -w 1 -r 5` 测得平均耗时约 `234.820 ms`
- 结合已发布的 MiniCPM-V-4.6 文本主干产物，在板端 `python/infer_axmodel.py` 单图 smoke test 中可以正常生成图片描述
- 新编译 Vision `axmodel` 可以正常产出可用结果，但与 Hugging Face 发布包中的已验证 `minicpmv4_6_vision_448.axmodel` 不是逐 bit 相同；如果你的目标是严格复现发布包结果，请直接使用发布包中的 Vision `axmodel`

## 生成 Vision calibration 数据

当前仓库提供固定 shape Vision calibration 生成脚本：

```bash
python prepare_calibration.py \
  --model_path ../python/MiniCPM-V-4.6 \
  --dataset_dir /path/to/image_dir_or_tar \
  --output_dir ./datasets/minicpmv4_6_vision_448_calibration \
  --input-shape 448x448
```

执行完成后，默认会生成：

- 中间 `.npy`：`datasets/minicpmv4_6_vision_448_calibration/*.pixel_values.npy`
- calibration tar：`datasets/minicpmv4_6_vision_448_calibration.tar`

说明：

- `dataset_dir` 支持图片目录，也支持包含图片的 `.tar`
- 当前脚本会复用 `../python/MiniCPM-V-4.6/` 中的 `preprocessor_config.json`
- 当前默认输出与 `pulsar2_configs/config_vision_448_npu3.json` 对齐

如果你希望先走统一入口脚本，也可以执行：

```bash
bash download_dataset.sh
```

`download_dataset.sh` 的作用是：

- 如果仓库发布页已经上传了 calibration 数据包，直接下载并解压
- 如果当前没有发布资产，或网络无法访问 GitHub，则回退提示你使用 `prepare_calibration.py`

因此：

- 不想自己准备 calibration 时，优先执行 `bash download_dataset.sh`
- 已经有自己的图片目录或图片 tar 时，直接执行 `python prepare_calibration.py ...`

## 继续编译 Vision axmodel

当前仓库提供 `448x448 / 16x` Vision 的 `pulsar2 build` 参考配置：

- `pulsar2_configs/config_vision_448_npu3.json`

如果你使用当前仓库的默认固定 profile，在 `model_convert/` 目录执行：

```bash
pulsar2 build \
  --output_dir ./compiled_output_448 \
  --config pulsar2_configs/config_vision_448_npu3.json \
  --npu_mode NPU3 \
  --input vit-models/minicpmv4_6_vision_448.onnx \
  --compiler.check 0 \
  --target_hardware AX650
```

如果你使用自己的标准 `pulsar2 build` 流程继续编译其他 Vision `axmodel`，请至少保证构建配置中包含：

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU3"
}
```

量化数据集、量化方法和其他编译参数可按你的实际环境配置，但 `npu_mode = NPU3` 是当前发布包对齐所需的关键条件。

## 准备输入模型

### BF16 权重

从 Hugging Face 下载 BF16 原始模型权重，并用 `$INPUT_PATH` 指向该目录：

```bash
hf download openbmb/MiniCPM-V-4.6 --local-dir ./hf-models/MiniCPM-V-4.6
export INPUT_PATH=./hf-models/MiniCPM-V-4.6
```

### GPTQ 权重

GPTQ 输入权重从 `openbmb/MiniCPM-V-4.6-GPTQ` 获取。该仓库是官方提供的 W4A16 GPTQModel 量化版本：

```bash
hf download openbmb/MiniCPM-V-4.6-GPTQ --local-dir ./hf-models/MiniCPM-V-4.6-GPTQ
export INPUT_PATH=./hf-models/MiniCPM-V-4.6-GPTQ
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

说明：

- 脚本默认在当前 shell 中直接执行 `pulsar2 llm_build`
- 如果设置了 `CONDA_SH`，脚本会先 `source "$CONDA_SH"` 并激活 `${CONDA_ENV:-npu}`
- GPTQ 与 BF16 的编译脚本相同，区别只在 `INPUT_PATH` 指向的原始模型目录

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

图像和视频能力最终仍需要把这里生成的 Vision `axmodel` 与 Hugging Face 发布包中的其他运行时文件整理到一起再做端到端验证。
