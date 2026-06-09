# hf-models

该目录用于放置从公开 Hugging Face 下载的原始模型权重。

推荐布局：

```text
hf-models/
├── MiniCPM-V-4.6/
└── MiniCPM-V-4.6-GPTQ/
```

推荐使用 `hf download` 直接把模型下载到当前仓库内：

```bash
hf download openbmb/MiniCPM-V-4.6 --local-dir ./hf-models/MiniCPM-V-4.6
hf download openbmb/MiniCPM-V-4.6-GPTQ --local-dir ./hf-models/MiniCPM-V-4.6-GPTQ
```

说明：

- 该目录中的大模型权重不提交到 git
- `export_onnx.py` 和 `llm_build_ax650.sh` 都可以直接使用这里的路径
- 如果你习惯使用 `git clone`，请确认本机已经安装并配置好 `git-lfs`
