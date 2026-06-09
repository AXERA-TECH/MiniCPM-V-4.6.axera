# vit-models

该目录用于放置 `export_onnx.py` 导出的固定 shape Vision ONNX、元数据 JSON，以及后续编译前的中间文件。

当前 MiniCPM-V-4.6 发布包默认使用：

- `448x448`
- `patch_size=14`
- `downsample_mode=16x`

推荐导出命令：

```bash
python export_onnx.py \
  --model ../../hf-models/MiniCPM-V-4.6 \
  --output ./vit-models/minicpmv4_6_vision_448.onnx
```

说明：

- `.onnx` 和 external data 权重文件不提交到 git
- 当前仓库只跟踪这个目录下的说明文件与必要元数据模板
