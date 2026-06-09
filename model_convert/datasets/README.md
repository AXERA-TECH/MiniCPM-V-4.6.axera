# datasets

该目录用于放置 `model_convert/` 相关的 calibration 输入与打包结果。

当前 MiniCPM-V-4.6 开发流程默认约定：

- Vision calibration tar:
  - `minicpmv4_6_vision_448_calibration.tar`

推荐流程：

1. 准备一个图片目录，或一个包含图片的 `.tar` 包
2. 在 `model_convert/` 目录执行：

```bash
python prepare_calibration.py \
  --model_path ../python/MiniCPM-V-4.6 \
  --dataset_dir /path/to/image_dir_or_tar \
  --output_dir ./datasets/minicpmv4_6_vision_448_calibration \
  --input-shape 448x448
```

说明：

- 当前仓库不提交实际 calibration 数据
- 当前仓库也没有发布可直接下载的 calibration 包
- `download_dataset.sh` 只是统一的入口脚本，会明确提示当前没有公开下载资产
