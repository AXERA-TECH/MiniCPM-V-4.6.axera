#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET_DATASET="datasets/minicpmv4_6_vision_448_calibration.tar"
SOURCE_HINT="datasets/imagenet-calib.tar"

REPO="${DATASET_REPO:-AXERA-TECH/MiniCPM-V-4.6.axera}"
TAG="${DATASET_RELEASE_TAG:-calibration}"
ASSET_NAME="${DATASET_ASSET_NAME:-minicpmv4_6-calibration-datasets.tar}"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"

if [[ -f "$TARGET_DATASET" ]]; then
  echo "calibration 数据已存在：$TARGET_DATASET"
  exit 0
fi

mkdir -p datasets
archive_path="datasets/${ASSET_NAME}"

echo "将尝试从当前仓库 GitHub Release 下载 calibration 数据集包："
echo "  ${URL}"

download_ok=0
if command -v wget >/dev/null 2>&1; then
  if wget -O "$archive_path" "$URL"; then
    download_ok=1
  fi
fi

if [[ "$download_ok" -ne 1 ]] && command -v curl >/dev/null 2>&1; then
  if curl -L "$URL" -o "$archive_path"; then
    download_ok=1
  fi
fi

if [[ "$download_ok" -eq 1 ]]; then
  tar -xf "$archive_path" -C "$SCRIPT_DIR"
  rm -f "$archive_path"
  if [[ -f "$TARGET_DATASET" ]]; then
    echo "下载并解压成功，目标 calibration 数据已就绪：$TARGET_DATASET"
    exit 0
  fi
  echo "错误：下载与解压完成后，仍未找到 $TARGET_DATASET"
  exit 1
fi

echo
echo "下载失败，说明当前 release 资产尚未上传，或者当前网络无法访问 GitHub。"
echo
if [[ -f "$SOURCE_HINT" ]]; then
  echo "检测到本地已经有原始图像 tar：$SOURCE_HINT"
  echo "你可以直接生成 calibration 包："
else
  echo "你可以自行准备一个图片目录或图片 tar 包，然后执行："
fi
echo
echo "  python prepare_calibration.py \\"
echo "    --model_path ../python/MiniCPM-V-4.6 \\"
if [[ -f "$SOURCE_HINT" ]]; then
  echo "    --dataset_dir $SOURCE_HINT \\"
else
  echo "    --dataset_dir /path/to/image_dir_or_tar \\"
fi
echo "    --output_dir ./datasets/minicpmv4_6_vision_448_calibration \\"
echo "    --input-shape 448x448"
echo
echo "生成完成后，标准 calibration 包路径应为："
echo "  $TARGET_DATASET"
exit 1
