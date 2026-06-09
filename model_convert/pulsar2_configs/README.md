# pulsar2_configs

该目录用于放置 Vision `pulsar2 build` 配置文件。

当前仓库提供一份面向固定 `448x448 / 16x` Vision ONNX 的参考配置：

- `config_vision_448_npu3.json`

该配置面向以下前提：

- ONNX 输入来自 `vit-models/minicpmv4_6_vision_448.onnx`
- calibration 数据来自 `datasets/minicpmv4_6_vision_448_calibration.tar`
- 编译目标为 `AX650 / NPU3`

如果你切换到其他 fixed shape，请同时修改：

- calibration tar 路径
- ONNX 输入路径
- 必要时的量化配置
