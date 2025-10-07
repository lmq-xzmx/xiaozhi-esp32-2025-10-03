# FP16模型量化脚本使用说明

## 概述

`quantize_fp16_models.py` 脚本用于将VAD和ASR模型量化为FP16格式，以减少内存使用并提升推理速度。

## 功能特性

- **VAD模型量化**: 支持Silero VAD的ONNX和PyTorch模型量化
- **ASR模型量化**: 支持SenseVoice模型的FP16量化
- **自动验证**: 量化后自动验证模型可用性
- **大小优化**: 通常可减少约50%的模型大小和内存使用

## 使用方法

### 基本用法

```bash
# 量化所有模型
python scripts/quantize_fp16_models.py

# 指定模型目录
python scripts/quantize_fp16_models.py --models-dir /path/to/models
```

### 选择性量化

```bash
# 仅量化VAD模型
python scripts/quantize_fp16_models.py --vad-only

# 仅量化ASR模型
python scripts/quantize_fp16_models.py --asr-only
```

### 验证模型

```bash
# 量化并验证模型
python scripts/quantize_fp16_models.py --validate
```

## 模型文件结构

### 量化前
```
models/
├── silero_vad.onnx          # VAD ONNX模型
├── silero_vad.pt            # VAD PyTorch模型（可选）
└── SenseVoiceSmall/         # ASR模型目录
    ├── model.pt             # 主模型文件
    ├── config.json          # 配置文件
    ├── tokenizer.json       # 分词器配置
    └── vocab.txt            # 词汇表
```

### 量化后
```
models/
├── silero_vad.onnx          # 原始VAD ONNX模型
├── silero_vad_fp16.onnx     # 量化VAD ONNX模型
├── silero_vad.pt            # 原始VAD PyTorch模型
├── silero_vad_fp16.pt       # 量化VAD PyTorch模型
├── SenseVoiceSmall/         # 原始ASR模型
└── SenseVoiceSmall_fp16/    # 量化ASR模型
    ├── model.pt             # 量化主模型文件
    ├── config.json          # 配置文件（复制）
    ├── tokenizer.json       # 分词器配置（复制）
    └── vocab.txt            # 词汇表（复制）
```

## 服务集成

### VAD服务
VAD服务会自动检测并优先使用FP16量化模型：
- 优先加载 `silero_vad_fp16.onnx`
- 如果不存在，回退到 `silero_vad.onnx`

### ASR服务
ASR服务需要在初始化时启用FP16选项：
```python
processor = SenseVoiceProcessor(
    model_path="models/SenseVoiceSmall",
    enable_fp16=True  # 启用FP16量化
)
```

## 性能优化效果

### 内存使用
- **VAD模型**: 减少约50%内存使用
- **ASR模型**: 减少约50%内存使用

### 推理速度
- **GPU环境**: 提升10-30%推理速度
- **CPU环境**: 轻微提升或持平

### 模型精度
- **量化损失**: 极小，通常不影响实际使用效果
- **兼容性**: 完全兼容现有服务接口

## 注意事项

1. **依赖要求**: 确保安装了以下依赖
   ```bash
   pip install torch onnx onnxruntime
   ```

2. **存储空间**: 量化过程会创建新的模型文件，确保有足够存储空间

3. **GPU支持**: FP16量化在GPU环境下效果最佳

4. **备份建议**: 量化前建议备份原始模型文件

## 故障排除

### 常见问题

1. **模型文件不存在**
   - 确保模型文件路径正确
   - 检查模型是否已下载

2. **量化失败**
   - 检查依赖是否正确安装
   - 确保有足够的内存和存储空间

3. **验证失败**
   - 检查量化模型文件是否完整
   - 重新运行量化过程

### 日志输出
脚本会输出详细的量化过程信息，包括：
- 模型大小对比
- 量化进度
- 验证结果
- 错误信息

## 示例输出

```
2024-01-01 12:00:00 - INFO - 开始量化VAD模型...
2024-01-01 12:00:01 - INFO - 开始量化ONNX模型: ./models/silero_vad.onnx
2024-01-01 12:00:02 - INFO - ONNX模型量化完成:
2024-01-01 12:00:02 - INFO -   原始大小: 42.5 MB
2024-01-01 12:00:02 - INFO -   量化大小: 21.8 MB
2024-01-01 12:00:02 - INFO -   大小减少: 48.7%
2024-01-01 12:00:03 - INFO - 开始量化ASR模型...
2024-01-01 12:00:05 - INFO - 所有模型量化完成！
```