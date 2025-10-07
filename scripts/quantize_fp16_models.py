#!/usr/bin/env python3
"""
FP16模型量化脚本
为VAD和ASR模型生成FP16量化版本，减少内存使用并提升推理速度
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging
from pathlib import Path
import argparse
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def quantize_pytorch_model(self, model_path: str, output_path: str) -> bool:
        """量化PyTorch模型到FP16"""
        try:
            logger.info(f"开始量化PyTorch模型: {model_path}")
            
            # 加载模型
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
                
            model = torch.load(model_path, map_location='cpu')
            
            # 转换为FP16
            if isinstance(model, torch.nn.Module):
                model = model.half()
            elif isinstance(model, dict):
                # 处理state_dict格式
                for key in model:
                    if isinstance(model[key], torch.Tensor) and model[key].dtype == torch.float32:
                        model[key] = model[key].half()
            
            # 保存量化模型
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(model, output_path)
            
            # 检查文件大小减少
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(output_path)
            reduction = (1 - quantized_size / original_size) * 100
            
            logger.info(f"PyTorch模型量化完成:")
            logger.info(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"  量化大小: {quantized_size / 1024 / 1024:.2f} MB")
            logger.info(f"  大小减少: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"PyTorch模型量化失败: {e}")
            return False
    
    def quantize_onnx_model(self, model_path: str, output_path: str) -> bool:
        """量化ONNX模型"""
        try:
            logger.info(f"开始量化ONNX模型: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"ONNX模型文件不存在: {model_path}")
                return False
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 动态量化到FP16
            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,  # 使用8位量化
                optimize_model=True
            )
            
            # 检查文件大小减少
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(output_path)
            reduction = (1 - quantized_size / original_size) * 100
            
            logger.info(f"ONNX模型量化完成:")
            logger.info(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"  量化大小: {quantized_size / 1024 / 1024:.2f} MB")
            logger.info(f"  大小减少: {reduction:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX模型量化失败: {e}")
            return False
    
    def quantize_vad_models(self) -> bool:
        """量化VAD模型"""
        logger.info("开始量化VAD模型...")
        success = True
        
        # VAD ONNX模型路径
        vad_onnx_path = self.models_dir / "silero_vad.onnx"
        vad_onnx_fp16_path = self.models_dir / "silero_vad_fp16.onnx"
        
        if vad_onnx_path.exists():
            if not self.quantize_onnx_model(str(vad_onnx_path), str(vad_onnx_fp16_path)):
                success = False
        else:
            logger.warning(f"VAD ONNX模型不存在: {vad_onnx_path}")
        
        # 检查是否有PyTorch VAD模型
        vad_pt_path = self.models_dir / "silero_vad.pt"
        vad_pt_fp16_path = self.models_dir / "silero_vad_fp16.pt"
        
        if vad_pt_path.exists():
            if not self.quantize_pytorch_model(str(vad_pt_path), str(vad_pt_fp16_path)):
                success = False
        else:
            logger.info("未找到PyTorch VAD模型，跳过")
        
        return success
    
    def quantize_asr_models(self) -> bool:
        """量化ASR模型"""
        logger.info("开始量化ASR模型...")
        success = True
        
        # SenseVoice模型路径
        sensevoice_dir = self.models_dir / "SenseVoiceSmall"
        sensevoice_fp16_dir = self.models_dir / "SenseVoiceSmall_fp16"
        
        if sensevoice_dir.exists():
            # 创建FP16模型目录
            sensevoice_fp16_dir.mkdir(exist_ok=True)
            
            # 复制配置文件
            for config_file in ["config.json", "tokenizer.json", "vocab.txt"]:
                src_path = sensevoice_dir / config_file
                dst_path = sensevoice_fp16_dir / config_file
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"复制配置文件: {config_file}")
            
            # 量化模型文件
            model_pt_path = sensevoice_dir / "model.pt"
            model_fp16_path = sensevoice_fp16_dir / "model.pt"
            
            if model_pt_path.exists():
                if not self.quantize_pytorch_model(str(model_pt_path), str(model_fp16_path)):
                    success = False
            else:
                logger.warning(f"SenseVoice模型文件不存在: {model_pt_path}")
                success = False
        else:
            logger.warning(f"SenseVoice模型目录不存在: {sensevoice_dir}")
            success = False
        
        return success
    
    def validate_quantized_models(self) -> bool:
        """验证量化模型"""
        logger.info("验证量化模型...")
        
        # 验证VAD模型
        vad_fp16_onnx = self.models_dir / "silero_vad_fp16.onnx"
        if vad_fp16_onnx.exists():
            try:
                # 尝试加载ONNX模型
                session = ort.InferenceSession(str(vad_fp16_onnx))
                logger.info("VAD FP16 ONNX模型验证成功")
            except Exception as e:
                logger.error(f"VAD FP16 ONNX模型验证失败: {e}")
                return False
        
        # 验证ASR模型
        asr_fp16_model = self.models_dir / "SenseVoiceSmall_fp16" / "model.pt"
        if asr_fp16_model.exists():
            try:
                # 尝试加载PyTorch模型
                model = torch.load(str(asr_fp16_model), map_location='cpu')
                logger.info("ASR FP16 PyTorch模型验证成功")
            except Exception as e:
                logger.error(f"ASR FP16 PyTorch模型验证失败: {e}")
                return False
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FP16模型量化脚本")
    parser.add_argument("--models-dir", default="./models", help="模型目录路径")
    parser.add_argument("--vad-only", action="store_true", help="仅量化VAD模型")
    parser.add_argument("--asr-only", action="store_true", help="仅量化ASR模型")
    parser.add_argument("--validate", action="store_true", help="验证量化模型")
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = ModelQuantizer(args.models_dir)
    
    success = True
    
    # 执行量化
    if not args.asr_only:
        if not quantizer.quantize_vad_models():
            success = False
    
    if not args.vad_only:
        if not quantizer.quantize_asr_models():
            success = False
    
    # 验证模型
    if args.validate or success:
        if not quantizer.validate_quantized_models():
            success = False
    
    if success:
        logger.info("所有模型量化完成！")
        logger.info("使用说明:")
        logger.info("1. VAD服务会自动尝试加载FP16量化模型")
        logger.info("2. ASR服务需要在初始化时启用FP16选项")
        logger.info("3. 量化模型可减少约50%的内存使用")
    else:
        logger.error("模型量化过程中出现错误")
        sys.exit(1)

if __name__ == "__main__":
    main()