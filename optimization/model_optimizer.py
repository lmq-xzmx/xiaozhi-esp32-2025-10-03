#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - AI模型优化工具集
包含模型量化、蒸馏、压缩和部署优化功能
"""

import os
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import onnx
import onnxruntime as ort
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import datasets
from datasets import Dataset

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    ONNX_CONVERSION = "onnx_conversion"
    TENSORRT_OPTIMIZATION = "tensorrt_optimization"

class QuantizationMethod(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    FP16 = "fp16"

@dataclass
class OptimizationConfig:
    """优化配置"""
    model_path: str
    output_path: str
    optimization_type: OptimizationType
    target_device: str = "cpu"  # cpu, cuda, edge
    compression_ratio: float = 0.5
    accuracy_threshold: float = 0.95
    batch_size: int = 32
    max_samples: int = 1000
    
    # 量化特定配置
    quantization_method: QuantizationMethod = QuantizationMethod.DYNAMIC
    calibration_dataset: Optional[str] = None
    
    # 蒸馏特定配置
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # 剪枝特定配置
    pruning_ratio: float = 0.3
    structured_pruning: bool = False

class ModelBenchmark:
    """模型性能基准测试"""
    
    def __init__(self):
        self.metrics = {}
    
    def benchmark_model(self, model_path: str, test_data: List[Any], device: str = "cpu") -> Dict[str, float]:
        """对模型进行基准测试"""
        logger.info(f"Benchmarking model: {model_path}")
        
        # 加载模型
        if model_path.endswith('.onnx'):
            model = self._load_onnx_model(model_path)
            results = self._benchmark_onnx_model(model, test_data)
        else:
            model = self._load_pytorch_model(model_path, device)
            results = self._benchmark_pytorch_model(model, test_data, device)
        
        return results
    
    def _load_onnx_model(self, model_path: str):
        """加载ONNX模型"""
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    
    def _load_pytorch_model(self, model_path: str, device: str):
        """加载PyTorch模型"""
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    
    def _benchmark_pytorch_model(self, model, test_data: List[Any], device: str) -> Dict[str, float]:
        """基准测试PyTorch模型"""
        model.to(device)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                if test_data:
                    dummy_input = test_data[0]
                    if isinstance(dummy_input, dict):
                        dummy_input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in dummy_input.items()}
                    elif torch.is_tensor(dummy_input):
                        dummy_input = dummy_input.to(device)
                    _ = model(dummy_input)
        
        # 实际测试
        latencies = []
        memory_usage = []
        
        for data in test_data[:100]:  # 测试前100个样本
            if torch.is_tensor(data):
                data = data.to(device)
            elif isinstance(data, dict):
                data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            
            # 测量内存使用
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # 测量推理时间
            start_time = time.time()
            with torch.no_grad():
                _ = model(data)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
            
            if device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - start_memory) / 1024 / 1024)  # MB
        
        results = {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_qps': 1000 / np.mean(latencies),
        }
        
        if memory_usage:
            results['avg_memory_mb'] = np.mean(memory_usage)
            results['peak_memory_mb'] = np.max(memory_usage)
        
        return results
    
    def _benchmark_onnx_model(self, session, test_data: List[Any]) -> Dict[str, float]:
        """基准测试ONNX模型"""
        input_name = session.get_inputs()[0].name
        
        # 预热
        for _ in range(10):
            if test_data:
                dummy_input = test_data[0]
                if torch.is_tensor(dummy_input):
                    dummy_input = dummy_input.numpy()
                _ = session.run(None, {input_name: dummy_input})
        
        # 实际测试
        latencies = []
        
        for data in test_data[:100]:
            if torch.is_tensor(data):
                data = data.numpy()
            
            start_time = time.time()
            _ = session.run(None, {input_name: data})
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_qps': 1000 / np.mean(latencies),
        }

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self):
        self.benchmark = ModelBenchmark()
    
    def quantize_model(self, config: OptimizationConfig) -> Dict[str, Any]:
        """量化模型"""
        logger.info(f"Starting model quantization: {config.quantization_method.value}")
        
        if config.quantization_method == QuantizationMethod.DYNAMIC:
            return self._dynamic_quantization(config)
        elif config.quantization_method == QuantizationMethod.STATIC:
            return self._static_quantization(config)
        elif config.quantization_method == QuantizationMethod.FP16:
            return self._fp16_quantization(config)
        else:
            raise ValueError(f"Unsupported quantization method: {config.quantization_method}")
    
    def _dynamic_quantization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """动态量化"""
        logger.info("Performing dynamic quantization")
        
        # 加载原始模型
        model = torch.load(config.model_path, map_location='cpu')
        model.eval()
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},  # 量化的层类型
            dtype=torch.qint8
        )
        
        # 保存量化模型
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        torch.save(quantized_model, config.output_path)
        
        # 计算模型大小
        original_size = os.path.getsize(config.model_path) / 1024 / 1024  # MB
        quantized_size = os.path.getsize(config.output_path) / 1024 / 1024  # MB
        compression_ratio = quantized_size / original_size
        
        logger.info(f"Dynamic quantization completed. Compression ratio: {compression_ratio:.2f}")
        
        return {
            'method': 'dynamic_quantization',
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'output_path': config.output_path
        }
    
    def _static_quantization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """静态量化（需要校准数据集）"""
        logger.info("Performing static quantization")
        
        if not config.calibration_dataset:
            raise ValueError("Static quantization requires calibration dataset")
        
        # 加载模型和校准数据
        model = torch.load(config.model_path, map_location='cpu')
        model.eval()
        
        # 准备量化
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # 校准（使用校准数据集运行模型）
        calibration_data = self._load_calibration_data(config.calibration_dataset)
        
        with torch.no_grad():
            for data in calibration_data[:config.max_samples]:
                model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        # 保存量化模型
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        torch.save(quantized_model, config.output_path)
        
        # 计算压缩比
        original_size = os.path.getsize(config.model_path) / 1024 / 1024
        quantized_size = os.path.getsize(config.output_path) / 1024 / 1024
        compression_ratio = quantized_size / original_size
        
        logger.info(f"Static quantization completed. Compression ratio: {compression_ratio:.2f}")
        
        return {
            'method': 'static_quantization',
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'output_path': config.output_path
        }
    
    def _fp16_quantization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """FP16量化"""
        logger.info("Performing FP16 quantization")
        
        # 加载模型
        model = torch.load(config.model_path, map_location='cpu')
        model.eval()
        
        # 转换为FP16
        model_fp16 = model.half()
        
        # 保存FP16模型
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        torch.save(model_fp16, config.output_path)
        
        # 计算压缩比
        original_size = os.path.getsize(config.model_path) / 1024 / 1024
        fp16_size = os.path.getsize(config.output_path) / 1024 / 1024
        compression_ratio = fp16_size / original_size
        
        logger.info(f"FP16 quantization completed. Compression ratio: {compression_ratio:.2f}")
        
        return {
            'method': 'fp16_quantization',
            'original_size_mb': original_size,
            'quantized_size_mb': fp16_size,
            'compression_ratio': compression_ratio,
            'output_path': config.output_path
        }
    
    def _load_calibration_data(self, dataset_path: str) -> List[torch.Tensor]:
        """加载校准数据集"""
        # 这里应该根据实际数据格式加载
        # 暂时返回随机数据作为示例
        calibration_data = []
        for _ in range(100):
            # 假设输入是形状为 (batch_size, seq_len, hidden_size) 的张量
            data = torch.randn(1, 128, 768)
            calibration_data.append(data)
        
        return calibration_data

class ModelDistiller:
    """模型蒸馏器"""
    
    def __init__(self):
        self.benchmark = ModelBenchmark()
    
    def distill_model(self, config: OptimizationConfig) -> Dict[str, Any]:
        """模型蒸馏"""
        logger.info("Starting model distillation")
        
        if not config.teacher_model_path:
            raise ValueError("Model distillation requires teacher model path")
        
        # 加载教师模型和学生模型
        teacher_model = self._load_teacher_model(config.teacher_model_path)
        student_model = self._create_student_model(teacher_model, config.compression_ratio)
        
        # 准备训练数据
        train_dataset = self._prepare_distillation_dataset(config)
        
        # 执行蒸馏训练
        distilled_model = self._train_student_model(
            teacher_model, student_model, train_dataset, config
        )
        
        # 保存蒸馏后的模型
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        torch.save(distilled_model.state_dict(), config.output_path)
        
        # 计算模型大小
        teacher_size = os.path.getsize(config.teacher_model_path) / 1024 / 1024
        student_size = os.path.getsize(config.output_path) / 1024 / 1024
        compression_ratio = student_size / teacher_size
        
        logger.info(f"Model distillation completed. Compression ratio: {compression_ratio:.2f}")
        
        return {
            'method': 'knowledge_distillation',
            'teacher_size_mb': teacher_size,
            'student_size_mb': student_size,
            'compression_ratio': compression_ratio,
            'output_path': config.output_path
        }
    
    def _load_teacher_model(self, model_path: str):
        """加载教师模型"""
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
    
    def _create_student_model(self, teacher_model, compression_ratio: float):
        """创建学生模型（更小的模型）"""
        # 这里应该根据教师模型的结构创建一个更小的学生模型
        # 暂时返回教师模型的简化版本
        
        # 获取教师模型的配置
        if hasattr(teacher_model, 'config'):
            config = teacher_model.config
            # 减少隐藏层大小和层数
            config.hidden_size = int(config.hidden_size * compression_ratio)
            config.num_hidden_layers = max(1, int(config.num_hidden_layers * compression_ratio))
            config.num_attention_heads = max(1, int(config.num_attention_heads * compression_ratio))
            
            # 创建新的模型
            student_model = type(teacher_model)(config)
        else:
            # 如果没有配置，创建一个简化的模型结构
            student_model = self._create_simple_student_model(teacher_model, compression_ratio)
        
        return student_model
    
    def _create_simple_student_model(self, teacher_model, compression_ratio: float):
        """创建简单的学生模型"""
        # 这是一个简化的实现，实际应用中需要根据具体模型结构调整
        class SimpleStudentModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # 估算输入输出大小
        input_size = 768  # 假设的输入大小
        hidden_size = int(512 * compression_ratio)
        output_size = 768  # 假设的输出大小
        
        return SimpleStudentModel(input_size, hidden_size, output_size)
    
    def _prepare_distillation_dataset(self, config: OptimizationConfig) -> Dataset:
        """准备蒸馏数据集"""
        # 这里应该加载实际的训练数据
        # 暂时创建模拟数据
        
        data = []
        for _ in range(config.max_samples):
            # 创建模拟的输入数据
            input_ids = torch.randint(0, 1000, (128,))  # 假设序列长度为128
            attention_mask = torch.ones(128)
            
            data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
        
        return Dataset.from_list(data)
    
    def _train_student_model(self, teacher_model, student_model, dataset, config: OptimizationConfig):
        """训练学生模型"""
        logger.info("Training student model with knowledge distillation")
        
        # 定义蒸馏损失函数
        def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
            # 软目标损失（蒸馏损失）
            soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
            soft_prob = torch.log_softmax(student_logits / temperature, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0]
            
            # 硬目标损失（原始损失）
            hard_targets_loss = torch.nn.functional.cross_entropy(student_logits, labels)
            
            # 组合损失
            return alpha * soft_targets_loss + (1 - alpha) * hard_targets_loss
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=os.path.dirname(config.output_path),
            num_train_epochs=3,
            per_device_train_batch_size=config.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(os.path.dirname(config.output_path), 'logs'),
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
        )
        
        # 自定义训练器
        class DistillationTrainer(Trainer):
            def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.temperature = temperature
                self.alpha = alpha
                self.teacher_model.eval()
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # 学生模型前向传播
                student_outputs = model(**inputs)
                student_logits = student_outputs.logits
                
                # 教师模型前向传播
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # 计算蒸馏损失
                labels = inputs.get('labels')
                if labels is None:
                    # 如果没有标签，使用教师模型的预测作为软标签
                    labels = torch.argmax(teacher_logits, dim=-1)
                
                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    self.temperature, self.alpha
                )
                
                return (loss, student_outputs) if return_outputs else loss
        
        # 创建训练器
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            temperature=config.distillation_temperature,
            alpha=config.distillation_alpha,
            model=student_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=None,  # 这里需要实际的tokenizer
                mlm=False
            )
        )
        
        # 开始训练
        trainer.train()
        
        return student_model

class ONNXConverter:
    """ONNX转换器"""
    
    def __init__(self):
        self.benchmark = ModelBenchmark()
    
    def convert_to_onnx(self, config: OptimizationConfig) -> Dict[str, Any]:
        """转换模型为ONNX格式"""
        logger.info("Converting model to ONNX format")
        
        # 加载PyTorch模型
        model = torch.load(config.model_path, map_location='cpu')
        model.eval()
        
        # 创建示例输入
        dummy_input = self._create_dummy_input(model)
        
        # 转换为ONNX
        onnx_path = config.output_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 优化ONNX模型
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        self._optimize_onnx_model(onnx_path, optimized_path)
        
        # 计算模型大小
        original_size = os.path.getsize(config.model_path) / 1024 / 1024
        onnx_size = os.path.getsize(optimized_path) / 1024 / 1024
        
        logger.info(f"ONNX conversion completed. Original: {original_size:.2f}MB, ONNX: {onnx_size:.2f}MB")
        
        return {
            'method': 'onnx_conversion',
            'original_size_mb': original_size,
            'onnx_size_mb': onnx_size,
            'output_path': optimized_path,
            'original_onnx_path': onnx_path
        }
    
    def _create_dummy_input(self, model):
        """创建示例输入"""
        # 这里需要根据模型的实际输入格式创建
        # 暂时使用通用的输入格式
        
        if hasattr(model, 'config'):
            # 对于Transformer模型
            seq_length = getattr(model.config, 'max_position_embeddings', 512)
            batch_size = 1
            return torch.randint(0, 1000, (batch_size, seq_length))
        else:
            # 对于其他模型，使用默认输入
            return torch.randn(1, 768)  # 假设输入维度
    
    def _optimize_onnx_model(self, input_path: str, output_path: str):
        """优化ONNX模型"""
        try:
            # 使用ONNX Runtime的优化工具
            from onnxruntime.tools import optimizer
            
            # 基本优化
            optimized_model = optimizer.optimize_model(
                input_path,
                model_type='bert',  # 可以根据实际模型类型调整
                num_heads=12,
                hidden_size=768
            )
            
            optimized_model.save_model_to_file(output_path)
            
        except ImportError:
            # 如果没有优化工具，直接复制文件
            shutil.copy2(input_path, output_path)
            logger.warning("ONNX optimization tools not available, using original model")

class ModelOptimizer:
    """模型优化器主类"""
    
    def __init__(self):
        self.quantizer = ModelQuantizer()
        self.distiller = ModelDistiller()
        self.onnx_converter = ONNXConverter()
        self.benchmark = ModelBenchmark()
    
    def optimize_model(self, config: OptimizationConfig) -> Dict[str, Any]:
        """优化模型"""
        logger.info(f"Starting model optimization: {config.optimization_type.value}")
        
        # 记录原始模型性能
        original_metrics = self._get_baseline_metrics(config)
        
        # 执行优化
        if config.optimization_type == OptimizationType.QUANTIZATION:
            result = self.quantizer.quantize_model(config)
        elif config.optimization_type == OptimizationType.DISTILLATION:
            result = self.distiller.distill_model(config)
        elif config.optimization_type == OptimizationType.ONNX_CONVERSION:
            result = self.onnx_converter.convert_to_onnx(config)
        else:
            raise ValueError(f"Unsupported optimization type: {config.optimization_type}")
        
        # 测试优化后的模型性能
        optimized_metrics = self._get_optimized_metrics(config, result['output_path'])
        
        # 计算性能对比
        performance_comparison = self._compare_performance(original_metrics, optimized_metrics)
        
        # 合并结果
        result.update({
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'performance_comparison': performance_comparison,
            'optimization_config': config.__dict__
        })
        
        # 保存优化报告
        self._save_optimization_report(result, config.output_path)
        
        return result
    
    def _get_baseline_metrics(self, config: OptimizationConfig) -> Dict[str, float]:
        """获取原始模型基准指标"""
        try:
            # 创建测试数据
            test_data = self._create_test_data(config)
            
            # 基准测试
            metrics = self.benchmark.benchmark_model(
                config.model_path, test_data, config.target_device
            )
            
            return metrics
        except Exception as e:
            logger.warning(f"Failed to get baseline metrics: {e}")
            return {}
    
    def _get_optimized_metrics(self, config: OptimizationConfig, optimized_path: str) -> Dict[str, float]:
        """获取优化后模型指标"""
        try:
            # 创建测试数据
            test_data = self._create_test_data(config)
            
            # 基准测试
            metrics = self.benchmark.benchmark_model(
                optimized_path, test_data, config.target_device
            )
            
            return metrics
        except Exception as e:
            logger.warning(f"Failed to get optimized metrics: {e}")
            return {}
    
    def _create_test_data(self, config: OptimizationConfig) -> List[torch.Tensor]:
        """创建测试数据"""
        test_data = []
        
        for _ in range(min(100, config.max_samples)):
            # 根据模型类型创建不同的测试数据
            if 'bert' in config.model_path.lower() or 'transformer' in config.model_path.lower():
                # 文本模型
                data = torch.randint(0, 1000, (1, 128))  # (batch_size, seq_len)
            elif 'resnet' in config.model_path.lower() or 'vision' in config.model_path.lower():
                # 视觉模型
                data = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
            else:
                # 默认数据
                data = torch.randn(1, 768)
            
            test_data.append(data)
        
        return test_data
    
    def _compare_performance(self, original: Dict[str, float], optimized: Dict[str, float]) -> Dict[str, Any]:
        """比较性能"""
        comparison = {}
        
        for metric in original.keys():
            if metric in optimized:
                original_val = original[metric]
                optimized_val = optimized[metric]
                
                if original_val > 0:
                    improvement = (original_val - optimized_val) / original_val * 100
                    comparison[f"{metric}_improvement_percent"] = improvement
                
                comparison[f"{metric}_ratio"] = optimized_val / original_val if original_val > 0 else 0
        
        return comparison
    
    def _save_optimization_report(self, result: Dict[str, Any], output_path: str):
        """保存优化报告"""
        report_path = output_path.replace('.pth', '_report.json').replace('.pt', '_report.json').replace('.onnx', '_report.json')
        
        # 确保所有数据都可以JSON序列化
        serializable_result = self._make_json_serializable(result)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization report saved to: {report_path}")
    
    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

def main():
    """主函数 - 示例用法"""
    # VAD模型优化示例
    vad_config = OptimizationConfig(
        model_path="/path/to/vad_model.pth",
        output_path="/path/to/optimized/vad_model_quantized.pth",
        optimization_type=OptimizationType.QUANTIZATION,
        quantization_method=QuantizationMethod.DYNAMIC,
        target_device="cpu",
        compression_ratio=0.5
    )
    
    # ASR模型优化示例
    asr_config = OptimizationConfig(
        model_path="/path/to/asr_model.pth",
        output_path="/path/to/optimized/asr_model.onnx",
        optimization_type=OptimizationType.ONNX_CONVERSION,
        target_device="cpu"
    )
    
    # LLM模型蒸馏示例
    llm_config = OptimizationConfig(
        model_path="/path/to/student_model.pth",
        output_path="/path/to/optimized/llm_model_distilled.pth",
        optimization_type=OptimizationType.DISTILLATION,
        teacher_model_path="/path/to/teacher_model.pth",
        compression_ratio=0.3,
        distillation_temperature=4.0,
        distillation_alpha=0.7
    )
    
    # 创建优化器
    optimizer = ModelOptimizer()
    
    # 执行优化
    configs = [vad_config, asr_config, llm_config]
    
    for config in configs:
        try:
            result = optimizer.optimize_model(config)
            print(f"Optimization completed for {config.model_path}")
            print(f"Compression ratio: {result.get('compression_ratio', 'N/A')}")
            print(f"Output path: {result['output_path']}")
            print("-" * 50)
        except Exception as e:
            print(f"Optimization failed for {config.model_path}: {e}")

if __name__ == "__main__":
    main()