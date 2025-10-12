#!/usr/bin/env python3
"""
ASR延迟优化配置
针对首字延迟的第二大瓶颈进行优化
"""

import os
import yaml
from typing import Dict, Any

class ASRLatencyOptimizer:
    """ASR延迟优化器"""
    
    def __init__(self):
        self.optimization_strategies = {
            # 快速响应优化
            'fast_response': {
                'model_settings': {
                    'model_type': 'streaming',     # 使用流式模型
                    'chunk_size': 160,             # 10ms音频块 (16kHz)
                    'overlap_size': 80,            # 5ms重叠
                    'vad_threshold': 0.3,          # 降低VAD阈值，更快检测
                    'silence_timeout': 0.5,        # 0.5秒静音超时
                },
                'inference_settings': {
                    'batch_size': 1,               # 单样本处理，最低延迟
                    'max_concurrent': 200,         # 高并发支持
                    'worker_threads': 16,          # 增加工作线程
                    'enable_fp16': True,           # FP16加速
                    'enable_int8': False,          # 关闭INT8，保证精度
                },
                'preprocessing': {
                    'skip_normalization': True,    # 跳过音频标准化
                    'fast_resampling': True,       # 快速重采样
                    'minimal_padding': True,       # 最小填充
                    'zero_copy': True,             # 零拷贝优化
                }
            },
            
            # 预测性优化
            'predictive': {
                'model_settings': {
                    'model_type': 'streaming',
                    'chunk_size': 320,             # 20ms音频块
                    'overlap_size': 160,           # 10ms重叠
                    'lookahead_frames': 3,         # 前瞻3帧
                    'early_prediction': True,      # 启用早期预测
                },
                'caching': {
                    'enable_model_cache': True,    # 模型缓存
                    'cache_size_mb': 8192,         # 8GB缓存
                    'enable_audio_cache': True,    # 音频特征缓存
                    'cache_ttl': 300,              # 5分钟TTL
                },
                'optimization': {
                    'model_warmup': True,          # 模型预热
                    'memory_pool': True,           # 内存池
                    'thread_affinity': True,       # CPU亲和性
                }
            },
            
            # 极限优化
            'extreme': {
                'model_settings': {
                    'model_type': 'streaming',
                    'chunk_size': 80,              # 5ms音频块，极低延迟
                    'overlap_size': 40,            # 2.5ms重叠
                    'vad_threshold': 0.2,          # 更低VAD阈值
                    'silence_timeout': 0.3,        # 0.3秒静音超时
                    'partial_results': True,       # 启用部分结果
                },
                'hardware_optimization': {
                    'use_gpu': True,               # 强制使用GPU
                    'gpu_memory_fraction': 0.8,    # 80% GPU内存
                    'mixed_precision': True,       # 混合精度
                    'tensorrt_optimization': True, # TensorRT优化
                },
                'system_optimization': {
                    'high_priority': True,         # 高优先级进程
                    'cpu_affinity': [0, 1, 2, 3],  # 绑定CPU核心
                    'memory_lock': True,           # 锁定内存
                    'disable_swap': True,          # 禁用交换
                }
            }
        }
    
    def generate_asr_config(self, strategy: str = 'fast_response') -> Dict[str, Any]:
        """生成ASR优化配置"""
        base_config = {
            'asr_service': {
                'provider': 'funasr',  # 或其他ASR provider
                'model_path': '/models/asr/streaming',
                'device': 'cuda:0',
                'timeout': 3.0,        # 减少超时时间
                'max_retries': 1,      # 减少重试
            },
            
            'performance_settings': {
                'max_concurrent_requests': 200,
                'request_queue_size': 500,
                'worker_threads': 16,
                'io_threads': 8,
                'connection_pool_size': 50,
            },
            
            'audio_processing': {
                'sample_rate': 16000,
                'channels': 1,
                'bit_depth': 16,
                'format': 'wav',
                'enable_vad': True,
                'vad_model': 'silero',
            },
            
            'latency_optimization': self.optimization_strategies.get(
                strategy,
                self.optimization_strategies['fast_response']
            )
        }
        
        return base_config
    
    def generate_environment_variables(self, strategy: str = 'fast_response') -> Dict[str, str]:
        """生成ASR优化环境变量"""
        base_env = {
            # ASR基础配置
            'ASR_MAX_CONCURRENT': '200',
            'ASR_WORKER_THREADS': '16',
            'ASR_IO_THREADS': '8',
            'ASR_TIMEOUT': '3',
            'ASR_MAX_RETRIES': '1',
            
            # 延迟优化
            'ASR_ENABLE_STREAMING': 'true',
            'ASR_CHUNK_SIZE': '160',
            'ASR_OVERLAP_SIZE': '80',
            'ASR_BATCH_SIZE': '1',
            
            # 性能优化
            'ASR_ENABLE_FP16': 'true',
            'ASR_ZERO_COPY': 'true',
            'ASR_MEMORY_POOL': 'true',
            'ASR_MODEL_WARMUP': 'true',
            
            # 缓存配置
            'ASR_ENABLE_CACHE': 'true',
            'ASR_CACHE_SIZE_MB': '8192',
            'ASR_CACHE_TTL': '300',
        }
        
        if strategy == 'extreme':
            base_env.update({
                'ASR_CHUNK_SIZE': '80',      # 5ms块
                'ASR_OVERLAP_SIZE': '40',    # 2.5ms重叠
                'ASR_VAD_THRESHOLD': '0.2',  # 更低阈值
                'ASR_SILENCE_TIMEOUT': '0.3', # 更短超时
                'ASR_HIGH_PRIORITY': 'true',
                'ASR_CPU_AFFINITY': '0,1,2,3',
                'ASR_TENSORRT': 'true',
            })
        elif strategy == 'predictive':
            base_env.update({
                'ASR_CHUNK_SIZE': '320',     # 20ms块
                'ASR_LOOKAHEAD_FRAMES': '3',
                'ASR_EARLY_PREDICTION': 'true',
                'ASR_CACHE_SIZE_MB': '8192',
            })
        
        return base_env
    
    def generate_startup_script(self, strategy: str = 'fast_response') -> str:
        """生成ASR优化启动脚本"""
        env_vars = self.generate_environment_variables(strategy)
        
        script_lines = [
            "#!/bin/bash",
            "# ASR延迟优化启动脚本",
            f"# 优化策略: {strategy}",
            "",
            "echo '🚀 启动ASR延迟优化配置'",
            "echo '=' * 50",
            "",
            "# 设置环境变量"
        ]
        
        for key, value in env_vars.items():
            script_lines.append(f"export {key}={value}")
        
        script_lines.extend([
            "",
            "# 系统优化",
            "echo '⚡ 应用系统优化...'",
        ])
        
        if strategy == 'extreme':
            script_lines.extend([
                "# 设置高优先级",
                "sudo renice -10 $$",
                "",
                "# 设置CPU亲和性",
                "taskset -cp 0,1,2,3 $$",
                "",
                "# 禁用CPU频率调节",
                "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "",
            ])
        
        script_lines.extend([
            "echo '📊 当前配置:'",
            f"echo '   策略: {strategy}'",
            "echo '   最大并发: $ASR_MAX_CONCURRENT'",
            "echo '   音频块大小: $ASR_CHUNK_SIZE'",
            "echo '   工作线程: $ASR_WORKER_THREADS'",
            "echo '   缓存大小: $ASR_CACHE_SIZE_MB MB'",
            "",
            "echo '🎯 预期延迟减少: 100-150ms'",
            "echo '⚠️  请监控系统资源使用情况'",
            "",
            "# 启动ASR服务",
            "echo '🚀 启动优化后的ASR服务...'",
            "python3 services/asr_service.py"
        ])
        
        return "\n".join(script_lines)
    
    def save_optimization_config(self, output_dir: str, strategy: str = 'fast_response'):
        """保存优化配置"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存配置文件
        config = self.generate_asr_config(strategy)
        config_path = os.path.join(output_dir, f"asr_{strategy}_config.yaml")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存启动脚本
        script_content = self.generate_startup_script(strategy)
        script_path = os.path.join(output_dir, f"start_asr_{strategy}.sh")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置脚本可执行权限
        os.chmod(script_path, 0o755)
        
        print(f"✅ ASR优化配置已保存:")
        print(f"   配置文件: {config_path}")
        print(f"   启动脚本: {script_path}")
        print(f"🎯 优化策略: {strategy}")
        print(f"📈 预期延迟减少: 100-150ms")

def main():
    """主函数"""
    optimizer = ASRLatencyOptimizer()
    
    # 生成不同策略的优化配置
    strategies = ['fast_response', 'predictive', 'extreme']
    
    for strategy in strategies:
        output_dir = f"/root/xiaozhi-server/optimization/asr_{strategy}"
        optimizer.save_optimization_config(output_dir, strategy)
        print("-" * 60)

if __name__ == "__main__":
    main()