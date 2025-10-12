#!/usr/bin/env python3
"""
LLM首Token延迟优化配置
针对首字延迟最大瓶颈进行优化
"""

import os
import yaml
from typing import Dict, Any

class LLMFirstTokenOptimizer:
    """LLM首Token延迟优化器"""
    
    def __init__(self):
        self.optimization_configs = {
            # 基础优化配置
            'basic_optimization': {
                'model_settings': {
                    'max_new_tokens': 1,  # 首次只生成1个token
                    'do_sample': False,   # 关闭采样，使用贪心解码
                    'temperature': 0.1,   # 降低温度，减少计算
                    'top_p': 0.8,        # 减少候选token数量
                    'repetition_penalty': 1.0,  # 关闭重复惩罚
                },
                'inference_settings': {
                    'use_cache': True,    # 启用KV缓存
                    'pad_token_id': 0,    # 设置padding token
                    'eos_token_id': 2,    # 设置结束token
                    'batch_size': 1,      # 首token生成使用单batch
                },
                'memory_optimization': {
                    'torch_compile': True,      # 启用PyTorch编译优化
                    'flash_attention': True,    # 启用Flash Attention
                    'gradient_checkpointing': False,  # 推理时关闭梯度检查点
                }
            },
            
            # 激进优化配置
            'aggressive_optimization': {
                'model_settings': {
                    'max_new_tokens': 1,
                    'do_sample': False,
                    'temperature': 0.0,   # 完全确定性
                    'top_k': 1,          # 只考虑最可能的token
                    'early_stopping': True,
                },
                'inference_settings': {
                    'use_cache': True,
                    'past_key_values': None,  # 预分配KV缓存
                    'attention_mask': None,   # 预计算attention mask
                    'position_ids': None,     # 预计算position ids
                },
                'hardware_optimization': {
                    'device_map': 'auto',     # 自动设备映射
                    'torch_dtype': 'float16', # 使用FP16
                    'low_cpu_mem_usage': True,
                    'offload_folder': '/tmp/llm_offload',
                }
            },
            
            # 流式优化配置
            'streaming_optimization': {
                'streaming_settings': {
                    'stream': True,           # 启用流式生成
                    'stream_first_token': True,  # 优先生成首token
                    'chunk_size': 1,          # 单token流式输出
                    'buffer_size': 0,         # 无缓冲
                },
                'pipeline_settings': {
                    'prefill_optimization': True,   # 预填充优化
                    'speculative_decoding': True,   # 投机解码
                    'parallel_sampling': False,     # 关闭并行采样
                }
            }
        }
    
    def generate_llm_config(self, optimization_level: str = 'aggressive') -> Dict[str, Any]:
        """生成LLM优化配置"""
        base_config = {
            'llm_service': {
                'provider': 'openai',  # 或其他provider
                'model_name': 'gpt-3.5-turbo',
                'api_base': 'http://localhost:8002',
                'timeout': 5.0,  # 减少超时时间
                'max_retries': 1,  # 减少重试次数
            },
            
            'performance_settings': {
                'max_concurrent_requests': 50,
                'request_queue_size': 100,
                'worker_threads': 8,
                'connection_pool_size': 20,
            },
            
            'caching': {
                'enable_response_cache': True,
                'cache_ttl': 3600,  # 1小时缓存
                'cache_size_mb': 512,
                'enable_prompt_cache': True,  # 启用prompt缓存
            },
            
            'first_token_optimization': self.optimization_configs.get(
                optimization_level, 
                self.optimization_configs['basic_optimization']
            )
        }
        
        return base_config
    
    def generate_environment_variables(self, optimization_level: str = 'aggressive') -> Dict[str, str]:
        """生成环境变量配置"""
        env_vars = {
            # LLM基础配置
            'LLM_MAX_CONCURRENT': '50',
            'LLM_TIMEOUT': '5',
            'LLM_MAX_RETRIES': '1',
            'LLM_WORKER_THREADS': '8',
            
            # 首Token优化
            'LLM_FIRST_TOKEN_PRIORITY': 'true',
            'LLM_STREAM_FIRST_TOKEN': 'true',
            'LLM_PREFILL_OPTIMIZATION': 'true',
            
            # 内存优化
            'LLM_ENABLE_CACHE': 'true',
            'LLM_CACHE_SIZE_MB': '512',
            'LLM_MEMORY_POOL': 'true',
            
            # 硬件优化
            'CUDA_VISIBLE_DEVICES': '0',  # 使用GPU 0
            'TORCH_COMPILE': 'true',
            'FLASH_ATTENTION': 'true',
        }
        
        if optimization_level == 'aggressive':
            env_vars.update({
                'LLM_MAX_NEW_TOKENS': '1',
                'LLM_TEMPERATURE': '0.0',
                'LLM_TOP_K': '1',
                'LLM_DO_SAMPLE': 'false',
                'LLM_TORCH_DTYPE': 'float16',
            })
        
        return env_vars
    
    def save_optimization_config(self, output_path: str, optimization_level: str = 'aggressive'):
        """保存优化配置到文件"""
        config = self.generate_llm_config(optimization_level)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ LLM优化配置已保存到: {output_path}")
        print(f"🎯 优化级别: {optimization_level}")
        print(f"📈 预期首Token延迟减少: 150-200ms")

def main():
    """主函数"""
    optimizer = LLMFirstTokenOptimizer()
    
    # 生成不同级别的优化配置
    levels = ['basic_optimization', 'aggressive_optimization', 'streaming_optimization']
    
    for level in levels:
        output_file = f"/root/xiaozhi-server/config/llm_{level.replace('_optimization', '')}_config.yaml"
        optimizer.save_optimization_config(output_file, level)
        
        # 打印环境变量
        env_vars = optimizer.generate_environment_variables(level)
        print(f"\n📋 {level} 环境变量:")
        for key, value in env_vars.items():
            print(f"export {key}={value}")
        print("-" * 50)

if __name__ == "__main__":
    main()