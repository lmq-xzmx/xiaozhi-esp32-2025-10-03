#!/usr/bin/env python3
"""
智能配置脚本：启用manager-api配置模式，但保持ASR使用本地配置以确保性能
"""
import yaml
import os

def configure_hybrid_mode():
    """配置混合模式：API配置 + ASR本地配置"""
    config_path = '/opt/xiaozhi-esp32-server/data/.config.yaml'
    
    # 读取当前配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== 当前配置状态 ===")
    print(f"read_config_from_api: {config.get('read_config_from_api')}")
    print(f"manager-api URL: {config.get('manager-api', {}).get('url', '未设置')}")
    
    # 启用manager-api配置模式
    config['read_config_from_api'] = True
    
    # 设置manager-api URL
    if 'manager-api' not in config:
        config['manager-api'] = {}
    
    config['manager-api']['url'] = 'http://xiaozhi-esp32-server-web:8002/xiaozhi'
    config['manager-api']['secret'] = '06af18d1-03a3-4abf-adab-b8386975508f'
    
    # 添加本地ASR配置覆盖，确保ASR仍使用本地配置
    config['local_override'] = {
        'ASR': {
            'SenseVoiceStream': {
                'type': 'sensevoice_stream',
                'model_dir': 'models/SenseVoiceSmall',
                'output_dir': 'tmp/',
                'vad_model': 'fsmn-vad',
                'enable_realtime': True,
                'chunk_size': 1024
            }
        },
        'selected_module': {
            'ASR': 'SenseVoiceStream'
        }
    }
    
    # 写回配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print("\n=== 配置更新完成 ===")
    print("✅ 启用manager-api配置模式")
    print("✅ 设置manager-api URL和secret")
    print("✅ 添加ASR本地配置覆盖")
    print("✅ LLM、TTS、VLLM等模块将从API获取配置")
    print("✅ ASR模块仍使用本地配置确保性能")
    
    # 验证配置
    with open(config_path, 'r', encoding='utf-8') as f:
        new_config = yaml.safe_load(f)
    
    print("\n=== 验证新配置 ===")
    print(f"read_config_from_api: {new_config.get('read_config_from_api')}")
    print(f"manager-api URL: {new_config.get('manager-api', {}).get('url')}")
    print(f"ASR本地覆盖: {new_config.get('local_override', {}).get('selected_module', {}).get('ASR')}")

if __name__ == "__main__":
    configure_hybrid_mode()