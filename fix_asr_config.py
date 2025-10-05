#!/usr/bin/env python3
import yaml
import re

def fix_asr_config():
    config_path = '/opt/xiaozhi-esp32-server/config.yaml'
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式找到并替换ASR配置
    # 查找selected_module部分的ASR配置
    pattern = r'(selected_module:.*?ASR:\s*)[^\n]*'
    replacement = r'\1sensevoice_stream'
    
    # 执行替换
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 写回文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("ASR配置已修复为: sensevoice_stream")
    
    # 验证YAML格式
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        print("YAML格式验证通过")
    except Exception as e:
        print(f"YAML格式错误: {e}")

if __name__ == "__main__":
    fix_asr_config()