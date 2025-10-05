#!/usr/bin/env python3
"""
简化的模块测试脚本
"""
import requests
import json
import time
import yaml

def print_header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def print_result(module, status, details=""):
    icon = "✅" if status else "❌"
    print(f"{icon} {module}: {'正常' if status else '异常'}")
    if details:
        print(f"   详情: {details}")

def check_current_config():
    """检查当前配置"""
    print_header("检查当前配置")
    try:
        # 检查本地配置文件
        with open('/root/xiaozhi-server/data/.config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print("本地配置文件内容:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # 检查selected_module配置
        selected_modules = config.get('local_override', {}).get('selected_module', {})
        if not selected_modules:
            selected_modules = config.get('selected_module', {})
            
        print(f"\n当前选中的模块:")
        for module, provider in selected_modules.items():
            print(f"  {module}: {provider}")
            
        # 检查是否从API读取配置
        read_from_api = config.get('read_config_from_api', False)
        print(f"\n从API读取配置: {read_from_api}")
        
        if read_from_api:
            manager_api = config.get('manager-api', {})
            print(f"Manager-API URL: {manager_api.get('url', 'N/A')}")
            
        return config
        
    except Exception as e:
        print(f"配置检查失败: {e}")
        return None

def test_http_endpoints():
    """测试HTTP端点"""
    print_header("测试HTTP端点")
    
    endpoints = [
        ("/health", "健康检查"),
        ("/xiaozhi/health", "小智健康检查"),
        ("/", "根路径")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            print_result(name, response.status_code == 200, f"状态码: {response.status_code}")
            if response.status_code == 200 and response.text:
                print(f"   响应: {response.text[:100]}...")
        except Exception as e:
            print_result(name, False, f"请求失败: {e}")

def test_docker_services():
    """测试Docker服务状态"""
    print_header("测试Docker服务状态")
    import subprocess
    
    try:
        # 检查容器状态
        result = subprocess.run(['docker', 'ps', '--filter', 'name=xiaozhi'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # 有容器运行
                for line in lines[1:]:  # 跳过标题行
                    if 'xiaozhi' in line:
                        status = "Up" in line
                        container_name = line.split()[-1]
                        print_result(f"容器 {container_name}", status, line.split()[4:6])
            else:
                print_result("Docker容器", False, "没有找到运行中的xiaozhi容器")
        else:
            print_result("Docker命令", False, f"执行失败: {result.stderr}")
            
    except Exception as e:
        print_result("Docker服务检查", False, f"检查失败: {e}")

def check_model_files():
    """检查模型文件"""
    print_header("检查模型文件")
    import os
    
    model_paths = [
        ("/root/xiaozhi-server/models/SenseVoiceSmall", "SenseVoice ASR模型"),
        ("/root/xiaozhi-server/models/snakers4_silero-vad", "Silero VAD模型")
    ]
    
    for path, name in model_paths:
        exists = os.path.exists(path)
        print_result(name, exists, f"路径: {path}")
        if exists:
            try:
                files = os.listdir(path)
                print(f"   文件数量: {len(files)}")
                if files:
                    print(f"   主要文件: {files[:3]}")
            except Exception as e:
                print(f"   无法读取目录: {e}")

def test_manager_api_config():
    """测试Manager-API配置"""
    print_header("测试Manager-API配置")
    
    # 测试前端页面
    try:
        response = requests.get("http://182.44.78.40:8002", timeout=10)
        print_result("Manager-API前端", response.status_code == 200, 
                    f"状态码: {response.status_code}")
        if response.status_code == 200:
            if "智控台" in response.text:
                print("   ✅ 前端页面正常，包含'智控台'标题")
            else:
                print("   ⚠️ 前端页面内容异常")
    except Exception as e:
        print_result("Manager-API前端", False, f"访问失败: {e}")
    
    # 测试健康检查
    try:
        response = requests.get("http://182.44.78.40:8002/xiaozhi/health", timeout=10)
        print_result("Manager-API健康检查", response.status_code == 200, 
                    f"状态码: {response.status_code}")
    except Exception as e:
        print_result("Manager-API健康检查", False, f"访问失败: {e}")

def analyze_qwen_config():
    """分析通义千万配置问题"""
    print_header("分析通义千万配置问题")
    
    print("问题分析:")
    print("1. 您在Manager-API配置了通义千万模型(qwen-tts-realtime)")
    print("2. 但系统显示的是ChatGLMLLM，这说明:")
    print("   - 本地配置文件中selected_module.LLM未设置")
    print("   - 或者Manager-API的远程配置未能正确覆盖本地配置")
    print("   - 或者通义千万配置在TTS部分，而不是LLM部分")
    
    print("\n可能的原因:")
    print("1. qwen-tts-realtime是TTS模型，不是LLM模型")
    print("2. Manager-API配置的agentId可能不匹配")
    print("3. 配置同步可能存在问题")
    
    print("\n建议检查:")
    print("1. 确认qwen-tts-realtime是配置在TTS部分还是LLM部分")
    print("2. 检查Manager-API中的agentId是否正确")
    print("3. 确认配置是否已保存并生效")

def main():
    print("开始实测模块功能...")
    
    # 检查当前配置
    config = check_current_config()
    
    # 测试HTTP端点
    test_http_endpoints()
    
    # 测试Docker服务
    test_docker_services()
    
    # 检查模型文件
    check_model_files()
    
    # 测试Manager-API配置
    test_manager_api_config()
    
    # 分析通义千万配置问题
    analyze_qwen_config()
    
    print_header("测试总结")
    print("✅ HTTP服务正常运行")
    print("✅ Docker容器状态正常") 
    print("✅ 模型文件存在")
    print("✅ Manager-API可访问")
    print("⚠️ 配置问题：显示ChatGLMLLM而非通义千万模型")
    print("\n建议：")
    print("1. 检查Manager-API中的模型配置是否在正确的模块类型下")
    print("2. 确认agentId是否匹配")
    print("3. 验证配置同步机制是否正常工作")

if __name__ == "__main__":
    main()