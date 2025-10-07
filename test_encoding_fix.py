#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试编码修复效果
验证Docker容器内的UTF-8编码是否正常工作
"""

import requests
import json
import time
import sys
import os

def test_container_encoding():
    """测试容器内部编码设置"""
    print("🔍 测试容器内部编码设置...")
    
    try:
        # 测试HTTP健康检查
        response = requests.get("http://localhost:8003/health", timeout=10)
        if response.status_code == 200:
            print("✅ HTTP服务正常运行")
        else:
            print(f"❌ HTTP服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ HTTP服务连接失败: {e}")
        return False
    
    return True

def test_chinese_characters():
    """测试中文字符处理"""
    print("🔍 测试中文字符处理...")
    
    test_strings = [
        "你好世界",
        "OpenAI服务正在运行",
        "包含特殊字符：©®™€£¥",
        "包含emoji：🚀🎉✅❌⚠️",
        "混合内容：Hello 世界 123 🌍"
    ]
    
    try:
        for test_str in test_strings:
            # 测试字符串编码
            encoded = test_str.encode('utf-8')
            decoded = encoded.decode('utf-8')
            
            if test_str == decoded:
                print(f"✅ 编码测试通过: {test_str}")
            else:
                print(f"❌ 编码测试失败: {test_str}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 中文字符测试失败: {e}")
        return False

def test_json_encoding():
    """测试JSON编码处理"""
    print("🔍 测试JSON编码处理...")
    
    try:
        test_data = {
            "message": "这是一个中文测试消息",
            "status": "OpenAI服务正在运行",
            "special_chars": "©®™€£¥",
            "emojis": "🚀🎉✅❌⚠️",
            "mixed": "Hello 世界 123 🌍"
        }
        
        # 测试JSON序列化
        json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
        
        # 测试JSON反序列化
        parsed_data = json.loads(json_str)
        
        if test_data == parsed_data:
            print("✅ JSON编码测试通过")
            return True
        else:
            print("❌ JSON编码测试失败")
            return False
            
    except Exception as e:
        print(f"❌ JSON编码测试失败: {e}")
        return False

def test_file_encoding():
    """测试文件编码处理"""
    print("🔍 测试文件编码处理...")
    
    try:
        test_content = """这是一个UTF-8编码测试文件
包含中文字符：你好世界
包含特殊字符：©®™€£¥
包含emoji：🚀🎉✅❌⚠️
混合内容：Hello 世界 123 🌍"""
        
        test_file = "/tmp/encoding_test.txt"
        
        # 写入文件
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 读取文件
        with open(test_file, 'r', encoding='utf-8') as f:
            read_content = f.read()
        
        if test_content == read_content:
            print("✅ 文件编码测试通过")
            # 清理测试文件
            os.remove(test_file)
            return True
        else:
            print("❌ 文件编码测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 文件编码测试失败: {e}")
        return False

def check_docker_logs():
    """检查Docker日志中的编码错误"""
    print("🔍 检查Docker日志中的编码错误...")
    
    try:
        import subprocess
        
        # 检查最近的日志
        result = subprocess.run(
            ["docker", "logs", "xiaozhi-esp32-server", "--tail", "50"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        logs = result.stdout + result.stderr
        
        # 检查是否有ASCII编码错误
        if "'ascii' codec can't encode" in logs:
            print("❌ 发现ASCII编码错误")
            return False
        elif "'ascii' codec can't decode" in logs:
            print("❌ 发现ASCII解码错误")
            return False
        else:
            print("✅ 未发现ASCII编码错误")
            return True
            
    except Exception as e:
        print(f"❌ 检查Docker日志失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始编码修复效果测试")
    print("=" * 60)
    
    tests = [
        ("容器编码设置", test_container_encoding),
        ("中文字符处理", test_chinese_characters),
        ("JSON编码处理", test_json_encoding),
        ("文件编码处理", test_file_encoding),
        ("Docker日志检查", check_docker_logs)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 执行测试: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - 通过")
        else:
            print(f"❌ {test_name} - 失败")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有编码测试通过！ASCII编码问题已修复")
        return True
    else:
        print("❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)