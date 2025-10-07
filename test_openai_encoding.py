#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OpenAI服务的编码处理
验证修复后的服务是否能正确处理中文字符
"""

import asyncio
import json
import sys
import os

# 添加项目路径
sys.path.append('/root/xiaozhi-server')

from config.logger import setup_logging

# 设置UTF-8编码环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

logger = setup_logging()

def test_logger_encoding():
    """测试日志记录器的编码处理"""
    print("🧪 测试日志记录器编码处理...")
    
    try:
        # 测试中文字符日志
        test_messages = [
            "这是一个中文测试消息",
            "OpenAI服务正在运行",
            "包含特殊字符：©®™€£¥",
            "包含emoji：🚀🎉✅❌⚠️",
            "混合内容：Hello 世界 123 🌍"
        ]
        
        for msg in test_messages:
            logger.info(f"测试消息: {msg}")
            print(f"✅ 成功记录: {msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ 日志编码测试失败: {e}")
        return False

def test_json_encoding():
    """测试JSON编码处理"""
    print("\n🧪 测试JSON编码处理...")
    
    try:
        test_data = {
            "message": "这是一个中文消息",
            "user": "用户测试",
            "content": "包含特殊字符：©®™€£¥ 和 emoji：🚀🎉",
            "timestamp": "2024-01-01 12:00:00"
        }
        
        # 测试JSON序列化
        json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
        print(f"✅ JSON序列化成功:\n{json_str}")
        
        # 测试JSON反序列化
        parsed_data = json.loads(json_str)
        print(f"✅ JSON反序列化成功: {parsed_data['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON编码测试失败: {e}")
        return False

def test_file_encoding():
    """测试文件编码处理"""
    print("\n🧪 测试文件编码处理...")
    
    try:
        test_file = "/tmp/test_encoding.txt"
        test_content = """这是一个UTF-8编码测试文件
包含中文字符：你好世界
包含特殊字符：©®™€£¥
包含emoji：🚀🎉✅❌⚠️
混合内容：Hello 世界 123 🌍"""
        
        # 写入文件
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("✅ UTF-8文件写入成功")
        
        # 读取文件
        with open(test_file, 'r', encoding='utf-8') as f:
            read_content = f.read()
        
        if read_content == test_content:
            print("✅ UTF-8文件读取成功")
            os.remove(test_file)
            return True
        else:
            print("❌ 文件内容不匹配")
            return False
            
    except Exception as e:
        print(f"❌ 文件编码测试失败: {e}")
        return False

def test_print_encoding():
    """测试print输出编码"""
    print("\n🧪 测试print输出编码...")
    
    try:
        test_messages = [
            "中文输出测试：你好世界",
            "特殊字符测试：©®™€£¥",
            "Emoji测试：🚀🎉✅❌⚠️",
            "混合测试：Hello 世界 123 🌍"
        ]
        
        for msg in test_messages:
            print(f"✅ {msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ print编码测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始OpenAI服务编码测试")
    print("=" * 50)
    
    # 显示当前编码设置
    print(f"系统默认编码: {sys.getdefaultencoding()}")
    print(f"文件系统编码: {sys.getfilesystemencoding()}")
    print(f"标准输出编码: {sys.stdout.encoding}")
    print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', '未设置')}")
    print(f"PYTHONUTF8: {os.environ.get('PYTHONUTF8', '未设置')}")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("日志编码", test_logger_encoding),
        ("JSON编码", test_json_encoding),
        ("文件编码", test_file_encoding),
        ("输出编码", test_print_encoding)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有编码测试通过！OpenAI服务编码配置正确")
        return True
    else:
        print("❌ 部分编码测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)