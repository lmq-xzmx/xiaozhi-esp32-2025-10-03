#!/usr/bin/env python3
"""
LLM性能测试配置
"""

def get_test_config():
    """获取测试配置"""
    return {
        "llm": {
            "apis": {
                "qwen": {
                    "api_key": "test-key",  # 使用测试密钥
                    "base_url": "http://182.44.78.40:8002/api/v1/chat/completions",
                    "model": "qwen3-235b",
                    "weight": 1.0
                }
            }
        },
        "module_test": {
            "test_sentences": [
                "你好，我今天心情不太好，能安慰一下我吗？",
                "帮我查一下明天的天气如何？",
                "我想听一个有趣的故事，你能给我讲一个吗？",
                "现在几点了？今天是星期几？",
                "我想设置一个明天早上8点的闹钟提醒我开会",
            ]
        }
    }