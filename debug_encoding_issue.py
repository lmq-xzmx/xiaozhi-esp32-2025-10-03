#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 从API返回的实际乱码数据
api_response_sample = {
    "content": "\u00e4\u00bb\u0160\u00e5\u00a4\u00a9\u00e5\u00a4\u00a9\u00e6\u2122\u00b4\u00e6\u0153\u2014\u00ef\u00bc\u0152\u00e6\u00b8\u00a9\u00e5\u00ba\u00a6\u00e9\u20ac\u201a\u00e5\u00ae\u0153\u00ef\u00bc\u0152\u00e6\u02dc\u00af\u00e4\u00b8\u00aa\u00e5\u2021\u00ba\u00e9\u2014\u00a8\u00e7\u0161\u201e\u00e5\u00a5\u00bd\u00e6\u2014\u00a5\u00e5\u00ad\u0080\u00e3\u20ac\u201a"
}

def has_valid_chinese(text):
    """检查文本是否包含有效的中文字符"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def analyze_content(content):
    print(f"原始内容: {repr(content[:50])}")
    print(f"内容长度: {len(content)}")
    
    # 检查中文字符
    chinese_chars = [char for char in content if '\u4e00' <= char <= '\u9fff']
    print(f"包含中文字符: {len(chinese_chars) > 0}")
    print(f"中文字符数量: {len(chinese_chars)}")
    
    if chinese_chars:
        print(f"中文字符: {chinese_chars[:10]}")  # 显示前10个中文字符
    
    # 分析字符分布
    char_ranges = {
        'ASCII (0-127)': 0,
        'Latin-1 (128-255)': 0, 
        'Chinese (19968-40959)': 0,
        'Other': 0
    }
    
    for char in content:
        code = ord(char)
        if 0 <= code <= 127:
            char_ranges['ASCII (0-127)'] += 1
        elif 128 <= code <= 255:
            char_ranges['Latin-1 (128-255)'] += 1
        elif 0x4e00 <= code <= 0x9fff:
            char_ranges['Chinese (19968-40959)'] += 1
        else:
            char_ranges['Other'] += 1
    
    print("字符分布:")
    for range_name, count in char_ranges.items():
        if count > 0:
            print(f"  {range_name}: {count}")
    
    # 显示前20个字符的详细信息
    print("前20个字符详情:")
    for i, char in enumerate(content[:20]):
        code = ord(char)
        if 0x4e00 <= code <= 0x9fff:
            char_type = "中文"
        elif 0 <= code <= 127:
            char_type = "ASCII"
        elif 128 <= code <= 255:
            char_type = "Latin-1"
        else:
            char_type = "其他"
        print(f"  {i+1:2d}: '{char}' U+{code:04X} ({char_type})")

print("分析API返回的乱码内容:")
print("=" * 60)
analyze_content(api_response_sample["content"])

print("\n" + "=" * 60)
print("测试编码修复算法:")

# 测试编码修复
original_content = api_response_sample["content"]

# 方法1: 智能字节序列修复
try:
    byte_array = []
    valid_conversion = True
    for char in original_content:
        char_code = ord(char)
        if char_code <= 255:  # 只处理单字节字符
            byte_array.append(char_code)
        else:
            valid_conversion = False
            break
    
    if valid_conversion and byte_array:
        fixed_content = bytes(byte_array).decode('utf-8')
        print(f"\n方法1修复结果:")
        print(f"修复后内容: {repr(fixed_content[:50])}")
        print(f"包含中文: {has_valid_chinese(fixed_content)}")
        if has_valid_chinese(fixed_content):
            print(f"修复成功: {fixed_content}")
    else:
        print("方法1: 无法转换为字节数组")
        
except Exception as e:
    print(f"方法1失败: {e}")