#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def has_valid_chinese_content(text):
    """检查文本是否包含有效的中文字符"""
    if not text:
        return False
    
    # 检查是否包含中文字符
    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    
    if not chinese_chars:
        return False
    
    # 计算中文字符比例
    chinese_ratio = len(chinese_chars) / len(text)
    
    # 如果中文字符比例太低，可能是乱码
    if chinese_ratio < 0.1:
        return False
    
    # 检查是否有连续的有效中文词汇
    # 简单检查：是否有连续的2个或以上中文字符
    consecutive_chinese = 0
    max_consecutive = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            consecutive_chinese += 1
            max_consecutive = max(max_consecutive, consecutive_chinese)
        else:
            consecutive_chinese = 0
    
    return max_consecutive >= 2

# 测试乱码文本
garbled_text = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤"

print("测试乱码文本:", garbled_text)
print("文本长度:", len(garbled_text))

# 分析每个字符
chinese_chars = []
for i, char in enumerate(garbled_text):
    unicode_val = ord(char)
    is_chinese = '\u4e00' <= char <= '\u9fff'
    print(f"字符 {i}: '{char}' (Unicode: {unicode_val}, 0x{unicode_val:04x}) - 中文: {is_chinese}")
    if is_chinese:
        chinese_chars.append(char)

print(f"\n发现的中文字符: {chinese_chars}")
print(f"中文字符数量: {len(chinese_chars)}")
print(f"中文字符比例: {len(chinese_chars) / len(garbled_text):.2%}")

result = has_valid_chinese_content(garbled_text)
print(f"\nhas_valid_chinese_content 结果: {result}")

# 测试正确的中文文本
correct_text = "这是AI对设备58:8c:81:65:52:48会话错误的回复"
print(f"\n正确中文文本: {correct_text}")
print(f"has_valid_chinese_content 结果: {has_valid_chinese_content(correct_text)}")