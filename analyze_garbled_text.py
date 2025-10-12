#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 分析乱码文本中的字符
garbled_text = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯"

print(f"乱码文本: {garbled_text}")
print(f"文本长度: {len(garbled_text)}")
print("\n字符分析:")

for i, char in enumerate(garbled_text):
    char_code = ord(char)
    print(f"  [{i:2d}] '{char}' -> {char_code} (0x{char_code:04x})")
    if char_code > 255:
        print(f"       ^^^ 超出255范围的字符!")

print("\n检查是否包含特殊字符:")
special_chars = {8482: '™', 8225: '‡'}
for char in garbled_text:
    char_code = ord(char)
    if char_code in special_chars:
        print(f"  发现特殊字符: '{char}' ({char_code}) -> {special_chars[char_code]}")

print("\n检查中文字符:")
chinese_chars = [char for char in garbled_text if '\u4e00' <= char <= '\u9fff']
print(f"  中文字符数量: {len(chinese_chars)}")
if chinese_chars:
    print(f"  中文字符: {chinese_chars}")

print("\n检查乱码指示符:")
garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï']
found_indicators = [char for char in garbled_text if char in garbled_indicators]
print(f"  乱码指示符: {found_indicators}")