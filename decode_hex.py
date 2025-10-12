#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从数据库查询得到的十六进制数据
hex_data = [
    "E5BD93E784B6E58FAFE4BBA5EFBC81E6A0B9E68DAEE682A8E79A84E99C80E6B182EFBC8CE68891E68EA8E88D90E4BBA5E4B88BE5ADA6E4B9A0E8B584E69699EFBC9A31",
    "E4BDA0E5A5BDEFBC8CE68891E683B3E4BA86E8A7A3E4B880E4B88BE4BB8AE5A4A9E79A84E5A4A9E6B094E68385E586B5E38082",
    "E8B0A2E8B0A2E4BDA0E79A84E5BBBAE8AEAEEFBC8CE982A3E6988EE5A4A9E591A2EFBC9F"
]

print("解码数据库中的十六进制数据:")
print("=" * 60)

for i, hex_str in enumerate(hex_data, 1):
    try:
        # 将十六进制字符串转换为字节
        byte_data = bytes.fromhex(hex_str)
        # 解码为UTF-8字符串
        decoded_text = byte_data.decode('utf-8')
        
        print(f"记录 {i}:")
        print(f"  十六进制: {hex_str[:50]}...")
        print(f"  解码结果: {decoded_text}")
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in decoded_text)
        print(f"  包含中文: {has_chinese}")
        print("-" * 40)
        
    except Exception as e:
        print(f"记录 {i} 解码失败: {e}")
        print("-" * 40)