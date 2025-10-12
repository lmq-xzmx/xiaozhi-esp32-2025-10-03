#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_encoding_fixes():
    original_content = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤"
    print(f"原始内容: {original_content}")
    
    # 方法1: 尝试将字符串当作latin-1编码的字节，然后解码为UTF-8
    try:
        print("\n方法1: Latin-1 -> UTF-8")
        # 过滤掉超出Latin-1范围的字符
        filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
        print(f"过滤后内容: {filtered_content}")
        
        # 将字符串编码为latin-1字节，然后解码为UTF-8
        fixed_content = filtered_content.encode('latin-1').decode('utf-8')
        print(f"修复结果: {fixed_content}")
        
        # 检查是否包含中文
        chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
        print(f"中文字符数: {chinese_count}")
        
    except Exception as e:
        print(f"方法1失败: {e}")
    
    # 方法2: 尝试Windows-1252编码
    try:
        print("\n方法2: Windows-1252 -> UTF-8")
        filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
        print(f"过滤后内容: {filtered_content}")
        
        fixed_content = filtered_content.encode('windows-1252').decode('utf-8')
        print(f"修复结果: {fixed_content}")
        
        chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
        print(f"中文字符数: {chinese_count}")
        
    except Exception as e:
        print(f"方法2失败: {e}")
    
    # 方法3: 尝试ISO-8859-1编码
    try:
        print("\n方法3: ISO-8859-1 -> UTF-8")
        filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
        print(f"过滤后内容: {filtered_content}")
        
        fixed_content = filtered_content.encode('iso-8859-1').decode('utf-8')
        print(f"修复结果: {fixed_content}")
        
        chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
        print(f"中文字符数: {chinese_count}")
        
    except Exception as e:
        print(f"方法3失败: {e}")
    
    # 方法4: 分析字符编码模式
    print("\n方法4: 字符编码分析")
    for i, char in enumerate(original_content[:20]):  # 只分析前20个字符
        print(f"位置{i}: '{char}' -> Unicode: {ord(char)} -> Hex: {hex(ord(char))}")

if __name__ == "__main__":
    test_encoding_fixes()