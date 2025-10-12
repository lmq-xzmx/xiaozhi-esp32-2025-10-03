#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def has_valid_chinese_content(text):
    """检查文本是否包含有效的中文字符，排除乱码"""
    if not text or len(text.strip()) == 0:
        return False
    
    # 统计中文字符
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

def has_valid_chinese(text):
    """检查文本是否包含有效的中文字符，排除乱码"""
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï']
    has_garbled = any(indicator in text for indicator in garbled_indicators)
    # 只有包含中文且乱码指示符较少时才认为修复成功
    return chinese_count > 0 and (not has_garbled or chinese_count > sum(1 for c in text if c in garbled_indicators))

def test_repair_logic():
    original_content = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤"
    
    print(f"原始内容: {original_content}")
    print(f"has_valid_chinese_content 检查结果: {has_valid_chinese_content(original_content)}")
    
    if not has_valid_chinese_content(original_content):
        print("开始修复...")
        
        # 方法1: 智能字节序列修复
        try:
            print("尝试方法1: 智能字节序列修复")
            # 将字符串转换为字节数组，处理混合字符
            byte_array = []
            for char in original_content:
                char_code = ord(char)
                print(f"字符: '{char}' -> 编码: {char_code}")
                if char_code <= 255:  # 单字节字符
                    byte_array.append(char_code)
                elif char_code == 8482:  # ™ 字符，映射为UTF-8字节序列
                    byte_array.extend([0xe2, 0x84, 0xa2])  # ™ 的UTF-8编码
                elif char_code == 8225:  # ‡ 字符，映射为UTF-8字节序列  
                    byte_array.extend([0xe2, 0x80, 0xa1])  # ‡ 的UTF-8编码
                elif char_code == 732:   # ˜ 字符，映射为UTF-8字节序列
                    byte_array.extend([0xcb, 0x9c])        # ˜ 的UTF-8编码
                elif char_code == 353:   # š 字符，映射为UTF-8字节序列
                    byte_array.extend([0xc5, 0xa1])        # š 的UTF-8编码
                else:
                    print(f"忽略字符: '{char}' (编码: {char_code})")
            
            print(f"字节数组: {byte_array}")
            
            if byte_array:
                fixed_content = bytes(byte_array).decode('utf-8')
                print(f"方法1修复结果: {fixed_content}")
                if has_valid_chinese(fixed_content):
                    print(f"✓ 方法1修复成功: {fixed_content}")
                else:
                    print(f"✗ 方法1修复后无有效中文")
            else:
                print("✗ 方法1失败: 无有效字节数组")
        except Exception as e:
            print(f"✗ 方法1失败: {str(e)}")

if __name__ == "__main__":
    test_repair_logic()