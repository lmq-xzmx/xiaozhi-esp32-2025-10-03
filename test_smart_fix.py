#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def smart_encoding_fix():
    original_content = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤"
    print(f"原始内容: {original_content}")
    
    # 智能修复：将字符转换为字节，跳过超出范围的字符，然后尝试UTF-8解码
    try:
        print("\n智能修复方法:")
        
        # 将字符串转换为字节数组，跳过超出Latin-1范围的字符
        byte_array = []
        skipped_chars = []
        
        for i, char in enumerate(original_content):
            char_code = ord(char)
            if char_code <= 255:  # Latin-1范围内的字符
                byte_array.append(char_code)
            else:
                skipped_chars.append((i, char, char_code))
        
        print(f"跳过的字符: {skipped_chars}")
        print(f"字节数组前20个: {byte_array[:20]}")
        
        # 尝试将字节数组解码为UTF-8
        try:
            fixed_content = bytes(byte_array).decode('utf-8')
            print(f"修复结果: {fixed_content}")
            
            # 检查中文字符
            chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
            print(f"中文字符数: {chinese_count}")
            
            if chinese_count > 0:
                print("✓ 修复成功！")
                return fixed_content
            else:
                print("✗ 修复后无中文字符")
                
        except UnicodeDecodeError as e:
            print(f"UTF-8解码失败: {e}")
            
            # 尝试错误处理策略
            try:
                fixed_content = bytes(byte_array).decode('utf-8', errors='ignore')
                print(f"忽略错误的修复结果: {fixed_content}")
                
                chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
                print(f"中文字符数: {chinese_count}")
                
                if chinese_count > 0:
                    print("✓ 忽略错误修复成功！")
                    return fixed_content
                    
            except Exception as e2:
                print(f"忽略错误也失败: {e2}")
    
    except Exception as e:
        print(f"智能修复失败: {e}")
    
    return None

def test_expected_result():
    """测试期望的结果"""
    # 这应该是正确的中文内容
    expected = "这是AI对设备58:8c:81:65:52:48会话错误的回复"
    print(f"\n期望结果: {expected}")
    
    # 将期望结果编码为UTF-8字节
    utf8_bytes = expected.encode('utf-8')
    print(f"UTF-8字节: {list(utf8_bytes)}")
    print(f"UTF-8字节(hex): {[hex(b) for b in utf8_bytes]}")

if __name__ == "__main__":
    result = smart_encoding_fix()
    test_expected_result()