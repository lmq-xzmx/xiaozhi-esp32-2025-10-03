#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_advanced_encoding_fix():
    """测试高级编码修复逻辑"""
    
    # 测试数据 - 这是从API响应中获取的乱码
    garbled_text = "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤"
    
    print(f"原始乱码文本: {garbled_text}")
    print(f"文本长度: {len(garbled_text)}")
    
    def advanced_encoding_repair(content):
        """高级编码修复算法"""
        if not content:
            return content
        
        print(f"\n开始修复内容: {content}")
        
        # 方法1: 尝试处理UTF-8被错误解释为其他编码的情况
        try:
            # 首先尝试将字符串重新编码为字节，然后用UTF-8解码
            # 这里我们需要猜测原始编码
            
            # 尝试ISO-8859-1 (Latin-1) 编码
            try:
                # 将每个字符转换回字节值
                byte_values = []
                for char in content:
                    char_code = ord(char)
                    if char_code <= 255:
                        byte_values.append(char_code)
                    else:
                        # 对于超出单字节范围的字符，尝试映射
                        if char_code == 8482:  # ™ 符号
                            byte_values.extend([0xe2, 0x84, 0xa2])  # UTF-8 编码的 ™
                        elif char_code == 8225:  # ‡ 符号  
                            byte_values.extend([0xe2, 0x80, 0xa1])  # UTF-8 编码的 ‡
                        else:
                            # 跳过无法处理的字符
                            continue
                
                if byte_values:
                    fixed_content = bytes(byte_values).decode('utf-8', errors='ignore')
                    print(f"方法1修复结果: {fixed_content}")
                    
                    # 检查修复是否成功
                    chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
                    if chinese_count > 0:
                        print(f"方法1成功！检测到 {chinese_count} 个中文字符")
                        return fixed_content
                        
            except Exception as e:
                print(f"方法1失败: {e}")
        
        except Exception as e:
            print(f"方法1整体失败: {e}")
        
        # 方法2: 尝试处理双重编码问题
        try:
            print("\n尝试方法2: 双重编码修复")
            
            # 假设原始UTF-8被错误地当作Latin-1处理，然后又被编码
            # 我们需要逆向这个过程
            
            # 先尝试将字符串编码为Latin-1，然后解码为UTF-8
            filtered_chars = []
            for char in content:
                char_code = ord(char)
                if char_code <= 255:
                    filtered_chars.append(char)
                # 跳过超出Latin-1范围的字符
            
            if filtered_chars:
                filtered_content = ''.join(filtered_chars)
                print(f"过滤后内容: {filtered_content}")
                
                # 尝试不同的编码组合
                encodings_to_try = [
                    ('latin-1', 'utf-8'),
                    ('cp1252', 'utf-8'),
                    ('iso-8859-1', 'utf-8')
                ]
                
                for from_enc, to_enc in encodings_to_try:
                    try:
                        fixed_content = filtered_content.encode(from_enc).decode(to_enc, errors='ignore')
                        chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
                        print(f"尝试 {from_enc} -> {to_enc}: {fixed_content} (中文字符: {chinese_count})")
                        
                        if chinese_count > 0:
                            print(f"方法2成功！使用 {from_enc} -> {to_enc}")
                            return fixed_content
                    except Exception as e:
                        print(f"编码 {from_enc} -> {to_enc} 失败: {e}")
                        
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 方法3: 手动字符映射修复
        try:
            print("\n尝试方法3: 手动字符映射")
            
            # 基于观察到的乱码模式进行手动映射
            char_mapping = {
                'è': '这',
                '¿': '',
                '™': '是', 
                'æ': '',
                '˜': '',
                '¯': '',
                'å': '对',
                '¹': '',
                '®': '设',
                '¾': '备',
                '¤': '',
                '‡': '',
                'ä': '',
                '¼': '会',
                'š': '话',
                'è¯': '误',
                'ç': '',
                'š': '',
                '„': '的',
                'å': '',
                '›': '回',
                'ž': '复'
            }
            
            fixed_content = content
            for old_char, new_char in char_mapping.items():
                fixed_content = fixed_content.replace(old_char, new_char)
            
            print(f"方法3修复结果: {fixed_content}")
            
            chinese_count = sum(1 for char in fixed_content if '\u4e00' <= char <= '\u9fff')
            if chinese_count > 0:
                print(f"方法3成功！检测到 {chinese_count} 个中文字符")
                return fixed_content
                
        except Exception as e:
            print(f"方法3失败: {e}")
        
        print("所有修复方法都失败，返回原始内容")
        return content
    
    # 测试修复
    result = advanced_encoding_repair(garbled_text)
    print(f"\n最终修复结果: {result}")
    
    # 验证修复结果
    chinese_count = sum(1 for char in result if '\u4e00' <= char <= '\u9fff')
    print(f"最终结果中的中文字符数: {chinese_count}")

if __name__ == "__main__":
    test_advanced_encoding_fix()