#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import re

def analyze_encoding_patterns():
    """分析编码模式并测试修复方法"""
    
    # 从实际API获取失败案例
    import requests
    
    try:
        response = requests.get('http://localhost:8092/api/export/chat-data/58:8c:81:65:4c:8c')
        if response.status_code == 200:
            data = response.json()
            records = data['data']['chat_records']
            
            failed_cases = []
            for record in records:
                content = record.get('content', '')
                # 检查是否包含中文字符
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
                if not has_chinese and len(content) > 10:  # 可能是乱码
                    failed_cases.append(content)
            
            print(f"找到 {len(failed_cases)} 个失败案例")
            
            for i, case in enumerate(failed_cases[:3], 1):  # 只分析前3个
                print(f"\n=== 案例 {i} ===")
                print(f"原始内容: {case[:100]}...")
                analyze_single_case(case)
                
    except Exception as e:
        print(f"获取数据失败: {e}")

def analyze_single_case(text):
    """分析单个案例的编码问题"""
    
    methods = [
        ("Latin-1 -> UTF-8", lambda t: t.encode('latin-1').decode('utf-8')),
        ("UTF-8 -> Latin-1 -> UTF-8", lambda t: t.encode('utf-8').decode('latin-1').encode('latin-1').decode('utf-8')),
        ("CP1252 -> UTF-8", lambda t: t.encode('cp1252').decode('utf-8')),
        ("ISO-8859-1 -> UTF-8", lambda t: t.encode('iso-8859-1').decode('utf-8')),
        ("Windows-1252 -> UTF-8", lambda t: codecs.decode(t.encode('windows-1252'), 'utf-8', errors='ignore')),
        ("直接字节转换", lambda t: bytes([ord(c) for c in t if ord(c) < 256]).decode('utf-8')),
        ("HTML实体解码", lambda t: decode_html_entities(t)),
        ("URL解码", lambda t: decode_url_encoding(t)),
    ]
    
    for method_name, method_func in methods:
        try:
            result = method_func(text)
            # 检查是否包含中文
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in result)
            if has_chinese:
                print(f"✓ {method_name}: {result[:50]}...")
                return result
            else:
                print(f"✗ {method_name}: 无中文字符")
        except Exception as e:
            print(f"✗ {method_name}: {str(e)[:50]}...")
    
    return None

def decode_html_entities(text):
    """解码HTML实体"""
    import html
    return html.unescape(text)

def decode_url_encoding(text):
    """解码URL编码"""
    import urllib.parse
    return urllib.parse.unquote(text)

def test_advanced_methods():
    """测试高级编码修复方法"""
    
    # 测试一些已知的乱码模式
    test_cases = [
        "ä»Šå¤©å¤©æ°",  # 今天天气
        "ä½ å¥½",        # 你好
        "è¯·ä»‹ç»",      # 请介绍
    ]
    
    print("\n=== 测试高级修复方法 ===")
    
    for case in test_cases:
        print(f"\n测试: {case}")
        
        # 方法1: 假设是UTF-8字节被误解为Latin-1字符
        try:
            # 将每个字符转换为其字节值，然后作为UTF-8解码
            byte_values = []
            for char in case:
                byte_val = ord(char)
                if byte_val < 256:  # 确保在Latin-1范围内
                    byte_values.append(byte_val)
            
            if byte_values:
                result = bytes(byte_values).decode('utf-8')
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in result)
                if has_chinese:
                    print(f"✓ 字节转换法: {result}")
                    continue
        except Exception as e:
            pass
        
        # 方法2: 尝试不同的编码组合
        encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                result = case.encode(enc).decode('utf-8')
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in result)
                if has_chinese:
                    print(f"✓ {enc}->UTF-8: {result}")
                    break
            except:
                continue

if __name__ == "__main__":
    print("开始分析编码问题...")
    analyze_encoding_patterns()
    test_advanced_methods()