#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def test_api_encoding_fix():
    """测试API编码修复功能"""
    url = "http://localhost:8092/api/chat-records/58:8c:81:65:52:48"
    
    print("调用API...")
    try:
        response = requests.get(url)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据类型: {type(data)}")
            
            if 'data' in data and 'records' in data['data']:
                records = data['data']['records']
                print(f"聊天记录数量: {len(records)}")
                
                # 检查前3条记录
                for i, record in enumerate(records[:3]):
                    content = record.get('content', '')
                    print(f"\n记录 {i+1}:")
                    print(f"  原始内容: {content}")
                    print(f"  内容长度: {len(content)}")
                    
                    # 检查中文字符
                    chinese_chars = [char for char in content if '\u4e00' <= char <= '\u9fff']
                    print(f"  中文字符数: {len(chinese_chars)}")
                    if chinese_chars:
                        print(f"  中文字符: {chinese_chars}")
                    
                    # 检查乱码指示符
                    garbled_indicators = ['è', '¿', 'æ', 'ä', '™', '‡', '®', '¾', '¹', '¤', '¯']
                    garbled_count = sum(1 for char in content if char in garbled_indicators)
                    print(f"  乱码指示符数量: {garbled_count}")
                    
                    if garbled_count > 0:
                        print("  ❌ 内容仍然包含乱码")
                    elif len(chinese_chars) > 0:
                        print("  ✅ 内容包含有效中文")
                    else:
                        print("  ⚠️  内容不包含中文字符")
            else:
                print("响应数据格式不正确")
        else:
            print(f"API调用失败: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_api_encoding_fix()