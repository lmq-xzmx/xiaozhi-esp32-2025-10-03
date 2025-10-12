#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def test_api_encoding_fix():
    """测试API编码修复功能"""
    
    api_url = "http://localhost:8092/api/chat-records/58:8c:81:65:52:48"
    
    print("正在调用API...")
    try:
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"API响应状态: {response.status_code}")
            print(f"响应数据结构: {type(data)}")
            
            if 'data' in data and 'records' in data['data']:
                records = data['data']['records']
                print(f"聊天记录数量: {len(records)}")
                
                for i, record in enumerate(records[:3]):  # 只检查前3条记录
                    content = record.get('content', '')
                    print(f"\n记录 {i+1}:")
                    print(f"  原始内容: {content}")
                    print(f"  内容长度: {len(content)}")
                    
                    # 检查是否包含中文字符
                    chinese_chars = [char for char in content if '\u4e00' <= char <= '\u9fff']
                    print(f"  中文字符数: {len(chinese_chars)}")
                    
                    if chinese_chars:
                        print(f"  中文字符: {chinese_chars[:10]}")  # 显示前10个中文字符
                    
                    # 检查是否包含乱码指示符
                    garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï', 'è', 'æ', '¿', '™', '¯', '¹', '®', '¾', '¤', '‡']
                    garbled_count = sum(1 for char in content if char in garbled_indicators)
                    print(f"  乱码指示符数量: {garbled_count}")
                    
                    if garbled_count > 0:
                        found_garbled = [char for char in content if char in garbled_indicators]
                        print(f"  发现的乱码字符: {set(found_garbled)}")
            else:
                print("响应数据格式不正确")
                print(f"响应内容: {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"调用API时发生错误: {e}")

if __name__ == "__main__":
    test_api_encoding_fix()