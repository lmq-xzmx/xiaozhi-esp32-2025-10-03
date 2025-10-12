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

# 测试数据
test_cases = [
    "当然可以！根据您的需求，我推荐以下学习资料：1. 在线课程平台 2. 专业书籍 3. 实践项目。您希望了解哪个方面的详细信息？",  # 正确的中文
    "è¿™æ˜¯AIå¯¹è®¾å¤‡58:8c:81:65:52:48ä¼šè¯¯çš„å›žå¤",  # 乱码
    "陈钰（晋二？）",  # 正确的中文
]

print("测试改进的中文字符检测逻辑：")
print("=" * 50)

for i, text in enumerate(test_cases, 1):
    result = has_valid_chinese_content(text)
    print(f"测试 {i}: {result}")
    print(f"内容: {text}")
    print(f"长度: {len(text)}")
    
    # 分析字符组成
    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    print(f"中文字符数: {len(chinese_chars)}")
    if chinese_chars:
        print(f"中文字符: {chinese_chars}")
        print(f"中文比例: {len(chinese_chars)/len(text):.2%}")
        
        # 检查连续中文字符
        consecutive_chinese = 0
        max_consecutive = 0
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                consecutive_chinese += 1
                max_consecutive = max(max_consecutive, consecutive_chinese)
            else:
                consecutive_chinese = 0
        
        print(f"最大连续中文字符数: {max_consecutive}")
    
    print("-" * 30)