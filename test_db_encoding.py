#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql
import json

# 数据库配置 - 连接到Docker容器
DB_CONFIG = {
    'host': '172.20.0.5',  # Docker容器IP
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True,
    'use_unicode': True,
    'init_command': "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci"
}

def test_db_encoding():
    """测试数据库编码问题"""
    print("测试数据库编码问题")
    print("=" * 60)
    
    try:
        # 连接数据库
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # 查询乱码设备的数据
        query = """
        SELECT id, device_id, content, LENGTH(content) as content_length
        FROM ai_agent_chat_history 
        WHERE device_id = '58:8c:81:65:4c:8c' 
        ORDER BY created_at DESC 
        LIMIT 3
        """
        
        cursor.execute(query)
        records = cursor.fetchall()
        
        print(f"查询到 {len(records)} 条记录:")
        print("-" * 60)
        
        for i, record in enumerate(records, 1):
            content = record['content']
            print(f"记录 {i} (ID: {record['id']}):")
            print(f"  设备ID: {record['device_id']}")
            print(f"  内容长度: {record['content_length']}")
            print(f"  内容类型: {type(content)}")
            print(f"  内容: {repr(content)}")
            print(f"  显示: {content}")
            
            # 检查是否包含中文
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
            print(f"  包含中文: {has_chinese}")
            
            # 尝试JSON序列化
            try:
                json_str = json.dumps({"content": content}, ensure_ascii=False)
                print(f"  JSON序列化: {json_str}")
            except Exception as e:
                print(f"  JSON序列化失败: {e}")
            
            print("-" * 40)
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"数据库连接或查询失败: {e}")

if __name__ == "__main__":
    test_db_encoding()