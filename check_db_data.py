#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql

# 数据库配置
config = {
    'host': 'xiaozhi-esp32-server-db',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4'
}

def check_chat_data():
    try:
        # 连接数据库
        connection = pymysql.connect(**config)
        cursor = connection.cursor()
        
        # 查询指定设备的聊天记录
        device_id = "58:8c:81:65:52:48"
        query = """
        SELECT id, device_id, content, created_at, LENGTH(content) as content_length
        FROM ai_agent_chat_history 
        WHERE device_id = %s 
        ORDER BY created_at DESC 
        LIMIT 5
        """
        
        cursor.execute(query, (device_id,))
        results = cursor.fetchall()
        
        print(f"设备 {device_id} 的聊天记录:")
        print(f"总查询记录数: {len(results)}")
        print("-" * 80)
        
        for i, record in enumerate(results, 1):
            content = record[2]
            created_at = record[3]
            content_length = record[4]
            
            # 检查是否包含中文字符
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
            
            # 获取前10个字符的Unicode编码
            unicode_chars = [f"U+{ord(char):04X}" for char in content[:10]]
            
            print(f"记录 {i}:")
            print(f"  ID: {record[0]}")
            print(f"  创建时间: {created_at}")
            print(f"  内容长度: {content_length}")
            print(f"  包含中文: {has_chinese}")
            print(f"  内容: {repr(content[:80])}")
            print(f"  前10字符Unicode: {unicode_chars}")
            print("-" * 40)
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"数据库连接或查询失败: {e}")

if __name__ == "__main__":
    check_chat_data()