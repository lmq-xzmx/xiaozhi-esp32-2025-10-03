#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql
import json
from flask import Flask, jsonify

# 模拟API服务器的完整流程
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 数据库配置
DB_CONFIG = {
    'host': '172.20.0.5',
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

def get_db_connection():
    """获取数据库连接"""
    connection = pymysql.connect(**DB_CONFIG)
    connection.set_charset('utf8mb4')
    return connection

def execute_query(query, params=None):
    """执行数据库查询"""
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
    finally:
        if connection:
            connection.close()

def debug_api_flow():
    """调试API完整流程"""
    print("调试API服务器完整数据流程")
    print("=" * 60)
    
    # 1. 从数据库查询数据
    query = """
    SELECT id, device_id, content, created_at
    FROM ai_agent_chat_history 
    WHERE device_id = %s 
    ORDER BY created_at DESC 
    LIMIT 3
    """
    
    device_id = '58:8c:81:65:4c:8c'
    chat_records = execute_query(query, (device_id,))
    
    print(f"步骤1: 从数据库查询到 {len(chat_records)} 条记录")
    print("-" * 40)
    
    for i, record in enumerate(chat_records, 1):
        print(f"记录 {i} (ID: {record['id']}):")
        content = record['content']
        
        # 2. 检查原始内容
        print(f"  原始内容类型: {type(content)}")
        print(f"  原始内容: {repr(content)}")
        print(f"  显示内容: {content}")
        
        # 3. 检查是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
        print(f"  包含中文: {has_chinese}")
        
        # 4. 处理时间格式
        if record['created_at']:
            record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        # 5. 模拟编码修复逻辑
        if has_chinese:
            print(f"  内容已包含中文字符，无需修复")
        else:
            print(f"  内容不包含中文，需要修复")
        
        # 6. JSON序列化测试
        try:
            json_str = json.dumps(record, ensure_ascii=False, default=str)
            print(f"  JSON序列化成功: {len(json_str)} 字符")
            
            # 7. Flask jsonify测试
            with app.app_context():
                response = jsonify(record)
                response_data = response.get_data(as_text=True)
                print(f"  Flask jsonify: {len(response_data)} 字符")
                print(f"  响应内容: {response_data[:100]}...")
                
        except Exception as e:
            print(f"  序列化失败: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    debug_api_flow()