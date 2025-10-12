#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import base64
import mysql.connector
from datetime import datetime
from typing import Optional, Dict, Any

class ChatHistoryWriter:
    """
    聊天记录写入器
    直接写入数据库，用于测试和演示
    """
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'xiaozhi123',
            'database': 'xiaozhi_esp32_server',
            'charset': 'utf8mb4'
        }
        
    def get_connection(self):
        """获取数据库连接"""
        return mysql.connector.connect(**self.db_config)
    
    def write_chat_record(self, 
                         mac_address: str, 
                         agent_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         chat_type: int = 1,  # 1=用户, 2=智能体
                         content: str = "",
                         audio_data: Optional[bytes] = None,
                         device_id: Optional[str] = None,
                         student_id: Optional[str] = None) -> bool:
        """
        写入聊天记录到数据库
        
        Args:
            mac_address: 设备MAC地址
            agent_id: 智能体ID
            session_id: 会话ID
            chat_type: 聊天类型 (1=用户, 2=智能体)
            content: 聊天内容
            audio_data: 音频数据
            device_id: 设备ID
            student_id: 学生ID
            
        Returns:
            bool: 写入是否成功
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 如果有音频数据，先插入音频表
            audio_id = None
            if audio_data:
                audio_id = self._insert_audio(cursor, audio_data)
            
            # 插入聊天记录
            now = datetime.now()
            insert_sql = """
                INSERT INTO ai_agent_chat_history 
                (mac_address, agent_id, session_id, chat_type, content, audio_id, 
                 created_at, updated_at, device_id, student_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                mac_address,
                agent_id,
                session_id or f"session_{int(time.time())}",
                chat_type,
                content,
                audio_id,
                now,
                now,
                device_id,
                student_id
            )
            
            cursor.execute(insert_sql, values)
            conn.commit()
            
            record_id = cursor.lastrowid
            print(f"聊天记录写入成功，ID: {record_id}")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"写入聊天记录失败: {e}")
            return False
    
    def _insert_audio(self, cursor, audio_data: bytes) -> Optional[int]:
        """
        插入音频数据
        
        Args:
            cursor: 数据库游标
            audio_data: 音频数据
            
        Returns:
            int: 音频记录ID
        """
        try:
            insert_sql = "INSERT INTO ai_agent_chat_audio (audio) VALUES (%s)"
            cursor.execute(insert_sql, (audio_data,))
            return cursor.lastrowid
        except Exception as e:
            print(f"插入音频数据失败: {e}")
            return None
    
    def add_test_data(self):
        """添加测试数据"""
        test_records = [
            {
                'mac_address': 'AA:BB:CC:DD:EE:01',
                'agent_id': 'agent_001',
                'session_id': 'test_session_001',
                'chat_type': 1,
                'content': '你好，我想了解一下今天的天气情况',
                'device_id': 'device_001',
                'student_id': 'student_001'
            },
            {
                'mac_address': 'AA:BB:CC:DD:EE:01',
                'agent_id': 'agent_001',
                'session_id': 'test_session_001',
                'chat_type': 2,
                'content': '今天天气晴朗，温度适宜，是个出门的好日子。',
                'device_id': 'device_001',
                'student_id': 'student_001'
            },
            {
                'mac_address': 'AA:BB:CC:DD:EE:02',
                'agent_id': 'agent_002',
                'session_id': 'test_session_002',
                'chat_type': 1,
                'content': '请帮我解释一下什么是人工智能',
                'device_id': 'device_002',
                'student_id': 'student_002'
            },
            {
                'mac_address': 'AA:BB:CC:DD:EE:02',
                'agent_id': 'agent_002',
                'session_id': 'test_session_002',
                'chat_type': 2,
                'content': '人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。',
                'device_id': 'device_002',
                'student_id': 'student_002'
            },
            {
                'mac_address': 'AA:BB:CC:DD:EE:03',
                'agent_id': 'agent_003',
                'session_id': 'test_session_003',
                'chat_type': 1,
                'content': '我想学习编程，有什么建议吗？',
                'device_id': 'device_003',
                'student_id': 'student_003'
            },
            {
                'mac_address': 'AA:BB:CC:DD:EE:03',
                'agent_id': 'agent_003',
                'session_id': 'test_session_003',
                'chat_type': 2,
                'content': '建议从Python开始学习，它语法简单易懂，适合初学者入门。',
                'device_id': 'device_003',
                'student_id': 'student_003'
            }
        ]
        
        success_count = 0
        for record in test_records:
            if self.write_chat_record(**record):
                success_count += 1
                time.sleep(0.1)  # 避免时间戳完全相同
        
        print(f"测试数据添加完成，成功添加 {success_count}/{len(test_records)} 条记录")
        return success_count == len(test_records)

# 全局实例
_writer = ChatHistoryWriter()

def write_chat_record(**kwargs) -> bool:
    """写入聊天记录的便捷函数"""
    return _writer.write_chat_record(**kwargs)

def add_test_data() -> bool:
    """添加测试数据的便捷函数"""
    return _writer.add_test_data()

if __name__ == "__main__":
    # 测试写入功能
    writer = ChatHistoryWriter()
    
    # 添加测试数据
    print("开始添加测试数据...")
    writer.add_test_data()
    
    print("测试数据添加完成！")