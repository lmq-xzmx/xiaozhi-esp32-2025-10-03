#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
import time
from datetime import datetime
from typing import Optional

class ChatHistoryDockerWriter:
    """
    通过Docker连接MySQL的聊天记录写入器
    """
    
    def __init__(self):
        self.container_name = "xiaozhi-esp32-server-db"
        self.database = "xiaozhi_esp32_server"
        
    def execute_sql(self, sql: str, params: tuple = None) -> dict:
        """
        通过Docker执行SQL命令
        
        Args:
            sql: SQL语句
            params: 参数元组
            
        Returns:
            dict: 执行结果
        """
        try:
            # 构建MySQL命令
            if params:
                # 简单的参数替换（仅用于演示，生产环境需要更安全的方式）
                formatted_sql = sql
                for param in params:
                    if isinstance(param, str):
                        formatted_sql = formatted_sql.replace('%s', f"'{param}'", 1)
                    elif param is None:
                        formatted_sql = formatted_sql.replace('%s', 'NULL', 1)
                    else:
                        formatted_sql = formatted_sql.replace('%s', str(param), 1)
            else:
                formatted_sql = sql
            
            # 执行Docker命令
            cmd = [
                'docker', 'exec', '-i', self.container_name,
                'mysql', '-u', 'root', '-p123456', self.database,
                '-e', formatted_sql
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout, 'error': None}
            else:
                return {'success': False, 'output': result.stdout, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'output': None, 'error': str(e)}
    
    def write_chat_record(self, 
                         mac_address: str, 
                         agent_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         chat_type: int = 1,  # 1=用户, 2=智能体
                         content: str = "",
                         device_id: Optional[str] = None,
                         student_id: Optional[str] = None) -> bool:
        """
        写入聊天记录到数据库
        """
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 构建插入SQL
            sql = """
                INSERT INTO ai_agent_chat_history 
                (mac_address, agent_id, session_id, chat_type, content, 
                 created_at, updated_at, device_id, student_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                mac_address,
                agent_id,
                session_id or f"session_{int(time.time())}",
                chat_type,
                content,
                now,
                now,
                device_id,
                student_id
            )
            
            result = self.execute_sql(sql, params)
            
            if result['success']:
                print(f"聊天记录写入成功: {content[:50]}...")
                return True
            else:
                print(f"写入聊天记录失败: {result['error']}")
                return False
                
        except Exception as e:
            print(f"写入聊天记录异常: {e}")
            return False
    
    def add_test_data(self):
        """添加测试数据"""
        test_records = [
            {
                'mac_address': '58:8c:81:65:4c:8c',
                'agent_id': 'agent_001',
                'session_id': 'test_session_001',
                'chat_type': 1,
                'content': '你好，我想了解一下今天的天气情况',
                'device_id': '58:8c:81:65:4c:8c',
                'student_id': 1001
            },
            {
                'mac_address': '58:8c:81:65:4c:8c',
                'agent_id': 'agent_001',
                'session_id': 'test_session_001',
                'chat_type': 2,
                'content': '今天天气晴朗，温度适宜，是个出门的好日子。',
                'device_id': '58:8c:81:65:4c:8c',
                'student_id': 1001
            },
            {
                'mac_address': '58:8c:81:65:4c:9c',
                'agent_id': 'agent_002',
                'session_id': 'test_session_002',
                'chat_type': 1,
                'content': '请帮我解释一下什么是人工智能',
                'device_id': '58:8c:81:65:4c:9c',
                'student_id': 1002
            },
            {
                'mac_address': '58:8c:81:65:4c:9c',
                'agent_id': 'agent_002',
                'session_id': 'test_session_002',
                'chat_type': 2,
                'content': '人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。',
                'device_id': '58:8c:81:65:4c:9c',
                'student_id': 1002
            },
            {
                'mac_address': '58:8c:81:65:4c:a0',
                'agent_id': 'agent_003',
                'session_id': 'test_session_003',
                'chat_type': 1,
                'content': '我想学习编程，有什么建议吗？',
                'device_id': '58:8c:81:65:4c:a0',
                'student_id': 1003
            },
            {
                'mac_address': '58:8c:81:65:4c:a0',
                'agent_id': 'agent_003',
                'session_id': 'test_session_003',
                'chat_type': 2,
                'content': '建议从Python开始学习，它语法简单易懂，适合初学者入门。',
                'device_id': '58:8c:81:65:4c:a0',
                'student_id': 1003
            }
        ]
        
        success_count = 0
        for record in test_records:
            if self.write_chat_record(**record):
                success_count += 1
                time.sleep(0.1)  # 避免时间戳完全相同
        
        print(f"测试数据添加完成，成功添加 {success_count}/{len(test_records)} 条记录")
        return success_count == len(test_records)
    
    def check_records_count(self):
        """检查记录数量"""
        result = self.execute_sql("SELECT COUNT(*) as total FROM ai_agent_chat_history")
        if result['success']:
            print(f"数据库查询结果: {result['output']}")
            return True
        else:
            print(f"查询失败: {result['error']}")
            return False

# 全局实例
_writer = ChatHistoryDockerWriter()

def write_chat_record(**kwargs) -> bool:
    """写入聊天记录的便捷函数"""
    return _writer.write_chat_record(**kwargs)

def add_test_data() -> bool:
    """添加测试数据的便捷函数"""
    return _writer.add_test_data()

if __name__ == "__main__":
    # 测试写入功能
    writer = ChatHistoryDockerWriter()
    
    # 检查当前记录数量
    print("检查当前记录数量...")
    writer.check_records_count()
    
    # 添加测试数据
    print("\n开始添加测试数据...")
    writer.add_test_data()
    
    # 再次检查记录数量
    print("\n检查添加后的记录数量...")
    writer.check_records_count()
    
    print("\n测试数据添加完成！")