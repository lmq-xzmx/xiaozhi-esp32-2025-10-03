#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建测试聊天记录数据
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.chat_history_service import ChatHistoryService
from datetime import datetime, timedelta
import uuid

def create_test_chat_data():
    """创建测试聊天记录数据"""
    service = ChatHistoryService()
    
    # 测试设备ID
    test_devices = [
        "58:8c:81:65:4c:ac",
        "58:8c:81:65:4c:b0", 
        "58:8c:81:65:52:48"
    ]
    
    # 测试智能体ID
    test_agents = [
        "2cc3f472aa2c453fbace644594a73aad",
        "28a35c49f0304a6dbb472b04e9b60fcc",
        "0be50cf09b8f4db5ac14b509501c7d79"
    ]
    
    print("开始创建测试聊天记录数据...")
    
    for i, device_id in enumerate(test_devices):
        agent_id = test_agents[i % len(test_agents)]
        
        # 为每个设备创建多个会话
        for session_num in range(1, 4):  # 3个会话
            session_id = f"session_{device_id}_{session_num}"
            
            # 每个会话创建多条聊天记录
            for chat_num in range(1, 6):  # 5条记录
                # 用户消息
                user_content = f"这是设备{device_id}会话{session_num}的第{chat_num}条用户消息"
                success = service.write_chat_record(
                    mac_address=device_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    chat_type=1,  # 用户消息
                    content=user_content,
                    device_id=device_id,
                    student_id=None
                )
                if success:
                    print(f"✓ 创建用户消息: {device_id} - {user_content[:30]}...")
                
                # AI回复
                ai_content = f"这是AI对设备{device_id}会话{session_num}第{chat_num}条消息的回复"
                success = service.write_chat_record(
                    mac_address=device_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    chat_type=2,  # AI消息
                    content=ai_content,
                    device_id=device_id,
                    student_id=None
                )
                if success:
                    print(f"✓ 创建AI回复: {device_id} - {ai_content[:30]}...")
    
    print("\n测试数据创建完成！")
    
    # 验证数据
    total_count = service.get_chat_records_count()
    print(f"总聊天记录数: {total_count}")
    
    for device_id in test_devices:
        recent_records = service.get_recent_chat_records(device_id, limit=3)
        print(f"\n设备 {device_id} 的最近记录数: {len(recent_records)}")
        for record in recent_records:
            print(f"  - [{record.get('chat_type')}] {record.get('content', '')[:50]}...")

if __name__ == "__main__":
    create_test_chat_data()