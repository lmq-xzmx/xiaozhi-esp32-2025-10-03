#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聊天记录集成示例
演示如何在主项目中集成聊天记录功能
"""

import time
import uuid
from core.chat_history_service import get_chat_history_service


def simulate_chat_conversation():
    """模拟一次聊天对话"""
    
    # 获取聊天记录服务
    chat_service = get_chat_history_service()
    
    # 模拟设备信息
    device_id = "58:8c:81:65:4c:8c"
    mac_address = "58:8c:81:65:4c:8c"
    agent_id = "agent_demo"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    student_id = 1001  # 使用现有的用户ID
    
    print(f"开始模拟聊天对话...")
    print(f"设备ID: {device_id}")
    print(f"会话ID: {session_id}")
    print("-" * 50)
    
    # 模拟用户提问
    user_question = "请介绍一下人工智能的发展历史"
    print(f"用户: {user_question}")
    
    # 记录用户输入 (chat_type=2)
    success = chat_service.write_chat_record(
        mac_address=mac_address,
        agent_id=agent_id,
        session_id=session_id,
        chat_type=2,  # 用户输入
        content=user_question,
        device_id=device_id,
        student_id=student_id
    )
    
    if success:
        print("✓ 用户输入记录成功")
    else:
        print("✗ 用户输入记录失败")
    
    # 模拟AI处理时间
    time.sleep(1)
    
    # 模拟AI回复
    ai_response = "人工智能的发展可以追溯到20世纪50年代。1950年，艾伦·图灵提出了著名的图灵测试。1956年，达特茅斯会议标志着人工智能学科的正式诞生。经过几十年的发展，AI经历了多次起伏，如今在深度学习和大模型的推动下迎来了新的黄金时代。"
    print(f"AI: {ai_response}")
    
    # 记录AI回复 (chat_type=1)
    success = chat_service.write_chat_record(
        mac_address=mac_address,
        agent_id=agent_id,
        session_id=session_id,
        chat_type=1,  # AI回复
        content=ai_response,
        device_id=device_id,
        student_id=student_id
    )
    
    if success:
        print("✓ AI回复记录成功")
    else:
        print("✗ AI回复记录失败")
    
    print("-" * 50)
    
    # 获取该设备的最近聊天记录
    recent_records = chat_service.get_recent_chat_records(device_id, limit=5)
    print(f"设备 {device_id} 的最近聊天记录:")
    
    for i, record in enumerate(recent_records, 1):
        chat_type_name = "AI回复" if record.get('chat_type') == '1' else "用户输入"
        content = record.get('content', '')[:50]
        created_at = record.get('created_at', '')
        print(f"{i}. [{chat_type_name}] {content}... ({created_at})")
    
    # 显示统计信息
    total_count = chat_service.get_chat_records_count()
    device_count = chat_service.get_chat_records_count(device_id)
    
    print(f"\n统计信息:")
    print(f"- 总聊天记录数: {total_count}")
    print(f"- 该设备聊天记录数: {device_count}")


def demonstrate_async_reporting():
    """演示异步报告功能"""
    
    chat_service = get_chat_history_service()
    
    print("\n" + "=" * 50)
    print("演示异步报告功能")
    print("=" * 50)
    
    # 模拟多个异步报告
    reports = [
        {
            "mac_address": "58:8c:81:65:4c:9c",
            "agent_id": "agent_async",
            "session_id": "async_session_001",
            "chat_type": 2,
            "content": "什么是机器学习？",
            "device_id": "58:8c:81:65:4c:9c",
            "student_id": 1002
        },
        {
            "mac_address": "58:8c:81:65:4c:9c",
            "agent_id": "agent_async",
            "session_id": "async_session_001",
            "chat_type": 1,
            "content": "机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。",
            "device_id": "58:8c:81:65:4c:9c",
            "student_id": 1002
        }
    ]
    
    for i, report in enumerate(reports, 1):
        print(f"异步报告 {i}: {report['content'][:30]}...")
        chat_service.report_chat_async(**report)
        time.sleep(0.5)  # 模拟异步间隔
    
    print("异步报告完成!")


def show_health_status():
    """显示服务健康状态"""
    
    chat_service = get_chat_history_service()
    
    print("\n" + "=" * 50)
    print("服务健康状态检查")
    print("=" * 50)
    
    health = chat_service.health_check()
    
    status_icon = "✓" if health.get("status") == "healthy" else "✗"
    print(f"{status_icon} 服务状态: {health.get('status')}")
    print(f"✓ 数据库连接: {'正常' if health.get('database_connected') else '异常'}")
    print(f"✓ 总聊天记录: {health.get('total_chat_records', 0)} 条")
    print(f"✓ 检查时间: {health.get('timestamp')}")


if __name__ == "__main__":
    print("聊天记录集成示例")
    print("=" * 50)
    
    try:
        # 1. 模拟聊天对话
        simulate_chat_conversation()
        
        # 2. 演示异步报告
        demonstrate_async_reporting()
        
        # 3. 显示健康状态
        show_health_status()
        
        print("\n" + "=" * 50)
        print("集成示例完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()