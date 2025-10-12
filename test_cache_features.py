#!/usr/bin/env python3
"""
测试缓存功能和实时同步服务
"""

import asyncio
import aiohttp
import json
import time

BASE_URL = "http://localhost:8001"

async def test_health_check():
    """测试健康检查"""
    print("1. 测试健康检查...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 健康检查通过: {data['status']}")
                return True
            else:
                print(f"   ✗ 健康检查失败: {response.status}")
                return False

async def test_cache_sessions():
    """测试会话缓存API"""
    print("2. 测试会话缓存API...")
    async with aiohttp.ClientSession() as session:
        # 获取所有会话
        async with session.get(f"{BASE_URL}/api/cache/sessions") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 获取所有会话成功: {len(data.get('sessions', []))} 个会话")
                return True
            else:
                print(f"   ✗ 获取所有会话失败: {response.status}")
                return False

async def test_cache_cleanup():
    """测试缓存清理"""
    print("3. 测试缓存清理...")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/api/cache/cleanup") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 缓存清理成功: {data.get('message', '')}")
                return True
            else:
                print(f"   ✗ 缓存清理失败: {response.status}")
                return False

async def test_realtime_sync_status():
    """测试实时同步状态"""
    print("4. 测试实时同步状态...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/realtime-sync/status") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 获取实时同步状态成功")
                sync_status = data.get('sync_status', {})
                print(f"     - 队列大小: {sync_status.get('queue_size', 0)}")
                print(f"     - 处理任务数: {sync_status.get('processed_tasks', 0)}")
                print(f"     - 失败任务数: {sync_status.get('failed_tasks', 0)}")
                return True
            else:
                print(f"   ✗ 获取实时同步状态失败: {response.status}")
                return False

async def test_realtime_sync_queue():
    """测试实时同步队列状态"""
    print("5. 测试实时同步队列状态...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/realtime-sync/queue") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 获取同步队列状态成功")
                queue = data.get('queue', {})
                print(f"     - 队列大小: {queue.get('size', 0)}")
                print(f"     - 待处理任务: {queue.get('pending', 0)}")
                return True
            else:
                print(f"   ✗ 获取同步队列状态失败: {response.status}")
                return False

async def test_start_realtime_sync():
    """测试启动实时同步"""
    print("6. 测试启动实时同步...")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/api/realtime-sync/start") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 启动实时同步成功: {data.get('message', '')}")
                return True
            else:
                print(f"   ✗ 启动实时同步失败: {response.status}")
                return False

async def test_stop_realtime_sync():
    """测试停止实时同步"""
    print("7. 测试停止实时同步...")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/api/realtime-sync/stop") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ 停止实时同步成功: {data.get('message', '')}")
                return True
            else:
                print(f"   ✗ 停止实时同步失败: {response.status}")
                return False

async def test_device_sessions():
    """测试设备会话查询"""
    print("8. 测试设备会话查询...")
    device_id = "test_device_001"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/cache/device/{device_id}/sessions") as response:
            if response.status == 200:
                data = await response.json()
                sessions = data.get('sessions', [])
                print(f"   ✓ 获取设备会话成功: 设备 {device_id} 有 {len(sessions)} 个会话")
                return True
            else:
                print(f"   ✗ 获取设备会话失败: {response.status}")
                return False

async def main():
    """主测试函数"""
    print("=" * 60)
    print("开始测试缓存功能和实时同步服务")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_cache_sessions,
        test_cache_cleanup,
        test_realtime_sync_status,
        test_realtime_sync_queue,
        test_start_realtime_sync,
        test_stop_realtime_sync,
        test_device_sessions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"   ✗ 测试异常: {e}")
            print()
    
    print("=" * 60)
    print(f"测试完成: {passed}/{total} 通过")
    if passed == total:
        print("所有测试通过！")
    else:
        print(f"有 {total - passed} 个测试失败")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())