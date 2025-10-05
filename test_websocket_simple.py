#!/usr/bin/env python3
"""
简单的WebSocket连接测试
"""
import asyncio
import websockets
import time

async def test_websocket_connection():
    """测试WebSocket基本连接"""
    print("测试WebSocket连接...")
    
    try:
        # 尝试连接WebSocket
        uri = "ws://localhost:8000/xiaozhi/v1/?device-id=test-device&client-id=test-client"
        
        print(f"连接到: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket连接成功")
            
            # 等待服务器的欢迎消息
            try:
                welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"收到欢迎消息: {welcome_msg}")
            except asyncio.TimeoutError:
                print("⚠️ 未收到欢迎消息（可能正常）")
            
            # 发送一些测试音频数据
            test_audio = b'\x00' * 1024  # 1KB的静音数据
            await websocket.send(test_audio)
            print("✅ 发送测试音频数据成功")
            
            # 等待可能的响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                print(f"收到响应: {len(response)} bytes")
            except asyncio.TimeoutError:
                print("⚠️ 未收到响应（可能正常，音频处理需要时间）")
            
            return True
            
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")
        return False

async def test_without_params():
    """测试不带参数的连接"""
    print("\n测试不带参数的WebSocket连接...")
    
    try:
        uri = "ws://localhost:8000/xiaozhi/v1/"
        
        async with websockets.connect(uri) as websocket:
            print("✅ 无参数连接成功")
            
            # 等待服务器消息
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                print(f"收到消息: {msg}")
            except asyncio.TimeoutError:
                print("⚠️ 未收到消息")
            
            return True
            
    except Exception as e:
        print(f"❌ 无参数连接失败: {e}")
        return False

async def main():
    print("开始WebSocket连接测试")
    print("="*50)
    
    # 测试带参数的连接
    result1 = await test_websocket_connection()
    
    # 测试不带参数的连接
    result2 = await test_without_params()
    
    print("\n" + "="*50)
    print("测试总结:")
    print(f"带参数连接: {'✅ 成功' if result1 else '❌ 失败'}")
    print(f"无参数连接: {'✅ 成功' if result2 else '❌ 失败'}")
    
    if result1 or result2:
        print("✅ WebSocket服务基本可用")
    else:
        print("❌ WebSocket服务不可用")

if __name__ == "__main__":
    asyncio.run(main())