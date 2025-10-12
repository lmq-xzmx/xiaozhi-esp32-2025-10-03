#!/usr/bin/env python3
"""
数据同步服务测试脚本
测试数据同步功能的各个方面
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSyncTester:
    """数据同步测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_sync_status(self):
        """测试获取同步状态"""
        logger.info("🔍 测试获取同步状态...")
        try:
            async with self.session.get(f"{self.base_url}/api/sync/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 同步状态获取成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 获取同步状态失败: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 获取同步状态异常: {e}")
            return False
    
    async def test_sync_history(self):
        """测试获取同步历史"""
        logger.info("🔍 测试获取同步历史...")
        try:
            async with self.session.get(f"{self.base_url}/api/sync/history?limit=10") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 同步历史获取成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 获取同步历史失败: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 获取同步历史异常: {e}")
            return False
    
    async def test_start_sync_daemon(self):
        """测试启动同步守护进程"""
        logger.info("🔍 测试启动同步守护进程...")
        try:
            async with self.session.post(f"{self.base_url}/api/sync/start") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 同步守护进程启动成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 启动同步守护进程失败: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 启动同步守护进程异常: {e}")
            return False
    
    async def test_force_sync(self, device_id: str = "test_device_001"):
        """测试强制同步"""
        logger.info(f"🔍 测试强制同步设备: {device_id}...")
        try:
            async with self.session.post(f"{self.base_url}/api/sync/force/{device_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 强制同步成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 强制同步失败: HTTP {resp.status}")
                    text = await resp.text()
                    logger.error(f"响应内容: {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ 强制同步异常: {e}")
            return False
    
    async def test_stop_sync_daemon(self):
        """测试停止同步守护进程"""
        logger.info("🔍 测试停止同步守护进程...")
        try:
            async with self.session.post(f"{self.base_url}/api/sync/stop") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 同步守护进程停止成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 停止同步守护进程失败: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 停止同步守护进程异常: {e}")
            return False
    
    async def test_health_check(self):
        """测试健康检查"""
        logger.info("🔍 测试服务器健康检查...")
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ 服务器健康检查成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    logger.error(f"❌ 服务器健康检查失败: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 服务器健康检查异常: {e}")
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始数据同步服务测试...")
        logger.info("=" * 60)
        
        results = []
        
        # 1. 健康检查
        results.append(await self.test_health_check())
        
        # 2. 获取同步状态
        results.append(await self.test_sync_status())
        
        # 3. 获取同步历史
        results.append(await self.test_sync_history())
        
        # 4. 启动同步守护进程
        results.append(await self.test_start_sync_daemon())
        
        # 等待一下让守护进程启动
        await asyncio.sleep(2)
        
        # 5. 强制同步（这个可能会失败，因为ESP32服务器不存在）
        results.append(await self.test_force_sync())
        
        # 6. 停止同步守护进程
        results.append(await self.test_stop_sync_daemon())
        
        # 统计结果
        passed = sum(results)
        total = len(results)
        
        logger.info("=" * 60)
        logger.info(f"📊 测试结果: {passed}/{total} 通过")
        
        if passed == total:
            logger.info("🎉 所有测试通过！")
        else:
            logger.warning(f"⚠️  有 {total - passed} 个测试失败")
        
        return passed == total

async def main():
    """主函数"""
    async with DataSyncTester() as tester:
        success = await tester.run_all_tests()
        return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)