#!/usr/bin/env python3
"""
监控系统功能测试脚本
测试监控、错误追踪和日志系统的API端点
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

class MonitoringFeaturesTester:
    def __init__(self):
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_api_endpoint(self, method: str, endpoint: str, data: dict = None, description: str = ""):
        """测试API端点"""
        try:
            url = f"{BASE_URL}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    result = await response.json()
                    status_code = response.status
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
                    status_code = response.status
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            success = status_code == 200 and result.get("status") == "success"
            
            test_result = {
                "description": description,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "success": success,
                "response": result
            }
            
            self.test_results.append(test_result)
            
            status_icon = "✅" if success else "❌"
            print(f"{status_icon} {description}")
            if not success:
                print(f"   错误: {result}")
            
            return success, result
            
        except Exception as e:
            test_result = {
                "description": description,
                "endpoint": endpoint,
                "method": method,
                "success": False,
                "error": str(e)
            }
            self.test_results.append(test_result)
            print(f"❌ {description} - 异常: {e}")
            return False, None
    
    async def test_monitoring_apis(self):
        """测试监控系统API"""
        print("\n🔍 测试监控系统API...")
        
        # 测试监控健康检查
        await self.test_api_endpoint(
            "GET", "/api/monitoring/health",
            description="监控系统健康检查"
        )
        
        # 测试获取系统指标
        await self.test_api_endpoint(
            "GET", "/api/monitoring/metrics",
            description="获取系统指标"
        )
        
        # 测试获取活跃告警
        await self.test_api_endpoint(
            "GET", "/api/monitoring/alerts",
            description="获取活跃告警"
        )
        
        # 测试启动监控守护进程
        await self.test_api_endpoint(
            "POST", "/api/monitoring/start",
            description="启动监控守护进程"
        )
        
        # 等待一下让监控系统运行
        await asyncio.sleep(2)
        
        # 再次检查系统指标
        await self.test_api_endpoint(
            "GET", "/api/monitoring/metrics",
            description="监控启动后获取系统指标"
        )
        
        # 测试停止监控守护进程
        await self.test_api_endpoint(
            "POST", "/api/monitoring/stop",
            description="停止监控守护进程"
        )
    
    async def test_error_tracking_apis(self):
        """测试错误追踪API"""
        print("\n🐛 测试错误追踪API...")
        
        # 测试获取最近错误
        await self.test_api_endpoint(
            "GET", "/api/errors/recent?limit=10",
            description="获取最近错误记录"
        )
        
        # 测试获取错误统计
        await self.test_api_endpoint(
            "GET", "/api/errors/statistics",
            description="获取错误统计信息"
        )
        
        # 模拟触发一个错误（通过访问不存在的错误ID）
        await self.test_api_endpoint(
            "GET", "/api/errors/nonexistent-error-id",
            description="获取不存在错误的详情（预期失败）"
        )
        
        # 再次获取错误统计，看是否有变化
        await self.test_api_endpoint(
            "GET", "/api/errors/statistics",
            description="错误后获取错误统计"
        )
    
    async def test_logging_apis(self):
        """测试日志系统API"""
        print("\n📝 测试日志系统API...")
        
        # 测试搜索日志
        await self.test_api_endpoint(
            "GET", "/api/logs/search?hours=1&limit=10",
            description="搜索最近1小时的日志"
        )
        
        # 测试按级别搜索日志
        await self.test_api_endpoint(
            "GET", "/api/logs/search?level=info&hours=1&limit=5",
            description="搜索INFO级别日志"
        )
        
        # 测试按关键词搜索日志
        await self.test_api_endpoint(
            "GET", "/api/logs/search?keyword=websocket&hours=1&limit=5",
            description="搜索包含websocket关键词的日志"
        )
        
        # 测试日志分析
        await self.test_api_endpoint(
            "GET", "/api/logs/analyze?hours=1",
            description="分析最近1小时的日志"
        )
        
        # 测试获取日志统计
        await self.test_api_endpoint(
            "GET", "/api/logs/statistics",
            description="获取日志统计信息"
        )
        
        # 测试导出日志
        await self.test_api_endpoint(
            "POST", "/api/logs/export",
            data={"hours": 1, "format": "json"},
            description="导出最近1小时的日志"
        )
    
    async def test_integration_scenarios(self):
        """测试集成场景"""
        print("\n🔄 测试集成场景...")
        
        # 启动监控系统
        success, _ = await self.test_api_endpoint(
            "POST", "/api/monitoring/start",
            description="启动监控系统进行集成测试"
        )
        
        if success:
            # 等待监控系统收集一些数据
            print("   等待监控系统收集数据...")
            await asyncio.sleep(3)
            
            # 检查监控指标
            await self.test_api_endpoint(
                "GET", "/api/monitoring/metrics",
                description="集成测试中获取监控指标"
            )
            
            # 检查是否有新的日志
            await self.test_api_endpoint(
                "GET", "/api/logs/search?category=system&hours=1&limit=5",
                description="搜索系统类别的日志"
            )
            
            # 停止监控系统
            await self.test_api_endpoint(
                "POST", "/api/monitoring/stop",
                description="停止监控系统"
            )
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始监控系统功能测试...")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 首先检查服务器健康状态
        await self.test_api_endpoint(
            "GET", "/health",
            description="服务器健康检查"
        )
        
        # 运行各个模块的测试
        await self.test_monitoring_apis()
        await self.test_error_tracking_apis()
        await self.test_logging_apis()
        await self.test_integration_scenarios()
        
        # 统计测试结果
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 测试结果统计:")
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for result in self.test_results:
                if not result.get("success", False):
                    print(f"  - {result['description']}")
                    if "error" in result:
                        print(f"    错误: {result['error']}")
                    elif "response" in result:
                        print(f"    响应: {result['response']}")
        
        return successful_tests == total_tests

async def main():
    """主函数"""
    try:
        async with MonitoringFeaturesTester() as tester:
            success = await tester.run_all_tests()
            
            if success:
                print(f"\n🎉 所有监控系统功能测试通过！")
                return 0
            else:
                print(f"\n⚠️  部分测试失败，请检查服务器状态和配置。")
                return 1
                
    except Exception as e:
        print(f"\n💥 测试过程中发生异常: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)