#!/usr/bin/env python3
"""
ASR性能监控脚本
用于监控ASR服务的性能指标和健康状态
"""

import requests
import time
import json
import psutil
import logging
from datetime import datetime
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/xiaozhi-server/logs/asr_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASRPerformanceMonitor:
    """ASR性能监控器"""
    
    def __init__(self, asr_url="http://localhost:8001", check_interval=30):
        self.asr_url = asr_url
        self.check_interval = check_interval
        self.alert_thresholds = {
            "max_response_time": 1.0,  # 最大响应时间1秒
            "max_error_rate": 0.05,    # 最大错误率5%
            "max_memory_usage": 1400,  # 最大内存使用1400MB
            "min_cache_hit_rate": 0.1  # 最小缓存命中率10%
        }
        self.stats_history = []
        
    def get_asr_health(self) -> Dict[str, Any]:
        """获取ASR服务健康状态"""
        try:
            response = requests.get(f"{self.asr_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_asr_stats(self) -> Dict[str, Any]:
        """获取ASR服务统计信息"""
        try:
            response = requests.get(f"{self.asr_url}/asr/stats", timeout=5)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统资源统计"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 网络统计
            network = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total // (1024*1024),  # MB
                    "used": memory.used // (1024*1024),    # MB
                    "percent": memory.percent,
                    "available": memory.available // (1024*1024)  # MB
                },
                "disk": {
                    "total": disk.total // (1024*1024*1024),  # GB
                    "used": disk.used // (1024*1024*1024),    # GB
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            return {}
    
    def check_alerts(self, asr_stats: Dict, system_stats: Dict) -> list:
        """检查告警条件"""
        alerts = []
        
        try:
            # 检查ASR服务统计
            if asr_stats.get("status") == "success":
                data = asr_stats["data"]
                
                # 检查平均响应时间
                avg_processing_time = data.get("processor", {}).get("avg_processing_time", 0)
                if avg_processing_time > self.alert_thresholds["max_response_time"]:
                    alerts.append({
                        "type": "high_response_time",
                        "message": f"ASR平均响应时间过高: {avg_processing_time:.3f}s",
                        "threshold": self.alert_thresholds["max_response_time"],
                        "current": avg_processing_time
                    })
                
                # 检查缓存命中率
                cache_hit_rate = data.get("processor", {}).get("cache_hit_rate", 0)
                if cache_hit_rate < self.alert_thresholds["min_cache_hit_rate"]:
                    alerts.append({
                        "type": "low_cache_hit_rate",
                        "message": f"ASR缓存命中率过低: {cache_hit_rate:.3f}",
                        "threshold": self.alert_thresholds["min_cache_hit_rate"],
                        "current": cache_hit_rate
                    })
            
            # 检查系统资源
            if system_stats:
                # 检查内存使用
                memory_used = system_stats.get("memory", {}).get("used", 0)
                if memory_used > self.alert_thresholds["max_memory_usage"]:
                    alerts.append({
                        "type": "high_memory_usage",
                        "message": f"系统内存使用过高: {memory_used}MB",
                        "threshold": self.alert_thresholds["max_memory_usage"],
                        "current": memory_used
                    })
                
                # 检查CPU使用率
                cpu_percent = system_stats.get("cpu", {}).get("percent", 0)
                if cpu_percent > 90:
                    alerts.append({
                        "type": "high_cpu_usage",
                        "message": f"CPU使用率过高: {cpu_percent}%",
                        "threshold": 90,
                        "current": cpu_percent
                    })
        
        except Exception as e:
            logger.error(f"告警检查失败: {e}")
        
        return alerts
    
    def log_stats(self, timestamp: str, health: Dict, stats: Dict, system: Dict, alerts: list):
        """记录统计信息"""
        log_entry = {
            "timestamp": timestamp,
            "health": health,
            "asr_stats": stats,
            "system_stats": system,
            "alerts": alerts
        }
        
        # 添加到历史记录
        self.stats_history.append(log_entry)
        
        # 保持最近100条记录
        if len(self.stats_history) > 100:
            self.stats_history.pop(0)
        
        # 记录到日志
        if health.get("status") == "healthy":
            logger.info(f"ASR服务健康 - 统计: {stats.get('status', 'unknown')}")
        else:
            logger.warning(f"ASR服务异常 - {health.get('error', 'unknown')}")
        
        # 记录告警
        for alert in alerts:
            logger.warning(f"告警: {alert['message']}")
    
    def print_dashboard(self, health: Dict, stats: Dict, system: Dict, alerts: list):
        """打印监控仪表板"""
        print("\n" + "="*60)
        print(f"🔍 ASR性能监控仪表板 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # 服务健康状态
        print(f"\n📊 服务状态:")
        if health.get("status") == "healthy":
            health_data = health.get("data", {})
            print(f"   ✅ ASR服务: {health_data.get('status', 'unknown')}")
            print(f"   🔄 当前并发: {health_data.get('current_concurrent', 0)}")
            print(f"   📈 最大并发: {health_data.get('max_concurrent', 0)}")
        else:
            print(f"   ❌ ASR服务: {health.get('error', 'unknown')}")
        
        # ASR统计信息
        print(f"\n📈 ASR统计:")
        if stats.get("status") == "success":
            data = stats["data"]
            processor = data.get("processor", {})
            service = data.get("service", {})
            
            print(f"   📊 总请求数: {service.get('total_requests', 0)}")
            print(f"   ⏱️  平均处理时间: {processor.get('avg_processing_time', 0):.3f}s")
            print(f"   🎯 缓存命中率: {processor.get('cache_hit_rate', 0):.3f}")
            print(f"   💾 缓存大小: {processor.get('cache_size', 0)}")
            print(f"   🔄 当前并发: {service.get('current_concurrent', 0)}")
        else:
            print(f"   ❌ 统计获取失败: {stats.get('error', 'unknown')}")
        
        # 系统资源
        print(f"\n💻 系统资源:")
        if system:
            cpu = system.get("cpu", {})
            memory = system.get("memory", {})
            disk = system.get("disk", {})
            
            print(f"   🖥️  CPU使用率: {cpu.get('percent', 0):.1f}%")
            print(f"   💾 内存使用: {memory.get('used', 0)}MB / {memory.get('total', 0)}MB ({memory.get('percent', 0):.1f}%)")
            print(f"   💿 磁盘使用: {disk.get('used', 0)}GB / {disk.get('total', 0)}GB ({disk.get('percent', 0):.1f}%)")
        
        # 告警信息
        if alerts:
            print(f"\n🚨 告警信息:")
            for alert in alerts:
                print(f"   ⚠️  {alert['message']}")
        else:
            print(f"\n✅ 无告警")
        
        print("="*60)
    
    def run_monitor(self, dashboard=True):
        """运行监控"""
        logger.info("启动ASR性能监控...")
        
        try:
            while True:
                timestamp = datetime.now().isoformat()
                
                # 获取各项统计
                health = self.get_asr_health()
                stats = self.get_asr_stats()
                system = self.get_system_stats()
                
                # 检查告警
                alerts = self.check_alerts(stats, system)
                
                # 记录统计
                self.log_stats(timestamp, health, stats, system, alerts)
                
                # 显示仪表板
                if dashboard:
                    self.print_dashboard(health, stats, system, alerts)
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("监控已停止")
        except Exception as e:
            logger.error(f"监控异常: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR性能监控")
    parser.add_argument("--url", default="http://localhost:8001", help="ASR服务URL")
    parser.add_argument("--interval", type=int, default=30, help="检查间隔(秒)")
    parser.add_argument("--no-dashboard", action="store_true", help="不显示仪表板")
    
    args = parser.parse_args()
    
    # 创建日志目录
    import os
    os.makedirs("/root/xiaozhi-server/logs", exist_ok=True)
    
    # 启动监控
    monitor = ASRPerformanceMonitor(args.url, args.interval)
    monitor.run_monitor(dashboard=not args.no_dashboard)

if __name__ == "__main__":
    main()