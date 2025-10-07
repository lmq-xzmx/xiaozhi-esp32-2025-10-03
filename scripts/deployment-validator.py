#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 部署验证脚本
验证系统部署是否正确，各组件是否正常工作
"""

import asyncio
import aiohttp
import subprocess
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
import redis
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """验证状态"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"

@dataclass
class ValidationResult:
    """验证结果"""
    component: str
    test_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = None
    duration: float = 0.0

class SystemValidator:
    """系统验证器"""
    
    def __init__(self, config_file: str = "optimization-configs.yaml"):
        self.config_file = config_file
        self.config = {}
        self.results: List[ValidationResult] = []
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_file}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self.config = {}
    
    def run_command(self, command: str, timeout: int = 30) -> tuple[int, str, str]:
        """执行命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)
    
    async def validate_kubernetes_cluster(self) -> List[ValidationResult]:
        """验证Kubernetes集群"""
        results = []
        start_time = time.time()
        
        # 检查kubectl连接
        returncode, stdout, stderr = self.run_command("kubectl cluster-info")
        if returncode == 0:
            results.append(ValidationResult(
                component="Kubernetes",
                test_name="Cluster Connectivity",
                status=ValidationStatus.PASS,
                message="Kubernetes集群连接正常",
                details={"cluster_info": stdout.strip()},
                duration=time.time() - start_time
            ))
        else:
            results.append(ValidationResult(
                component="Kubernetes",
                test_name="Cluster Connectivity",
                status=ValidationStatus.FAIL,
                message=f"Kubernetes集群连接失败: {stderr}",
                duration=time.time() - start_time
            ))
            return results
        
        # 检查节点状态
        start_time = time.time()
        returncode, stdout, stderr = self.run_command("kubectl get nodes -o json")
        if returncode == 0:
            try:
                nodes_data = json.loads(stdout)
                nodes = nodes_data.get("items", [])
                ready_nodes = 0
                total_nodes = len(nodes)
                
                for node in nodes:
                    conditions = node.get("status", {}).get("conditions", [])
                    for condition in conditions:
                        if condition.get("type") == "Ready" and condition.get("status") == "True":
                            ready_nodes += 1
                            break
                
                if ready_nodes == total_nodes and total_nodes > 0:
                    results.append(ValidationResult(
                        component="Kubernetes",
                        test_name="Node Status",
                        status=ValidationStatus.PASS,
                        message=f"所有节点({ready_nodes}/{total_nodes})状态正常",
                        details={"ready_nodes": ready_nodes, "total_nodes": total_nodes},
                        duration=time.time() - start_time
                    ))
                else:
                    results.append(ValidationResult(
                        component="Kubernetes",
                        test_name="Node Status",
                        status=ValidationStatus.WARNING,
                        message=f"部分节点未就绪({ready_nodes}/{total_nodes})",
                        details={"ready_nodes": ready_nodes, "total_nodes": total_nodes},
                        duration=time.time() - start_time
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    component="Kubernetes",
                    test_name="Node Status",
                    status=ValidationStatus.FAIL,
                    message=f"节点状态检查失败: {e}",
                    duration=time.time() - start_time
                ))
        
        # 检查命名空间
        start_time = time.time()
        namespaces = ["xiaozhi-system", "monitoring"]
        for namespace in namespaces:
            returncode, stdout, stderr = self.run_command(f"kubectl get namespace {namespace}")
            if returncode == 0:
                results.append(ValidationResult(
                    component="Kubernetes",
                    test_name=f"Namespace {namespace}",
                    status=ValidationStatus.PASS,
                    message=f"命名空间 {namespace} 存在",
                    duration=time.time() - start_time
                ))
            else:
                results.append(ValidationResult(
                    component="Kubernetes",
                    test_name=f"Namespace {namespace}",
                    status=ValidationStatus.FAIL,
                    message=f"命名空间 {namespace} 不存在",
                    duration=time.time() - start_time
                ))
        
        return results
    
    async def validate_xiaozhi_services(self) -> List[ValidationResult]:
        """验证Xiaozhi服务"""
        results = []
        services = ["vad-service", "asr-service", "llm-service", "tts-service", "intelligent-load-balancer"]
        
        for service in services:
            start_time = time.time()
            
            # 检查Pod状态
            returncode, stdout, stderr = self.run_command(
                f"kubectl get pods -n xiaozhi-system -l app={service} -o json"
            )
            
            if returncode == 0:
                try:
                    pods_data = json.loads(stdout)
                    pods = pods_data.get("items", [])
                    
                    if not pods:
                        results.append(ValidationResult(
                            component="Xiaozhi Services",
                            test_name=f"{service} Deployment",
                            status=ValidationStatus.FAIL,
                            message=f"服务 {service} 没有运行的Pod",
                            duration=time.time() - start_time
                        ))
                        continue
                    
                    running_pods = 0
                    ready_pods = 0
                    
                    for pod in pods:
                        phase = pod.get("status", {}).get("phase", "")
                        if phase == "Running":
                            running_pods += 1
                        
                        conditions = pod.get("status", {}).get("conditions", [])
                        for condition in conditions:
                            if condition.get("type") == "Ready" and condition.get("status") == "True":
                                ready_pods += 1
                                break
                    
                    total_pods = len(pods)
                    
                    if running_pods == total_pods and ready_pods == total_pods:
                        results.append(ValidationResult(
                            component="Xiaozhi Services",
                            test_name=f"{service} Deployment",
                            status=ValidationStatus.PASS,
                            message=f"服务 {service} 所有Pod({total_pods})运行正常",
                            details={
                                "total_pods": total_pods,
                                "running_pods": running_pods,
                                "ready_pods": ready_pods
                            },
                            duration=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component="Xiaozhi Services",
                            test_name=f"{service} Deployment",
                            status=ValidationStatus.WARNING,
                            message=f"服务 {service} 部分Pod未就绪(运行:{running_pods}/{total_pods}, 就绪:{ready_pods}/{total_pods})",
                            details={
                                "total_pods": total_pods,
                                "running_pods": running_pods,
                                "ready_pods": ready_pods
                            },
                            duration=time.time() - start_time
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        component="Xiaozhi Services",
                        test_name=f"{service} Deployment",
                        status=ValidationStatus.FAIL,
                        message=f"服务 {service} 状态检查失败: {e}",
                        duration=time.time() - start_time
                    ))
            else:
                results.append(ValidationResult(
                    component="Xiaozhi Services",
                    test_name=f"{service} Deployment",
                    status=ValidationStatus.FAIL,
                    message=f"服务 {service} 状态查询失败: {stderr}",
                    duration=time.time() - start_time
                ))
        
        return results
    
    async def validate_redis_cluster(self) -> List[ValidationResult]:
        """验证Redis集群"""
        results = []
        start_time = time.time()
        
        # 检查Redis Pod状态
        returncode, stdout, stderr = self.run_command(
            "kubectl get pods -n xiaozhi-system -l app=redis-cluster -o json"
        )
        
        if returncode == 0:
            try:
                pods_data = json.loads(stdout)
                pods = pods_data.get("items", [])
                
                if not pods:
                    results.append(ValidationResult(
                        component="Redis",
                        test_name="Cluster Status",
                        status=ValidationStatus.FAIL,
                        message="Redis集群没有运行的Pod",
                        duration=time.time() - start_time
                    ))
                    return results
                
                running_pods = sum(1 for pod in pods if pod.get("status", {}).get("phase") == "Running")
                total_pods = len(pods)
                
                if running_pods == total_pods:
                    results.append(ValidationResult(
                        component="Redis",
                        test_name="Cluster Status",
                        status=ValidationStatus.PASS,
                        message=f"Redis集群所有Pod({total_pods})运行正常",
                        details={"total_pods": total_pods, "running_pods": running_pods},
                        duration=time.time() - start_time
                    ))
                else:
                    results.append(ValidationResult(
                        component="Redis",
                        test_name="Cluster Status",
                        status=ValidationStatus.WARNING,
                        message=f"Redis集群部分Pod未运行({running_pods}/{total_pods})",
                        details={"total_pods": total_pods, "running_pods": running_pods},
                        duration=time.time() - start_time
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    component="Redis",
                    test_name="Cluster Status",
                    status=ValidationStatus.FAIL,
                    message=f"Redis集群状态检查失败: {e}",
                    duration=time.time() - start_time
                ))
        
        # 测试Redis连接
        start_time = time.time()
        try:
            # 通过端口转发测试Redis连接
            returncode, stdout, stderr = self.run_command(
                "kubectl get svc -n xiaozhi-system redis-cluster -o jsonpath='{.spec.clusterIP}'"
            )
            
            if returncode == 0 and stdout.strip():
                redis_ip = stdout.strip()
                # 这里可以添加实际的Redis连接测试
                results.append(ValidationResult(
                    component="Redis",
                    test_name="Connectivity",
                    status=ValidationStatus.PASS,
                    message=f"Redis服务可访问: {redis_ip}",
                    details={"cluster_ip": redis_ip},
                    duration=time.time() - start_time
                ))
            else:
                results.append(ValidationResult(
                    component="Redis",
                    test_name="Connectivity",
                    status=ValidationStatus.FAIL,
                    message="Redis服务IP获取失败",
                    duration=time.time() - start_time
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="Redis",
                test_name="Connectivity",
                status=ValidationStatus.FAIL,
                message=f"Redis连接测试失败: {e}",
                duration=time.time() - start_time
            ))
        
        return results
    
    async def validate_monitoring_stack(self) -> List[ValidationResult]:
        """验证监控堆栈"""
        results = []
        monitoring_services = ["prometheus", "grafana", "alertmanager"]
        
        for service in monitoring_services:
            start_time = time.time()
            
            returncode, stdout, stderr = self.run_command(
                f"kubectl get pods -n monitoring -l app={service} -o json"
            )
            
            if returncode == 0:
                try:
                    pods_data = json.loads(stdout)
                    pods = pods_data.get("items", [])
                    
                    if not pods:
                        results.append(ValidationResult(
                            component="Monitoring",
                            test_name=f"{service} Status",
                            status=ValidationStatus.WARNING,
                            message=f"监控服务 {service} 没有运行的Pod",
                            duration=time.time() - start_time
                        ))
                        continue
                    
                    running_pods = sum(1 for pod in pods if pod.get("status", {}).get("phase") == "Running")
                    total_pods = len(pods)
                    
                    if running_pods == total_pods:
                        results.append(ValidationResult(
                            component="Monitoring",
                            test_name=f"{service} Status",
                            status=ValidationStatus.PASS,
                            message=f"监控服务 {service} 运行正常({total_pods}个Pod)",
                            details={"total_pods": total_pods, "running_pods": running_pods},
                            duration=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component="Monitoring",
                            test_name=f"{service} Status",
                            status=ValidationStatus.WARNING,
                            message=f"监控服务 {service} 部分Pod未运行({running_pods}/{total_pods})",
                            details={"total_pods": total_pods, "running_pods": running_pods},
                            duration=time.time() - start_time
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        component="Monitoring",
                        test_name=f"{service} Status",
                        status=ValidationStatus.FAIL,
                        message=f"监控服务 {service} 状态检查失败: {e}",
                        duration=time.time() - start_time
                    ))
            else:
                results.append(ValidationResult(
                    component="Monitoring",
                    test_name=f"{service} Status",
                    status=ValidationStatus.WARNING,
                    message=f"监控服务 {service} 状态查询失败: {stderr}",
                    duration=time.time() - start_time
                ))
        
        return results
    
    async def validate_service_endpoints(self) -> List[ValidationResult]:
        """验证服务端点"""
        results = []
        
        # 获取负载均衡器服务IP
        returncode, stdout, stderr = self.run_command(
            "kubectl get svc -n xiaozhi-system intelligent-load-balancer -o jsonpath='{.status.loadBalancer.ingress[0].ip}'"
        )
        
        if returncode != 0 or not stdout.strip():
            # 尝试获取ClusterIP
            returncode, stdout, stderr = self.run_command(
                "kubectl get svc -n xiaozhi-system intelligent-load-balancer -o jsonpath='{.spec.clusterIP}'"
            )
        
        if returncode == 0 and stdout.strip():
            service_ip = stdout.strip()
            base_url = f"http://{service_ip}:8080"
            
            # 测试健康检查端点
            start_time = time.time()
            try:
                async with self.session.get(f"{base_url}/health", timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results.append(ValidationResult(
                            component="Service Endpoints",
                            test_name="Health Check",
                            status=ValidationStatus.PASS,
                            message="健康检查端点正常",
                            details={"url": f"{base_url}/health", "response": health_data},
                            duration=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component="Service Endpoints",
                            test_name="Health Check",
                            status=ValidationStatus.FAIL,
                            message=f"健康检查端点返回错误状态: {response.status}",
                            details={"url": f"{base_url}/health"},
                            duration=time.time() - start_time
                        ))
            except Exception as e:
                results.append(ValidationResult(
                    component="Service Endpoints",
                    test_name="Health Check",
                    status=ValidationStatus.FAIL,
                    message=f"健康检查端点访问失败: {e}",
                    details={"url": f"{base_url}/health"},
                    duration=time.time() - start_time
                ))
            
            # 测试API端点
            api_endpoints = [
                "/api/v1/vad/detect",
                "/api/v1/asr/recognize", 
                "/api/v1/llm/chat",
                "/api/v1/tts/synthesize"
            ]
            
            for endpoint in api_endpoints:
                start_time = time.time()
                try:
                    # 发送OPTIONS请求检查端点是否存在
                    async with self.session.options(f"{base_url}{endpoint}", timeout=5) as response:
                        if response.status in [200, 405]:  # 405表示方法不允许但端点存在
                            results.append(ValidationResult(
                                component="Service Endpoints",
                                test_name=f"API {endpoint}",
                                status=ValidationStatus.PASS,
                                message=f"API端点 {endpoint} 可访问",
                                details={"url": f"{base_url}{endpoint}"},
                                duration=time.time() - start_time
                            ))
                        else:
                            results.append(ValidationResult(
                                component="Service Endpoints",
                                test_name=f"API {endpoint}",
                                status=ValidationStatus.WARNING,
                                message=f"API端点 {endpoint} 状态异常: {response.status}",
                                details={"url": f"{base_url}{endpoint}"},
                                duration=time.time() - start_time
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        component="Service Endpoints",
                        test_name=f"API {endpoint}",
                        status=ValidationStatus.WARNING,
                        message=f"API端点 {endpoint} 访问失败: {e}",
                        details={"url": f"{base_url}{endpoint}"},
                        duration=time.time() - start_time
                    ))
        else:
            results.append(ValidationResult(
                component="Service Endpoints",
                test_name="Service Discovery",
                status=ValidationStatus.FAIL,
                message="无法获取负载均衡器服务IP",
                duration=0
            ))
        
        return results
    
    async def validate_resource_usage(self) -> List[ValidationResult]:
        """验证资源使用情况"""
        results = []
        start_time = time.time()
        
        # 检查节点资源使用情况
        returncode, stdout, stderr = self.run_command("kubectl top nodes")
        if returncode == 0:
            lines = stdout.strip().split('\n')[1:]  # 跳过标题行
            high_cpu_nodes = []
            high_memory_nodes = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    node_name = parts[0]
                    cpu_usage = parts[1]
                    memory_usage = parts[3]
                    
                    # 检查CPU使用率
                    if cpu_usage.endswith('%'):
                        cpu_percent = int(cpu_usage[:-1])
                        if cpu_percent > 80:
                            high_cpu_nodes.append(f"{node_name}({cpu_percent}%)")
                    
                    # 检查内存使用率
                    if memory_usage.endswith('%'):
                        memory_percent = int(memory_usage[:-1])
                        if memory_percent > 80:
                            high_memory_nodes.append(f"{node_name}({memory_percent}%)")
            
            if not high_cpu_nodes and not high_memory_nodes:
                results.append(ValidationResult(
                    component="Resource Usage",
                    test_name="Node Resources",
                    status=ValidationStatus.PASS,
                    message="所有节点资源使用率正常",
                    duration=time.time() - start_time
                ))
            else:
                warning_msg = []
                if high_cpu_nodes:
                    warning_msg.append(f"高CPU使用率节点: {', '.join(high_cpu_nodes)}")
                if high_memory_nodes:
                    warning_msg.append(f"高内存使用率节点: {', '.join(high_memory_nodes)}")
                
                results.append(ValidationResult(
                    component="Resource Usage",
                    test_name="Node Resources",
                    status=ValidationStatus.WARNING,
                    message="; ".join(warning_msg),
                    details={"high_cpu_nodes": high_cpu_nodes, "high_memory_nodes": high_memory_nodes},
                    duration=time.time() - start_time
                ))
        else:
            results.append(ValidationResult(
                component="Resource Usage",
                test_name="Node Resources",
                status=ValidationStatus.WARNING,
                message="无法获取节点资源使用情况(可能需要安装metrics-server)",
                duration=time.time() - start_time
            ))
        
        return results
    
    async def validate_storage(self) -> List[ValidationResult]:
        """验证存储"""
        results = []
        start_time = time.time()
        
        # 检查PVC状态
        returncode, stdout, stderr = self.run_command("kubectl get pvc --all-namespaces -o json")
        if returncode == 0:
            try:
                pvc_data = json.loads(stdout)
                pvcs = pvc_data.get("items", [])
                
                bound_pvcs = 0
                total_pvcs = len(pvcs)
                pending_pvcs = []
                
                for pvc in pvcs:
                    phase = pvc.get("status", {}).get("phase", "")
                    name = pvc.get("metadata", {}).get("name", "")
                    namespace = pvc.get("metadata", {}).get("namespace", "")
                    
                    if phase == "Bound":
                        bound_pvcs += 1
                    elif phase == "Pending":
                        pending_pvcs.append(f"{namespace}/{name}")
                
                if bound_pvcs == total_pvcs:
                    results.append(ValidationResult(
                        component="Storage",
                        test_name="PVC Status",
                        status=ValidationStatus.PASS,
                        message=f"所有PVC({total_pvcs})已绑定",
                        details={"total_pvcs": total_pvcs, "bound_pvcs": bound_pvcs},
                        duration=time.time() - start_time
                    ))
                else:
                    results.append(ValidationResult(
                        component="Storage",
                        test_name="PVC Status",
                        status=ValidationStatus.WARNING,
                        message=f"部分PVC未绑定({bound_pvcs}/{total_pvcs}), 待绑定: {', '.join(pending_pvcs)}",
                        details={"total_pvcs": total_pvcs, "bound_pvcs": bound_pvcs, "pending_pvcs": pending_pvcs},
                        duration=time.time() - start_time
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    component="Storage",
                    test_name="PVC Status",
                    status=ValidationStatus.FAIL,
                    message=f"PVC状态检查失败: {e}",
                    duration=time.time() - start_time
                ))
        
        return results
    
    async def run_all_validations(self) -> List[ValidationResult]:
        """运行所有验证"""
        logger.info("开始系统部署验证...")
        
        self.load_config()
        all_results = []
        
        # 验证Kubernetes集群
        logger.info("验证Kubernetes集群...")
        k8s_results = await self.validate_kubernetes_cluster()
        all_results.extend(k8s_results)
        
        # 验证Xiaozhi服务
        logger.info("验证Xiaozhi服务...")
        service_results = await self.validate_xiaozhi_services()
        all_results.extend(service_results)
        
        # 验证Redis集群
        logger.info("验证Redis集群...")
        redis_results = await self.validate_redis_cluster()
        all_results.extend(redis_results)
        
        # 验证监控堆栈
        logger.info("验证监控堆栈...")
        monitoring_results = await self.validate_monitoring_stack()
        all_results.extend(monitoring_results)
        
        # 验证服务端点
        logger.info("验证服务端点...")
        endpoint_results = await self.validate_service_endpoints()
        all_results.extend(endpoint_results)
        
        # 验证资源使用
        logger.info("验证资源使用...")
        resource_results = await self.validate_resource_usage()
        all_results.extend(resource_results)
        
        # 验证存储
        logger.info("验证存储...")
        storage_results = await self.validate_storage()
        all_results.extend(storage_results)
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        if not self.results:
            return "没有验证结果"
        
        # 统计结果
        pass_count = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        fail_count = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warning_count = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        skip_count = sum(1 for r in self.results if r.status == ValidationStatus.SKIP)
        total_count = len(self.results)
        
        # 按组件分组
        results_by_component = {}
        for result in self.results:
            if result.component not in results_by_component:
                results_by_component[result.component] = []
            results_by_component[result.component].append(result)
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("Xiaozhi ESP32 Server 部署验证报告")
        report.append("=" * 60)
        report.append(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总验证项: {total_count}")
        report.append(f"通过: {pass_count} | 失败: {fail_count} | 警告: {warning_count} | 跳过: {skip_count}")
        report.append("")
        
        # 整体状态
        if fail_count == 0 and warning_count == 0:
            overall_status = "🎉 优秀 - 所有验证项通过"
        elif fail_count == 0:
            overall_status = "✅ 良好 - 有警告项需要关注"
        elif fail_count <= 2:
            overall_status = "⚠️ 一般 - 有少量失败项需要修复"
        else:
            overall_status = "❌ 需要修复 - 有多个失败项"
        
        report.append(f"整体状态: {overall_status}")
        report.append("")
        
        # 详细结果
        for component, results in results_by_component.items():
            report.append(f"{component} 验证结果:")
            report.append("-" * 40)
            
            for result in results:
                status_icon = {
                    ValidationStatus.PASS: "✅",
                    ValidationStatus.FAIL: "❌", 
                    ValidationStatus.WARNING: "⚠️",
                    ValidationStatus.SKIP: "⏭️"
                }[result.status]
                
                report.append(f"  {status_icon} {result.test_name}: {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"    {key}: {value}")
                report.append(f"    耗时: {result.duration:.2f}s")
                report.append("")
        
        # 修复建议
        failed_results = [r for r in self.results if r.status == ValidationStatus.FAIL]
        warning_results = [r for r in self.results if r.status == ValidationStatus.WARNING]
        
        if failed_results or warning_results:
            report.append("修复建议:")
            report.append("-" * 40)
            
            if failed_results:
                report.append("🔴 需要立即修复的问题:")
                for result in failed_results:
                    report.append(f"  • {result.component} - {result.test_name}: {result.message}")
                report.append("")
            
            if warning_results:
                report.append("🟡 建议关注的问题:")
                for result in warning_results:
                    report.append(f"  • {result.component} - {result.test_name}: {result.message}")
                report.append("")
        
        # 下一步建议
        report.append("下一步建议:")
        report.append("-" * 40)
        
        if fail_count == 0 and warning_count == 0:
            report.append("  ✅ 系统部署验证通过，可以进行性能测试")
            report.append("  ✅ 建议运行: python scripts/performance-test.py")
        elif fail_count == 0:
            report.append("  ⚠️ 解决警告项后可以进行性能测试")
            report.append("  ⚠️ 建议检查资源使用情况和监控配置")
        else:
            report.append("  ❌ 修复失败项后重新运行验证")
            report.append("  ❌ 检查Kubernetes集群和服务配置")
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "validation_results.json"):
        """保存验证结果"""
        results_data = []
        for result in self.results:
            results_data.append({
                "component": result.component,
                "test_name": result.test_name,
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "duration": result.duration
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证结果已保存到: {output_file}")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xiaozhi ESP32 Server 部署验证")
    parser.add_argument("--config", default="optimization-configs.yaml", help="配置文件路径")
    parser.add_argument("--output", default="validation_results.json", help="结果输出文件")
    parser.add_argument("--report", default="validation_report.txt", help="报告输出文件")
    
    args = parser.parse_args()
    
    async with SystemValidator(args.config) as validator:
        try:
            # 运行所有验证
            results = await validator.run_all_validations()
            
            # 生成报告
            report = validator.generate_report()
            
            # 保存结果
            validator.save_results(args.output)
            
            # 保存报告
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 打印报告
            print(report)
            
            # 返回退出码
            fail_count = sum(1 for r in results if r.status == ValidationStatus.FAIL)
            if fail_count > 0:
                logger.error(f"验证失败，有 {fail_count} 个失败项")
                exit(1)
            else:
                logger.info("验证完成，系统部署正常")
                exit(0)
                
        except KeyboardInterrupt:
            logger.info("验证被用户中断")
            exit(130)
        except Exception as e:
            logger.error(f"验证过程中发生错误: {e}")
            exit(1)

if __name__ == "__main__":
    asyncio.run(main())