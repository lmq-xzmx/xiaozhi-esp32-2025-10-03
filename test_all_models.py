#!/usr/bin/env python3
"""
简化模型测试脚本
测试服务状态和配置
"""

import requests
import json
import yaml
import os
from datetime import datetime

class ModelTester:
    def __init__(self):
        self.test_results = {
            "主服务": {"status": "未测试", "details": ""},
            "Manager-API": {"status": "未测试", "details": ""},
            "配置文件": {"status": "未测试", "details": ""},
            "Docker服务": {"status": "未测试", "details": ""}
        }
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
    def print_result(self, module, status, details=""):
        status_icon = "✅" if status == "正常" else "❌" if status == "异常" else "⏳"
        print(f"{status_icon} {module}: {status}")
        if details:
            print(f"   详情: {details}")
    
    def test_main_service(self):
        """测试主服务状态"""
        self.print_header("测试主服务状态")
        
        ports_to_test = [8080, 8000, 3000]
        
        for port in ports_to_test:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    self.test_results["主服务"]["status"] = "正常"
                    self.test_results["主服务"]["details"] = f"端口{port}响应正常"
                    self.print_result("主服务", "正常", f"端口{port}响应正常")
                    return
                else:
                    print(f"⚠️ 端口{port}状态码: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"❌ 端口{port}无法连接")
            except Exception as e:
                print(f"❌ 端口{port}测试失败: {e}")
        
        # 如果所有端口都失败
        self.test_results["主服务"]["status"] = "异常"
        self.test_results["主服务"]["details"] = "所有测试端口都无法连接"
        self.print_result("主服务", "异常", "所有测试端口都无法连接")
    
    def test_manager_api(self):
        """测试Manager-API状态"""
        self.print_header("测试Manager-API状态")
        
        api_urls = [
            "http://xiaozhi-esp32-server-web:8002/xiaozhi/health",
            "http://localhost:8002/xiaozhi/health",
            "http://182.44.78.40:8002/xiaozhi/health"
        ]
        
        for url in api_urls:
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 401:
                    self.test_results["Manager-API"]["status"] = "正常"
                    self.test_results["Manager-API"]["details"] = f"API运行正常({url})，需要认证"
                    self.print_result("Manager-API", "正常", f"API运行正常，需要认证")
                    return
                elif response.status_code == 200:
                    self.test_results["Manager-API"]["status"] = "正常"
                    self.test_results["Manager-API"]["details"] = f"API运行正常({url})"
                    self.print_result("Manager-API", "正常", f"API运行正常")
                    return
                else:
                    print(f"⚠️ {url} 状态码: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ 无法连接到: {url}")
            except Exception as e:
                print(f"❌ 测试失败 {url}: {e}")
        
        # 如果所有URL都失败
        self.test_results["Manager-API"]["status"] = "异常"
        self.test_results["Manager-API"]["details"] = "所有API端点都无法访问"
        self.print_result("Manager-API", "异常", "所有API端点都无法访问")
    
    def test_config_files(self):
        """测试配置文件状态"""
        self.print_header("测试配置文件状态")
        
        config_file = "/root/xiaozhi-server/data/.config.yaml"
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 检查关键配置
                checks = []
                
                # 检查manager-api配置
                if 'manager-api' in config:
                    api_config = config['manager-api']
                    if 'url' in api_config and 'secret' in api_config:
                        checks.append("✅ Manager-API配置完整")
                    else:
                        checks.append("❌ Manager-API配置不完整")
                else:
                    checks.append("❌ 缺少Manager-API配置")
                
                # 检查read_config_from_api
                if config.get('read_config_from_api', False):
                    checks.append("✅ 启用远程配置读取")
                else:
                    checks.append("⚠️ 未启用远程配置读取")
                
                # 检查local_override
                if 'local_override' in config:
                    overrides = config['local_override']
                    if 'asr' in overrides:
                        checks.append("✅ ASR本地覆盖配置存在")
                    else:
                        checks.append("⚠️ 无ASR本地覆盖配置")
                else:
                    checks.append("⚠️ 无本地覆盖配置")
                
                self.test_results["配置文件"]["status"] = "正常"
                self.test_results["配置文件"]["details"] = "; ".join(checks)
                self.print_result("配置文件", "正常", "; ".join(checks))
                
                # 打印配置详情
                print(f"\n📋 配置文件内容:")
                print(f"   Manager-API URL: {config.get('manager-api', {}).get('url', '未配置')}")
                print(f"   远程配置读取: {config.get('read_config_from_api', False)}")
                print(f"   本地覆盖: {list(config.get('local_override', {}).keys())}")
                
            else:
                self.test_results["配置文件"]["status"] = "异常"
                self.test_results["配置文件"]["details"] = "配置文件不存在"
                self.print_result("配置文件", "异常", "配置文件不存在")
                
        except Exception as e:
            self.test_results["配置文件"]["status"] = "异常"
            self.test_results["配置文件"]["details"] = f"读取配置文件失败: {e}"
            self.print_result("配置文件", "异常", f"读取配置文件失败: {e}")
    
    def test_docker_services(self):
        """测试Docker服务状态"""
        self.print_header("测试Docker服务状态")
        
        try:
            import subprocess
            
            # 检查docker-compose服务
            result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\\t{{.Status}}'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout
                print("🐳 Docker容器状态:")
                print(output)
                
                # 检查关键服务
                if 'xiaozhi-esp32-server' in output:
                    if 'Up' in output:
                        self.test_results["Docker服务"]["status"] = "正常"
                        self.test_results["Docker服务"]["details"] = "主要容器运行正常"
                        self.print_result("Docker服务", "正常", "主要容器运行正常")
                    else:
                        self.test_results["Docker服务"]["status"] = "异常"
                        self.test_results["Docker服务"]["details"] = "容器状态异常"
                        self.print_result("Docker服务", "异常", "容器状态异常")
                else:
                    self.test_results["Docker服务"]["status"] = "异常"
                    self.test_results["Docker服务"]["details"] = "未找到主要容器"
                    self.print_result("Docker服务", "异常", "未找到主要容器")
            else:
                self.test_results["Docker服务"]["status"] = "异常"
                self.test_results["Docker服务"]["details"] = "Docker命令执行失败"
                self.print_result("Docker服务", "异常", "Docker命令执行失败")
                
        except subprocess.TimeoutExpired:
            self.test_results["Docker服务"]["status"] = "异常"
            self.test_results["Docker服务"]["details"] = "Docker命令超时"
            self.print_result("Docker服务", "异常", "Docker命令超时")
        except Exception as e:
            self.test_results["Docker服务"]["status"] = "异常"
            self.test_results["Docker服务"]["details"] = f"Docker测试失败: {e}"
            self.print_result("Docker服务", "异常", f"Docker测试失败: {e}")
    
    def test_vad_config(self):
        """测试VAD配置"""
        self.print_header("测试VAD配置")
        
        try:
            # 检查VAD配置文件
            vad_config_file = "/root/xiaozhi-server/data/.wakeup_words.yaml"
            if os.path.exists(vad_config_file):
                with open(vad_config_file, 'r', encoding='utf-8') as f:
                    vad_config = yaml.safe_load(f)
                print(f"✅ VAD配置文件存在")
                print(f"   配置内容: {vad_config}")
            else:
                print(f"⚠️ VAD配置文件不存在")
            
            # 检查VAD模型文件
            vad_model_path = "/root/xiaozhi-server/models/snakers4_silero-vad"
            if os.path.exists(vad_model_path):
                print(f"✅ VAD模型文件存在: {vad_model_path}")
            else:
                print(f"❌ VAD模型文件不存在: {vad_model_path}")
                
        except Exception as e:
            print(f"❌ VAD配置检查失败: {e}")
    
    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试结果总结")
        
        for module, result in self.test_results.items():
            self.print_result(module, result["status"], result["details"])
        
        # 统计
        total = len(self.test_results)
        normal = sum(1 for r in self.test_results.values() if r["status"] == "正常")
        abnormal = sum(1 for r in self.test_results.values() if r["status"] == "异常")
        untested = sum(1 for r in self.test_results.values() if r["status"] == "未测试")
        
        print(f"\n📊 测试统计:")
        print(f"   总计: {total} 个模块")
        print(f"   正常: {normal} 个")
        print(f"   异常: {abnormal} 个")
        print(f"   未测试: {untested} 个")
        
        if abnormal == 0 and untested == 0:
            print(f"\n🎉 所有模块测试通过！")
        elif abnormal > 0:
            print(f"\n⚠️ 发现 {abnormal} 个模块异常，需要检查")
        else:
            print(f"\n⏳ 还有 {untested} 个模块未完成测试")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"🚀 开始系统状态测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 测试Docker服务
        self.test_docker_services()
        
        # 2. 测试配置文件
        self.test_config_files()
        
        # 3. 测试主服务
        self.test_main_service()
        
        # 4. 测试Manager-API
        self.test_manager_api()
        
        # 5. 测试VAD配置
        self.test_vad_config()
        
        # 6. 打印总结
        self.print_summary()

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_all_tests()