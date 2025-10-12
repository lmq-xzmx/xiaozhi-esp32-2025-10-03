#!/usr/bin/env python3
"""
代理服务器 - 解决前端跨端口访问问题
将Java API (8005) 和 Python API (8092) 代理到同一端口 (8080)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# 后端服务配置
JAVA_API_BASE = "http://localhost:8005"
PYTHON_API_BASE = "http://localhost:8092"

@app.route('/xiaozhi/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_java_api(path):
    """代理Java API请求"""
    try:
        url = f"{JAVA_API_BASE}/xiaozhi/{path}"
        
        # 转发请求
        if request.method == 'GET':
            response = requests.get(url, params=request.args)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), params=request.args)
        elif request.method == 'PUT':
            response = requests.put(url, json=request.get_json(), params=request.args)
        elif request.method == 'DELETE':
            response = requests.delete(url, params=request.args)
        
        # 返回响应
        return response.json(), response.status_code
    except Exception as e:
        return {"error": f"Java API proxy error: {str(e)}"}, 500

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_python_api(path):
    """代理Python API请求"""
    try:
        url = f"{PYTHON_API_BASE}/api/{path}"
        
        # 转发请求
        if request.method == 'GET':
            response = requests.get(url, params=request.args)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), params=request.args)
        elif request.method == 'PUT':
            response = requests.put(url, json=request.get_json(), params=request.args)
        elif request.method == 'DELETE':
            response = requests.delete(url, params=request.args)
        
        # 返回响应
        return response.json(), response.status_code
    except Exception as e:
        return {"error": f"Python API proxy error: {str(e)}"}, 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 检查Java API
        java_health = requests.get(f"{JAVA_API_BASE}/xiaozhi/health", timeout=5)
        java_status = java_health.status_code == 200
    except:
        java_status = False
    
    try:
        # 检查Python API
        python_health = requests.get(f"{PYTHON_API_BASE}/api/health", timeout=5)
        python_status = python_health.status_code == 200
    except:
        python_status = False
    
    return {
        "proxy_status": "healthy",
        "java_api": "healthy" if java_status else "unhealthy",
        "python_api": "healthy" if python_status else "unhealthy"
    }

if __name__ == '__main__':
    print("=" * 60)
    print("API代理服务器启动中...")
    print("=" * 60)
    print("代理地址: http://localhost:8081")
    print("Java API代理: /xiaozhi/* -> http://localhost:8005/xiaozhi/*")
    print("Python API代理: /api/* -> http://localhost:8092/api/*")
    print("健康检查: /health")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8081, debug=True)