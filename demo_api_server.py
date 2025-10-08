#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载演示数据
def load_demo_data():
    try:
        with open('demo_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "agents": [],
            "students": [],
            "devices": [],
            "device_bindings": [],
            "chat_records": []
        }

# 保存演示数据
def save_demo_data(data):
    with open('demo_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/api/demo-data-summary', methods=['GET'])
def get_demo_data_summary():
    """获取演示数据摘要"""
    data = load_demo_data()
    
    summary = {
        "total_devices": len(data.get("devices", [])),
        "total_agents": len(data.get("agents", [])),
        "total_students": len(data.get("students", [])),
        "total_chats": len(data.get("chat_records", [])),
        "device_bindings": data.get("device_bindings", []),
        "chat_records": data.get("chat_records", [])
    }
    
    return jsonify(summary)

@app.route('/api/devices', methods=['GET', 'POST'])
def handle_devices():
    """处理设备相关请求"""
    data = load_demo_data()
    
    if request.method == 'GET':
        return jsonify(data.get("devices", []))
    
    elif request.method == 'POST':
        new_device = request.json
        new_device['id'] = len(data.get("devices", [])) + 1
        new_device['created_at'] = datetime.now().isoformat()
        new_device['is_online'] = False
        new_device['last_connected_at'] = None
        
        if "devices" not in data:
            data["devices"] = []
        data["devices"].append(new_device)
        save_demo_data(data)
        
        return jsonify(new_device), 201

@app.route('/api/devices/<int:device_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_device(device_id):
    """处理单个设备的请求"""
    data = load_demo_data()
    devices = data.get("devices", [])
    
    device = next((d for d in devices if d['id'] == device_id), None)
    if not device:
        return jsonify({"error": "设备未找到"}), 404
    
    if request.method == 'GET':
        return jsonify(device)
    
    elif request.method == 'PUT':
        update_data = request.json
        device.update(update_data)
        save_demo_data(data)
        return jsonify(device)
    
    elif request.method == 'DELETE':
        devices.remove(device)
        save_demo_data(data)
        return jsonify({"message": "设备已删除"})

@app.route('/api/agents', methods=['GET', 'POST'])
def handle_agents():
    """处理智能体相关请求"""
    data = load_demo_data()
    
    if request.method == 'GET':
        return jsonify(data.get("agents", []))
    
    elif request.method == 'POST':
        new_agent = request.json
        new_agent['id'] = len(data.get("agents", [])) + 1
        new_agent['created_at'] = datetime.now().isoformat()
        
        if "agents" not in data:
            data["agents"] = []
        data["agents"].append(new_agent)
        save_demo_data(data)
        
        return jsonify(new_agent), 201

@app.route('/api/agents/<int:agent_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_agent(agent_id):
    """处理单个智能体的请求"""
    data = load_demo_data()
    agents = data.get("agents", [])
    
    agent = next((a for a in agents if a['id'] == agent_id), None)
    if not agent:
        return jsonify({"error": "智能体未找到"}), 404
    
    if request.method == 'GET':
        return jsonify(agent)
    
    elif request.method == 'PUT':
        update_data = request.json
        agent.update(update_data)
        save_demo_data(data)
        return jsonify(agent)
    
    elif request.method == 'DELETE':
        agents.remove(agent)
        save_demo_data(data)
        return jsonify({"message": "智能体已删除"})

@app.route('/api/students', methods=['GET', 'POST'])
def handle_students():
    """处理学员相关请求"""
    data = load_demo_data()
    
    if request.method == 'GET':
        return jsonify(data.get("students", []))
    
    elif request.method == 'POST':
        new_student = request.json
        new_student['id'] = len(data.get("students", [])) + 1
        new_student['created_at'] = datetime.now().isoformat()
        
        if "students" not in data:
            data["students"] = []
        data["students"].append(new_student)
        save_demo_data(data)
        
        return jsonify(new_student), 201

@app.route('/api/students/<int:student_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_student(student_id):
    """处理单个学员的请求"""
    data = load_demo_data()
    students = data.get("students", [])
    
    student = next((s for s in students if s['id'] == student_id), None)
    if not student:
        return jsonify({"error": "学员未找到"}), 404
    
    if request.method == 'GET':
        return jsonify(student)
    
    elif request.method == 'PUT':
        update_data = request.json
        student.update(update_data)
        save_demo_data(data)
        return jsonify(student)
    
    elif request.method == 'DELETE':
        students.remove(student)
        save_demo_data(data)
        return jsonify({"message": "学员已删除"})

@app.route('/api/bindings', methods=['GET', 'POST'])
def handle_bindings():
    """处理绑定关系相关请求"""
    data = load_demo_data()
    
    if request.method == 'GET':
        return jsonify(data.get("device_bindings", []))
    
    elif request.method == 'POST':
        new_binding = request.json
        new_binding['bind_time'] = datetime.now().isoformat()
        new_binding['bind_status'] = 'active'
        new_binding['is_current'] = True
        new_binding['bind_duration_days'] = 0
        
        if "device_bindings" not in data:
            data["device_bindings"] = []
        data["device_bindings"].append(new_binding)
        save_demo_data(data)
        
        return jsonify(new_binding), 201

@app.route('/api/bindings/<int:device_id>/<int:student_id>', methods=['DELETE'])
def unbind_device(device_id, student_id):
    """解绑设备和学员"""
    data = load_demo_data()
    bindings = data.get("device_bindings", [])
    
    binding = next((b for b in bindings if b['device_id'] == device_id and b['student_id'] == student_id and b['bind_status'] == 'active'), None)
    if not binding:
        return jsonify({"error": "绑定关系未找到"}), 404
    
    binding['bind_status'] = 'inactive'
    binding['unbind_time'] = datetime.now().isoformat()
    binding['is_current'] = False
    
    # 计算绑定天数
    bind_time = datetime.fromisoformat(binding['bind_time'])
    unbind_time = datetime.now()
    binding['bind_duration_days'] = (unbind_time - bind_time).days
    
    save_demo_data(data)
    return jsonify({"message": "设备已解绑"})

@app.route('/api/chat-stats', methods=['GET'])
def get_chat_stats():
    """获取聊天统计"""
    data = load_demo_data()
    chat_records = data.get("chat_records", [])
    
    # 按设备分组统计
    stats_by_device = {}
    for chat in chat_records:
        device_id = chat['device_id']
        if device_id not in stats_by_device:
            stats_by_device[device_id] = {
                'device_id': device_id,
                'total_chats': 0,
                'user_messages': 0,
                'ai_messages': 0,
                'first_chat_time': None,
                'last_chat_time': None
            }
        
        stats = stats_by_device[device_id]
        stats['total_chats'] += 1
        
        if chat['chat_type'] == 'user':
            stats['user_messages'] += 1
        elif chat['chat_type'] == 'ai':
            stats['ai_messages'] += 1
        
        chat_time = chat['created_at']
        if not stats['first_chat_time'] or chat_time < stats['first_chat_time']:
            stats['first_chat_time'] = chat_time
        if not stats['last_chat_time'] or chat_time > stats['last_chat_time']:
            stats['last_chat_time'] = chat_time
    
    return jsonify(list(stats_by_device.values()))

@app.route('/api/chat-records/<int:device_id>', methods=['GET'])
def get_device_chat_records(device_id):
    """获取指定设备的聊天记录"""
    data = load_demo_data()
    chat_records = data.get("chat_records", [])
    
    device_chats = [chat for chat in chat_records if chat['device_id'] == device_id]
    device_chats.sort(key=lambda x: x['created_at'])
    
    return jsonify(device_chats)

@app.route('/api/export/chat-data/<int:device_id>', methods=['GET'])
def export_chat_data(device_id):
    """导出指定设备的聊天数据"""
    data = load_demo_data()
    chat_records = data.get("chat_records", [])
    
    device_chats = [chat for chat in chat_records if chat['device_id'] == device_id]
    
    # 获取设备信息
    devices = data.get("devices", [])
    device = next((d for d in devices if d['id'] == device_id), None)
    
    export_data = {
        "device_info": device,
        "chat_records": device_chats,
        "export_time": datetime.now().isoformat(),
        "total_records": len(device_chats)
    }
    
    return jsonify(export_data)

@app.route('/api/statistics/overview', methods=['GET'])
def get_overview_statistics():
    """获取概览统计数据"""
    data = load_demo_data()
    
    # 计算各种统计数据
    devices = data.get("devices", [])
    agents = data.get("agents", [])
    students = data.get("students", [])
    bindings = data.get("device_bindings", [])
    chats = data.get("chat_records", [])
    
    # 在线设备数
    online_devices = len([d for d in devices if d.get('is_online', False)])
    
    # 活跃绑定数
    active_bindings = len([b for b in bindings if b.get('bind_status') == 'active'])
    
    # 今日聊天数
    today = datetime.now().date()
    today_chats = len([c for c in chats if datetime.fromisoformat(c['created_at']).date() == today])
    
    # 智能体使用统计
    agent_usage = {}
    for device in devices:
        agent_id = device.get('agent_id')
        if agent_id:
            agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1
    
    statistics = {
        "total_devices": len(devices),
        "online_devices": online_devices,
        "total_agents": len(agents),
        "total_students": len(students),
        "active_bindings": active_bindings,
        "total_chats": len(chats),
        "today_chats": today_chats,
        "agent_usage": agent_usage
    }
    
    return jsonify(statistics)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "一对多关系管理API"
    })

if __name__ == '__main__':
    print("启动一对多关系管理API服务器...")
    print("访问地址: http://localhost:8090")
    print("API文档:")
    print("  - GET  /api/devices          - 获取设备列表")
    print("  - POST /api/devices          - 创建新设备")
    print("  - GET  /api/agents           - 获取智能体列表")
    print("  - POST /api/agents           - 创建新智能体")
    print("  - GET  /api/students         - 获取学员列表")
    print("  - POST /api/students         - 创建新学员")
    print("  - GET  /api/bindings         - 获取绑定关系")
    print("  - POST /api/bindings         - 创建新绑定")
    print("  - GET  /api/chat-stats       - 获取聊天统计")
    print("  - GET  /api/demo-data-summary - 获取演示数据摘要")
    print("  - GET  /health               - 健康检查")
    
    app.run(host='0.0.0.0', port=8090, debug=True)