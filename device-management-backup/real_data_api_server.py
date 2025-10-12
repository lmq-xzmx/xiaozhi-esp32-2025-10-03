#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据API服务器
连接MySQL数据库，提供一对多关系管理的RESTful API接口
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql
import logging
from datetime import datetime, timedelta
import json
from decimal import Decimal

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保JSON响应支持中文
CORS(app)  # 允许跨域请求

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True,
    'use_unicode': True,
    'init_command': "SET sql_mode='STRICT_TRANS_TABLES'"
}

def get_db_connection():
    """获取数据库连接"""
    try:
        # 尝试连接Docker容器中的MySQL (使用正确的容器IP)
        config = DB_CONFIG.copy()
        config['host'] = '172.20.0.5'  # 更新为正确的MySQL容器IP
        connection = pymysql.connect(**config)
        # 设置连接字符集
        connection.set_charset('utf8mb4')
        return connection
    except Exception as e:
        logger.error(f"Docker IP连接失败: {e}")
        # 如果IP连接失败，尝试通过容器名连接
        try:
            config = DB_CONFIG.copy()
            config['host'] = 'xiaozhi-esp32-server-db'
            connection = pymysql.connect(**config)
            # 设置连接字符集
            connection.set_charset('utf8mb4')
            return connection
        except Exception as e2:
            logger.error(f"容器名连接也失败: {e2}")
            # 最后尝试localhost
            try:
                connection = pymysql.connect(**DB_CONFIG)
                return connection
            except Exception as e3:
                logger.error(f"localhost连接也失败: {e3}")
                raise e3

def execute_query(query, params=None, fetch_one=False):
    """执行数据库查询"""
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            if fetch_one:
                result = cursor.fetchone()
            else:
                result = cursor.fetchall()
            connection.commit()
            return result
    except Exception as e:
        logger.error(f"查询执行失败: {e}")
        if connection:
            connection.rollback()
        raise e
    finally:
        if connection:
            connection.close()

def json_serializer(obj):
    """JSON序列化器，处理datetime和Decimal类型"""
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        execute_query("SELECT 1")
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """获取设备列表"""
    try:
        query = """
        SELECT 
            d.id,
            d.mac_address,
            d.alias,
            d.agent_id,
            d.student_id,
            d.bind_status,
            d.bind_time,
            d.last_connected_at,
            d.app_version,
            d.board,
            d.remark,
            a.agent_name,
            u.real_name as student_name,
            u.username as student_username
        FROM ai_device d
        LEFT JOIN ai_agent a ON d.agent_id = a.id
        LEFT JOIN sys_user u ON d.student_id = u.id
        ORDER BY d.create_date DESC
        """
        devices = execute_query(query)
        
        # 处理数据格式
        for device in devices:
            device['bind_status_text'] = '已绑定' if device['bind_status'] == 1 else '未绑定'
            if device['last_connected_at']:
                device['last_connected_at'] = device['last_connected_at'].strftime('%Y-%m-%d %H:%M:%S')
            if device['bind_time']:
                device['bind_time'] = device['bind_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': devices,
            'total': len(devices)
        })
    except Exception as e:
        logger.error(f"获取设备列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """获取智能体列表"""
    try:
        query = """
        SELECT 
            a.id,
            a.agent_code,
            a.agent_name,
            a.system_prompt,
            a.lang_code,
            a.language,
            a.created_at,
            a.updated_at,
            COUNT(d.id) as device_count
        FROM ai_agent a
        LEFT JOIN ai_device d ON a.id = d.agent_id
        GROUP BY a.id, a.agent_code, a.agent_name, a.system_prompt, a.lang_code, a.language, a.created_at, a.updated_at
        ORDER BY a.created_at DESC
        """
        agents = execute_query(query)
        
        # 处理数据格式
        for agent in agents:
            if agent['created_at']:
                agent['created_at'] = agent['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if agent['updated_at']:
                agent['updated_at'] = agent['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
            # 截断系统提示词
            if agent['system_prompt'] and len(agent['system_prompt']) > 100:
                agent['system_prompt_short'] = agent['system_prompt'][:100] + '...'
            else:
                agent['system_prompt_short'] = agent['system_prompt']
        
        return jsonify({
            'success': True,
            'data': agents,
            'total': len(agents)
        })
    except Exception as e:
        logger.error(f"获取智能体列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    """获取学生列表"""
    try:
        query = """
        SELECT 
            u.id,
            u.username,
            u.real_name,
            u.school_name,
            u.student_id,
            u.join_grade,
            u.current_grade,
            u.class_name,
            u.contact_phone,
            u.contact_email,
            u.enrollment_date,
            u.create_date,
            COUNT(d.id) as device_count
        FROM sys_user u
        LEFT JOIN ai_device d ON u.id = d.student_id
        WHERE u.id != 1975423707193643010  -- 排除root用户
        GROUP BY u.id, u.username, u.real_name, u.school_name, u.student_id, 
                 u.join_grade, u.current_grade, u.class_name, u.contact_phone, 
                 u.contact_email, u.enrollment_date, u.create_date
        ORDER BY u.create_date DESC
        """
        students = execute_query(query)
        
        # 处理数据格式
        for student in students:
            if student['enrollment_date']:
                student['enrollment_date'] = student['enrollment_date'].strftime('%Y-%m-%d')
            if student['create_date']:
                student['create_date'] = student['create_date'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': students,
            'total': len(students)
        })
    except Exception as e:
        logger.error(f"获取学生列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students', methods=['POST'])
def create_student():
    """创建新学生"""
    try:
        data = request.get_json()
        
        # 验证必填字段
        required_fields = ['username', 'real_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'{field} 是必填字段'
                }), 400
        
        # 检查用户名是否已存在
        check_query = "SELECT id FROM sys_user WHERE username = %s"
        existing = execute_query(check_query, (data['username'],), fetch_one=True)
        if existing:
            return jsonify({
                'success': False,
                'error': '用户名已存在'
            }), 400
        
        # 插入新学生
        insert_query = """
        INSERT INTO sys_user (
            username, real_name, school_name, student_id, join_grade, 
            current_grade, class_name, contact_phone, contact_email, 
            enrollment_date, create_date, update_date
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
        )
        """
        
        params = (
            data['username'],
            data['real_name'],
            data.get('school_name'),
            data.get('student_id'),
            data.get('join_grade'),
            data.get('current_grade'),
            data.get('class_name'),
            data.get('contact_phone'),
            data.get('contact_email'),
            data.get('enrollment_date')
        )
        
        execute_query(insert_query, params)
        
        return jsonify({
            'success': True,
            'message': '学生创建成功'
        }), 201
        
    except Exception as e:
        logger.error(f"创建学生失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    """获取单个学生信息"""
    try:
        query = """
        SELECT 
            u.id,
            u.username,
            u.real_name,
            u.school_name,
            u.student_id,
            u.join_grade,
            u.current_grade,
            u.class_name,
            u.contact_phone,
            u.contact_email,
            u.enrollment_date,
            u.create_date,
            u.update_date,
            COUNT(d.id) as device_count
        FROM sys_user u
        LEFT JOIN ai_device d ON u.id = d.student_id
        WHERE u.id = %s
        GROUP BY u.id, u.username, u.real_name, u.school_name, u.student_id, 
                 u.join_grade, u.current_grade, u.class_name, u.contact_phone, 
                 u.contact_email, u.enrollment_date, u.create_date, u.update_date
        """
        
        student = execute_query(query, (student_id,), fetch_one=True)
        
        if not student:
            return jsonify({
                'success': False,
                'error': '学生不存在'
            }), 404
        
        # 处理数据格式
        if student['enrollment_date']:
            student['enrollment_date'] = student['enrollment_date'].strftime('%Y-%m-%d')
        if student['create_date']:
            student['create_date'] = student['create_date'].strftime('%Y-%m-%d %H:%M:%S')
        if student['update_date']:
            student['update_date'] = student['update_date'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': student
        })
        
    except Exception as e:
        logger.error(f"获取学生信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students/<int:student_id>', methods=['PUT'])
def update_student(student_id):
    """更新学生信息"""
    try:
        data = request.get_json()
        
        # 检查学生是否存在
        check_query = "SELECT id FROM sys_user WHERE id = %s"
        existing = execute_query(check_query, (student_id,), fetch_one=True)
        if not existing:
            return jsonify({
                'success': False,
                'error': '学生不存在'
            }), 404
        
        # 如果要更新用户名，检查是否与其他用户冲突
        if 'username' in data:
            username_check = "SELECT id FROM sys_user WHERE username = %s AND id != %s"
            conflict = execute_query(username_check, (data['username'], student_id), fetch_one=True)
            if conflict:
                return jsonify({
                    'success': False,
                    'error': '用户名已被其他用户使用'
                }), 400
        
        # 构建更新查询
        update_fields = []
        params = []
        
        allowed_fields = [
            'username', 'real_name', 'school_name', 'student_id', 
            'join_grade', 'current_grade', 'class_name', 
            'contact_phone', 'contact_email', 'enrollment_date'
        ]
        
        for field in allowed_fields:
            if field in data:
                update_fields.append(f"{field} = %s")
                params.append(data[field])
        
        if not update_fields:
            return jsonify({
                'success': False,
                'error': '没有提供要更新的字段'
            }), 400
        
        update_fields.append("update_date = NOW()")
        params.append(student_id)
        
        update_query = f"UPDATE sys_user SET {', '.join(update_fields)} WHERE id = %s"
        execute_query(update_query, params)
        
        return jsonify({
            'success': True,
            'message': '学生信息更新成功'
        })
        
    except Exception as e:
        logger.error(f"更新学生信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    """删除学生"""
    try:
        # 检查学生是否存在
        check_query = "SELECT id FROM sys_user WHERE id = %s"
        existing = execute_query(check_query, (student_id,), fetch_one=True)
        if not existing:
            return jsonify({
                'success': False,
                'error': '学生不存在'
            }), 404
        
        # 检查是否有绑定的设备
        device_check = "SELECT COUNT(*) as count FROM ai_device WHERE student_id = %s"
        device_count = execute_query(device_check, (student_id,), fetch_one=True)
        if device_count['count'] > 0:
            return jsonify({
                'success': False,
                'error': '该学生还有绑定的设备，请先解绑设备后再删除'
            }), 400
        
        # 删除学生
        delete_query = "DELETE FROM sys_user WHERE id = %s"
        execute_query(delete_query, (student_id,))
        
        return jsonify({
            'success': True,
            'message': '学生删除成功'
        })
        
    except Exception as e:
        logger.error(f"删除学生失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/bind-relations', methods=['GET'])
def get_bind_relations():
    """获取绑定关系列表"""
    try:
        query = """
        SELECT 
            d.id as device_id,
            d.mac_address,
            d.alias as device_alias,
            d.student_id,
            d.agent_id,
            d.bind_status,
            d.bind_time,
            d.remark,
            u.real_name as student_name,
            u.username as student_username,
            u.school_name,
            u.class_name,
            a.agent_name,
            a.agent_code
        FROM ai_device d
        LEFT JOIN sys_user u ON d.student_id = u.id
        LEFT JOIN ai_agent a ON d.agent_id = a.id
        WHERE d.bind_status = 1  -- 只显示已绑定的设备
        ORDER BY d.bind_time DESC
        """
        relations = execute_query(query)
        
        # 处理数据格式
        for relation in relations:
            if relation['bind_time']:
                relation['bind_time'] = relation['bind_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': relations,
            'total': len(relations)
        })
    except Exception as e:
        logger.error(f"获取绑定关系失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat-statistics', methods=['GET'])
def get_chat_statistics():
    """获取聊天统计数据"""
    try:
        # 获取基本统计
        stats_query = """
        SELECT 
            COUNT(*) as total_chats,
            COUNT(DISTINCT device_id) as active_devices,
            COUNT(DISTINCT student_id) as active_students,
            COUNT(DISTINCT agent_id) as active_agents,
            DATE(created_at) as chat_date
        FROM ai_agent_chat_history 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY DATE(created_at)
        ORDER BY chat_date DESC
        LIMIT 30
        """
        daily_stats = execute_query(stats_query)
        
        # 获取设备聊天统计
        device_stats_query = """
        SELECT 
            d.id as device_id,
            d.mac_address,
            d.alias,
            u.real_name as student_name,
            a.agent_name,
            COUNT(c.id) as chat_count,
            MAX(c.created_at) as last_chat_time
        FROM ai_device d
        LEFT JOIN ai_agent_chat_history c ON d.id = c.device_id
        LEFT JOIN sys_user u ON d.student_id = u.id
        LEFT JOIN ai_agent a ON d.agent_id = a.id
        WHERE c.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY d.id, d.mac_address, d.alias, u.real_name, a.agent_name
        HAVING chat_count > 0
        ORDER BY chat_count DESC
        LIMIT 20
        """
        device_stats = execute_query(device_stats_query)
        
        # 处理数据格式
        for stat in daily_stats:
            if stat['chat_date']:
                stat['chat_date'] = stat['chat_date'].strftime('%Y-%m-%d')
        
        for stat in device_stats:
            if stat['last_chat_time']:
                stat['last_chat_time'] = stat['last_chat_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': {
                'daily_statistics': daily_stats,
                'device_statistics': device_stats
            }
        })
    except Exception as e:
        logger.error(f"获取聊天统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat-records/<device_id>', methods=['GET'])
def get_device_chat_records(device_id):
    """获取指定设备的聊天记录"""
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        offset = (page - 1) * per_page
        
        # 获取聊天记录
        query = """
        SELECT 
            c.id,
            c.device_id,
            c.student_id,
            c.agent_id,
            c.chat_type,
            c.content,
            c.created_at,
            c.updated_at,
            d.mac_address,
            d.alias as device_alias,
            u.real_name as student_name,
            a.agent_name
        FROM ai_agent_chat_history c
        LEFT JOIN ai_device d ON c.device_id = d.id
        LEFT JOIN sys_user u ON c.student_id = u.id
        LEFT JOIN ai_agent a ON c.agent_id = a.id
        WHERE c.device_id = %s
        ORDER BY c.created_at DESC
        LIMIT %s OFFSET %s
        """
        chat_records = execute_query(query, (device_id, per_page, offset))
        
        # 获取总记录数
        count_query = """
        SELECT COUNT(*) as total
        FROM ai_agent_chat_history
        WHERE device_id = %s
        """
        total_result = execute_query(count_query, (device_id,), fetch_one=True)
        total = total_result['total'] if total_result else 0
        
        # 处理数据格式
        for record in chat_records:
            if record['created_at']:
                record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if record['updated_at']:
                record['updated_at'] = record['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': {
                'records': chat_records,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        })
    except Exception as e:
        logger.error(f"获取设备聊天记录失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/chat-data/<device_id>', methods=['GET'])
def export_chat_data(device_id):
    """导出指定设备的聊天数据"""
    try:
        # 获取设备信息
        device_query = """
        SELECT 
            d.id,
            d.mac_address,
            d.alias,
            d.agent_id,
            d.student_id,
            d.bind_status,
            d.bind_time,
            d.last_connected_at,
            a.agent_name,
            u.real_name as student_name,
            u.username as student_username
        FROM ai_device d
        LEFT JOIN ai_agent a ON d.agent_id = a.id
        LEFT JOIN sys_user u ON d.student_id = u.id
        WHERE d.id = %s
        """
        device_info = execute_query(device_query, (device_id,), fetch_one=True)
        
        if not device_info:
            return jsonify({
                'success': False,
                'error': f'设备 {device_id} 不存在'
            }), 404
        
        # 获取所有聊天记录
        chat_query = """
        SELECT 
            c.id,
            c.device_id,
            c.student_id,
            c.agent_id,
            c.chat_type,
            c.content,
            c.created_at,
            c.updated_at
        FROM ai_agent_chat_history c
        WHERE c.device_id = %s
        ORDER BY c.created_at ASC
        """
        chat_records = execute_query(chat_query, (device_id,))
        
        # 处理数据格式
        if device_info['bind_time']:
            device_info['bind_time'] = device_info['bind_time'].strftime('%Y-%m-%d %H:%M:%S')
        if device_info['last_connected_at']:
            device_info['last_connected_at'] = device_info['last_connected_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        for record in chat_records:
            if record['created_at']:
                record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if record['updated_at']:
                record['updated_at'] = record['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        export_data = {
            "device_info": device_info,
            "chat_records": chat_records,
            "export_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_records": len(chat_records)
        }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
    except Exception as e:
        logger.error(f"导出聊天数据失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """获取数据摘要"""
    try:
        # 总体统计
        summary_query = """
        SELECT 
            (SELECT COUNT(*) FROM ai_device) as total_devices,
            (SELECT COUNT(*) FROM ai_device WHERE bind_status = 1) as bound_devices,
            (SELECT COUNT(*) FROM ai_agent) as total_agents,
            (SELECT COUNT(*) FROM sys_user WHERE id != 1975423707193643010) as total_students,
            (SELECT COUNT(*) FROM ai_agent_chat_history WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)) as chats_today,
            (SELECT COUNT(*) FROM ai_agent_chat_history WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)) as chats_week
        """
        summary = execute_query(summary_query, fetch_one=True)
        
        # 最近活动
        recent_activity_query = """
        SELECT 
            'chat' as activity_type,
            c.created_at,
            d.mac_address as device_mac,
            u.real_name as student_name,
            a.agent_name,
            c.content
        FROM ai_agent_chat_history c
        LEFT JOIN ai_device d ON c.device_id = d.id
        LEFT JOIN sys_user u ON c.student_id = u.id
        LEFT JOIN ai_agent a ON c.agent_id = a.id
        WHERE c.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        ORDER BY c.created_at DESC
        LIMIT 10
        """
        recent_activities = execute_query(recent_activity_query)
        
        # 处理数据格式
        for activity in recent_activities:
            if activity['created_at']:
                activity['created_at'] = activity['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            # 截断聊天内容
            if activity['content'] and len(activity['content']) > 50:
                activity['content'] = activity['content'][:50] + '...'
        
        return jsonify({
            'success': True,
            'data': {
                'summary': summary,
                'recent_activities': recent_activities
            }
        })
    except Exception as e:
        logger.error(f"获取数据摘要失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/device/<device_id>/bind', methods=['POST'])
def bind_device(device_id):
    """绑定设备到学生"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        agent_id = data.get('agent_id')
        remark = data.get('remark', '')
        
        if not student_id:
            return jsonify({
                'success': False,
                'error': '学生ID不能为空'
            }), 400
        
        # 更新设备绑定信息
        update_query = """
        UPDATE ai_device 
        SET student_id = %s, agent_id = %s, bind_status = 1, 
            bind_time = NOW(), remark = %s, update_date = NOW()
        WHERE id = %s
        """
        execute_query(update_query, (student_id, agent_id, remark, device_id))
        
        # 记录绑定历史
        history_query = """
        INSERT INTO ai_device_student_bind_history 
        (device_id, student_id, agent_id, action_type, operator_name, remark)
        VALUES (%s, %s, %s, 1, 'API', %s)
        """
        execute_query(history_query, (device_id, student_id, agent_id, remark))
        
        return jsonify({
            'success': True,
            'message': '设备绑定成功'
        })
    except Exception as e:
        logger.error(f"设备绑定失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/device/<device_id>/unbind', methods=['POST'])
def unbind_device(device_id):
    """解绑设备"""
    try:
        data = request.get_json()
        remark = data.get('remark', '通过API解绑')
        
        # 获取当前绑定信息
        current_query = "SELECT student_id, agent_id FROM ai_device WHERE id = %s"
        current_bind = execute_query(current_query, (device_id,), fetch_one=True)
        
        if not current_bind:
            return jsonify({
                'success': False,
                'error': '设备不存在'
            }), 404
        
        # 更新设备绑定信息
        update_query = """
        UPDATE ai_device 
        SET student_id = NULL, bind_status = 0, remark = %s, update_date = NOW()
        WHERE id = %s
        """
        execute_query(update_query, (remark, device_id))
        
        # 记录解绑历史
        if current_bind['student_id']:
            history_query = """
            INSERT INTO ai_device_student_bind_history 
            (device_id, old_student_id, agent_id, action_type, operator_name, remark)
            VALUES (%s, %s, %s, 0, 'API', %s)
            """
            execute_query(history_query, (device_id, current_bind['student_id'], current_bind['agent_id'], remark))
        
        return jsonify({
            'success': True,
            'message': '设备解绑成功'
        })
    except Exception as e:
        logger.error(f"设备解绑失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("真实数据API服务器启动中...")
    print("=" * 60)
    print("服务地址: http://localhost:8091")
    print("API文档:")
    print("  GET  /api/health                        - 健康检查")
    print("  GET  /api/devices                       - 获取设备列表")
    print("  GET  /api/agents                        - 获取智能体列表")
    print("  GET  /api/students                      - 获取学生列表")
    print("  POST /api/students                      - 创建新学生")
    print("  GET  /api/students/<id>                 - 获取单个学生信息")
    print("  PUT  /api/students/<id>                 - 更新学生信息")
    print("  DELETE /api/students/<id>               - 删除学生")
    print("  GET  /api/bind-relations                - 获取绑定关系")
    print("  GET  /api/chat-statistics               - 获取聊天统计")
    print("  GET  /api/chat-records/<device_id>      - 获取设备聊天记录")
    print("  GET  /api/export/chat-data/<device_id>  - 导出设备聊天数据")
    print("  GET  /api/summary                       - 获取数据摘要")
    print("  POST /api/device/<id>/bind              - 绑定设备")
    print("  POST /api/device/<id>/unbind            - 解绑设备")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8091, debug=True)