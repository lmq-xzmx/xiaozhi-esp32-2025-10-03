#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据API服务器
连接MySQL数据库，提供一对多关系管理的RESTful API接口
"""

from flask import Flask, jsonify, request, Response
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
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
CORS(app)  # 允许跨域请求

# 设置响应头确保UTF-8编码
@app.after_request
def after_request(response):
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

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
    'init_command': "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci"
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

def create_json_response(data, status_code=200):
    """创建正确编码的JSON响应，确保中文字符正确显示"""
    json_str = json.dumps(data, ensure_ascii=False, default=json_serializer, indent=2)
    response = Response(
        json_str,
        status=status_code,
        mimetype='application/json; charset=utf-8'
    )
    return response

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

@app.route('/api/device/<device_id>/sessions', methods=['GET'])
def get_device_sessions(device_id):
    """获取指定设备的会话列表"""
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        offset = (page - 1) * per_page
        
        # 获取会话列表
        query = """
        SELECT 
            session_id,
            device_id,
            COUNT(*) as chat_count,
            MIN(created_at) as created_at,
            MAX(created_at) as last_message_at
        FROM ai_agent_chat_history
        WHERE device_id = %s
        GROUP BY session_id, device_id
        ORDER BY last_message_at DESC
        LIMIT %s OFFSET %s
        """
        sessions = execute_query(query, (device_id, per_page, offset))
        
        # 获取总会话数
        count_query = """
        SELECT COUNT(DISTINCT session_id) as total
        FROM ai_agent_chat_history
        WHERE device_id = %s
        """
        total_result = execute_query(count_query, (device_id,), fetch_one=True)
        total = total_result['total'] if total_result else 0
        
        # 处理数据格式
        for session in sessions:
            if session['created_at']:
                session['created_at'] = session['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if session['last_message_at']:
                session['last_message_at'] = session['last_message_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'data': {
                'list': sessions,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page,
                    'has_more': page * per_page < total
                }
            }
        })
    except Exception as e:
        logger.error(f"获取设备会话列表失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取设备会话列表失败: {str(e)}'
        }), 500

@app.route('/api/device/<device_id>/session/<session_id>/chat-history', methods=['GET'])
def get_session_chat_history(device_id, session_id):
    """获取指定设备和会话的聊天记录"""
    try:
        # 获取聊天记录
        query = """
        SELECT 
            c.id,
            c.device_id,
            c.session_id,
            c.student_id,
            c.agent_id,
            c.chat_type,
            c.content,
            c.audio_id,
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
        WHERE c.device_id = %s AND c.session_id = %s
        ORDER BY c.created_at ASC
        """
        chat_records = execute_query(query, (device_id, session_id))
        
        # 处理数据格式
        for record in chat_records:
            if record['created_at']:
                record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if record['updated_at']:
                record['updated_at'] = record['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 确保文本内容正确编码
            if record['content'] and isinstance(record['content'], str):
                original_content = record['content']
                
                # 使用改进的中文字符检测逻辑
                def has_valid_chinese_content(text):
                    """检查文本是否包含有效的中文字符"""
                    if not text:
                        return False
                    
                    # 检查是否包含中文字符
                    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
                    
                    if not chinese_chars:
                        return False
                    
                    # 计算中文字符比例
                    chinese_ratio = len(chinese_chars) / len(text)
                    
                    # 如果中文字符比例太低，可能是乱码
                    if chinese_ratio < 0.1:
                        return False
                    
                    # 检查是否有连续的有效中文词汇
                    # 简单检查：是否有连续的2个或以上中文字符
                    consecutive_chinese = 0
                    max_consecutive = 0
                    
                    for char in text:
                        if '\u4e00' <= char <= '\u9fff':
                            consecutive_chinese += 1
                            max_consecutive = max(max_consecutive, consecutive_chinese)
                        else:
                            consecutive_chinese = 0
                    
                    return max_consecutive >= 2
                
                # 检查是否已经包含有效中文字符，如果是则不需要修复
                has_valid_chinese_result = has_valid_chinese_content(original_content)
                print(f"has_valid_chinese_content 检查结果: {has_valid_chinese_result}")
                if has_valid_chinese_result:
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"内容已包含有效中文字符，无需修复: {original_short}")
                else:
                    # 超强编码修复算法 - 针对复杂乱码问题
                    fixed = False
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"原始内容: {original_short}")
                    
                    def has_valid_chinese(text):
                        """检查文本是否包含有效的中文字符，排除乱码"""
                        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                        garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï']
                        has_garbled = any(indicator in text for indicator in garbled_indicators)
                        # 只有包含中文且乱码指示符较少时才认为修复成功
                        return chinese_count > 0 and (not has_garbled or chinese_count > sum(1 for c in text if c in garbled_indicators))
                    
                    # 方法1: 改进的智能字节序列修复
                    try:
                        # 将字符串转换为字节数组，处理混合字符
                        byte_array = []
                        for char in original_content:
                            char_code = ord(char)
                            if char_code <= 255:
                                byte_array.append(char_code)
                            else:
                                # 对于超出单字节范围的字符，尝试映射到UTF-8字节序列
                                if char_code == 8482:  # ™ 符号
                                    byte_array.extend([0xe2, 0x84, 0xa2])  # UTF-8 编码的 ™
                                elif char_code == 8225:  # ‡ 符号  
                                    byte_array.extend([0xe2, 0x80, 0xa1])  # UTF-8 编码的 ‡
                                # 跳过其他无法处理的字符
                        
                        if byte_array:
                            fixed_content = bytes(byte_array).decode('utf-8', errors='ignore')
                            # 使用改进的中文检测逻辑
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法1修复成功: {fixed_short}")
                                fixed = True
                    except (UnicodeDecodeError, ValueError) as e:
                        print(f"✗ 方法1失败: {str(e)[:100]}")
                    
                    # 方法2: 多重编码修复 - 处理UTF-8被误解为Latin-1的情况
                    if not fixed:
                        try:
                            # 先过滤掉超出Latin-1范围的字符，只处理可编码的部分
                            filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
                            if filtered_content:
                                # 假设原始UTF-8字节被错误地解释为Latin-1字符
                                fixed_content = filtered_content.encode('latin-1').decode('utf-8')
                                if has_valid_chinese(fixed_content):
                                    record['content'] = fixed_content
                                    fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                    print(f"✓ 方法2修复成功: {fixed_short}")
                                    fixed = True
                                else:
                                    print(f"✗ 方法2修复后无有效中文: {fixed_content[:50]}")
                            else:
                                print("✗ 方法2失败: 没有可处理的Latin-1字符")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法2失败: {str(e)[:100]}")
                    
                    # 方法3: 处理Windows-1252编码问题
                    if not fixed:
                        try:
                            # 尝试Windows-1252到UTF-8的转换
                            filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
                            if filtered_content:
                                fixed_content = filtered_content.encode('windows-1252').decode('utf-8')
                                if has_valid_chinese(fixed_content):
                                    record['content'] = fixed_content
                                    fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                    print(f"✓ 方法3修复成功: {fixed_short}")
                                    fixed = True
                                else:
                                    print(f"✗ 方法3修复后无有效中文: {fixed_content[:50]}")
                            else:
                                print("✗ 方法3失败: 没有可处理的Windows-1252字符")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法3失败: {str(e)[:100]}")
                    
                    if not fixed:
                        print(f"✗ 所有修复方法都失败，保持原始内容: {original_short}")
            # 确保文本内容正确编码
            if record['content'] and isinstance(record['content'], str):
                original_content = record['content']
                print(f"原始内容: {original_content}")
                
                # 检查是否已经是正确的中文
                if any('\u4e00' <= char <= '\u9fff' for char in original_content):
                    print(f"内容已包含中文字符，无需修复: {original_content}")
                else:
                    # 尝试多种编码修复方法
                    fixed = False
                    
                    # 方法1: 尝试将乱码字符串当作UTF-8字节序列处理
                    try:
                        # 将字符串的每个字符转换为字节，然后解码为UTF-8
                        byte_data = bytes(ord(c) for c in original_content if ord(c) < 256)
                        fixed_content = byte_data.decode('utf-8')
                        # 检查修复后的内容是否包含中文字符
                        if any('\u4e00' <= char <= '\u9fff' for char in fixed_content):
                            record['content'] = fixed_content
                            print(f"方法1修复成功: {record['content']}")
                            fixed = True
                    except (UnicodeDecodeError, ValueError) as e:
                        print(f"方法1失败: {e}")
                    
                    # 方法2: 尝试latin1编码修复（仅对可编码字符）
                    if not fixed:
                        try:
                            # 检查是否所有字符都可以用latin1编码
                            if all(ord(c) < 256 for c in original_content):
                                fixed_content = original_content.encode('latin1').decode('utf-8')
                                if any('\u4e00' <= char <= '\u9fff' for char in fixed_content):
                                    record['content'] = fixed_content
                                    print(f"方法2修复成功: {record['content']}")
                                    fixed = True
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"方法2失败: {e}")
                    
                    if not fixed:
                        print(f"所有修复方法都失败，保持原样: {original_content}")
        
        return jsonify({
            'success': True,
            'data': chat_records
        })
    except Exception as e:
        logger.error(f"获取会话聊天记录失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取会话聊天记录失败: {str(e)}'
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
            
            # 确保文本内容正确编码
            if record['content'] and isinstance(record['content'], str):
                original_content = record['content']
                
                # 使用改进的中文字符检测逻辑
                def has_valid_chinese_content(text):
                    """检查文本是否包含有效的中文字符"""
                    if not text:
                        return False
                    
                    # 检查是否包含中文字符
                    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
                    
                    if not chinese_chars:
                        return False
                    
                    # 计算中文字符比例
                    chinese_ratio = len(chinese_chars) / len(text)
                    
                    # 如果中文字符比例太低，可能是乱码
                    if chinese_ratio < 0.1:
                        return False
                    
                    # 检查是否有连续的有效中文词汇
                    # 简单检查：是否有连续的2个或以上中文字符
                    consecutive_chinese = 0
                    max_consecutive = 0
                    
                    for char in text:
                        if '\u4e00' <= char <= '\u9fff':
                            consecutive_chinese += 1
                            max_consecutive = max(max_consecutive, consecutive_chinese)
                        else:
                            consecutive_chinese = 0
                    
                    return max_consecutive >= 2
                
                # 检查是否已经包含有效中文字符，如果是则不需要修复
                has_valid_chinese_result = has_valid_chinese_content(original_content)
                print(f"has_valid_chinese_content 检查结果: {has_valid_chinese_result}")
                if has_valid_chinese_result:
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"内容已包含有效中文字符，无需修复: {original_short}")
                else:
                    # 超强编码修复算法 - 针对复杂乱码问题
                    fixed = False
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"原始内容: {original_short}")
                    
                    def has_valid_chinese(text):
                        """检查文本是否包含有效的中文字符，排除乱码"""
                        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                        garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï']
                        has_garbled = any(indicator in text for indicator in garbled_indicators)
                        # 只有包含中文且乱码指示符较少时才认为修复成功
                        return chinese_count > 0 and (not has_garbled or chinese_count > sum(1 for c in text if c in garbled_indicators))
                    
                    # 方法1: 智能字节序列修复 - 跳过超出范围字符，使用错误忽略策略
                    try:
                        print("尝试方法1: 智能字节序列修复")
                        # 将字符串转换为字节数组，跳过超出Latin-1范围的字符
                        byte_array = []
                        skipped_count = 0
                        
                        for char in original_content:
                            char_code = ord(char)
                            if char_code <= 255:  # Latin-1范围内的字符
                                byte_array.append(char_code)
                            else:
                                skipped_count += 1
                        
                        print(f"跳过了 {skipped_count} 个超出范围的字符")
                        
                        if byte_array:
                            try:
                                # 首先尝试正常UTF-8解码
                                fixed_content = bytes(byte_array).decode('utf-8')
                                print(f"方法1修复结果: {fixed_content[:50]}...")
                            except UnicodeDecodeError:
                                # 如果失败，使用错误忽略策略
                                print("正常解码失败，使用错误忽略策略")
                                fixed_content = bytes(byte_array).decode('utf-8', errors='ignore')
                                print(f"方法1忽略错误修复结果: {fixed_content[:50]}...")
                            
                            # 使用改进的中文检测逻辑
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法1修复成功: {fixed_short}")
                                fixed = True
                            else:
                                print(f"✗ 方法1修复后无有效中文")
                        else:
                            print("✗ 方法1失败: 无有效字节数组")
                    except Exception as e:
                        print(f"✗ 方法1失败: {str(e)[:100]}")
                    
                    # 方法2: 多重编码修复 - 处理UTF-8被误解为Latin-1的情况
                    if not fixed:
                        try:
                            # 先过滤掉超出Latin-1范围的字符，只处理可编码的部分
                            filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
                            if filtered_content:
                                # 假设原始UTF-8字节被错误地解释为Latin-1字符
                                fixed_content = filtered_content.encode('latin-1').decode('utf-8')
                                if has_valid_chinese(fixed_content):
                                    record['content'] = fixed_content
                                    fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                    print(f"✓ 方法2修复成功: {fixed_short}")
                                    fixed = True
                                else:
                                    print(f"✗ 方法2修复后无有效中文: {fixed_content[:50]}")
                            else:
                                print("✗ 方法2失败: 没有可处理的Latin-1字符")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法2失败: {str(e)[:100]}")
                    
                    # 方法3: 处理Windows-1252编码问题
                    if not fixed:
                        try:
                            # 尝试Windows-1252到UTF-8的转换
                            filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
                            if filtered_content:
                                fixed_content = filtered_content.encode('windows-1252').decode('utf-8')
                                if has_valid_chinese(fixed_content):
                                    record['content'] = fixed_content
                                    fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                    print(f"✓ 方法3修复成功: {fixed_short}")
                                    fixed = True
                                else:
                                    print(f"✗ 方法3修复后无有效中文: {fixed_content[:50]}")
                            else:
                                print("✗ 方法3失败: 没有可处理的Windows-1252字符")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法3失败: {str(e)[:100]}")
                    
                    if not fixed:
                        print(f"✗ 所有修复方法都失败，保持原始内容: {original_short}")
        
        return create_json_response({
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
        return create_json_response({
            'success': False,
            'message': f'获取设备聊天记录失败: {str(e)}'
        }, 500)

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
            # 确保文本内容正确编码
            if record['content'] and isinstance(record['content'], str):
                original_content = record['content']
                
                # 使用改进的中文字符检测逻辑
                def has_valid_chinese_content(text):
                    """检查文本是否包含有效的中文字符"""
                    if not text:
                        return False
                    
                    # 检查是否包含中文字符
                    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
                    
                    if not chinese_chars:
                        return False
                    
                    # 计算中文字符比例
                    chinese_ratio = len(chinese_chars) / len(text)
                    
                    # 如果中文字符比例太低，可能是乱码
                    if chinese_ratio < 0.1:
                        return False
                    
                    # 检查是否有连续的有效中文词汇
                    # 简单检查：是否有连续的2个或以上中文字符
                    consecutive_chinese = 0
                    max_consecutive = 0
                    
                    for char in text:
                        if '\u4e00' <= char <= '\u9fff':
                            consecutive_chinese += 1
                            max_consecutive = max(max_consecutive, consecutive_chinese)
                        else:
                            consecutive_chinese = 0
                    
                    return max_consecutive >= 2
                
                # 检查是否已经包含有效中文字符，如果是则不需要修复
                if has_valid_chinese_content(original_content):
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"内容已包含有效中文字符，无需修复: {original_short}")
                else:
                    # 超强编码修复算法 - 针对复杂乱码问题
                    fixed = False
                    original_short = original_content[:50] + "..." if len(original_content) > 50 else original_content
                    print(f"原始内容: {original_short}")
                    
                    def has_valid_chinese(text):
                        """检查文本是否包含有效的中文字符，排除乱码"""
                        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                        garbled_indicators = ['ä', 'ï', 'â', 'Â', 'Ã', 'Ë', 'Ï']
                        has_garbled = any(indicator in text for indicator in garbled_indicators)
                        # 只有包含中文且乱码指示符较少时才认为修复成功
                        return chinese_count > 0 and (not has_garbled or chinese_count > sum(1 for c in text if c in garbled_indicators))
                    
                    # 方法1: 智能字节序列修复
                    try:
                        # 将字符串转换为字节数组，只处理ASCII范围内的字符
                        byte_array = []
                        valid_conversion = True
                        for char in original_content:
                            char_code = ord(char)
                            if char_code <= 255:  # 只处理单字节字符
                                byte_array.append(char_code)
                            else:
                                valid_conversion = False
                                break
                        
                        if valid_conversion and byte_array:
                            fixed_content = bytes(byte_array).decode('utf-8')
                            # 使用改进的中文检测逻辑
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法1修复成功: {fixed_short}")
                                fixed = True
                    except (UnicodeDecodeError, ValueError) as e:
                        print(f"✗ 方法1失败: {str(e)[:100]}")
                    
                    # 方法2: 多重编码修复 - 处理UTF-8被误解为Latin-1的情况
                    if not fixed:
                        try:
                            # 先过滤掉超出Latin-1范围的字符，只处理可编码的部分
                            filtered_content = ''.join(char for char in original_content if ord(char) <= 255)
                            if filtered_content:
                                # 假设原始UTF-8字节被错误地解释为Latin-1字符
                                fixed_content = filtered_content.encode('latin-1').decode('utf-8')
                                if has_valid_chinese(fixed_content):
                                    record['content'] = fixed_content
                                    fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                    print(f"✓ 方法2修复成功: {fixed_short}")
                                    fixed = True
                                else:
                                    print(f"✗ 方法2修复后无有效中文: {fixed_content[:50]}")
                            else:
                                print("✗ 方法2失败: 没有可处理的Latin-1字符")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法2失败: {str(e)[:100]}")
                    
                    # 方法3: 处理Windows-1252编码问题
                    if not fixed:
                        try:
                            # 尝试Windows-1252到UTF-8的转换
                            fixed_content = original_content.encode('windows-1252').decode('utf-8')
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法3修复成功: {fixed_short}")
                                fixed = True
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法3失败: {str(e)[:100]}")
                    
                    # 方法4: 处理特殊字符映射
                    if not fixed:
                        try:
                            # 创建特殊字符映射表，处理常见的乱码字符
                            char_map = {
                                'â€™': "'",  # 右单引号
                                'â€œ': '"',  # 左双引号  
                                'â€': '"',   # 右双引号
                                'â€"': '—',  # 长破折号
                                'â€"': '–',  # 短破折号
                                'Â': '',     # 删除多余的Â
                                'ï¼Œ': '，', # 中文逗号
                                'ï¼': '！',  # 中文感叹号
                                'ï¼Ÿ': '？', # 中文问号
                            }
                            
                            fixed_content = original_content
                            for old_char, new_char in char_map.items():
                                fixed_content = fixed_content.replace(old_char, new_char)
                            
                            # 再次尝试Latin-1到UTF-8转换
                            try:
                                fixed_content = fixed_content.encode('latin-1').decode('utf-8')
                            except:
                                pass
                            
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法4修复成功: {fixed_short}")
                                fixed = True
                        except Exception as e:
                            print(f"✗ 方法4失败: {str(e)[:100]}")
                    
                    # 方法5: 尝试CP1252编码
                    if not fixed:
                        try:
                            fixed_content = original_content.encode('cp1252').decode('utf-8')
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法5修复成功: {fixed_short}")
                                fixed = True
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            print(f"✗ 方法5失败: {str(e)[:100]}")
                    
                    # 方法6: 暴力修复 - 逐字符尝试
                    if not fixed:
                        try:
                            # 尝试将每个字符的Unicode码点当作字节值处理
                            result_chars = []
                            temp_bytes = []
                            
                            for char in original_content:
                                char_code = ord(char)
                                if char_code <= 255:
                                    temp_bytes.append(char_code)
                                else:
                                    # 如果遇到非ASCII字符，先处理之前累积的字节
                                    if temp_bytes:
                                        try:
                                            decoded = bytes(temp_bytes).decode('utf-8')
                                            result_chars.append(decoded)
                                            temp_bytes = []
                                        except:
                                            # 如果解码失败，保持原字符
                                            result_chars.extend([chr(b) for b in temp_bytes])
                                            temp_bytes = []
                                    result_chars.append(char)
                            
                            # 处理剩余的字节
                            if temp_bytes:
                                try:
                                    decoded = bytes(temp_bytes).decode('utf-8')
                                    result_chars.append(decoded)
                                except:
                                    result_chars.extend([chr(b) for b in temp_bytes])
                            
                            fixed_content = ''.join(result_chars)
                            if has_valid_chinese(fixed_content):
                                record['content'] = fixed_content
                                fixed_short = fixed_content[:50] + "..." if len(fixed_content) > 50 else fixed_content
                                print(f"✓ 方法6修复成功: {fixed_short}")
                                fixed = True
                        except Exception as e:
                            print(f"✗ 方法6失败: {str(e)[:100]}")
                    
                    if not fixed:
                        print(f"✗ 所有修复方法都失败，保持原样: {original_short}")
                    print("---")
        
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

@app.route('/api/test/create-sample-data', methods=['POST'])
def create_sample_data():
    """创建测试数据"""
    try:
        # 获取第一个设备ID
        device_query = "SELECT id FROM ai_device LIMIT 1"
        device = execute_query(device_query, fetch_one=True)
        
        if not device:
            return jsonify({
                'success': False,
                'message': '没有可用的设备'
            }), 400
        
        device_id = device['id']
        
        # 创建测试聊天记录
        test_messages = [
            {
                'session_id': 'test_session_001',
                'chat_type': '1',  # 用户消息
                'content': '你好，我想了解一下今天的天气情况。',
                'audio_id': None
            },
            {
                'session_id': 'test_session_001',
                'chat_type': '2',  # AI回复
                'content': '您好！今天天气晴朗，温度适宜，是个不错的天气。建议您外出时注意防晒。',
                'audio_id': 'audio_001'
            },
            {
                'session_id': 'test_session_001',
                'chat_type': '1',  # 用户消息
                'content': '谢谢你的建议，那明天呢？',
                'audio_id': None
            },
            {
                'session_id': 'test_session_001',
                'chat_type': '2',  # AI回复
                'content': '明天预计会有小雨，建议您出门时携带雨具。温度会比今天稍低一些。',
                'audio_id': 'audio_002'
            },
            {
                'session_id': 'test_session_002',
                'chat_type': '1',  # 用户消息
                'content': '能帮我推荐一些学习资料吗？',
                'audio_id': None
            },
            {
                'session_id': 'test_session_002',
                'chat_type': '2',  # AI回复
                'content': '当然可以！根据您的需求，我推荐以下学习资料：1. 在线课程平台 2. 专业书籍 3. 实践项目。您希望了解哪个方面的详细信息？',
                'audio_id': 'audio_003'
            }
        ]
        
        # 插入测试数据
        for msg in test_messages:
            insert_query = """
            INSERT INTO ai_agent_chat_history 
            (device_id, session_id, student_id, agent_id, chat_type, content, audio_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """
            execute_query(insert_query, (
                device_id,
                msg['session_id'],
                None,  # 学生ID设为NULL
                None,  # 智能体ID设为NULL
                msg['chat_type'],
                msg['content'],
                msg['audio_id']
            ))
        
        return jsonify({
            'success': True,
            'message': f'成功创建 {len(test_messages)} 条测试数据',
            'device_id': device_id
        })
    except Exception as e:
        logger.error(f"创建测试数据失败: {e}")
        return jsonify({
            'success': False,
            'message': f'创建测试数据失败: {str(e)}'
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