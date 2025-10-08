from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# 数据库配置
DB_CONFIG = {
    'host': 'xiaozhi-esp32-server-db',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

@app.route('/xiaozhi/device-student-bind/list', methods=['GET'])
def get_device_list():
    """获取设备绑定列表"""
    try:
        page = int(request.args.get('page', 1))
        size = int(request.args.get('size', 10))
        agent_id = request.args.get('agentId', '')
        device_name = request.args.get('deviceName', '')
        student_name = request.args.get('studentName', '')
        bind_status = request.args.get('bindStatus', '')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'code': 500, 'message': '数据库连接失败'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 构建查询条件
        where_conditions = []
        params = []
        
        base_sql = """
            SELECT d.id as device_id, d.device_name, d.device_code, d.bind_status,
                   d.bind_time, d.remark, a.agent_name, a.id as agent_id,
                   u.username as student_name, u.real_name as student_real_name,
                   u.id as student_id
            FROM ai_device d
            LEFT JOIN ai_agent a ON d.agent_id = a.id
            LEFT JOIN sys_user u ON d.student_id = u.id
        """
        
        if agent_id:
            where_conditions.append("a.id = %s")
            params.append(agent_id)
        if device_name:
            where_conditions.append("d.device_name LIKE %s")
            params.append(f"%{device_name}%")
        if student_name:
            where_conditions.append("u.username LIKE %s")
            params.append(f"%{student_name}%")
        if bind_status:
            where_conditions.append("d.bind_status = %s")
            params.append(bind_status)
            
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # 查询总数
        count_sql = f"SELECT COUNT(*) as total FROM ({base_sql} {where_clause}) as t"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['total']
        
        # 查询数据
        offset = (page - 1) * size
        data_sql = f"{base_sql} {where_clause} ORDER BY d.create_time DESC LIMIT %s OFFSET %s"
        params.extend([size, offset])
        cursor.execute(data_sql, params)
        records = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'code': 200,
            'message': '查询成功',
            'data': {
                'records': records,
                'total': total,
                'size': size,
                'current': page,
                'pages': (total + size - 1) // size
            }
        })
        
    except Exception as e:
        print(f"查询设备列表失败: {e}")
        return jsonify({'code': 500, 'message': f'查询失败: {str(e)}'}), 500

@app.route('/xiaozhi/device-student-bind/bind', methods=['POST'])
def bind_student():
    """绑定学员到设备"""
    try:
        data = request.get_json()
        device_id = data.get('deviceId')
        student_id = data.get('studentId')
        remark = data.get('remark', '')
        
        if not device_id or not student_id:
            return jsonify({'code': 400, 'message': '设备ID和学员ID不能为空'}), 400
            
        conn = get_db_connection()
        if not conn:
            return jsonify({'code': 500, 'message': '数据库连接失败'}), 500
            
        cursor = conn.cursor()
        
        # 更新设备绑定信息
        cursor.execute("""
            UPDATE ai_device 
            SET student_id = %s, bind_status = 'bound', bind_time = NOW(), remark = %s
            WHERE id = %s
        """, (student_id, remark, device_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'code': 200,
            'message': '绑定成功'
        })
        
    except Exception as e:
        print(f"绑定学员失败: {e}")
        return jsonify({'code': 500, 'message': f'绑定失败: {str(e)}'}), 500

@app.route('/xiaozhi/device-student-bind/unbind', methods=['POST'])
def unbind_student():
    """解绑学员"""
    try:
        data = request.get_json()
        device_id = data.get('deviceId')
        remark = data.get('remark', '')
        
        if not device_id:
            return jsonify({'code': 400, 'message': '设备ID不能为空'}), 400
            
        conn = get_db_connection()
        if not conn:
            return jsonify({'code': 500, 'message': '数据库连接失败'}), 500
            
        cursor = conn.cursor()
        
        # 更新设备绑定状态
        cursor.execute("""
            UPDATE ai_device 
            SET student_id = NULL, bind_status = 'unbound', bind_time = NULL, remark = %s
            WHERE id = %s
        """, (remark, device_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'code': 200,
            'message': '解绑成功'
        })
        
    except Exception as e:
        print(f"解绑学员失败: {e}")
        return jsonify({'code': 500, 'message': f'解绑失败: {str(e)}'}), 500

@app.route('/xiaozhi/device-student-bind/students/search', methods=['GET'])
def search_students():
    """搜索可用学员"""
    try:
        keyword = request.args.get('keyword', '')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'code': 500, 'message': '数据库连接失败'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 搜索学员
        sql = "SELECT id, username, real_name, email, phone FROM sys_user WHERE user_type = 'student'"
        params = []
        
        if keyword:
            sql += " AND (username LIKE %s OR real_name LIKE %s OR email LIKE %s)"
            params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            
        sql += " ORDER BY create_time DESC LIMIT 20"
        
        cursor.execute(sql, params)
        students = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'code': 200,
            'message': '查询成功',
            'data': students
        })
        
    except Exception as e:
        print(f"搜索学员失败: {e}")
        return jsonify({'code': 500, 'message': f'搜索失败: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'code': 200,
        'message': '服务正常',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True)
