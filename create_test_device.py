#!/usr/bin/env python3
"""
创建测试设备记录脚本
用于解决聊天记录外键约束问题
"""

import mysql.connector
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': '172.20.0.5',  # MySQL容器内部IP
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4'
}

def create_test_device():
    """创建测试设备记录"""
    conn = None
    cursor = None
    try:
        # 连接数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查设备是否已存在
        check_sql = "SELECT id FROM ai_device WHERE mac_address = %s"
        cursor.execute(check_sql, ('test_device_001',))
        existing_device = cursor.fetchone()
        
        if existing_device:
            logger.info(f"✅ 测试设备已存在，ID: {existing_device[0]}")
            return existing_device[0]
        
        # 插入测试设备
        insert_sql = """
        INSERT INTO ai_device (
            mac_address, 
            alias, 
            agent_id, 
            student_id, 
            bind_status, 
            bind_time,
            last_connected_at,
            app_version,
            board,
            remark,
            create_date,
            update_date
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        current_time = datetime.now()
        values = (
            'test_device_001',           # mac_address
            '测试设备001',                # alias
            1,                          # agent_id (假设存在ID为1的智能体)
            1001,                       # student_id (假设存在ID为1001的学生)
            1,                          # bind_status (1=已绑定)
            current_time,               # bind_time
            current_time,               # last_connected_at
            'v1.0.0',                   # app_version
            'ESP32',                    # board
            '用于WebSocket聊天测试的设备',  # remark
            current_time,               # create_date
            current_time                # update_date
        )
        
        cursor.execute(insert_sql, values)
        device_id = cursor.lastrowid
        
        # 提交事务
        conn.commit()
        
        logger.info(f"✅ 测试设备创建成功，ID: {device_id}")
        return device_id
        
    except mysql.connector.Error as e:
        logger.error(f"❌ 数据库操作失败: {e}")
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        logger.error(f"❌ 创建测试设备失败: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def create_test_agent():
    """创建测试智能体记录"""
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查智能体是否已存在
        check_sql = "SELECT id FROM ai_agent WHERE id = %s"
        cursor.execute(check_sql, (1,))
        existing_agent = cursor.fetchone()
        
        if existing_agent:
            logger.info(f"✅ 测试智能体已存在，ID: {existing_agent[0]}")
            return existing_agent[0]
        
        # 插入测试智能体
        insert_sql = """
        INSERT INTO ai_agent (
            id,
            agent_name, 
            agent_code, 
            status,
            create_date,
            update_date
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        )
        """
        
        current_time = datetime.now()
        values = (
            1,                          # id
            '测试智能体',                 # agent_name
            'test_agent_001',           # agent_code
            1,                          # status (1=启用)
            current_time,               # create_date
            current_time                # update_date
        )
        
        cursor.execute(insert_sql, values)
        conn.commit()
        
        logger.info(f"✅ 测试智能体创建成功，ID: 1")
        return 1
        
    except mysql.connector.Error as e:
        logger.error(f"❌ 创建智能体失败: {e}")
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        logger.error(f"❌ 创建测试智能体失败: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    """主函数"""
    logger.info("🚀 开始创建测试数据...")
    
    # 创建测试智能体
    agent_id = create_test_agent()
    if not agent_id:
        logger.error("❌ 智能体创建失败，退出")
        return
    
    # 创建测试设备
    device_id = create_test_device()
    if not device_id:
        logger.error("❌ 设备创建失败，退出")
        return
    
    logger.info("✅ 测试数据创建完成！")
    logger.info(f"   - 智能体ID: {agent_id}")
    logger.info(f"   - 设备ID: {device_id}")
    logger.info("   - 设备MAC: test_device_001")
    logger.info("   - 现在可以进行聊天记录测试了")

if __name__ == "__main__":
    main()