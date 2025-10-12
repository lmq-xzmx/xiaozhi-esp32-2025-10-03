#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的数据库连接服务
支持直接MySQL连接，解决容器网络连接问题
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import yaml

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.warning("mysql-connector-python 未安装，将使用Docker exec模式")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDatabaseService:
    """增强的数据库服务类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化增强数据库服务
        
        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml'
        )
        self.config = self._load_config()
        self.connection_pool = None
        self._setup_connection_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件 {self.config_path}: {e}")
            return {}
    
    def _setup_connection_config(self):
        """设置数据库连接配置"""
        db_config = self.config.get('database', {})
        
        # 优先使用容器IP配置
        self.db_config = {
            'host': '172.20.0.5',  # 使用容器IP
            'port': db_config.get('port', 3306),
            'user': db_config.get('user', 'root'),
            'password': db_config.get('password', '123456'),
            'database': db_config.get('database', 'xiaozhi_esp32_server'),
            'charset': db_config.get('charset', 'utf8mb4'),
            'autocommit': True,
            'connect_timeout': 30,
            'read_timeout': 30,
            'write_timeout': 30,
            'pool_name': 'xiaozhi_pool',
            'pool_size': 5,
            'pool_reset_session': True
        }
        
        # 备用配置（Docker exec模式）
        self.container_name = db_config.get('container_name', 'xiaozhi-esp32-server-db')
        
        logger.info(f"数据库配置: host={self.db_config['host']}, database={self.db_config['database']}")
    
    def _get_connection_pool(self):
        """获取连接池"""
        if not MYSQL_AVAILABLE:
            return None
            
        if self.connection_pool is None:
            try:
                self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**self.db_config)
                logger.info("MySQL连接池创建成功")
            except MySQLError as e:
                logger.error(f"创建MySQL连接池失败: {e}")
                return None
        return self.connection_pool
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict[str, Any]]]:
        """
        执行查询SQL
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表或None
        """
        # 尝试直接MySQL连接
        if MYSQL_AVAILABLE:
            try:
                pool = self._get_connection_pool()
                if pool:
                    connection = pool.get_connection()
                    cursor = connection.cursor(dictionary=True)
                    
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    result = cursor.fetchall()
                    cursor.close()
                    connection.close()
                    
                    logger.debug(f"MySQL查询成功，返回 {len(result)} 条记录")
                    return result
                    
            except MySQLError as e:
                logger.error(f"MySQL查询失败: {e}")
                # 降级到Docker exec模式
                return self._execute_query_docker(query, params)
        
        # 降级到Docker exec模式
        return self._execute_query_docker(query, params)
    
    def execute_update(self, query: str, params: tuple = None) -> bool:
        """
        执行更新SQL
        
        Args:
            query: SQL更新语句
            params: 更新参数
            
        Returns:
            是否执行成功
        """
        # 尝试直接MySQL连接
        if MYSQL_AVAILABLE:
            try:
                pool = self._get_connection_pool()
                if pool:
                    connection = pool.get_connection()
                    cursor = connection.cursor()
                    
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    connection.commit()
                    affected_rows = cursor.rowcount
                    cursor.close()
                    connection.close()
                    
                    logger.debug(f"MySQL更新成功，影响 {affected_rows} 行")
                    return True
                    
            except MySQLError as e:
                logger.error(f"MySQL更新失败: {e}")
                # 降级到Docker exec模式
                return self._execute_update_docker(query, params)
        
        # 降级到Docker exec模式
        return self._execute_update_docker(query, params)
    
    def _execute_query_docker(self, query: str, params: tuple = None) -> Optional[List[Dict[str, Any]]]:
        """Docker exec模式执行查询"""
        import subprocess
        
        try:
            # 构建完整的SQL命令
            if params:
                # 简单的参数替换（生产环境需要更安全的处理）
                formatted_query = query % params
            else:
                formatted_query = query
            
            cmd = [
                'docker', 'exec', '-i', self.container_name,
                'mysql', '-u', self.db_config['user'], 
                f'-p{self.db_config["password"]}', 
                self.db_config['database'],
                '-e', formatted_query
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                # 解析输出为字典列表
                lines = result.stdout.strip().split('\n')
                if len(lines) < 2:
                    return []
                
                headers = lines[0].split('\t')
                records = []
                for line in lines[1:]:
                    values = line.split('\t')
                    record = dict(zip(headers, values))
                    records.append(record)
                
                logger.debug(f"Docker查询成功，返回 {len(records)} 条记录")
                return records
            else:
                logger.error(f"Docker查询失败: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Docker查询超时")
            return None
        except Exception as e:
            logger.error(f"Docker查询异常: {e}")
            return None
    
    def _execute_update_docker(self, query: str, params: tuple = None) -> bool:
        """Docker exec模式执行更新"""
        import subprocess
        
        try:
            # 构建完整的SQL命令
            if params:
                formatted_query = query % params
            else:
                formatted_query = query
            
            cmd = [
                'docker', 'exec', '-i', self.container_name,
                'mysql', '-u', self.db_config['user'], 
                f'-p{self.db_config["password"]}', 
                self.db_config['database'],
                '-e', formatted_query
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                logger.debug("Docker更新成功")
                return True
            else:
                logger.error(f"Docker更新失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Docker更新超时")
            return False
        except Exception as e:
            logger.error(f"Docker更新异常: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        result = {
            'mysql_direct': False,
            'docker_exec': False,
            'error': None
        }
        
        # 测试直接MySQL连接
        if MYSQL_AVAILABLE:
            try:
                pool = self._get_connection_pool()
                if pool:
                    connection = pool.get_connection()
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                    connection.close()
                    result['mysql_direct'] = True
                    logger.info("MySQL直连测试成功")
            except Exception as e:
                result['error'] = f"MySQL直连失败: {str(e)}"
                logger.warning(f"MySQL直连测试失败: {e}")
        
        # 测试Docker exec连接
        try:
            test_result = self._execute_query_docker("SELECT 1 as test")
            if test_result:
                result['docker_exec'] = True
                logger.info("Docker exec测试成功")
        except Exception as e:
            if not result['error']:
                result['error'] = f"Docker exec失败: {str(e)}"
            logger.warning(f"Docker exec测试失败: {e}")
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        connection_test = self.test_connection()
        
        health_status = {
            'status': 'healthy' if (connection_test['mysql_direct'] or connection_test['docker_exec']) else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'connection_methods': {
                'mysql_direct': connection_test['mysql_direct'],
                'docker_exec': connection_test['docker_exec']
            },
            'config': {
                'host': self.db_config['host'],
                'database': self.db_config['database'],
                'container_name': self.container_name
            }
        }
        
        if connection_test['error']:
            health_status['error'] = connection_test['error']
        
        return health_status


# 全局实例
_enhanced_db_service = None

def get_enhanced_db_service() -> EnhancedDatabaseService:
    """获取增强数据库服务的全局实例"""
    global _enhanced_db_service
    if _enhanced_db_service is None:
        _enhanced_db_service = EnhancedDatabaseService()
    return _enhanced_db_service


if __name__ == "__main__":
    # 测试增强数据库服务
    service = EnhancedDatabaseService()
    
    # 健康检查
    health = service.health_check()
    print("增强数据库服务健康检查结果:")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # 测试查询
    try:
        records = service.execute_query("SELECT COUNT(*) as total FROM chat_records LIMIT 1")
        if records:
            print(f"\n聊天记录总数: {records[0]['total']}")
        else:
            print("\n无法获取聊天记录数量")
    except Exception as e:
        print(f"\n查询测试失败: {e}")