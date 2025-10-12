#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聊天记录服务
集成聊天记录的写入、查询和管理功能
"""

import os
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatHistoryService:
    """聊天记录服务类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化聊天记录服务
        
        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml'
        )
        self.config = self._load_config()
        self.container_name = self.config.get('database', {}).get('container_name', 'xiaozhi-esp32-server-db')
        self.database = self.config.get('database', {}).get('database', 'xiaozhi_esp32_server')
        self.password = self.config.get('database', {}).get('password', '123456')
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件 {self.config_path}: {e}")
            return {}
    
    def execute_sql(self, sql: str) -> Optional[str]:
        """
        执行SQL命令
        
        Args:
            sql: SQL命令
            
        Returns:
            执行结果或None
        """
        try:
            cmd = [
                'docker', 'exec', '-i', self.container_name,
                'mysql', '-u', 'root', f'-p{self.password}', self.database,
                '-e', sql
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"SQL执行失败: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("SQL执行超时")
            return None
        except Exception as e:
            logger.error(f"SQL执行异常: {e}")
            return None
    
    def write_chat_record(self, 
                         mac_address: str,
                         agent_id: str,
                         session_id: str,
                         chat_type: int,
                         content: str,
                         device_id: str,
                         student_id: Optional[int] = None,
                         audio_id: Optional[str] = None) -> bool:
        """
        写入聊天记录
        
        Args:
            mac_address: MAC地址
            agent_id: 代理ID
            session_id: 会话ID
            chat_type: 聊天类型 (1=AI回复, 2=用户输入)
            content: 聊天内容
            device_id: 设备ID
            student_id: 学生ID (可选)
            audio_id: 音频ID (可选)
            
        Returns:
            是否写入成功
        """
        try:
            # 构建SQL插入语句
            sql_parts = [
                "INSERT INTO ai_agent_chat_history",
                "(mac_address, agent_id, session_id, chat_type, content, device_id"
            ]
            
            values_parts = [
                f"('{mac_address}', '{agent_id}', '{session_id}', {chat_type}, '{content}', '{device_id}'"
            ]
            
            if student_id is not None:
                sql_parts.append(", student_id")
                values_parts.append(f", {student_id}")
                
            if audio_id is not None:
                sql_parts.append(", audio_id")
                values_parts.append(f", '{audio_id}'")
            
            sql_parts.append(") VALUES ")
            values_parts.append(")")
            
            sql = "".join(sql_parts) + "".join(values_parts) + ";"
            
            result = self.execute_sql(sql)
            
            if result is not None:
                logger.info(f"聊天记录写入成功: {content[:30]}...")
                return True
            else:
                logger.error(f"聊天记录写入失败: {content[:30]}...")
                return False
                
        except Exception as e:
            logger.error(f"写入聊天记录异常: {e}")
            return False
    
    def get_chat_records_count(self, device_id: Optional[str] = None) -> int:
        """
        获取聊天记录数量
        
        Args:
            device_id: 设备ID，如果为None则获取总数
            
        Returns:
            记录数量
        """
        try:
            if device_id:
                sql = f"SELECT COUNT(*) as total FROM ai_agent_chat_history WHERE device_id = '{device_id}';"
            else:
                sql = "SELECT COUNT(*) as total FROM ai_agent_chat_history;"
                
            result = self.execute_sql(sql)
            
            if result:
                lines = result.strip().split('\n')
                if len(lines) >= 2:
                    return int(lines[1])
            
            return 0
            
        except Exception as e:
            logger.error(f"获取聊天记录数量异常: {e}")
            return 0
    
    def get_recent_chat_records(self, device_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的聊天记录
        
        Args:
            device_id: 设备ID
            limit: 记录数量限制
            
        Returns:
            聊天记录列表
        """
        try:
            sql = f"SELECT id, mac_address, agent_id, session_id, chat_type, content, created_at, device_id, student_id FROM ai_agent_chat_history WHERE device_id = '{device_id}' ORDER BY created_at DESC LIMIT {limit};"
            
            result = self.execute_sql(sql)
            
            if not result:
                return []
            
            lines = result.strip().split('\n')
            if len(lines) < 2:
                return []
            
            # 解析表头
            headers = lines[0].split('\t')
            records = []
            
            # 解析数据行
            for line in lines[1:]:
                values = line.split('\t')
                if len(values) == len(headers):
                    record = dict(zip(headers, values))
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"获取聊天记录异常: {e}")
            return []
    
    def get_chat_history(self, device_id: str, since_time: Optional[datetime] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        获取聊天历史记录
        
        Args:
            device_id: 设备ID
            since_time: 起始时间，可选
            limit: 记录数量限制
            
        Returns:
            聊天记录列表
        """
        try:
            if since_time:
                sql = f"SELECT id, mac_address, agent_id, session_id, chat_type, content, created_at, device_id, student_id FROM ai_agent_chat_history WHERE device_id = '{device_id}' AND created_at >= '{since_time.isoformat()}' ORDER BY created_at DESC LIMIT {limit};"
            else:
                sql = f"SELECT id, mac_address, agent_id, session_id, chat_type, content, created_at, device_id, student_id FROM ai_agent_chat_history WHERE device_id = '{device_id}' ORDER BY created_at DESC LIMIT {limit};"
            
            result = self.execute_sql(sql)
            
            if not result:
                return []
            
            lines = result.strip().split('\n')
            if len(lines) < 2:
                return []
            
            # 解析表头
            headers = lines[0].split('\t')
            records = []
            
            # 解析数据行
            for line in lines[1:]:
                values = line.split('\t')
                if len(values) == len(headers):
                    record = dict(zip(headers, values))
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"获取聊天历史记录异常: {e}")
            return []
    
    def report_chat_async(self, 
                         mac_address: str,
                         agent_id: str,
                         session_id: str,
                         chat_type: int,
                         content: str,
                         device_id: str,
                         student_id: Optional[int] = None,
                         audio_data: Optional[bytes] = None) -> None:
        """
        异步报告聊天记录（模拟原项目的报告机制）
        
        Args:
            mac_address: MAC地址
            agent_id: 代理ID
            session_id: 会话ID
            chat_type: 聊天类型
            content: 聊天内容
            device_id: 设备ID
            student_id: 学生ID
            audio_data: 音频数据（暂不处理）
        """
        try:
            # 在实际应用中，这里可以放入队列进行异步处理
            # 现在直接同步写入
            success = self.write_chat_record(
                mac_address=mac_address,
                agent_id=agent_id,
                session_id=session_id,
                chat_type=chat_type,
                content=content,
                device_id=device_id,
                student_id=student_id
            )
            
            if success:
                logger.info(f"异步聊天记录报告成功: {content[:30]}...")
            else:
                logger.error(f"异步聊天记录报告失败: {content[:30]}...")
                
        except Exception as e:
            logger.error(f"异步聊天记录报告异常: {e}")
    
    async def record_user_input(self, 
                               device_id: str, 
                               student_id: str, 
                               user_input: str, 
                               timestamp: datetime) -> bool:
        """
        记录用户输入
        
        Args:
            device_id: 设备ID
            student_id: 学生ID
            user_input: 用户输入内容
            timestamp: 时间戳
            
        Returns:
            是否成功记录
        """
        try:
            session_id = f"{device_id}_{student_id}_{int(timestamp.timestamp())}"
            
            success = self.write_chat_record(
                mac_address=device_id,
                agent_id="default_agent",
                session_id=session_id,
                chat_type=1,  # 1表示用户输入
                content=user_input,
                device_id=device_id,
                student_id=int(student_id) if student_id.isdigit() else None
            )
            
            if success:
                logger.info(f"✅ 用户输入记录成功: {device_id} - {user_input[:50]}...")
            else:
                logger.error(f"❌ 用户输入记录失败: {device_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ 记录用户输入异常: {e}")
            return False
    
    async def record_ai_response(self, 
                               device_id: str, 
                               student_id: str, 
                               ai_response: str, 
                               timestamp: datetime) -> bool:
        """
        记录AI响应
        
        Args:
            device_id: 设备ID
            student_id: 学生ID
            ai_response: AI响应内容
            timestamp: 时间戳
            
        Returns:
            是否成功记录
        """
        try:
            session_id = f"{device_id}_{student_id}_{int(timestamp.timestamp())}"
            
            success = self.write_chat_record(
                mac_address=device_id,
                agent_id="default_agent",
                session_id=session_id,
                chat_type=2,  # 2表示AI响应
                content=ai_response,
                device_id=device_id,
                student_id=int(student_id) if student_id.isdigit() else None
            )
            
            if success:
                logger.info(f"✅ AI响应记录成功: {device_id} - {ai_response[:50]}...")
            else:
                logger.error(f"❌ AI响应记录失败: {device_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ 记录AI响应异常: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        try:
            # 检查数据库连接
            result = self.execute_sql("SELECT 1;")
            db_connected = result is not None
            
            # 获取总记录数
            total_records = self.get_chat_records_count()
            
            return {
                "status": "healthy" if db_connected else "unhealthy",
                "database_connected": db_connected,
                "total_chat_records": total_records,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# 全局服务实例
_chat_history_service = None


def get_chat_history_service() -> ChatHistoryService:
    """获取聊天记录服务实例（单例模式）"""
    global _chat_history_service
    if _chat_history_service is None:
        _chat_history_service = ChatHistoryService()
    return _chat_history_service


if __name__ == "__main__":
    # 测试代码
    service = ChatHistoryService()
    
    # 健康检查
    health = service.health_check()
    print("健康检查结果:")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # 获取记录数量
    count = service.get_chat_records_count()
    print(f"\n总聊天记录数: {count}")
    
    # 获取某个设备的最近记录
    if count > 0:
        records = service.get_recent_chat_records("58:8c:81:65:4c:8c", limit=3)
        print(f"\n设备 58:8c:81:65:4c:8c 的最近记录:")
        for record in records:
            print(f"- [{record.get('chat_type')}] {record.get('content', '')[:50]}...")