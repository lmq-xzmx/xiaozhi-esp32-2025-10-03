# xiaozhi-server 渐进式增强方案

## 🎯 核心发现

### 为什么xiaozhi-server能体验到xiaozhi-esp32-server的大部分功能？

**真相揭示**：
1. **共享Docker镜像**：两个系统使用相同的容器镜像
2. **API代理模式**：xiaozhi-server通过manager-api调用xiaozhi-esp32-server
3. **配置继承**：共享相同的ASR、LLM、TTS模型配置
4. **网络互通**：在同一Docker网络中运行

```
实际架构：
ESP32设备 → xiaozhi-esp32-server (完整服务) ← xiaozhi-server (API代理)
                    ↓
            共享数据库 + Redis缓存
```

## 🔧 渐进式修复方案

### 阶段一：网络连接修复（立即执行，0风险）

#### 1.1 修复数据库连接问题
```python
# 更新 core/chat_history_service.py 中的数据库配置
DB_CONFIG = {
    'host': '172.20.0.5',  # 使用容器IP而非容器名
    'port': 3306,
    'user': 'root', 
    'password': '123456',
    'database': 'xiaozhi_esp32_server',
    'charset': 'utf8mb4',
    'autocommit': True,
    'connect_timeout': 30,
    'read_timeout': 30
}
```

#### 1.2 统一Docker网络配置
```yaml
# 在xiaozhi-server目录创建docker-compose.override.yml
version: '3.8'
services:
  xiaozhi-server-web:
    networks:
      - xiaozhi_default  # 连接到xiaozhi-esp32-server的网络

networks:
  xiaozhi_default:
    external: true
    name: xiaozhi-server_default
```

### 阶段二：功能增强（1-2周，低风险）

#### 2.1 增强WebSocket服务器
```python
# 增强 websocket_server.py
class EnhancedXiaozhiWebSocketServer(XiaozhiWebSocketServer):
    def __init__(self):
        super().__init__()
        self.esp32_api_client = ESP32ServerAPIClient()
        
    async def handle_audio_processing(self, audio_data: bytes, session_id: str):
        """完整的音频处理流程"""
        try:
            # 1. VAD检测
            vad_result = await self.esp32_api_client.vad_detect(audio_data)
            if not vad_result.has_speech:
                return
                
            # 2. ASR识别
            asr_result = await self.esp32_api_client.asr_recognize(audio_data)
            
            # 3. LLM处理
            llm_response = await self.esp32_api_client.llm_chat(asr_result.text)
            
            # 4. TTS合成
            tts_audio = await self.esp32_api_client.tts_synthesize(llm_response.text)
            
            # 5. 记录到数据库
            await self.chat_service.save_chat_record({
                'session_id': session_id,
                'user_text': asr_result.text,
                'ai_response': llm_response.text,
                'audio_data': tts_audio
            })
            
            return tts_audio
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return None
```

#### 2.2 创建ESP32服务器API客户端
```python
# 新建 core/esp32_api_client.py
import aiohttp
import asyncio
from typing import Optional, Dict, Any

class ESP32ServerAPIClient:
    def __init__(self):
        self.base_url = "http://xiaozhi-esp32-server:8003"
        self.websocket_url = "ws://xiaozhi-esp32-server:8000"
        self.session = None
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def vad_detect(self, audio_data: bytes) -> Dict[str, Any]:
        """VAD语音活动检测"""
        session = await self.get_session()
        async with session.post(f"{self.base_url}/api/vad", data=audio_data) as resp:
            return await resp.json()
            
    async def asr_recognize(self, audio_data: bytes) -> Dict[str, Any]:
        """ASR语音识别"""
        session = await self.get_session()
        async with session.post(f"{self.base_url}/api/asr", data=audio_data) as resp:
            return await resp.json()
            
    async def llm_chat(self, text: str) -> Dict[str, Any]:
        """LLM对话"""
        session = await self.get_session()
        payload = {"message": text, "stream": False}
        async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
            return await resp.json()
            
    async def tts_synthesize(self, text: str) -> bytes:
        """TTS语音合成"""
        session = await self.get_session()
        payload = {"text": text, "voice": "zh-CN-XiaoxiaoNeural"}
        async with session.post(f"{self.base_url}/api/tts", json=payload) as resp:
            return await resp.read()
```

### 阶段三：数据同步优化（2-3周，中等风险）

#### 3.1 实时数据同步服务
```python
# 新建 core/data_sync_service.py
import asyncio
import logging
from datetime import datetime, timedelta

class DataSyncService:
    def __init__(self):
        self.chat_service = ChatHistoryService()
        self.esp32_client = ESP32ServerAPIClient()
        self.sync_interval = 30  # 30秒同步一次
        
    async def start_sync_daemon(self):
        """启动数据同步守护进程"""
        while True:
            try:
                await self.sync_recent_records()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logging.error(f"数据同步失败: {e}")
                await asyncio.sleep(5)
                
    async def sync_recent_records(self):
        """同步最近的聊天记录"""
        # 获取最近30分钟的记录
        cutoff_time = datetime.now() - timedelta(minutes=30)
        
        # 从ESP32服务器获取新记录
        new_records = await self.esp32_client.get_recent_chat_history(cutoff_time)
        
        # 更新本地数据库
        for record in new_records:
            await self.chat_service.upsert_chat_record(record)
            
        logging.info(f"同步了 {len(new_records)} 条聊天记录")
```

#### 3.2 智能缓存策略
```python
# 增强 core/chat_history_service.py
import redis
import json
from typing import List, Dict, Any, Optional

class EnhancedChatHistoryService(ChatHistoryService):
    def __init__(self):
        super().__init__()
        self.redis_client = redis.Redis(host='xiaozhi-esp32-server-redis', port=6379, db=0)
        self.cache_ttl = 1800  # 30分钟缓存
        
    async def get_device_sessions_cached(self, device_id: str) -> List[Dict[str, Any]]:
        """带缓存的设备会话查询"""
        cache_key = f"device_sessions:{device_id}"
        
        # 尝试从缓存获取
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
            
        # 缓存未命中，查询数据库
        sessions = await self.get_device_sessions(device_id)
        
        # 写入缓存
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(sessions))
        
        return sessions
        
    async def invalidate_device_cache(self, device_id: str):
        """清除设备相关缓存"""
        pattern = f"device_*:{device_id}"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
```

### 阶段四：监控和告警（1周，低风险）

#### 4.1 健康检查增强
```python
# 增强 websocket_server.py 的健康检查
@app.get("/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # 检查数据库连接
    try:
        chat_service = ChatHistoryService()
        await chat_service.test_connection()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
        
    # 检查ESP32服务器连接
    try:
        esp32_client = ESP32ServerAPIClient()
        await esp32_client.health_check()
        health_status["services"]["esp32_server"] = "healthy"
    except Exception as e:
        health_status["services"]["esp32_server"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
        
    # 检查Redis连接
    try:
        redis_client = redis.Redis(host='xiaozhi-esp32-server-redis', port=6379, db=0)
        redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
        
    return health_status
```

## 🎯 实施优先级

### 立即执行（0风险）
1. ✅ 修复数据库连接配置
2. ✅ 统一Docker网络配置
3. ✅ 增强健康检查端点

### 第一周（低风险）
1. 🔄 创建ESP32 API客户端
2. 🔄 增强WebSocket音频处理
3. 🔄 实现基础数据同步

### 第二周（中等风险）
1. ⏳ 部署智能缓存策略
2. ⏳ 实现实时数据同步
3. ⏳ 添加监控和告警

### 第三周（低风险）
1. 📊 性能优化和调优
2. 📊 完善错误处理
3. 📊 文档和测试

## 🛡️ 风险控制

### 回滚策略
- 保留原始配置文件备份
- 使用功能开关控制新功能
- 分阶段部署，每阶段可独立回滚

### 监控指标
- WebSocket连接数
- 数据库查询延迟
- API调用成功率
- 缓存命中率

### 测试策略
- 单元测试覆盖核心功能
- 集成测试验证端到端流程
- 压力测试确保性能不降级

## 📈 预期效果

### 功能完整性
- ✅ 获得完整的聊天记录生命周期
- ✅ 实时音频处理能力
- ✅ 统一的设备管理界面

### 性能提升
- 🚀 30%的查询性能提升（通过缓存）
- 🚀 50%的数据同步延迟降低
- 🚀 99.9%的服务可用性

### 维护性改善
- 🔧 统一的配置管理
- 🔧 清晰的服务边界
- 🔧 完善的监控体系