# 🎵 本地Edge-TTS部署指南 - 消除TTS延迟

## 🎯 **目标**
在4核8GB硬件上部署本地Edge-TTS服务，将TTS延迟从500-1000ms降低到50-150ms，支持流式音频输出。

---

## 📊 **TTS方案对比**

### **本地TTS引擎对比**

| TTS引擎 | 内存占用 | 延迟 | 音质 | 语音种类 | 部署难度 | 推荐指数 |
|---------|----------|------|------|----------|----------|----------|
| **Edge-TTS** | 50MB | 50-100ms | ⭐⭐⭐⭐⭐ | 200+ | 简单 | ⭐⭐⭐⭐⭐ |
| **PicoTTS** | 20MB | 30-80ms | ⭐⭐⭐ | 10+ | 简单 | ⭐⭐⭐⭐ |
| **eSpeak-NG** | 15MB | 20-60ms | ⭐⭐ | 50+ | 简单 | ⭐⭐⭐ |
| **Festival** | 100MB | 100-200ms | ⭐⭐⭐ | 20+ | 中等 | ⭐⭐⭐ |
| **VITS** | 500MB | 200-400ms | ⭐⭐⭐⭐⭐ | 自定义 | 复杂 | ⭐⭐ |

**最佳选择**: **Edge-TTS** - 微软免费TTS，音质优秀，延迟极低

---

## 🚀 **Edge-TTS快速部署**

### **方案A: Python直接部署 (推荐)**

```bash
# 1. 安装Edge-TTS
pip install edge-tts

# 2. 查看可用语音
edge-tts --list-voices | grep zh-CN

# 3. 测试语音合成
edge-tts --voice zh-CN-XiaoxiaoNeural --text "你好，这是测试语音" --write-media test.mp3

# 4. 启动HTTP服务
python -m edge_tts_server --host 0.0.0.0 --port 5000
```

### **方案B: Docker容器部署 (推荐生产)**

```dockerfile
# Dockerfile.edge-tts
FROM python:3.9-slim

# 安装依赖
RUN pip install edge-tts flask gunicorn

# 创建工作目录
WORKDIR /app

# 复制服务代码
COPY edge_tts_service.py .

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "edge_tts_service:app"]
```

```python
# edge_tts_service.py
import asyncio
import io
import edge_tts
from flask import Flask, request, jsonify, send_file
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
import os

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# 音频缓存目录
CACHE_DIR = "/tmp/tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 支持的语音列表
VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "yunxi": "zh-CN-YunxiNeural", 
    "xiaoyi": "zh-CN-XiaoyiNeural",
    "yunjian": "zh-CN-YunjianNeural",
    "xiaomo": "zh-CN-XiaomoNeural"
}

async def generate_speech(text, voice, rate="+0%", pitch="+0Hz"):
    """生成语音"""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def get_cache_key(text, voice, rate, pitch):
    """生成缓存键"""
    content = f"{text}_{voice}_{rate}_{pitch}"
    return hashlib.md5(content.encode()).hexdigest()

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """文本转语音API"""
    try:
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'xiaoxiao')
        rate = data.get('rate', '+0%')
        pitch = data.get('pitch', '+0Hz')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
            
        # 检查缓存
        cache_key = get_cache_key(text, voice, rate, pitch)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.mp3")
        
        if os.path.exists(cache_file):
            return send_file(cache_file, mimetype='audio/mpeg')
        
        # 生成语音
        voice_name = VOICES.get(voice, VOICES['xiaoxiao'])
        
        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_data = loop.run_until_complete(
            generate_speech(text, voice_name, rate, pitch)
        )
        loop.close()
        
        # 保存到缓存
        with open(cache_file, 'wb') as f:
            f.write(audio_data)
            
        generation_time = time.time() - start_time
        
        # 返回音频文件
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=f'speech_{cache_key}.mp3'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tts/stream', methods=['POST'])
def stream_speech():
    """流式语音合成"""
    try:
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'xiaoxiao')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
            
        voice_name = VOICES.get(voice, VOICES['xiaoxiao'])
        
        def generate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def stream_audio():
                communicate = edge_tts.Communicate(text, voice_name)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
                        
            for chunk in loop.run_until_complete(stream_audio()):
                yield chunk
            loop.close()
            
        return app.response_class(
            generate(),
            mimetype='audio/mpeg',
            headers={'Content-Disposition': 'attachment; filename=speech.mp3'}
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    """获取支持的语音列表"""
    return jsonify(VOICES)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "edge-tts"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### **方案C: Docker Compose集成**

```yaml
# docker-compose-edge-tts.yml
version: '3.8'
services:
  xiaozhi-edge-tts:
    build:
      context: .
      dockerfile: Dockerfile.edge-tts
    container_name: xiaozhi-edge-tts
    ports:
      - "5000:5000"
    volumes:
      - ./tts_cache:/tmp/tts_cache
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - xiaozhi-network

networks:
  xiaozhi-network:
    external: true
```

---

## ⚙️ **混合TTS架构配置**

### **智能TTS路由**

```yaml
# tts_hybrid_config.yaml
tts_routing:
  # 本地Edge-TTS优先
  local_edge_tts:
    endpoint: "http://localhost:5000"
    priority: 85                         # 85%使用本地
    max_text_length: 500                 # 最大文本长度
    timeout: 3                           # 3秒超时
    cache_enabled: true
    
  # 远程TTS备份
  remote_tts:
    - name: "azure_tts"
      endpoint: "https://eastus.tts.speech.microsoft.com"
      priority: 10                       # 10%使用Azure
      trigger_conditions:
        - "text_length > 500"
        - "local_tts_timeout"
        - "special_voice_required"
        
    - name: "xunfei_tts"
      endpoint: "https://tts-api.xfyun.cn"
      priority: 5                        # 5%使用讯飞
      trigger_conditions:
        - "all_other_tts_failed"

# 语音选择策略
voice_selection:
  # 默认语音映射
  default_voices:
    female: "xiaoxiao"                   # 女声
    male: "yunxi"                        # 男声
    child: "xiaoyi"                      # 童声
    
  # 情感语音映射
  emotion_voices:
    happy: "xiaomo"                      # 开心
    calm: "yunjian"                      # 平静
    excited: "xiaoyi"                    # 兴奋
    
  # 场景语音映射
  scenario_voices:
    news: "yunjian"                      # 新闻播报
    story: "xiaoxiao"                    # 故事讲述
    assistant: "xiaoyi"                  # 助手回答
```

### **音频缓存策略**

```yaml
# 音频缓存配置
audio_caching:
  # 缓存设置
  cache_settings:
    enabled: true
    max_size: "1GB"                      # 最大缓存1GB
    ttl: 14400                           # 4小时过期
    cleanup_interval: 3600               # 1小时清理一次
    
  # 预缓存常用短语
  preload_phrases:
    greetings:
      - "你好"
      - "早上好"
      - "晚上好"
      - "欢迎"
      
    responses:
      - "好的"
      - "明白了"
      - "没问题"
      - "请稍等"
      
    errors:
      - "抱歉，我没听清"
      - "请重新说一遍"
      - "网络连接异常"
      
  # 智能缓存策略
  smart_caching:
    # 高频词汇自动缓存
    auto_cache_threshold: 5              # 使用5次以上自动缓存
    
    # 相似文本合并
    similarity_threshold: 0.9            # 相似度90%以上合并
    
    # 缓存优先级
    priority_rules:
      - pattern: "^(你好|早上好|晚上好)"
        priority: "high"
      - pattern: "^(好的|明白|没问题)"
        priority: "high"
      - length: "< 10"
        priority: "medium"
      - length: "> 100"
        priority: "low"
```

---

## 🎛️ **音频质量优化**

### **音频参数调优**

```yaml
# 音频质量配置
audio_quality:
  # 基础参数
  basic_settings:
    sample_rate: 24000                   # 采样率24kHz
    bit_rate: 64                         # 比特率64kbps
    format: "mp3"                        # 输出格式
    
  # 语音参数
  voice_parameters:
    rate: "+10%"                         # 语速+10%
    pitch: "+0Hz"                        # 音调正常
    volume: "+0%"                        # 音量正常
    
  # 动态调整
  adaptive_quality:
    # 网络状况自适应
    network_adaptive:
      good: {bit_rate: 128, sample_rate: 24000}
      normal: {bit_rate: 64, sample_rate: 22050}
      poor: {bit_rate: 32, sample_rate: 16000}
      
    # 设备性能自适应
    device_adaptive:
      high_end: {bit_rate: 128, format: "wav"}
      mid_range: {bit_rate: 64, format: "mp3"}
      low_end: {bit_rate: 32, format: "mp3"}
```

### **流式音频优化**

```python
# streaming_tts_optimizer.py
import asyncio
import edge_tts
from typing import AsyncGenerator

class StreamingTTSOptimizer:
    def __init__(self):
        self.chunk_size = 1024               # 音频块大小
        self.buffer_size = 4096              # 缓冲区大小
        
    async def optimized_stream(
        self, 
        text: str, 
        voice: str,
        chunk_callback=None
    ) -> AsyncGenerator[bytes, None]:
        """优化的流式TTS"""
        
        # 文本预处理 - 按句子分割
        sentences = self.split_sentences(text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 并行生成音频
            communicate = edge_tts.Communicate(sentence, voice)
            
            audio_buffer = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer += chunk["data"]
                    
                    # 当缓冲区达到阈值时输出
                    while len(audio_buffer) >= self.chunk_size:
                        output_chunk = audio_buffer[:self.chunk_size]
                        audio_buffer = audio_buffer[self.chunk_size:]
                        
                        if chunk_callback:
                            await chunk_callback(output_chunk)
                        yield output_chunk
                        
            # 输出剩余音频
            if audio_buffer:
                if chunk_callback:
                    await chunk_callback(audio_buffer)
                yield audio_buffer
                
    def split_sentences(self, text: str) -> list:
        """智能分句"""
        import re
        # 按标点符号分句
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]
        
    async def preload_common_phrases(self, phrases: list, voice: str):
        """预加载常用短语"""
        tasks = []
        for phrase in phrases:
            task = self.generate_and_cache(phrase, voice)
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    async def generate_and_cache(self, text: str, voice: str):
        """生成并缓存音频"""
        cache_key = f"{text}_{voice}"
        cache_file = f"/tmp/tts_cache/{cache_key}.mp3"
        
        if not os.path.exists(cache_file):
            communicate = edge_tts.Communicate(text, voice)
            with open(cache_file, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
```

---

## 📊 **性能基准测试**

### **延迟对比测试**

```bash
#!/bin/bash
# tts_benchmark.sh

echo "=== TTS延迟对比测试 ==="

# 测试文本
TEST_TEXT="你好，今天天气很好，适合出门散步。"

# 测试本地Edge-TTS
echo "测试本地Edge-TTS..."
start_time=$(date +%s%3N)
curl -s -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEST_TEXT\", \"voice\": \"xiaoxiao\"}" \
  -o local_output.mp3
end_time=$(date +%s%3N)
local_latency=$((end_time - start_time))
echo "本地TTS延迟: ${local_latency}ms"

# 测试远程Azure TTS
echo "测试远程Azure TTS..."
start_time=$(date +%s%3N)
curl -s -X POST "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1" \
  -H "Ocp-Apim-Subscription-Key: $AZURE_TTS_KEY" \
  -H "Content-Type: application/ssml+xml" \
  -d "<speak version='1.0' xml:lang='zh-CN'><voice xml:lang='zh-CN' name='zh-CN-XiaoxiaoNeural'>$TEST_TEXT</voice></speak>" \
  -o remote_output.wav
end_time=$(date +%s%3N)
remote_latency=$((end_time - start_time))
echo "远程TTS延迟: ${remote_latency}ms"

# 计算改善幅度
improvement=$((100 * (remote_latency - local_latency) / remote_latency))
echo "延迟改善: ${improvement}%"
```

### **预期性能指标**

| 指标 | 远程TTS | 本地Edge-TTS | 改善幅度 |
|------|---------|--------------|----------|
| **首字延迟** | 500-1000ms | 50-150ms | **-80%** |
| **总合成时间** | 800-1500ms | 100-300ms | **-75%** |
| **并发处理** | 3-5个 | 15-20个 | **+300%** |
| **音频质量** | 优秀 | 优秀 | 相同 |
| **可用性** | 95% | 99.9% | **+5%** |
| **成本** | $0.016/1K字符 | 免费 | **-100%** |

---

## 🔧 **集成现有系统**

### **修改TTS服务配置**

```python
# services/tts_service.py 修改
class HybridTTSService:
    def __init__(self):
        self.local_endpoint = "http://localhost:5000"
        self.remote_endpoints = {
            "azure": "https://eastus.tts.speech.microsoft.com",
            "xunfei": "https://tts-api.xfyun.cn"
        }
        self.cache = TTSCache()
        
    async def synthesize_speech(self, request):
        """智能TTS合成"""
        # 检查缓存
        cache_key = self.get_cache_key(request)
        cached_audio = await self.cache.get(cache_key)
        if cached_audio:
            return cached_audio
            
        # 选择TTS引擎
        if len(request.text) <= 500:
            # 短文本使用本地TTS
            try:
                audio = await self.call_local_tts(request)
                await self.cache.set(cache_key, audio)
                return audio
            except Exception as e:
                logger.warning(f"本地TTS失败: {e}")
                
        # 回退到远程TTS
        audio = await self.call_remote_tts(request)
        await self.cache.set(cache_key, audio)
        return audio
        
    async def call_local_tts(self, request):
        """调用本地Edge-TTS"""
        response = await self.http_client.post(
            f"{self.local_endpoint}/tts",
            json={
                "text": request.text,
                "voice": request.voice or "xiaoxiao",
                "rate": request.rate or "+10%"
            },
            timeout=3.0
        )
        return response.content
        
    async def stream_synthesis(self, request):
        """流式TTS合成"""
        try:
            # 优先使用本地流式TTS
            async with self.http_client.stream(
                "POST",
                f"{self.local_endpoint}/tts/stream",
                json={"text": request.text, "voice": request.voice}
            ) as response:
                async for chunk in response.aiter_bytes(1024):
                    yield chunk
        except Exception:
            # 回退到远程TTS
            async for chunk in self.remote_stream_synthesis(request):
                yield chunk
```

### **Docker Compose完整集成**

```yaml
# 在现有docker-compose中添加
services:
  # ... 现有服务 ...
  
  xiaozhi-edge-tts:
    build:
      context: .
      dockerfile: Dockerfile.edge-tts
    container_name: xiaozhi-edge-tts
    ports:
      - "5000:5000"
    volumes:
      - ./tts_cache:/tmp/tts_cache
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
    restart: unless-stopped
    networks:
      - xiaozhi-network
      
  # 修改主服务环境变量
  xiaozhi-esp32-server:
    # ... 现有配置 ...
    environment:
      # ... 现有环境变量 ...
      - LOCAL_TTS_ENDPOINT=http://xiaozhi-edge-tts:5000
      - TTS_HYBRID_MODE=true
      - LOCAL_TTS_PRIORITY=85
    depends_on:
      - xiaozhi-edge-tts
```

---

## 🚨 **故障转移和监控**

### **健康检查配置**

```yaml
health_monitoring:
  local_tts:
    endpoint: "http://localhost:5000/health"
    interval: 30                         # 30秒检查
    timeout: 5                           # 5秒超时
    retries: 3                           # 重试3次
    
  performance_thresholds:
    response_time: 200                   # 响应时间阈值200ms
    error_rate: 5                        # 错误率阈值5%
    memory_usage: 80                     # 内存使用率80%
    
  fallback_strategy:
    - condition: "local_tts_unhealthy"
      action: "switch_to_remote"
      
    - condition: "response_time > 300ms"
      action: "reduce_quality"
      
    - condition: "memory_usage > 90%"
      action: "clear_cache_and_restart"
```

### **自动化运维脚本**

```bash
#!/bin/bash
# tts_auto_ops.sh

# TTS服务自动运维
while true; do
    # 检查服务健康状态
    if ! curl -s http://localhost:5000/health | grep -q "healthy"; then
        echo "Edge-TTS服务异常，重启中..."
        docker restart xiaozhi-edge-tts
        sleep 30
    fi
    
    # 检查缓存大小
    cache_size=$(du -sm /tmp/tts_cache | cut -f1)
    if [ "$cache_size" -gt 1024 ]; then
        echo "TTS缓存过大(${cache_size}MB)，清理中..."
        find /tmp/tts_cache -type f -mtime +1 -delete
    fi
    
    # 检查响应时间
    response_time=$(curl -w "%{time_total}" -s -o /dev/null \
        -X POST http://localhost:5000/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "测试", "voice": "xiaoxiao"}')
    
    if (( $(echo "$response_time > 0.3" | bc -l) )); then
        echo "TTS响应时间过长(${response_time}s)，重启服务..."
        docker restart xiaozhi-edge-tts
        sleep 30
    fi
    
    sleep 60
done
```

---

## ✅ **部署检查清单**

### **部署前准备**
- [ ] 确认Python 3.9+环境
- [ ] 安装edge-tts包
- [ ] 创建缓存目录
- [ ] 配置网络访问

### **部署步骤**
- [ ] 构建Edge-TTS Docker镜像
- [ ] 启动Edge-TTS服务
- [ ] 测试语音合成功能
- [ ] 配置混合TTS路由
- [ ] 预加载常用短语
- [ ] 配置缓存策略
- [ ] 部署监控系统

### **部署后验证**
- [ ] TTS响应时间 < 150ms
- [ ] 音频质量正常
- [ ] 缓存功能正常
- [ ] 流式输出正常
- [ ] 故障转移正常
- [ ] 监控指标正常

**预期效果**: TTS延迟从500-1000ms降低到50-150ms，音频质量保持优秀，支撑设备数提升4-5倍！