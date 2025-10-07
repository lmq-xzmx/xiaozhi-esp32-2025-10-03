# 🧠 本地LLM部署指南 - 解决首字反馈延迟

## 🎯 **目标**
在4核8GB硬件限制下，部署本地轻量级LLM模型，将首字反馈延迟从1.5-3秒降低到200-400ms。

---

## 📊 **模型选择对比**

### **推荐模型排序**

| 模型 | 内存占用 | 推理延迟 | 智能程度 | 部署难度 | 推荐指数 |
|------|----------|----------|----------|----------|----------|
| **Qwen2-1.5B-INT4** | 1.2GB | 200-300ms | ⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐⭐ |
| **TinyLlama-1.1B-INT4** | 800MB | 150-250ms | ⭐⭐⭐ | 简单 | ⭐⭐⭐⭐ |
| **ChatGLM3-6B-INT4** | 4.5GB | 400-600ms | ⭐⭐⭐⭐⭐ | 中等 | ⭐⭐⭐ |
| **Phi-3-mini-INT4** | 1.8GB | 250-350ms | ⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐ |

**最佳选择**: **Qwen2-1.5B-INT4** - 平衡性能、智能程度和资源占用

---

## 🚀 **快速部署方案**

### **方案A: Ollama部署 (推荐)**

```bash
# 1. 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 拉取量化模型
ollama pull qwen2:1.5b-instruct-q4_0

# 3. 启动服务
ollama serve --host 0.0.0.0 --port 11434

# 4. 测试模型
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen2:1.5b-instruct-q4_0", "prompt": "你好"}'
```

### **方案B: vLLM部署 (高性能)**

```bash
# 1. 安装vLLM
pip install vllm

# 2. 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-1.5B-Instruct \
  --quantization awq \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8000

# 3. 测试API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2-1.5B-Instruct", "messages": [{"role": "user", "content": "你好"}]}'
```

### **方案C: Docker容器部署 (推荐生产)**

```yaml
# docker-compose-local-llm.yml
version: '3.8'
services:
  local-llm:
    image: ollama/ollama:latest
    container_name: xiaozhi-local-llm
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_MODELS=/root/.ollama/models
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1.5G
    restart: unless-stopped
    command: >
      sh -c "
        ollama serve &
        sleep 10 &&
        ollama pull qwen2:1.5b-instruct-q4_0 &&
        wait
      "
```

---

## ⚙️ **混合架构配置**

### **智能路由策略**

```yaml
# llm_hybrid_config.yaml
llm_routing:
  # 本地模型优先
  local_model:
    endpoint: "http://localhost:11434"
    model: "qwen2:1.5b-instruct-q4_0"
    priority: 80                         # 80%请求使用本地
    max_tokens: 1024
    timeout: 5                           # 5秒超时
    
  # 远程模型备份
  remote_models:
    - name: "qwen_cloud"
      endpoint: "https://dashscope.aliyuncs.com"
      priority: 15                       # 15%使用云端
      trigger_conditions:
        - "local_model_confidence < 0.8"
        - "query_complexity > 0.7"
        - "local_model_timeout"
        
    - name: "openai_backup"
      endpoint: "https://api.openai.com"
      priority: 5                        # 5%使用OpenAI
      trigger_conditions:
        - "all_other_models_failed"

# 路由决策逻辑
routing_logic:
  # 简单查询 -> 本地模型
  simple_queries:
    patterns:
      - "你好|hello|hi"
      - "谢谢|thank you"
      - "再见|goodbye|bye"
      - "天气|weather"
    route_to: "local_model"
    
  # 复杂查询 -> 云端模型
  complex_queries:
    patterns:
      - "写代码|编程|programming"
      - "翻译|translate"
      - "分析|analysis"
      - "创作|creative"
    route_to: "remote_models"
    
  # 默认策略
  default_strategy:
    first_try: "local_model"
    fallback: "remote_models"
    confidence_threshold: 0.8
```

### **性能优化配置**

```yaml
# 本地模型优化
local_optimization:
  # 模型加载优化
  model_loading:
    preload: true                        # 预加载模型
    keep_alive: "24h"                    # 24小时保持活跃
    num_ctx: 2048                        # 上下文长度
    num_predict: 512                     # 最大生成长度
    
  # 推理优化
  inference:
    num_thread: 4                        # 使用4个线程
    num_gpu: 0                           # CPU推理
    batch_size: 1                        # 批处理大小
    temperature: 0.7                     # 温度参数
    top_p: 0.9                          # Top-p采样
    
  # 内存优化
  memory:
    mmap: true                           # 内存映射
    mlock: false                         # 不锁定内存
    numa: false                          # 禁用NUMA
    
# 缓存策略
caching_strategy:
  # 本地模型缓存
  local_cache:
    enable: true
    ttl: 3600                           # 1小时缓存
    max_size: 10000                     # 1万条缓存
    
  # 语义缓存
  semantic_cache:
    enable: true
    similarity_threshold: 0.9           # 相似度阈值
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📈 **性能基准测试**

### **延迟对比测试**

```bash
# 测试脚本
#!/bin/bash

echo "=== LLM延迟对比测试 ==="

# 测试本地模型
echo "测试本地模型..."
time curl -s http://localhost:11434/api/generate \
  -d '{"model": "qwen2:1.5b-instruct-q4_0", "prompt": "你好，今天天气怎么样？"}' \
  | jq -r '.response'

# 测试远程模型
echo "测试远程模型..."
time curl -s https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -d '{"model": "qwen-turbo", "input": {"messages": [{"role": "user", "content": "你好，今天天气怎么样？"}]}}' \
  | jq -r '.output.text'
```

### **预期性能指标**

| 指标 | 远程模型 | 本地模型 | 改善幅度 |
|------|----------|----------|----------|
| **首字延迟** | 1500-3000ms | 200-400ms | **-80%** |
| **总响应时间** | 2000-4000ms | 300-600ms | **-75%** |
| **并发处理** | 5-10个 | 15-20个 | **+200%** |
| **可用性** | 95% | 99.9% | **+5%** |
| **成本** | $0.01/1K tokens | 免费 | **-100%** |

---

## 🔧 **集成配置**

### **修改现有LLM服务**

```python
# services/llm_service.py 修改
class HybridLLMService:
    def __init__(self):
        self.local_endpoint = "http://localhost:11434"
        self.remote_endpoints = [
            {"name": "qwen", "url": "https://dashscope.aliyuncs.com"},
            {"name": "openai", "url": "https://api.openai.com"}
        ]
        
    async def route_request(self, request):
        # 智能路由逻辑
        if self.is_simple_query(request.messages):
            return await self.call_local_model(request)
        else:
            try:
                return await self.call_local_model(request)
            except Exception:
                return await self.call_remote_model(request)
                
    async def call_local_model(self, request):
        # 调用本地模型
        response = await self.http_client.post(
            f"{self.local_endpoint}/api/generate",
            json={
                "model": "qwen2:1.5b-instruct-q4_0",
                "prompt": self.format_prompt(request.messages),
                "stream": False
            },
            timeout=5.0
        )
        return response.json()
```

### **Docker Compose集成**

```yaml
# 在现有docker-compose中添加
services:
  # ... 现有服务 ...
  
  xiaozhi-local-llm:
    image: ollama/ollama:latest
    container_name: xiaozhi-local-llm
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    restart: unless-stopped
    networks:
      - xiaozhi-network
      
  # 修改主服务环境变量
  xiaozhi-esp32-server:
    # ... 现有配置 ...
    environment:
      # ... 现有环境变量 ...
      - LOCAL_LLM_ENDPOINT=http://xiaozhi-local-llm:11434
      - LLM_HYBRID_MODE=true
      - LOCAL_LLM_PRIORITY=80
```

---

## 🚨 **故障转移机制**

### **健康检查配置**

```yaml
health_check:
  local_model:
    endpoint: "http://localhost:11434/api/tags"
    interval: 30                         # 30秒检查一次
    timeout: 5                           # 5秒超时
    retries: 3                           # 重试3次
    
  fallback_strategy:
    - condition: "local_model_unhealthy"
      action: "switch_to_remote"
      
    - condition: "local_model_slow"
      threshold: "response_time > 1000ms"
      action: "hybrid_mode"
      
    - condition: "local_model_overload"
      threshold: "cpu_usage > 90%"
      action: "reduce_local_traffic"
```

### **自动恢复机制**

```bash
#!/bin/bash
# auto_recovery.sh

# 监控本地LLM健康状态
while true; do
    # 检查服务状态
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "本地LLM服务异常，尝试重启..."
        docker restart xiaozhi-local-llm
        sleep 30
    fi
    
    # 检查内存使用
    memory_usage=$(docker stats --no-stream xiaozhi-local-llm | awk 'NR==2 {print $7}' | sed 's/%//')
    if [ "$memory_usage" -gt 90 ]; then
        echo "内存使用率过高，重启服务..."
        docker restart xiaozhi-local-llm
        sleep 30
    fi
    
    sleep 60
done
```

---

## 📊 **监控和调优**

### **关键指标监控**

```yaml
monitoring_metrics:
  # 性能指标
  performance:
    - local_model_response_time
    - local_model_throughput
    - local_model_error_rate
    - memory_usage
    - cpu_usage
    
  # 业务指标
  business:
    - local_vs_remote_ratio
    - user_satisfaction_score
    - first_response_time
    - conversation_completion_rate
    
  # 告警阈值
  alerts:
    - metric: "local_model_response_time"
      threshold: "> 500ms"
      action: "investigate_performance"
      
    - metric: "local_model_error_rate"
      threshold: "> 5%"
      action: "switch_to_remote"
      
    - metric: "memory_usage"
      threshold: "> 85%"
      action: "restart_service"
```

### **性能调优建议**

```yaml
tuning_recommendations:
  # 如果响应时间 > 400ms
  slow_response:
    - "减少num_ctx到1024"
    - "降低temperature到0.5"
    - "启用更激进的缓存"
    
  # 如果内存使用 > 85%
  high_memory:
    - "减少keep_alive时间"
    - "清理模型缓存"
    - "重启服务"
    
  # 如果错误率 > 5%
  high_error_rate:
    - "检查模型文件完整性"
    - "增加超时时间"
    - "切换到远程模型"
```

---

## ✅ **部署检查清单**

### **部署前准备**
- [ ] 确认可用内存 > 2GB
- [ ] 确认可用CPU > 2核
- [ ] 下载模型文件 (约1.2GB)
- [ ] 配置网络访问权限

### **部署步骤**
- [ ] 安装Ollama或vLLM
- [ ] 拉取Qwen2-1.5B模型
- [ ] 启动本地LLM服务
- [ ] 配置混合路由
- [ ] 测试本地模型响应
- [ ] 配置故障转移
- [ ] 部署监控系统

### **部署后验证**
- [ ] 本地模型响应时间 < 400ms
- [ ] 内存使用率 < 80%
- [ ] 错误率 < 2%
- [ ] 故障转移正常工作
- [ ] 监控指标正常

**预期效果**: 首字反馈延迟从1.5-3秒降低到200-400ms，支撑设备数提升3-5倍！