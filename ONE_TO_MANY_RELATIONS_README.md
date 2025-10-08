# 一对多关系管理系统

## 概述

本系统提供了设备、智能体、学员之间一对多关系的完整管理功能，包括可视化界面和RESTful API接口。

## 快速开始

### 1. 启动服务

```bash
# 启动HTTP文件服务器（用于访问Web界面）
python3 -m http.server 8080

# 启动API服务器（用于数据管理）
python3 demo_api_server.py
```

### 2. 访问地址

- **Web管理界面**: http://localhost:8080/one-to-many-relations.html
- **API服务**: http://localhost:8090/api/
- **现有系统**: http://localhost:8002/#/home

## 功能特性

### 一对多关系概览
- 📊 **实时统计**: 设备、智能体、学员总数统计
- 🔗 **关系可视化**: 直观展示各实体间的关联关系
- 📈 **使用情况**: 智能体使用统计和设备在线状态

### 设备管理
- 📱 **设备列表**: 查看所有设备及其基本信息
- 🔍 **搜索过滤**: 按设备别名、MAC地址搜索
- 💬 **聊天统计**: 查看每个设备的聊天记录数量
- 🟢 **在线状态**: 实时显示设备在线/离线状态

### 智能体管理
- 🤖 **智能体列表**: 查看所有可用的智能体
- 📊 **服务统计**: 显示每个智能体服务的设备数量
- ➕ **添加智能体**: 支持创建新的智能体

### 学员管理
- 👥 **学员列表**: 查看所有学员信息
- 📱 **绑定设备**: 显示每个学员绑定的设备数量
- 🔗 **绑定历史**: 查看详细的设备绑定历史记录

### 绑定关系管理
- 🔗 **关系查看**: 查看设备与学员的绑定关系
- ⏰ **时间管理**: 显示绑定时间、解绑时间
- 📊 **状态跟踪**: 跟踪绑定状态（活跃/已解绑）

### 聊天统计分析
- 💬 **聊天概览**: 按设备统计聊天记录
- 📈 **消息分析**: 区分用户消息和AI回复
- 🕒 **时间统计**: 显示首次和最后聊天时间

## API接口文档

### 基础信息
- **基础URL**: `http://localhost:8090/api`
- **数据格式**: JSON
- **支持CORS**: 是

### 主要接口

#### 设备相关
```
GET  /api/devices          - 获取设备列表
POST /api/devices          - 创建新设备
GET  /api/devices/{id}     - 获取单个设备信息
```

#### 智能体相关
```
GET  /api/agents           - 获取智能体列表
POST /api/agents           - 创建新智能体
GET  /api/agents/{id}      - 获取单个智能体信息
```

#### 学员相关
```
GET  /api/students         - 获取学员列表
POST /api/students         - 创建新学员
GET  /api/students/{id}    - 获取单个学员信息
```

#### 绑定关系
```
GET  /api/bindings         - 获取绑定关系列表
POST /api/bindings         - 创建新绑定关系
```

#### 统计分析
```
GET  /api/chat-stats       - 获取聊天统计
GET  /api/demo-data-summary - 获取数据摘要
```

#### 系统状态
```
GET  /health               - 健康检查
```

## 数据结构

### 设备 (Device)
```json
{
  "id": "dev_001",
  "mac_address": "AA:BB:CC:DD:EE:01",
  "device_alias": "智能设备01",
  "agent_id": "agent_004",
  "is_online": false,
  "last_connected_at": "2025-10-03 10:33:42",
  "created_at": "2025-09-03 14:23:38"
}
```

### 智能体 (Agent)
```json
{
  "id": "agent_001",
  "name": "小智助手",
  "description": "专业的AI助手，擅长回答各种问题"
}
```

### 学员 (Student)
```json
{
  "id": "student_001",
  "name": "张三",
  "class": "三年级一班",
  "grade": "三年级"
}
```

### 绑定关系 (Binding)
```json
{
  "device_id": "dev_001",
  "student_id": "student_001",
  "bind_time": "2025-09-01 08:00:00",
  "unbind_time": null,
  "bind_status": "active",
  "is_current": true,
  "bind_duration_days": 37
}
```

## 使用场景

### 1. 设备管理员
- 查看所有设备的运行状态
- 监控设备与智能体的关联情况
- 分析设备使用统计

### 2. 教育管理员
- 管理学员与设备的绑定关系
- 查看学员的设备使用历史
- 分析学员的学习活跃度

### 3. 系统运维
- 监控系统整体运行状况
- 分析智能体的负载分布
- 优化资源配置

## 技术架构

### 前端技术
- **HTML5 + CSS3**: 响应式界面设计
- **JavaScript (ES6+)**: 交互逻辑实现
- **Fetch API**: 异步数据请求
- **CSS Grid/Flexbox**: 现代布局技术

### 后端技术
- **Python 3.9+**: 服务器端开发
- **Flask**: 轻量级Web框架
- **Flask-CORS**: 跨域请求支持
- **JSON**: 数据交换格式

### 数据存储
- **JSON文件**: 演示数据存储
- **MySQL**: 生产环境数据库（可选）

## 部署说明

### 开发环境
```bash
# 克隆项目
git clone <repository-url>
cd xiaozhi-server

# 安装依赖
pip install flask flask-cors

# 启动服务
python3 demo_api_server.py
python3 -m http.server 8080
```

### Docker部署
```bash
# 使用docker-compose
docker-compose up -d relations-api relations-web
```

### 生产环境
- 使用Nginx作为反向代理
- 使用Gunicorn作为WSGI服务器
- 配置SSL证书
- 设置日志轮转

## 配置选项

### 环境变量
```bash
# API服务配置
RELATIONS_API_PORT=8090
RELATIONS_API_HOST=0.0.0.0
RELATIONS_DEBUG=false

# 数据文件路径
DEMO_DATA_PATH=./demo_data.json
```

### 配置文件
可以通过修改 `demo_api_server.py` 中的配置来调整服务行为。

## 故障排除

### 常见问题

1. **页面无法加载数据**
   - 检查API服务器是否启动
   - 确认端口8090没有被占用
   - 查看浏览器控制台错误信息

2. **CORS错误**
   - 确认Flask-CORS已正确安装
   - 检查API服务器的CORS配置

3. **数据显示异常**
   - 检查demo_data.json文件格式
   - 确认文件权限正确

### 调试命令
```bash
# 检查端口占用
netstat -tlnp | grep 8090

# 测试API接口
curl http://localhost:8090/api/health

# 查看服务日志
tail -f /var/log/relations-api.log
```

## 扩展开发

### 添加新功能
1. 在API服务器中添加新的路由
2. 在前端页面中添加对应的UI组件
3. 更新数据结构和接口文档

### 集成现有系统
参考 `ONE_TO_MANY_IMPLEMENTATION_GUIDE.md` 中的详细集成指南。

## 许可证

本项目遵循与主项目相同的许可证。

## 支持

如有问题或建议，请提交Issue或联系开发团队。