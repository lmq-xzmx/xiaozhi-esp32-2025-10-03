# Device Management 页面数据备份文档

## 备份概述

本文档详细说明了 `device-management` 页面的完整数据备份，包括所有相关的前端、后端文件以及数据结构。

**备份时间**: 2024年12月19日  
**备份位置**: `/root/xiaozhi-server/device-management-backup/`  
**页面访问地址**: `http://localhost:8002/#/device-management?agentId={agentId}`

## 页面功能概述

Device Management 页面是一个设备管理界面，主要功能包括：

1. **设备列表管理**: 显示智能体下的所有设备
2. **设备-学员绑定**: 管理设备与学员的绑定关系
3. **聊天统计**: 显示设备的聊天记录统计
4. **批量操作**: 支持批量绑定、导入导出等功能
5. **历史记录**: 查看设备绑定历史

## 数据源和API接口

### 主要数据库表

1. **ai_device** - 设备信息表
   - 字段: id, mac_address, device_alias, last_connected_at, board, app_version, bind_status
   
2. **ai_agent** - 智能体信息表
   - 字段: id, agent_name, agent_code, status
   
3. **ai_student** - 学员信息表
   - 字段: id, username, real_name, school_name, current_grade, class_name, contact_phone, contact_email
   
4. **ai_device_student_bind_history** - 设备学员绑定历史表
   - 字段: device_id, student_id, agent_id, action_type, bind_time, unbind_time, operator_name, remark
   
5. **ai_agent_chat_history** - 聊天记录表
   - 字段: device_id, user_message, ai_message, created_at
   
6. **v_device_chat_statistics** - 设备聊天统计视图
   - 字段: device_id, total_chat_count, user_message_count, ai_message_count, first_chat_time, last_chat_time

### Java后端API接口

**基础路径**: `/xiaozhi/device-student-bind`

1. **GET** `/list/{agentId}` - 获取智能体的设备学员绑定列表
2. **POST** `/bind` - 绑定学员到设备
3. **POST** `/unbind/{deviceId}` - 解绑设备的学员
4. **POST** `/transfer` - 转移设备绑定
5. **POST** `/batch-bind` - 批量绑定学员到设备
6. **POST** `/import-excel` - 通过Excel文件批量导入绑定关系
7. **GET** `/chat-statistics/{deviceId}` - 获取设备的聊天记录统计
8. **GET** `/agent-statistics/{agentId}` - 获取智能体的整体统计信息
9. **GET** `/bind-history/{deviceId}` - 获取设备绑定历史记录
10. **GET** `/search-students` - 搜索可绑定的学员
11. **GET** `/device-detail/{deviceId}` - 获取设备详细信息
12. **POST** `/validate-bind` - 验证绑定关系的有效性
13. **GET** `/export-excel/{agentId}` - 导出绑定关系到Excel

### Python API接口

**基础路径**: `/api`

1. **GET** `/agents` - 获取智能体列表
2. **GET** `/devices` - 获取设备列表
3. **GET** `/students` - 获取学员列表
4. **GET** `/demo-data-summary` - 获取数据摘要统计
5. **POST** `/device/{device_id}/bind` - 绑定设备到学生
6. **DELETE** `/bindings/{device_id}/{student_id}` - 解绑设备和学员

## 备份文件清单

### 后端文件

#### Java Spring Boot 后端
```
java-backend/
├── controller/
│   └── DeviceStudentBindController.java    # 设备学员绑定控制器
├── dto/
│   └── DeviceStudentBindDTO.java           # 数据传输对象
├── vo/
│   └── DeviceStudentVO.java                # 视图对象
├── entity/                                 # 实体类目录（空）
├── service/                                # 服务层目录（空）
└── common/                                 # 通用类目录
```

#### Python API 服务器
```
real_data_api_server.py                     # 真实数据API服务器
demo_api_server.py                          # 演示数据API服务器
```

### 前端文件

```
real-data-relations.html                    # 真实数据关系管理页面
one-to-many-relations.html                  # 一对多关系管理页面
student-management.html                     # 学员管理页面
```

### 配置文件

```
docker-compose.yml                          # Docker编排配置
pom.xml                                     # Java项目依赖配置
Dockerfile                                  # Java后端Docker镜像配置
```

### 文档文件

```
ONE_TO_MANY_IMPLEMENTATION_GUIDE.md         # 一对多关系实现指南
ONE_TO_MANY_RELATIONS_README.md             # 一对多关系说明文档
demo_data.json                              # 演示数据文件
```

## 数据结构详解

### DeviceStudentVO 数据结构

```java
public class DeviceStudentVO {
    // 设备信息
    private String deviceId;           // 设备ID
    private String macAddress;         // MAC地址
    private String deviceAlias;        // 设备别名
    private LocalDateTime lastConnectedAt; // 最后连接时间
    private String board;              // 开发板类型
    private String appVersion;         // 应用版本
    private Integer bindStatus;        // 绑定状态 (0-未绑定，1-已绑定，2-已解绑)
    
    // 智能体信息
    private String agentId;            // 智能体ID
    private String agentName;          // 智能体名称
    private String agentCode;          // 智能体代码
    
    // 学员信息
    private Long studentId;            // 学员ID
    private String studentUsername;    // 学员用户名
    private String studentRealName;    // 学员真实姓名
    private String schoolName;         // 学校名称
    private String currentGrade;       // 当前年级
    private String className;          // 班级名称
    private String contactPhone;       // 联系电话
    private String contactEmail;       // 联系邮箱
    private String studentId_;         // 学号
    
    // 绑定信息
    private LocalDateTime bindTime;    // 绑定时间
    private String bindRemark;         // 绑定备注
    
    // 聊天统计信息
    private Long totalChatCount;       // 总聊天数
    private Long userMessageCount;     // 用户消息数
    private Long aiMessageCount;       // AI消息数
    private LocalDateTime lastChatTime;    // 最后聊天时间
    private LocalDateTime firstChatTime;   // 首次聊天时间
}
```

### DeviceStudentBindDTO 数据结构

```java
public class DeviceStudentBindDTO {
    private String deviceId;           // 设备ID (必填)
    private Long studentId;            // 学员ID (必填)
    private String agentId;            // 智能体ID (必填)
    private String remark;             // 备注
    private Long operatorId;           // 操作员ID
    private String operatorName;       // 操作员姓名
}
```

## 部署架构

### Docker 容器架构

1. **xiaozhi-esp32-server-web** (端口 8002)
   - 基于 Java Spring Boot
   - 提供 Web 管理界面
   - 包含 device-management 页面

2. **xiaozhi-esp32-server-db** (MySQL)
   - 存储所有业务数据
   - 包含设备、智能体、学员、绑定关系等表

3. **xiaozhi-esp32-server-redis**
   - 缓存服务
   - 会话管理

### 访问路径

- **主要管理界面**: `http://localhost:8002/#/home`
- **设备管理页面**: `http://localhost:8002/#/device-management?agentId={agentId}`
- **一对多关系管理**: `http://localhost:8081` 或 `http://localhost:8080/one-to-many-relations.html`

## 数据备份策略

### 1. 数据库备份
```sql
-- 备份设备表
SELECT * FROM ai_device;

-- 备份智能体表
SELECT * FROM ai_agent;

-- 备份学员表
SELECT * FROM ai_student;

-- 备份绑定历史表
SELECT * FROM ai_device_student_bind_history;

-- 备份聊天记录表
SELECT * FROM ai_agent_chat_history;
```

### 2. 文件系统备份
- 所有相关源代码文件已备份到当前目录
- 配置文件已包含在备份中
- 文档文件已完整保存

### 3. 配置备份
- Docker 编排配置
- Java 项目依赖配置
- 数据库连接配置

## 恢复指南

### 1. 环境准备
```bash
# 确保 Docker 和 Docker Compose 已安装
docker --version
docker-compose --version
```

### 2. 恢复步骤
```bash
# 1. 恢复源代码
cp -r java-backend/* /path/to/xiaozhi-server/java-backend/

# 2. 恢复前端文件
cp *.html /path/to/xiaozhi-server/

# 3. 恢复Python API
cp *.py /path/to/xiaozhi-server/

# 4. 启动服务
docker-compose up -d xiaozhi-esp32-server-web
```

### 3. 数据库恢复
```sql
-- 恢复数据库表结构和数据
-- 根据具体的SQL备份文件执行恢复操作
```

## 注意事项

1. **环境依赖**: 确保 MySQL、Redis 服务正常运行
2. **端口冲突**: 检查 8002、8003 端口是否被占用
3. **权限设置**: 确保文件具有正确的读写权限
4. **数据一致性**: 恢复时注意数据的完整性和一致性
5. **版本兼容**: 确保 Java、Spring Boot 版本兼容

## 联系信息

如有问题或需要技术支持，请联系开发团队。

---
**备份完成时间**: 2024年12月19日  
**文档版本**: v1.0