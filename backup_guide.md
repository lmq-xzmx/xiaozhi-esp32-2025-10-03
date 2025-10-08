# 小智ESP32服务器数据备份操作指南

## 概述
本文档提供了小智ESP32服务器用户数据的完整备份和恢复方案，包括本地备份和远程备份的详细操作步骤。

## 需要备份的数据清单

### 1. 核心配置数据
- **位置**: `/root/xiaozhi-server/data/`
- **文件**:
  - `.config.yaml` - 主配置文件（API密钥、服务器配置）
  - `.config_backup.yaml` - 配置备份文件
  - `.wakeup_words.yaml` - 唤醒词配置
- **用途**: 系统核心配置，包含API密钥、服务器设置、唤醒词等

### 2. MySQL数据库
- **位置**: `/root/xiaozhi-server/mysql/data/`
- **内容**: 
  - 用户账户信息
  - 设备绑定关系
  - 智能体配置
  - 对话历史记录
  - 系统参数设置
- **用途**: 存储所有用户数据和系统状态

### 3. 上传文件
- **位置**: `/root/xiaozhi-server/uploadfile/`
- **内容**: 用户上传的文件、头像、音频等
- **用途**: 用户个性化内容存储

### 4. 模型文件（可选）
- **位置**: `/root/xiaozhi-server/models/`
- **内容**: SenseVoice模型文件
- **用途**: ASR语音识别模型（可重新下载）

## 本地备份操作

### 1. 创建备份目录
```bash
mkdir -p /backup/xiaozhi-$(date +%Y%m%d_%H%M%S)
cd /backup/xiaozhi-$(date +%Y%m%d_%H%M%S)
```

### 2. 备份配置文件
```bash
# 备份配置目录
cp -r /root/xiaozhi-server/data ./config_backup
```

### 3. 备份MySQL数据库
```bash
# 方法1: 使用mysqldump（推荐）
docker exec xiaozhi-esp32-server-db mysqldump -u root -p123456 xiaozhi_esp32_server > mysql_backup.sql

# 方法2: 直接备份数据文件（需要停止服务）
# docker-compose down
# cp -r /root/xiaozhi-server/mysql/data ./mysql_data_backup
# docker-compose up -d
```

### 4. 备份上传文件
```bash
# 备份用户上传文件
cp -r /root/xiaozhi-server/uploadfile ./uploadfile_backup
```

### 5. 创建备份清单
```bash
# 生成备份信息文件
cat > backup_info.txt << EOF
备份时间: $(date)
服务器版本: $(docker images | grep xiaozhi-esp32-server | head -1)
备份内容:
- 配置文件: config_backup/
- 数据库备份: mysql_backup.sql
- 上传文件: uploadfile_backup/
EOF
```

## 远程备份操作

### 方案1: 使用rsync同步到远程服务器

#### 1. 安装rsync（如果未安装）
```bash
apt update && apt install -y rsync
```

#### 2. 配置SSH密钥认证（推荐）
```bash
# 生成SSH密钥对
ssh-keygen -t rsa -b 4096 -C "xiaozhi-backup"

# 将公钥复制到远程服务器
ssh-copy-id -i ~/.ssh/id_rsa.pub user@remote-server.com
```

#### 3. 创建远程备份脚本
```bash
cat > /root/backup_to_remote.sh << 'EOF'
#!/bin/bash

# 配置变量
REMOTE_USER="backup_user"
REMOTE_HOST="your-backup-server.com"
REMOTE_PATH="/backup/xiaozhi-server"
LOCAL_BACKUP_DIR="/backup/xiaozhi-$(date +%Y%m%d_%H%M%S)"

# 创建本地备份目录
mkdir -p $LOCAL_BACKUP_DIR
cd $LOCAL_BACKUP_DIR

echo "开始创建本地备份..."

# 备份配置文件
cp -r /root/xiaozhi-server/data ./config_backup
echo "✓ 配置文件备份完成"

# 备份数据库
docker exec xiaozhi-esp32-server-db mysqldump -u root -p123456 xiaozhi_esp32_server > mysql_backup.sql
echo "✓ 数据库备份完成"

# 备份上传文件
cp -r /root/xiaozhi-server/uploadfile ./uploadfile_backup
echo "✓ 上传文件备份完成"

# 生成备份信息
cat > backup_info.txt << EOL
备份时间: $(date)
服务器IP: $(hostname -I | awk '{print $1}')
备份大小: $(du -sh . | cut -f1)
EOL

echo "开始上传到远程服务器..."

# 同步到远程服务器
rsync -avz --progress $LOCAL_BACKUP_DIR/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$(basename $LOCAL_BACKUP_DIR)/

if [ $? -eq 0 ]; then
    echo "✓ 远程备份完成: $REMOTE_HOST:$REMOTE_PATH/$(basename $LOCAL_BACKUP_DIR)/"
    
    # 可选：删除本地备份（保留最近3个）
    find /backup -name "xiaozhi-*" -type d | sort | head -n -3 | xargs rm -rf
else
    echo "✗ 远程备份失败"
    exit 1
fi
EOF

chmod +x /root/backup_to_remote.sh
```

#### 4. 执行远程备份
```bash
# 手动执行备份
/root/backup_to_remote.sh

# 设置定时备份（每天凌晨2点）
echo "0 2 * * * /root/backup_to_remote.sh >> /var/log/xiaozhi_backup.log 2>&1" | crontab -
```

### 方案2: 使用云存储服务

#### 使用阿里云OSS
```bash
# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod 755 ossutil64
mv ossutil64 /usr/local/bin/ossutil

# 配置OSS
ossutil config -e oss-cn-hangzhou.aliyuncs.com -i YOUR_ACCESS_KEY_ID -k YOUR_ACCESS_KEY_SECRET

# 创建OSS备份脚本
cat > /root/backup_to_oss.sh << 'EOF'
#!/bin/bash

BUCKET_NAME="xiaozhi-backup"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
LOCAL_BACKUP_DIR="/backup/xiaozhi-$BACKUP_DATE"

# 创建本地备份
mkdir -p $LOCAL_BACKUP_DIR
cd $LOCAL_BACKUP_DIR

# 备份数据
cp -r /root/xiaozhi-server/data ./config_backup
docker exec xiaozhi-esp32-server-db mysqldump -u root -p123456 xiaozhi_esp32_server > mysql_backup.sql
cp -r /root/xiaozhi-server/uploadfile ./uploadfile_backup

# 打包备份
tar -czf xiaozhi_backup_$BACKUP_DATE.tar.gz -C /backup xiaozhi-$BACKUP_DATE

# 上传到OSS
ossutil cp xiaozhi_backup_$BACKUP_DATE.tar.gz oss://$BUCKET_NAME/backups/

echo "备份已上传到OSS: oss://$BUCKET_NAME/backups/xiaozhi_backup_$BACKUP_DATE.tar.gz"
EOF

chmod +x /root/backup_to_oss.sh
```

## 数据恢复操作

### 1. 从本地备份恢复
```bash
# 停止服务
docker-compose down

# 恢复配置文件
cp -r /backup/xiaozhi-YYYYMMDD_HHMMSS/config_backup/* /root/xiaozhi-server/data/

# 恢复数据库
docker-compose up -d xiaozhi-esp32-server-db
sleep 30
docker exec -i xiaozhi-esp32-server-db mysql -u root -p123456 xiaozhi_esp32_server < /backup/xiaozhi-YYYYMMDD_HHMMSS/mysql_backup.sql

# 恢复上传文件
cp -r /backup/xiaozhi-YYYYMMDD_HHMMSS/uploadfile_backup/* /root/xiaozhi-server/uploadfile/

# 重启所有服务
docker-compose up -d
```

### 2. 从远程备份恢复
```bash
# 从远程服务器下载备份
rsync -avz backup_user@remote-server.com:/backup/xiaozhi-server/xiaozhi-YYYYMMDD_HHMMSS/ /backup/restore/

# 按照本地恢复流程操作
```

## 备份策略建议

### 1. 备份频率
- **配置文件**: 每次修改后立即备份
- **数据库**: 每日备份
- **上传文件**: 每周备份
- **完整备份**: 每月备份

### 2. 备份保留策略
- **本地备份**: 保留最近7天
- **远程备份**: 保留最近30天
- **归档备份**: 每月归档一次，保留12个月

### 3. 监控和告警
```bash
# 创建备份监控脚本
cat > /root/check_backup.sh << 'EOF'
#!/bin/bash

LAST_BACKUP=$(find /backup -name "xiaozhi-*" -type d | sort | tail -1)
LAST_BACKUP_TIME=$(stat -c %Y "$LAST_BACKUP" 2>/dev/null || echo 0)
CURRENT_TIME=$(date +%s)
HOURS_SINCE_BACKUP=$(( (CURRENT_TIME - LAST_BACKUP_TIME) / 3600 ))

if [ $HOURS_SINCE_BACKUP -gt 25 ]; then
    echo "警告: 最后一次备份超过25小时前！"
    # 这里可以添加邮件或微信通知
fi
EOF

# 每小时检查一次
echo "0 * * * * /root/check_backup.sh" | crontab -
```

## 安全注意事项

1. **加密备份**: 对敏感数据进行加密
2. **访问控制**: 限制备份文件的访问权限
3. **网络安全**: 使用VPN或专用网络传输备份
4. **定期测试**: 定期验证备份文件的完整性和可恢复性

## 故障排除

### 常见问题
1. **MySQL备份失败**: 检查容器状态和密码
2. **远程传输中断**: 使用rsync的断点续传功能
3. **磁盘空间不足**: 定期清理旧备份文件
4. **权限问题**: 确保备份脚本有足够的文件访问权限

### 紧急恢复
如果系统完全损坏，可以使用以下步骤快速恢复：
```bash
# 1. 重新部署基础环境
git clone https://github.com/xinnan-tech/xiaozhi-esp32-server.git
cd xiaozhi-esp32-server

# 2. 恢复备份数据
# 按照上述恢复流程操作

# 3. 启动服务
docker-compose up -d
```