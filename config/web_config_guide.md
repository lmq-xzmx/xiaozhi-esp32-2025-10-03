# TTS引擎Web配置指南

## 配置页面地址
**TTS引擎配置现在通过以下Web界面统一管理：**
- 配置页面：http://182.44.78.40:8002/#/model-config
- 页面名称：智控台

## 配置说明

### 当前配置状态
- **Edge TTS**: 100%权重，本地免费引擎
- **远程TTS服务**: 已移除Azure TTS和讯飞TTS配置

### 使用方法
1. 访问配置页面：http://182.44.78.40:8002/#/model-config
2. 在TTS模型配置部分进行设置
3. 根据需要调整引擎权重和参数
4. 保存配置后系统将自动应用新设置

### 优势
- **统一管理**: 所有TTS引擎配置集中在一个Web界面
- **实时更新**: 配置修改后立即生效，无需重启服务
- **可视化操作**: 直观的Web界面，操作简便
- **成本优化**: 主要使用免费的Edge TTS，降低API调用成本

### 注意事项
- 配置修改前请确保了解各引擎的特性和成本
- 建议保持Edge TTS作为主要引擎以确保服务稳定性
- 如需添加新的TTS引擎，请通过Web配置页面进行操作

## 相关文件
以下配置文件已更新为使用Web配置管理：
- `/root/xiaozhi-server/services/tts_service.py`
- `/root/xiaozhi-server/config/llm_tts_resource_reservation.yaml`
- `/root/xiaozhi-server/config/settings.py`
- `/root/xiaozhi-server/config/config_loader.py`