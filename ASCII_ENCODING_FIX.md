# ASCII编码问题解决方案

## 问题描述

在Docker容器中运行的小智ESP32服务器出现ASCII编码错误：
```
'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
```

这个错误主要发生在处理包含中文字符的日志输出和OpenAI服务响应时。

## 根本原因

1. **Docker容器默认编码设置不当**：容器内部默认使用`C.UTF-8`或`POSIX`编码
2. **Python环境变量缺失**：缺少`PYTHONIOENCODING`和`PYTHONUTF8`环境变量
3. **系统locale设置不完整**：缺少完整的UTF-8 locale配置

## 解决方案

### 1. Docker Compose环境变量配置

在所有Docker Compose文件中添加UTF-8编码环境变量：

```yaml
environment:
  # UTF-8编码设置，防止ASCII编码错误
  - LANG=en_US.UTF-8
  - LC_ALL=en_US.UTF-8
  - PYTHONIOENCODING=utf-8
  - PYTHONUTF8=1
```

### 2. 已修复的文件

- `docker-compose.yml` - 主配置文件
- `docker-compose_optimized.yml` - 优化配置文件
- `docker-compose_dedicated.yml` - 专用服务器配置文件

### 3. 日志配置优化

在`config/logger.py`中已实现UTF-8编码支持：

```python
def setup_logging(name: str = __name__, level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
    # 确保控制台处理器使用UTF-8编码
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 确保UTF-8编码
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(console_handler.stream, 'buffer'):
        console_handler.stream = io.TextIOWrapper(
            console_handler.stream.buffer,
            encoding='utf-8',
            errors='replace',
            newline=None,
            line_buffering=console_handler.stream.line_buffering
        )
```

### 4. 系统级UTF-8环境脚本

创建了`set_utf8_env.sh`脚本用于设置UTF-8环境：

```bash
#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
```

## 验证方法

### 1. 检查容器环境变量

```bash
docker exec xiaozhi-esp32-server env | grep -E "(LANG|LC_|PYTHON)"
```

预期输出：
```
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
```

### 2. 检查Docker日志

```bash
docker logs xiaozhi-esp32-server --tail 50 | grep -i "ascii\|encoding"
```

应该没有ASCII编码错误输出。

### 3. 运行编码测试

```bash
python test_encoding_fix.py
```

所有测试应该通过。

## 预防措施

### 1. 新Docker配置检查清单

在创建新的Docker配置时，确保包含以下环境变量：
- [ ] `LANG=en_US.UTF-8`
- [ ] `LC_ALL=en_US.UTF-8`
- [ ] `PYTHONIOENCODING=utf-8`
- [ ] `PYTHONUTF8=1`

### 2. 代码开发规范

1. **文件操作**：始终指定`encoding='utf-8'`
   ```python
   with open(file_path, 'r', encoding='utf-8') as f:
       content = f.read()
   ```

2. **JSON处理**：使用`ensure_ascii=False`
   ```python
   json.dumps(data, ensure_ascii=False, indent=2)
   ```

3. **日志输出**：使用配置好的logger而不是直接print
   ```python
   from config.logger import setup_logging
   logger = setup_logging()
   logger.info("包含中文的日志消息")
   ```

### 3. 测试要求

在部署前运行编码测试：
- 中文字符处理测试
- JSON编码/解码测试
- 文件读写测试
- 日志输出测试

## 技术细节

### 环境变量说明

- `LANG`: 系统默认语言和编码
- `LC_ALL`: 覆盖所有locale设置
- `PYTHONIOENCODING`: Python I/O操作的默认编码
- `PYTHONUTF8`: 启用Python UTF-8模式（Python 3.7+）

### Docker重启要求

修改环境变量后需要重启容器：
```bash
docker-compose down && docker-compose up -d
```

## 故障排除

### 问题：环境变量未生效

**解决方案**：
1. 确认Docker Compose文件语法正确
2. 完全重启容器：`docker-compose down && docker-compose up -d`
3. 检查容器内环境变量：`docker exec container_name env`

### 问题：仍有ASCII编码错误

**解决方案**：
1. 检查代码中是否有硬编码的ASCII编码
2. 确认所有文件操作都指定了UTF-8编码
3. 检查第三方库的编码设置

### 问题：日志乱码

**解决方案**：
1. 确认日志配置使用UTF-8编码
2. 检查终端/控制台的编码设置
3. 使用`errors='replace'`处理编码错误

## 更新历史

- 2024-01-XX: 初始版本，修复ASCII编码问题
- 修复了Docker Compose配置中的UTF-8环境变量
- 优化了日志配置以支持UTF-8编码
- 创建了编码测试脚本和环境设置脚本