#!/usr/bin/env python3
import yaml
import re

# 读取配置文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

# 删除所有SenseVoiceStream相关的行
lines = content.split('\n')
cleaned_lines = []
skip_lines = False

for line in lines:
    if 'SenseVoiceStream:' in line:
        skip_lines = True
        continue
    elif skip_lines and (line.startswith('  ') and not line.startswith('    ')):
        # 遇到下一个同级配置项，停止跳过
        skip_lines = False
    elif skip_lines:
        continue
    
    cleaned_lines.append(line)

# 重新组合内容
content = '\n'.join(cleaned_lines)

# 在FunASRServer之前插入正确的SenseVoiceStream配置
sensevoice_config = """  SenseVoiceStream:
    type: sensevoice_stream
    model_dir: models/SenseVoiceSmall
    output_dir: tmp/
    vad_model: fsmn-vad
    enable_realtime: true
    chunk_size: 1024

"""

# 找到FunASRServer的位置并插入
content = content.replace('  FunASRServer:', sensevoice_config + '  FunASRServer:')

# 写回文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'w', encoding='utf-8') as f:
    f.write(content)

print("配置文件最终修复完成")