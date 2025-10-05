#!/usr/bin/env python3
import re

# 读取配置文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

# 首先删除所有现有的SenseVoiceStream配置
content = re.sub(r'  SenseVoiceStream:.*?(?=  \w+:|\Z)', '', content, flags=re.DOTALL)

# 在FunASRServer之前插入正确格式的SenseVoiceStream配置
sensevoice_config = """  SenseVoiceStream:
    type: sensevoice_stream
    model_dir: models/SenseVoiceSmall
    output_dir: tmp/
    vad_model: fsmn-vad
    enable_realtime: true
    chunk_size: 1024

"""

# 找到FunASRServer并在其前面插入配置
content = content.replace('  FunASRServer:', sensevoice_config + '  FunASRServer:')

# 写回文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'w', encoding='utf-8') as f:
    f.write(content)

print("配置文件修复完成")