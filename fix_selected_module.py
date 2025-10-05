#!/usr/bin/env python3
import re

# 读取配置文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到selected_module部分并重新格式化
# 使用正则表达式匹配selected_module部分
pattern = r'(selected_module:\s*\n)(.*?)(\n\w+:)'
def replace_selected_module(match):
    start = match.group(1)
    end = match.group(3)
    
    new_content = """  # 语音活动检测模块，默认使用SileroVAD模型
  VAD: SileroVAD
  # 语音识别模块，默认使用FunASR本地模型
  ASR: SenseVoiceStream
  # 将根据配置名称对应的type调用实际的LLM适配器
  LLM: ChatGLMLLM
  # 视觉语言大模型
  VLLM: ChatGLMVLLM
  # TTS将根据配置名称对应的type调用实际的TTS适配器
  TTS: EdgeTTS
  # 记忆模块，默认不开启记忆；如果想使用超长记忆，推荐使用mem0ai；如果注重隐私，请使用本地的mem_local_short
  Memory: nomem
  # 意图识别模块开启后，可以播放音乐、控制音量、识别退出指令。
  Intent: nointent
"""
    
    return start + new_content + end

content = re.sub(pattern, replace_selected_module, content, flags=re.DOTALL)

# 写回文件
with open('/opt/xiaozhi-esp32-server/config.yaml', 'w', encoding='utf-8') as f:
    f.write(content)

print("selected_module配置修复完成")