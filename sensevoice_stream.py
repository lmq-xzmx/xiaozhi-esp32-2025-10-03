import time
import os
import sys
import io
import psutil
import asyncio
import threading
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.base import ASRProviderBase
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import shutil
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()

MAX_RETRIES = 2
RETRY_DELAY = 1  # 重试延迟（秒）


class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        
        # 内存检测，要求大于2G
        min_mem_bytes = 2 * 1024 * 1024 * 1024
        total_mem = psutil.virtual_memory().total
        if total_mem < min_mem_bytes:
            logger.bind(tag=TAG).error(f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB，可能无法启动SenseVoice")
        
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir", "/opt/xiaozhi-esp32-server/models/SenseVoiceSmall")
        self.output_dir = config.get("output_dir", "/tmp")
        self.delete_audio_file = delete_audio_file
        
        # 移除全局锁，改为基于连接的并发控制
        # self.processing_lock = threading.Lock()  # 移除全局锁
        # self.is_processing = False  # 移除全局处理状态
        
        # VAD相关配置
        vad_kwargs = {
            "vad_model": config.get("vad_model", "fsmn-vad"),
            "vad_kwargs": {"max_single_segment_time": 30000}
        }
        
        logger.bind(tag=TAG).info(f"正在初始化SenseVoice流式ASR模型，模型路径: {self.model_dir}")
        
        try:
            with CaptureOutput():
                self.model = AutoModel(
                    model=self.model_dir,
                    **vad_kwargs
                )
            logger.bind(tag=TAG).info("SenseVoice流式ASR模型初始化成功")
        except Exception as e:
            logger.bind(tag=TAG).error(f"SenseVoice流式ASR模型初始化失败: {e}")
            raise

    async def open_audio_channels(self, conn):
        """打开音频通道，开始流式处理"""
        await super().open_audio_channels(conn)
        logger.bind(tag=TAG).info("开启SenseVoice流式音频通道")
        
        # 初始化连接的音频缓冲区和处理状态
        if not hasattr(conn, 'asr_audio'):
            conn.asr_audio = []
        if not hasattr(conn, 'asr_audio_for_voiceprint'):
            conn.asr_audio_for_voiceprint = []
        # 为每个连接添加独立的处理锁和状态
        if not hasattr(conn, 'asr_processing_lock'):
            conn.asr_processing_lock = threading.Lock()
        if not hasattr(conn, 'asr_is_processing'):
            conn.asr_is_processing = False

    async def receive_audio(self, conn, audio, audio_have_voice):
        """接收音频数据进行流式处理"""
        # 存储音频数据（保持与原有系统兼容）
        conn.asr_audio.append(audio)
        conn.asr_audio = conn.asr_audio[-20:]  # 保留最近20个音频块
        
        # 存储音频数据用于声纹识别
        if not hasattr(conn, 'asr_audio_for_voiceprint'):
            conn.asr_audio_for_voiceprint = []
        conn.asr_audio_for_voiceprint.append(audio)
        
        # 当没有音频数据时处理完整语音片段
        if not audio and len(conn.asr_audio_for_voiceprint) > 0:
            await self.handle_voice_stop(conn, conn.asr_audio_for_voiceprint)
            conn.asr_audio_for_voiceprint = []
        
        # 如果有声音且有足够的音频数据，进行实时处理
        if audio_have_voice and len(conn.asr_audio) >= 5:
            await self._process_audio_realtime(conn)

    async def _process_audio_realtime(self, conn):
        """实时处理音频数据 - 改为基于连接的并发控制"""
        # 使用连接级别的锁避免同一连接的并发处理
        if conn.asr_is_processing:
            return
        
        with conn.asr_processing_lock:
            if conn.asr_is_processing:
                return
            conn.asr_is_processing = True
        
        try:
            # 合并最近的音频数据
            combined_audio = b''.join(conn.asr_audio[-10:])
            
            if len(combined_audio) < 1000:  # 音频数据太少，跳过
                return
            
            # 在线程池中异步处理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._recognize_audio_chunk, 
                combined_audio,
                f"realtime_{id(conn)}_{int(time.time() * 1000)}"
            )
            
            if result:
                logger.bind(tag=TAG).debug(f"连接{id(conn)}实时识别结果: {result}")
                # 这里可以发送部分结果给客户端
                # 具体实现取决于系统架构
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"连接{id(conn)}实时音频处理失败: {e}")
        finally:
            conn.asr_is_processing = False

    def _recognize_audio_chunk(self, audio_data, session_id):
        """识别音频块"""
        try:
            # 保存音频数据到临时文件
            file_path = f"{self.output_dir}/chunk_{session_id}.wav"
            
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            start_time = time.time()
            result = self.model.generate(
                input=file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            
            text = rich_transcription_postprocess(result[0]["text"]) if result else ""
            
            logger.bind(tag=TAG).debug(
                f"SenseVoice块识别耗时: {time.time() - start_time:.3f}s | 结果: {text}"
            )
            
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return text
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"音频块识别失败: {e}")
            return ""

    async def handle_voice_stop(self, conn, audio_data_list):
        """处理语音停止事件"""
        try:
            # 合并所有音频数据
            combined_audio = b''.join(audio_data_list)
            
            if len(combined_audio) < 1000:  # 音频数据太少
                return
            
            # 在线程池中处理完整音频
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._recognize_complete_audio, 
                combined_audio,
                f"complete_{id(conn)}_{int(time.time() * 1000)}"
            )
            
            if result:
                logger.bind(tag=TAG).info(f"连接{id(conn)}完整语音识别结果: {result}")
                # 发送最终结果给文本处理器
                if hasattr(conn, 'handle_text_message'):
                    await conn.handle_text_message(result)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"连接{id(conn)}完整语音处理失败: {e}")

    def _recognize_complete_audio(self, audio_data, session_id):
        """识别完整音频"""
        return self._recognize_audio_chunk(audio_data, session_id)

    async def close_audio_channels(self, conn):
        """关闭音频通道"""
        await super().close_audio_channels(conn)
        logger.bind(tag=TAG).info(f"关闭连接{id(conn)}的SenseVoice流式音频通道")
        
        # 清理连接相关的缓冲区
        if hasattr(conn, 'asr_audio'):
            conn.asr_audio = []
        if hasattr(conn, 'asr_audio_for_voiceprint'):
            conn.asr_audio_for_voiceprint = []
        # 清理连接级别的处理状态
        if hasattr(conn, 'asr_is_processing'):
            conn.asr_is_processing = False

    def speech_to_text(self, pcm_data, session_id):
        """兼容原有接口的语音转文本方法"""
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                # 保存PCM数据到临时文件
                file_path = f"{self.output_dir}/audio_{session_id}_{int(time.time() * 1000)}.wav"
                
                with open(file_path, 'wb') as f:
                    f.write(pcm_data)
                
                start_time = time.time()
                result = self.model.generate(
                    input=file_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                )
                
                text = rich_transcription_postprocess(result[0]["text"]) if result else ""
                
                logger.bind(tag=TAG).debug(
                    f"SenseVoice语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}"
                )
                
                return text, file_path
                
            except OSError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.bind(tag=TAG).error(
                        f"SenseVoice语音识别失败（已重试{retry_count}次）: {e}", exc_info=True
                    )
                    return "", ""
                logger.bind(tag=TAG).warning(
                    f"SenseVoice语音识别失败，正在重试（{retry_count}/{MAX_RETRIES}）: {e}"
                )
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                logger.bind(tag=TAG).error(f"SenseVoice语音识别失败: {e}", exc_info=True)
                return "", ""
            finally:
                # 清理临时文件
                if self.delete_audio_file and 'file_path' in locals() and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"文件删除失败: {file_path} | 错误: {e}")
        
        return "", ""