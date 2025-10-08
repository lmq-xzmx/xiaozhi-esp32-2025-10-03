#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置加载器
from config.config_loader import get_private_config_from_api

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    try:
        # 加载配置
        logger.info("Loading configuration...")
        config = get_private_config_from_api()
        
        # 确保server配置存在
        if "server" not in config:
            config["server"] = {
                "host": "0.0.0.0",
                "port": 8000,
                "websocket_port": 8003,
                "auth_key": "",
                "debug": False
            }
        
        # 设置auth_key
        auth_key = os.getenv("AUTH_KEY", "")
        if auth_key:
            config["server"]["auth_key"] = auth_key
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Server will run on {config['server']['host']}:{config['server']['port']}")
        
        # 这里应该启动实际的服务器
        # 为了测试，我们只是打印配置并保持运行
        logger.info("Server starting...")
        
        # 保持服务运行
        while True:
            await asyncio.sleep(60)
            logger.info("Server is running...")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)