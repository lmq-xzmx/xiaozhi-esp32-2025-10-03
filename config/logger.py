#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger configuration module for xiaozhi-server
Provides setup_logging function to configure logging with UTF-8 encoding
"""

import logging
import sys
from typing import Optional


def setup_logging(
    name: str = __name__,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging with UTF-8 encoding to avoid ASCII encoding errors
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Ensure UTF-8 encoding for the handler
    if hasattr(console_handler.stream, 'reconfigure'):
        # Python 3.7+
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(console_handler.stream, 'buffer'):
        # For older Python versions, wrap the stream
        import io
        console_handler.stream = io.TextIOWrapper(
            console_handler.stream.buffer,
            encoding='utf-8',
            errors='replace',
            newline=None,
            line_buffering=console_handler.stream.line_buffering
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get or create a logger with UTF-8 encoding
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logging(name)

def build_module_string(module_name: str, class_name: str = None) -> str:
    """
    构建模块字符串
    
    Args:
        module_name: 模块名称
        class_name: 类名称（可选）
        
    Returns:
        格式化的模块字符串
    """
    if class_name:
        return f"{module_name}.{class_name}"
    return module_name

def create_connection_logger(connection_id: str) -> logging.Logger:
    """
    为连接创建专用的日志记录器
    
    Args:
        connection_id: 连接ID
        
    Returns:
        连接专用的日志记录器
    """
    logger_name = f"connection.{connection_id}"
    return setup_logging(logger_name)