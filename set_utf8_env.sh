#!/bin/bash
# 设置UTF-8编码环境变量，防止ASCII编码错误

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# 确保Python使用UTF-8编码
export PYTHONUTF8=1

echo "UTF-8编码环境变量已设置"
echo "LANG=$LANG"
echo "LC_ALL=$LC_ALL"
echo "PYTHONIOENCODING=$PYTHONIOENCODING"
echo "PYTHONUTF8=$PYTHONUTF8"