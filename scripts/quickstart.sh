#!/bin/bash
# QuantumFlow 快速启动脚本

set -e

echo "=========================================="
echo "  QuantumFlow 快速启动"
echo "=========================================="
echo ""

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查Redis
echo ""
echo "检查Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✓ Redis运行中"
    else
        echo "⚠ Redis未运行，启动中..."
        redis-server --daemonize yes
    fi
else
    echo "⚠ Redis未安装，跳过"
fi

# 检查GPU
echo ""
echo "检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo "✓ GPU可用"
else
    echo "⚠ GPU不可用"
fi

# 安装依赖
echo ""
echo "安装依赖..."
if command -v poetry &> /dev/null; then
    poetry install
elif command -v pip &> /dev/null; then
    pip install -e .
else
    echo "⚠ 无法安装依赖，请手动运行 pip install -e ."
fi

# 创建配置文件
echo ""
echo "检查配置文件..."
if [ ! -f "config.yaml" ]; then
    if [ -f "configs/default.yaml" ]; then
        cp configs/default.yaml config.yaml
        echo "✓ 配置文件已创建: config.yaml"
    fi
fi

# 启动服务
echo ""
echo "=========================================="
echo "  启动服务"
echo "=========================================="

# 启动API服务器
echo "启动API服务器..."
python3 -m quantumflow.cli serve --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "API服务器 PID: $API_PID"
echo ""
echo "服务已启动!"
echo "  - API: http://localhost:8000"
echo "  - 文档: http://localhost:8000/docs"
echo "  - 状态页: http://localhost:8000/static/status.html"
echo ""
echo "按 Ctrl+C 停止服务"

# 等待
wait $API_PID
