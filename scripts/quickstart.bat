@echo off
REM QuantumFlow 快速启动脚本 (Windows)

echo ==========================================
echo   QuantumFlow 快速启动
echo ==========================================
echo.

REM 检查Python版本
python --version
echo.

REM 启动Redis (如果可用)
where redis-server >nul 2>nul
if %errorlevel% equ 0 (
    echo 启动Redis...
    start /B redis-server --daemonize yes
) else (
    echo Redis未安装，跳过
)
echo.

REM 安装依赖
echo 安装依赖...
pip install -e . --quiet
echo.

REM 检查配置文件
if not exist "config.yaml" (
    if exist "configs\default.yaml" (
        copy configs\default.yaml config.yaml
        echo 配置文件已创建: config.yaml
    )
)
echo.

echo ==========================================
echo   启动服务
echo ==========================================

REM 启动API服务器
echo 启动API服务器...
start "QuantumFlow API" python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

echo.
echo 服务已启动!
echo   - API: http://localhost:8000
echo   - 文档: http://localhost:8000/docs
echo   - 状态页: http://localhost:8000/static/status.html
echo.
echo 按任意键退出...
pause >nul
