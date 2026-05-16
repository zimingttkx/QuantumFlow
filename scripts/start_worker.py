#!/usr/bin/env python3
"""启动Worker节点"""

import sys
import os
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantumflow.worker import WorkerNode, WorkerConfig
from quantumflow.inference.backends import VLLMEngine
from quantumflow.utils.logging import setup_logging
import click


def main():
    import argparse

    parser = argparse.ArgumentParser(description="启动QuantumFlow Worker")
    parser.add_argument("--controller-url", default="http://localhost:8000", help="Controller URL")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "tgi", "sglang"], help="推理后端")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    # 创建引擎
    if args.backend == "vllm":
        engine = VLLMEngine()
    else:
        engine = None

    # 创建Worker配置
    config = WorkerConfig(
        node_id=f"worker-{args.port}",
        host=args.host,
        port=args.port,
    )

    # 创建并启动Worker
    worker = WorkerNode(config=config, engine=engine)

    async def run():
        await worker.start(controller_url=args.controller_url)
        print(f"Worker已启动: {args.host}:{args.port}")
        print(f"Controller: {args.controller_url}")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("正在停止Worker...")
            await worker.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
