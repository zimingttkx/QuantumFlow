#!/usr/bin/env python3
"""启动Controller（API服务器）"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantumflow.api.server import create_app
import uvicorn


def main():
    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
