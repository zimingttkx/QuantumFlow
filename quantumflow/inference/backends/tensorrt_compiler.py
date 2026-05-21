"""TensorRT-LLM 模型编译工具"""

import os
import subprocess
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="tensorrt_compiler")


class TensorRTCompiler:
    """TensorRT-LLM 模型编译器

    将 HuggingFace 模型编译为 TensorRT Engine。
    """

    def __init__(self, engine_dir: str = "/tmp/tensorrt_engines"):
        self.engine_dir = engine_dir

    def compile(
        self,
        model_path: str,
        model_name: str,
        tensor_parallel: int = 1,
        dtype: str = "float16",
        **kwargs,
    ) -> str:
        """编译模型为 TensorRT Engine

        Args:
            model_path: HuggingFace 模型路径或本地路径
            model_name: 模型名称（用于目录命名）
            tensor_parallel: 张量并行数
            dtype: 数据类型 (float16, bfloat16)

        Returns:
            Engine 目录路径
        """
        engine_path = os.path.join(
            self.engine_dir,
            f"{model_name.replace('/', '_')}_tp{tensor_parallel}_{dtype}"
        )

        if os.path.exists(engine_path):
            logger.info("engine_already_exists", path=engine_path)
            return engine_path

        logger.info("compiling_model", model=model_path, output=engine_path)

        # 使用 tensorrt_llm.commands.build_checkpoint 构建
        cmd = [
            "python", "-m", "tensorrt_llm.commands.build",
            "--model_dir", model_path,
            "--output_dir", engine_path,
            "--tp_size", str(tensor_parallel),
            "--dtype", dtype,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("compilation_success", path=engine_path)
            return engine_path
        except subprocess.CalledProcessError as e:
            logger.error("compilation_failed", error=e.stderr)
            raise RuntimeError(f"TensorRT-LLM compilation failed: {e.stderr}") from e

    def get_engine_path(
        self,
        model_name: str,
        tensor_parallel: int = 1,
        dtype: str = "float16",
    ) -> str | None:
        """获取已编译的 Engine 路径"""
        engine_path = os.path.join(
            self.engine_dir,
            f"{model_name.replace('/', '_')}_tp{tensor_parallel}_{dtype}"
        )
        return engine_path if os.path.exists(engine_path) else None