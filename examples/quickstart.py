"""快速开始示例"""

import asyncio
import structlog

from quantumflow.scheduler import Scheduler, SchedulingRequest, NodeResource, GPUResource
from quantumflow.cluster import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.inference import VLLMEngine, ModelConfig, SamplingParams
from quantumflow.utils.logging import setup_logging

logger = structlog.get_logger()


async def demo_scheduler():
    """调度器示例"""
    logger.info("=== 调度器示例 ===")

    # 创建调度器
    scheduler = Scheduler(default_strategy="adaptive")

    # 添加节点
    for i in range(3):
        node = NodeResource(
            node_id=f"gpu-node-{i}",
            hostname=f"server-{i}",
            ip=f"192.168.1.{100 + i}",
            status="healthy",
            gpu_count=4,
            gpus=[
                GPUResource(
                    gpu_id=j,
                    memory_total=24 * 1024**3,
                    memory_used=10 * 1024**3,
                    utilization=0.5,
                    temperature=45.0,
                    node_id=f"gpu-node-{i}",
                )
                for j in range(4)
            ],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )
        await scheduler.register_node(node)

    # 启动调度器
    await scheduler.start()

    # 提交请求
    for i in range(5):
        request = SchedulingRequest(
            request_id=f"req_{i:04d}",
            model="Qwen2.5-7B-Instruct",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
            prompt=f"这是测试请求 {i}",
            priority=i % 3,
        )
        await scheduler.submit(request)

    # 等待调度
    await asyncio.sleep(2)

    # 查看统计
    stats = scheduler.get_stats()
    logger.info("调度统计", stats=stats)

    # 停止调度器
    await scheduler.stop()


async def demo_inference():
    """推理引擎示例"""
    logger.info("=== 推理引擎示例 ===")

    # 创建引擎
    engine = VLLMEngine()

    # 初始化
    if not await engine.initialize():
        logger.error("引擎初始化失败")
        return

    # 加载模型（需要先下载模型）
    # config = ModelConfig(
    #     model_name="Qwen2.5-7B-Instruct",
    #     model_path="Qwen/Qwen2.5-7B-Instruct",
    #     tensor_parallel=1,
    # )
    # await engine.load_model(config)

    # 生成（示例）
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    # results = await engine.generate(
    #     "Qwen2.5-7B-Instruct",
    #     ["Hello, how are you?"],
    #     sampling_params,
    # )
    # logger.info("生成结果", results=results)


async def demo_cluster():
    """集群管理示例"""
    logger.info("=== 集群管理示例 ===")

    # 创建集群管理器
    manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)

    # 启动
    await manager.start()

    # 注册节点
    node_info = {
        "node_id": "gpu-node-1",
        "hostname": "server-1",
        "ip": "192.168.1.101",
        "port": 8001,
        "gpu_count": 4,
        "gpu_info": [
            {
                "gpu_id": i,
                "name": "NVIDIA RTX 4090",
                "memory_total": 24 * 1024**3,
                "memory_used": 10 * 1024**3,
                "utilization": 0.5,
                "temperature": 45.0,
            }
            for i in range(4)
        ],
        "labels": {"zone": "zone-a", "gpu_type": "RTX4090"},
    }

    node = await manager.register_node(node_info)
    logger.info("节点已注册", node_id=node.node_id)

    # 获取集群统计
    stats = await manager.get_cluster_stats()
    logger.info("集群统计", stats=stats)

    # 获取健康节点
    healthy = await manager.get_healthy_nodes()
    logger.info("健康节点数量", count=len(healthy))

    # 停止
    await manager.stop()


async def main():
    """主函数"""
    setup_logging(log_level="INFO", log_format="console")

    logger.info("QuantumFlow 快速开始示例")
    logger.info("=" * 50)

    # 运行各个示例
    await demo_cluster()
    await demo_scheduler()
    # await demo_inference()

    logger.info("示例完成")


if __name__ == "__main__":
    asyncio.run(main())
