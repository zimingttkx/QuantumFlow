"""集群管理路由"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Query
import structlog

from quantumflow.api.models import (
    NodeInfo,
    ClusterStatus,
    GPUInfo,
)

logger = structlog.get_logger().bind(component="api_cluster")

router = APIRouter(prefix="/cluster", tags=["Cluster"])

# 模拟节点数据
_mock_nodes = {
    "node-1": NodeInfo(
        node_id="node-1",
        hostname="gpu-server-1",
        ip="192.168.1.101",
        port=8001,
        status="healthy",
        gpu_count=4,
        gpu_info=[
            GPUInfo(
                gpu_id=i,
                name="NVIDIA RTX 4090",
                memory_total=24 * 1024**3,
                memory_used=10 * 1024**3,
                memory_free=14 * 1024**3,
                utilization=0.5,
                temperature=45.0,
            )
            for i in range(4)
        ],
        cpu_count=32,
        memory_total=128 * 1024**3,
        memory_available=64 * 1024**3,
        disk_total=2 * 1024**4,
        disk_available=1 * 1024**4,
        current_load=0.3,
        labels={"zone": "zone-a", "gpu_type": "RTX4090"},
        version="1.0.0",
        uptime_seconds=86400,
        last_heartbeat=datetime.now(),
        loaded_models=["Qwen2.5-7B-Instruct"],
    ),
    "node-2": NodeInfo(
        node_id="node-2",
        hostname="gpu-server-2",
        ip="192.168.1.102",
        port=8001,
        status="healthy",
        gpu_count=4,
        gpu_info=[
            GPUInfo(
                gpu_id=i,
                name="NVIDIA RTX 4090",
                memory_total=24 * 1024**3,
                memory_used=8 * 1024**3,
                memory_free=16 * 1024**3,
                utilization=0.4,
                temperature=42.0,
            )
            for i in range(4)
        ],
        cpu_count=32,
        memory_total=128 * 1024**3,
        memory_available=80 * 1024**3,
        disk_total=2 * 1024**4,
        disk_available=1.5 * 1024**4,
        current_load=0.2,
        labels={"zone": "zone-a", "gpu_type": "RTX4090"},
        version="1.0.0",
        uptime_seconds=72000,
        last_heartbeat=datetime.now(),
        loaded_models=["Qwen2.5-7B-Instruct"],
    ),
    "node-3": NodeInfo(
        node_id="node-3",
        hostname="gpu-server-3",
        ip="192.168.1.103",
        port=8001,
        status="healthy",
        gpu_count=8,
        gpu_info=[
            GPUInfo(
                gpu_id=i,
                name="NVIDIA A100",
                memory_total=80 * 1024**3,
                memory_used=40 * 1024**3,
                memory_free=40 * 1024**3,
                utilization=0.6,
                temperature=50.0,
            )
            for i in range(8)
        ],
        cpu_count=64,
        memory_total=512 * 1024**3,
        memory_available=256 * 1024**3,
        disk_total=4 * 1024**4,
        disk_available=2 * 1024**4,
        current_load=0.5,
        labels={"zone": "zone-b", "gpu_type": "A100"},
        version="1.0.0",
        uptime_seconds=100000,
        last_heartbeat=datetime.now(),
        loaded_models=["Qwen2.5-72B-Instruct"],
    ),
}


@router.get(
    "/status",
    response_model=ClusterStatus,
    summary="获取集群状态",
    description="获取整个集群的状态概览",
)
async def get_cluster_status() -> ClusterStatus:
    """获取集群状态"""
    nodes = list(_mock_nodes.values())

    total_gpus = sum(n.gpu_count for n in nodes)
    available_gpus = sum(
        len(n.available_gpus) if hasattr(n, "available_gpus") else n.gpu_count
        for n in nodes
    )

    healthy = sum(1 for n in nodes if n.status == "healthy")
    unhealthy = sum(1 for n in nodes if n.status == "unhealthy")
    draining = sum(1 for n in nodes if n.status == "draining")

    return ClusterStatus(
        total_nodes=len(nodes),
        healthy_nodes=healthy,
        unhealthy_nodes=unhealthy,
        draining_nodes=draining,
        total_gpus=total_gpus,
        available_gpus=available_gpus,
        active_models=2,  # TODO: 从模型管理器获取
        pending_jobs=0,  # TODO: 从调度器获取
        running_jobs=0,  # TODO: 从调度器获取
        system_metrics={
            "cpu_usage": 0.35,
            "memory_usage": 0.45,
            "gpu_usage": 0.5,
        },
        uptime_seconds=100000,
    )


@router.get(
    "/nodes",
    response_model=List[NodeInfo],
    summary="列出节点",
    description="列出所有计算节点",
)
async def list_nodes(
    status_filter: Optional[str] = Query(None, description="状态过滤"),
    zone: Optional[str] = Query(None, description="可用区过滤"),
) -> List[NodeInfo]:
    """列出所有节点"""
    nodes = list(_mock_nodes.values())

    if status_filter:
        nodes = [n for n in nodes if n.status == status_filter]

    if zone:
        nodes = [n for n in nodes if n.labels.get("zone") == zone]

    return nodes


@router.get(
    "/nodes/{node_id}",
    response_model=NodeInfo,
    summary="获取节点信息",
    description="获取指定节点的详细信息",
)
async def get_node(node_id: str) -> NodeInfo:
    """获取节点信息"""
    if node_id not in _mock_nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "NODE_NOT_FOUND",
                    "message": f"Node not found: {node_id}",
                }
            },
        )

    return _mock_nodes[node_id]


@router.post(
    "/nodes/{node_id}/action",
    summary="节点操作",
    description="对节点执行操作（drain, uncordon）",
)
async def node_action(node_id: str, action: str) -> dict:
    """节点操作"""
    if node_id not in _mock_nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "NODE_NOT_FOUND",
                    "message": f"Node not found: {node_id}",
                }
            },
        )

    logger.info("node_action", node_id=node_id, action=action)

    # TODO: 执行实际操作
    if action == "drain":
        _mock_nodes[node_id].status = "draining"
    elif action == "uncordon":
        _mock_nodes[node_id].status = "healthy"
    elif action == "restart":
        pass  # TODO: 实现重启逻辑
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_ACTION",
                    "message": f"Invalid action: {action}",
                }
            },
        )

    return {
        "node_id": node_id,
        "action": action,
        "status": "completed",
    }
