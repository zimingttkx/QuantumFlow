"""集群管理路由"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Query
import platform
import socket
import os
import time
import psutil
import structlog

from quantumflow.api.models import (
    NodeInfo,
    ClusterStatus,
    GPUInfo,
)

logger = structlog.get_logger().bind(component="api_cluster")

router = APIRouter(prefix="/cluster", tags=["Cluster"])


def _get_gpu_info() -> List[GPUInfo]:
    """获取真实GPU信息"""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpus.append(GPUInfo(
                gpu_id=i,
                name=name,
                memory_total=mem.total,
                memory_used=mem.used,
                memory_free=mem.free,
                utilization=util.gpu / 100.0,
                temperature=float(temp),
            ))
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory
                    allocated = torch.cuda.memory_allocated(i)
                    gpus.append(GPUInfo(
                        gpu_id=i,
                        name=props.name,
                        memory_total=total,
                        memory_used=allocated,
                        memory_free=total - allocated,
                        utilization=0.0,
                        temperature=0.0,
                    ))
        except Exception:
            pass

    return gpus


def _build_local_node() -> NodeInfo:
    """基于真实本机信息构建节点"""
    gpus = _get_gpu_info()
    cpu_count = os.cpu_count() or 1
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    uptime = time.time() - psutil.boot_time()

    from quantumflow.inference import get_engine_manager
    engine_manager = get_engine_manager()
    loaded_models = engine_manager.get_loaded_models()

    return NodeInfo(
        node_id="local-node",
        hostname=socket.gethostname(),
        ip=socket.gethostbyname(socket.gethostname()),
        port=8000,
        status="healthy",
        gpu_count=len(gpus),
        gpu_info=gpus if gpus else None,
        cpu_count=cpu_count,
        memory_total=mem.total,
        memory_available=mem.available,
        disk_total=disk.total,
        disk_available=disk.free,
        current_load=psutil.cpu_percent(interval=0.1) / 100.0,
        labels={"platform": platform.system(), "host": "local"},
        version="1.0.0",
        uptime_seconds=int(uptime),
        last_heartbeat=datetime.now(),
        loaded_models=loaded_models,
    )


@router.get(
    "/status",
    response_model=ClusterStatus,
    summary="获取集群状态",
    description="获取整个集群的状态概览",
)
async def get_cluster_status() -> ClusterStatus:
    """获取集群状态"""
    node = _build_local_node()
    nodes = [node]

    total_gpus = node.gpu_count
    available_gpus = node.gpu_count

    healthy = 1 if node.status == "healthy" else 0
    unhealthy = 1 if node.status == "unhealthy" else 0

    return ClusterStatus(
        total_nodes=1,
        healthy_nodes=healthy,
        unhealthy_nodes=unhealthy,
        draining_nodes=0,
        total_gpus=total_gpus,
        available_gpus=available_gpus,
        active_models=len(node.loaded_models),
        pending_jobs=0,
        running_jobs=0,
        system_metrics={
            "cpu_usage": node.current_load,
            "memory_usage": 1.0 - (node.memory_available / node.memory_total) if node.memory_total else 0,
            "gpu_usage": sum(g.utilization for g in (node.gpu_info or [])) / len(node.gpu_info) if node.gpu_info else 0,
        },
        uptime_seconds=node.uptime_seconds,
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
    node = _build_local_node()
    nodes = [node]

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
    node = _build_local_node()
    if node_id == node.node_id:
        return node

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": {
                "code": "NODE_NOT_FOUND",
                "message": f"Node not found: {node_id}",
            }
        },
    )


_local_node_status = "healthy"


@router.post(
    "/nodes/{node_id}/action",
    summary="节点操作",
    description="对节点执行操作（drain, uncordon）",
)
async def node_action(node_id: str, action: str) -> dict:
    """节点操作"""
    global _local_node_status
    node = _build_local_node()
    if node_id != node.node_id:
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

    if action == "drain":
        _local_node_status = "draining"
    elif action == "uncordon":
        _local_node_status = "healthy"
    elif action == "restart":
        pass
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
