"""集群管理路由"""

import os
import platform
import socket
import time
from datetime import datetime

import psutil
import structlog
from fastapi import APIRouter, HTTPException, Query, status

from quantumflow.api.models import (
    ClusterStatus,
    GPUInfo,
    NodeInfo,
)
from quantumflow.cluster import get_cluster_manager
from quantumflow.core.constants import NodeStatus as NodeStatusEnum

logger = structlog.get_logger().bind(component="api_cluster")

router = APIRouter(prefix="/cluster", tags=["Cluster"])


def _get_gpu_info() -> list[GPUInfo]:
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
            gpus.append(
                GPUInfo(
                    gpu_id=i,
                    name=name,
                    memory_total=mem.total,
                    memory_used=mem.used,
                    memory_free=mem.free,
                    utilization=util.gpu / 100.0,
                    temperature=float(temp),
                )
            )
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
                    gpus.append(
                        GPUInfo(
                            gpu_id=i,
                            name=props.name,
                            memory_total=total,
                            memory_used=allocated,
                            memory_free=total - allocated,
                            utilization=0.0,
                            temperature=0.0,
                        )
                    )
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


def _node_to_node_info(node) -> NodeInfo:
    """将 ClusterManager Node 转换为 API NodeInfo"""
    return NodeInfo(
        node_id=node.node_id,
        hostname=node.hostname,
        ip=node.ip,
        port=node.port,
        status=node.status.value,
        gpu_count=node.gpu_count,
        gpu_info=(
            [
                GPUInfo(
                    gpu_id=gpu.gpu_id,
                    name=gpu.name,
                    memory_total=gpu.memory_total,
                    memory_used=gpu.memory_used,
                    memory_free=gpu.memory_total - gpu.memory_used,
                    utilization=gpu.utilization,
                    temperature=gpu.temperature,
                )
                for gpu in node.gpu_info
            ]
            if node.gpu_info
            else []
        ),
        cpu_count=node.cpu_count,
        memory_total=node.memory_total,
        memory_available=node.memory_available,
        disk_total=node.disk_total,
        disk_available=node.disk_available,
        current_load=node.current_load,
        labels=node.labels,
        version=node.version,
        uptime_seconds=int((datetime.now() - datetime.fromtimestamp(0)).total_seconds()),
        last_heartbeat=node.last_heartbeat,
        loaded_models=node.loaded_models,
    )


@router.get(
    "/status",
    response_model=ClusterStatus,
    summary="获取集群状态",
    description="获取整个集群的状态概览",
)
async def get_cluster_status() -> ClusterStatus:
    """获取集群状态"""
    cluster_mgr = get_cluster_manager()
    nodes = await cluster_mgr.get_nodes()

    total_gpus = sum(n.gpu_count for n in nodes)
    available_gpus = sum(len(n.available_gpus) for n in nodes)

    healthy = sum(1 for n in nodes if n.status == NodeStatusEnum.HEALTHY)
    unhealthy = sum(1 for n in nodes if n.status == NodeStatusEnum.UNHEALTHY)

    return ClusterStatus(
        total_nodes=len(nodes),
        healthy_nodes=healthy,
        unhealthy_nodes=unhealthy,
        draining_nodes=0,
        total_gpus=total_gpus,
        available_gpus=available_gpus,
        active_models=len({m for n in nodes for m in n.loaded_models}),
        pending_jobs=0,
        running_jobs=0,
        system_metrics={
            "cpu_usage": psutil.cpu_percent(interval=0.1) / 100.0,
            "memory_usage": (
                1.0 - (psutil.virtual_memory().available / psutil.virtual_memory().total)
                if psutil.virtual_memory().total
                else 0
            ),
            "gpu_usage": 0.0,
        },
        uptime_seconds=int(time.time() - psutil.boot_time()),
    )


@router.get(
    "/nodes",
    response_model=list[NodeInfo],
    summary="列出节点",
    description="列出所有计算节点",
)
async def list_nodes(
    status_filter: str | None = Query(None, description="状态过滤"),
    zone: str | None = Query(None, description="可用区过滤"),
) -> list[NodeInfo]:
    """列出所有节点"""
    cluster_mgr = get_cluster_manager()

    # 获取节点
    if status_filter:
        status_enum = NodeStatusEnum(status_filter)
        nodes = await cluster_mgr.get_nodes(status=status_enum)
    else:
        nodes = await cluster_mgr.get_nodes()

    # 标签过滤
    if zone:
        nodes = [n for n in nodes if n.labels.get("zone") == zone]

    return [_node_to_node_info(n) for n in nodes]


@router.get(
    "/nodes/{node_id}",
    response_model=NodeInfo,
    summary="获取节点信息",
    description="获取指定节点的详细信息",
)
async def get_node(node_id: str) -> NodeInfo:
    """获取节点信息"""
    cluster_mgr = get_cluster_manager()
    node = await cluster_mgr.get_node(node_id)

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "NODE_NOT_FOUND",
                    "message": f"Node not found: {node_id}",
                }
            },
        )

    return _node_to_node_info(node)


@router.post(
    "/heartbeat",
    summary="接收Worker心跳",
    description="Worker节点定期发送心跳以表明其存活状态",
)
async def receive_heartbeat(node_info: dict) -> dict:
    """
    接收Worker心跳并更新节点状态

    Worker在启动时会发送注册请求，之后定期发送心跳更新状态
    """
    cluster_mgr = get_cluster_manager()

    node_id = node_info.get("node_id")
    if not node_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "INVALID_REQUEST", "message": "node_id is required"}},
        )

    # 检查节点是否已注册
    existing_node = await cluster_mgr.get_node(node_id)

    if existing_node:
        # 更新节点信息
        await cluster_mgr.update_node_info(
            node_id=node_id,
            gpu_info=node_info.get("gpu_info", []),
            load=node_info.get("current_load"),
        )
        # 更新已加载模型
        loaded_models = node_info.get("loaded_models", [])
        for model in loaded_models:
            if model not in existing_node.loaded_models:
                await cluster_mgr.add_loaded_model(node_id, model)

        logger.debug("heartbeat_received", node_id=node_id)
    else:
        # 注册新节点
        await cluster_mgr.register_node(node_info)
        logger.info("node_registered_via_heartbeat", node_id=node_id)

    return {"status": "ok", "node_id": node_id}


@router.post(
    "/nodes/{node_id}/action",
    summary="节点操作",
    description="对节点执行操作（drain, uncordon）",
)
async def node_action(node_id: str, action: str) -> dict:
    """节点操作"""
    cluster_mgr = get_cluster_manager()
    node = await cluster_mgr.get_node(node_id)

    if not node:
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
        await cluster_mgr.update_node_status(node_id, NodeStatusEnum.DRAINING)
    elif action == "uncordon":
        await cluster_mgr.update_node_status(node_id, NodeStatusEnum.HEALTHY)
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


@router.delete(
    "/nodes/{node_id}",
    summary="注销节点",
    description="从集群中移除节点",
)
async def unregister_node(node_id: str) -> dict:
    """注销节点"""
    cluster_mgr = get_cluster_manager()
    node = await cluster_mgr.get_node(node_id)

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "NODE_NOT_FOUND",
                    "message": f"Node not found: {node_id}",
                }
            },
        )

    await cluster_mgr.unregister_node(node_id)
    logger.info("node_unregistered_via_api", node_id=node_id)

    return {
        "node_id": node_id,
        "status": "unregistered",
    }
