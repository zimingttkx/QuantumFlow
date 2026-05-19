"""集群模块"""

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo

# 全局单例
_cluster_manager: ClusterManager = None


def get_cluster_manager() -> ClusterManager:
    """获取全局 ClusterManager 单例"""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


def set_cluster_manager(manager: ClusterManager):
    """设置全局 ClusterManager（用于测试）"""
    global _cluster_manager
    _cluster_manager = manager


__all__ = [
    "ClusterManager",
    "Node",
    "NodeStatus",
    "GPUInfo",
    "get_cluster_manager",
    "set_cluster_manager",
]
