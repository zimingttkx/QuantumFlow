"""副本策略配置"""

from dataclasses import dataclass, field


@dataclass
class HealthThresholds:
    """健康检测阈值配置"""

    # GPU 相关阈值
    gpu_temp_threshold: float = 90.0  # GPU 温度阈值（摄氏度）
    gpu_mem_threshold: float = 0.95  # GPU 显存使用率阈值
    gpu_util_threshold: float = 0.99  # GPU 利用率阈值
    gpu_temp_warning: float = 80.0  # GPU 温度警告阈值

    # 心跳和超时
    heartbeat_timeout: int = 30  # 心跳超时（秒）
    heartbeat_interval: int = 5  # 心跳发送间隔（秒）

    # 模型相关超时
    model_load_timeout: int = 600  # 模型加载超时（秒）
    inference_timeout: int = 300  # 推理超时（秒）

    # 通信超时
    comm_timeout: int = 10  # 节点通信超时（秒）
    rpc_timeout: int = 30  # RPC 调用超时（秒）

    # 故障判定
    failure_threshold: int = 3  # 连续失败次数阈值，触发故障
    degraded_threshold: int = 2  # 降级阈值
    jitter_window: int = 3  # 抖动判定窗口（秒）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "gpu_temp_threshold": self.gpu_temp_threshold,
            "gpu_mem_threshold": self.gpu_mem_threshold,
            "gpu_util_threshold": self.gpu_util_threshold,
            "gpu_temp_warning": self.gpu_temp_warning,
            "heartbeat_timeout": self.heartbeat_timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "model_load_timeout": self.model_load_timeout,
            "inference_timeout": self.inference_timeout,
            "comm_timeout": self.comm_timeout,
            "rpc_timeout": self.rpc_timeout,
            "failure_threshold": self.failure_threshold,
            "degraded_threshold": self.degraded_threshold,
            "jitter_window": self.jitter_window,
        }


@dataclass
class ReplicaPolicy:
    """副本策略配置"""

    # 副本数量
    default_replica_count: int = 2  # 默认副本数量
    min_replica_count: int = 1  # 最小副本数量
    max_replica_count: int = 5  # 最大副本数量

    # 同步配置
    sync_interval_seconds: int = 30  # 同步间隔（秒）
    sync_timeout_seconds: int = 300  # 同步超时（秒）
    sync_batch_size: int = 10  # 同步批次大小

    # 健康检测配置
    health_check_interval_seconds: int = 10  # 健康检测间隔（秒）

    # 故障转移配置
    failover_timeout_seconds: int = 60  # 故障转移超时（秒）
    election_timeout_seconds: int = 10  # 选举超时（秒）
    leader_lease_seconds: int = 30  # Leader 租约（秒）

    # 副本角色
    primary_preference: str = "any"  # 主节点选择偏好: "any", "local", "low_latency"

    # 脑裂防护
    lock_ttl_seconds: int = 30  # 分布式锁 TTL
    lock_retry_count: int = 3  # 锁获取重试次数
    lock_retry_delay_seconds: float = 1.0  # 锁重试延迟（秒）

    # 降级策略
    allow_degraded_replica: bool = True  # 允许降级副本
    auto_promote_secondary: bool = True  # 主节点故障时自动提升备节点

    # 资源预留
    reserve_gpu_memory_mb: int = 2048  # 预留 GPU 显存（MB）
    reserve_cpu_memory_mb: int = 8192  # 预留 CPU 内存（MB）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "default_replica_count": self.default_replica_count,
            "min_replica_count": self.min_replica_count,
            "max_replica_count": self.max_replica_count,
            "sync_interval_seconds": self.sync_interval_seconds,
            "sync_timeout_seconds": self.sync_timeout_seconds,
            "sync_batch_size": self.sync_batch_size,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "failover_timeout_seconds": self.failover_timeout_seconds,
            "election_timeout_seconds": self.election_timeout_seconds,
            "leader_lease_seconds": self.leader_lease_seconds,
            "primary_preference": self.primary_preference,
            "lock_ttl_seconds": self.lock_ttl_seconds,
            "lock_retry_count": self.lock_retry_count,
            "lock_retry_delay_seconds": self.lock_retry_delay_seconds,
            "allow_degraded_replica": self.allow_degraded_replica,
            "auto_promote_secondary": self.auto_promote_secondary,
            "reserve_gpu_memory_mb": self.reserve_gpu_memory_mb,
            "reserve_cpu_memory_mb": self.reserve_cpu_memory_mb,
        }


@dataclass
class FailoverPolicy:
    """故障转移策略配置"""

    # 自动故障转移
    auto_failover_enabled: bool = True  # 是否启用自动故障转移
    failover_delay_seconds: float = 0.0  # 故障转移延迟（秒）

    # 手动故障转移
    require_manual_confirmation: bool = False  # 是否需要手动确认

    # 故障检测增强
    multi_threshold_failover: bool = True  # 多阈值判定
    gpu_isolation_check: bool = True  # GPU 隔离检查

    # 恢复配置
    auto_recovery_enabled: bool = True  # 是否启用自动恢复
    recovery_delay_seconds: int = 300  # 恢复延迟（秒）

    # 通知配置
    notify_on_failover: bool = True  # 故障转移时发送通知
    notify_on_recovery: bool = True  # 恢复时发送通知

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "auto_failover_enabled": self.auto_failover_enabled,
            "failover_delay_seconds": self.failover_delay_seconds,
            "require_manual_confirmation": self.require_manual_confirmation,
            "multi_threshold_failover": self.multi_threshold_failover,
            "gpu_isolation_check": self.gpu_isolation_check,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "recovery_delay_seconds": self.recovery_delay_seconds,
            "notify_on_failover": self.notify_on_failover,
            "notify_on_recovery": self.notify_on_recovery,
        }
