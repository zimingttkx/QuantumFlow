"""QuantumFlow 核心异常定义"""


class QuantumFlowError(Exception):
    """QuantumFlow 基础异常类"""

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN",
        details: dict[str, object] | None = None,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, object]:
        """转换为字典格式"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class SchedulerError(QuantumFlowError):
    """调度器相关错误"""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(message, code="SCHEDULER_ERROR", details=details)


class SchedulerNodeUnavailableError(SchedulerError):
    """没有可用节点"""

    def __init__(self, required_gpus: int, available_gpus: int):
        super().__init__(
            f"Insufficient GPUs: required {required_gpus}, available {available_gpus}",
            details={
                "required_gpus": required_gpus,
                "available_gpus": available_gpus,
            },
        )
        self.code = "INSUFFICIENT_RESOURCES"


class SchedulerQueueFullError(SchedulerError):
    """队列已满"""

    def __init__(self, max_size: int):
        super().__init__(
            f"Request queue is full (max size: {max_size})",
            details={"max_size": max_size},
        )
        self.code = "QUEUE_FULL"


class SchedulerTimeoutError(SchedulerError):
    """调度超时"""

    def __init__(self, request_id: str, timeout: float):
        super().__init__(
            f"Request {request_id} scheduling timeout after {timeout}s",
            details={"request_id": request_id, "timeout": timeout},
        )
        self.code = "SCHEDULING_TIMEOUT"


class NodeError(QuantumFlowError):
    """节点相关错误"""

    def __init__(
        self,
        message: str,
        node_id: str | None = None,
        details: dict[str, object] | None = None,
    ):
        details = details or {}
        if node_id:
            details["node_id"] = node_id
        super().__init__(message, code="NODE_ERROR", details=details)


class NodeNotFoundError(NodeError):
    """节点未找到"""

    def __init__(self, node_id: str):
        super().__init__(f"Node not found: {node_id}", node_id=node_id)
        self.code = "NODE_NOT_FOUND"


class NodeUnhealthyError(NodeError):
    """节点不健康"""

    def __init__(self, node_id: str, reason: str | None = None):
        details = {"reason": reason} if reason else {}
        super().__init__(f"Node {node_id} is unhealthy", node_id=node_id, details=details)
        self.code = "NODE_UNHEALTHY"


class NodeConnectionError(NodeError):
    """节点连接失败"""

    def __init__(self, node_id: str, host: str, port: int):
        super().__init__(
            f"Failed to connect to node {node_id} at {host}:{port}",
            node_id=node_id,
            details={"host": host, "port": port},
        )
        self.code = "NODE_CONNECTION_ERROR"


class ModelError(QuantumFlowError):
    """模型相关错误"""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        details: dict[str, object] | None = None,
    ):
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(message, code="MODEL_ERROR", details=details)


class ModelNotFoundError(ModelError):
    """模型未找到"""

    def __init__(self, model: str):
        super().__init__(f"Model not found: {model}", model=model)
        self.code = "MODEL_NOT_FOUND"


class ModelLoadError(ModelError):
    """模型加载失败"""

    def __init__(self, model: str, reason: str):
        super().__init__(
            f"Failed to load model {model}: {reason}",
            model=model,
            details={"reason": reason},
        )
        self.code = "MODEL_LOAD_ERROR"


class ModelAlreadyLoadedError(ModelError):
    """模型已加载"""

    def __init__(self, model: str, node_id: str):
        super().__init__(
            f"Model {model} is already loaded on node {node_id}",
            model=model,
            details={"node_id": node_id},
        )
        self.code = "MODEL_ALREADY_LOADED"


class InferenceError(QuantumFlowError):
    """推理相关错误"""

    def __init__(
        self,
        message: str,
        request_id: str | None = None,
        details: dict[str, object] | None = None,
    ):
        details = details or {}
        if request_id:
            details["request_id"] = request_id
        super().__init__(message, code="INFERENCE_ERROR", details=details)


class InferenceTimeoutError(InferenceError):
    """推理超时"""

    def __init__(self, request_id: str, timeout: float):
        super().__init__(
            f"Inference request {request_id} timeout after {timeout}s",
            request_id=request_id,
            details={"timeout": timeout},
        )
        self.code = "INFERENCE_TIMEOUT"


class InferenceFailedError(InferenceError):
    """推理执行失败"""

    def __init__(self, request_id: str, reason: str):
        super().__init__(
            f"Inference request {request_id} failed: {reason}",
            request_id=request_id,
            details={"reason": reason},
        )
        self.code = "INFERENCE_FAILED"


class ResourceError(QuantumFlowError):
    """资源相关错误"""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        details: dict[str, object] | None = None,
    ):
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(message, code="RESOURCE_ERROR", details=details)


class GPUOutOfMemoryError(ResourceError):
    """GPU显存不足"""

    def __init__(self, node_id: str, gpu_id: int, required: int, available: int):
        super().__init__(
            f"GPU {gpu_id} on node {node_id} out of memory",
            resource_type="gpu_memory",
            details={
                "node_id": node_id,
                "gpu_id": gpu_id,
                "required_mb": required // (1024 * 1024),
                "available_mb": available // (1024 * 1024),
            },
        )
        self.code = "GPU_OOM"


class ValidationError(QuantumFlowError):
    """验证错误"""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: object | None = None,
    ):
        details: dict[str, object] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ConfigurationError(QuantumFlowError):
    """配置错误"""

    def __init__(self, message: str, config_key: str | None = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class StorageError(QuantumFlowError):
    """存储相关错误"""

    def __init__(
        self,
        message: str,
        storage_type: str | None = None,
        details: dict[str, object] | None = None,
    ):
        details = details or {}
        if storage_type:
            details["storage_type"] = storage_type
        super().__init__(message, code="STORAGE_ERROR", details=details)
