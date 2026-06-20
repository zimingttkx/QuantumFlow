"""系统配置检测与模型推荐"""

import structlog

logger = structlog.get_logger().bind(component="system_profiler")


class SystemCapability:
    """系统能力描述"""

    def __init__(self):
        self.gpu_count: int = 0
        self.gpu_names: list[str] = []
        self.total_vram_gb: float = 0.0  # 总GPU显存（GB）
        self.free_vram_gb: float = 0.0  # 可用GPU显存（GB）
        self.ram_total_gb: float = 0.0  # 系统内存（GB）
        self.ram_free_gb: float = 0.0
        self.disk_free_gb: float = 0.0  # 磁盘可用空间（GB）
        self.cuda_version: str = ""
        self.pytorch_version: str = ""
        self.has_cuda: bool = False

    def to_dict(self) -> dict:
        return {
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "total_vram_gb": round(self.total_vram_gb, 1),
            "free_vram_gb": round(self.free_vram_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_free_gb": round(self.ram_free_gb, 1),
            "disk_free_gb": round(self.disk_free_gb, 1),
            "cuda_version": self.cuda_version,
            "pytorch_version": self.pytorch_version,
            "has_cuda": self.has_cuda,
        }


def detect_system() -> SystemCapability:
    """检测系统配置"""
    cap = SystemCapability()

    # GPU检测 - pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        cap.gpu_count = pynvml.nvmlDeviceGetCount()
        for i in range(cap.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            cap.gpu_names.append(name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cap.total_vram_gb += mem.total / (1024**3)
            cap.free_vram_gb += mem.free / (1024**3)
        pynvml.nvmlShutdown()
        cap.has_cuda = cap.gpu_count > 0
    except Exception:
        pass

    # PyTorch GPU
    if cap.gpu_count == 0:
        try:
            import torch

            if torch.cuda.is_available():
                cap.gpu_count = torch.cuda.device_count()
                cap.has_cuda = True
                for i in range(cap.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    cap.gpu_names.append(props.name)
                    cap.total_vram_gb += props.total_memory / (1024**3)
                    # rough estimate for free
                    cap.free_vram_gb += (props.total_memory - torch.cuda.memory_allocated(i)) / (
                        1024**3
                    )
            cap.cuda_version = torch.version.cuda or ""
            cap.pytorch_version = torch.__version__
        except Exception:
            pass

    # 系统内存
    try:
        import psutil

        mem = psutil.virtual_memory()
        cap.ram_total_gb = mem.total / (1024**3)
        cap.ram_free_gb = mem.available / (1024**3)
        cap.disk_free_gb = psutil.disk_usage("/").free / (1024**3)
    except Exception:
        pass

    logger.info(
        "system_detected",
        gpu_count=cap.gpu_count,
        total_vram_gb=round(cap.total_vram_gb, 1),
        free_vram_gb=round(cap.free_vram_gb, 1),
        ram_total_gb=round(cap.ram_total_gb, 1),
    )

    return cap


def recommend_models(
    capability: SystemCapability = None,
    popular_models: list[dict] = None,
) -> dict:
    """基于系统配置推荐模型

    推荐策略：
    - 如果能跑CUDA且有充足显存 → 推荐GPU模型
    - 如果只有CPU/MPS → 推荐小模型
    - 始终按显存做安全边界（70%使用率）

    Returns:
        {
            "system": {...},
            "recommendations": [...],
            "exceeds_capacity": [...],
        }
    """
    if capability is None:
        capability = detect_system()

    available_vram = capability.free_vram_gb * 0.7  # 安全边界
    if available_vram < 0.5 and capability.has_cuda:
        available_vram = capability.total_vram_gb * 0.7

    # 已知常用模型及显存需求
    known_models = [
        {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "name": "Qwen2.5-0.5B",
            "params": 0.5,
            "vram_gb": 1.5,
            "description": "超轻量模型，适合CPU或低显存GPU",
            "tags": ["轻量", "CPU可运行", "入门"],
        },
        {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "name": "Qwen2.5-1.5B",
            "params": 1.5,
            "vram_gb": 4.0,
            "description": "轻量级模型，最低配置可运行",
            "tags": ["轻量", "已验证", "推荐"],
        },
        {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "name": "Qwen2.5-3B",
            "params": 3.0,
            "vram_gb": 8.0,
            "description": "小模型，适合中等配置GPU",
            "tags": ["实用", "多任务"],
        },
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen2.5-7B",
            "params": 7.0,
            "vram_gb": 16.0,
            "description": "主流开源模型，性能和资源平衡",
            "tags": ["主流", "高性能"],
        },
        {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "name": "Llama-3.2-1B",
            "params": 1.0,
            "vram_gb": 3.0,
            "description": "Meta轻量级模型",
            "tags": ["轻量", "Meta"],
        },
        {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "name": "Llama-3.2-3B",
            "params": 3.0,
            "vram_gb": 8.0,
            "description": "Meta小型模型",
            "tags": ["实用", "Meta"],
        },
        {
            "model_id": "meta-llama/Llama-3-8B-Instruct",
            "name": "Llama-3-8B",
            "params": 8.0,
            "vram_gb": 18.0,
            "description": "Meta 8B模型，需较高配置",
            "tags": ["主流", "Meta"],
        },
        {
            "model_id": "google/gemma-2-2b-it",
            "name": "Gemma-2-2B",
            "params": 2.0,
            "vram_gb": 5.0,
            "description": "Google轻量模型",
            "tags": ["轻量", "Google"],
        },
        {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "name": "Phi-3.5-mini",
            "params": 3.8,
            "vram_gb": 9.0,
            "description": "微软小模型，推理能力强",
            "tags": ["实用", "Microsoft"],
        },
        {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "name": "DeepSeek-R1-1.5B",
            "params": 1.5,
            "vram_gb": 4.0,
            "description": "DeepSeek推理蒸馏版，轻量",
            "tags": ["推理", "轻量"],
        },
        {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "name": "DeepSeek-R1-7B",
            "params": 7.0,
            "vram_gb": 16.0,
            "description": "DeepSeek推理蒸馏版，7B",
            "tags": ["推理", "主流"],
        },
        {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "name": "Mistral-7B-v0.3",
            "params": 7.0,
            "vram_gb": 16.0,
            "description": "Mistral 7B，经典开源模型",
            "tags": ["主流", "Mistral"],
        },
    ]

    recommendations = []
    exceeds = []

    for m in known_models:
        status = "compatible"
        if capability.has_cuda:
            if m["vram_gb"] <= available_vram:
                status = "compatible"
            elif m["vram_gb"] <= capability.total_vram_gb * 0.9:
                status = "tight"  # 勉强能跑
            else:
                status = "exceeds"
        else:
            # CPU模式
            if m["params"] <= 3.0:
                status = "compatible" if capability.ram_free_gb > m["vram_gb"] * 1.5 else "tight"
            elif m["params"] <= 7.0:
                status = "tight"
            else:
                status = "exceeds"

        entry = {
            **m,
            "status": status,
        }

        if status == "exceeds":
            exceeds.append(entry)
        else:
            recommendations.append(entry)

    # 排序：compatible优先，然后按vram升序
    recommendations.sort(key=lambda x: (0 if x["status"] == "compatible" else 1, x["vram_gb"]))
    exceeds.sort(key=lambda x: x["vram_gb"])

    # 合并流行模型（如果提供了）
    hub_recommendations = []
    if popular_models:
        for pm in popular_models:
            model_id = pm.get("model_id", "")
            # 推测显存
            tags_str = " ".join(pm.get("tags", []) or [])
            combined = (model_id + " " + tags_str).lower()
            vram_est = _estimate_vram_from_name(combined)
            if vram_est > 0:
                param_est = _estimate_params_from_name(combined)
                status = "compatible"
                if capability.has_cuda:
                    if vram_est > capability.total_vram_gb * 0.9:
                        status = "exceeds"
                    elif vram_est > available_vram:
                        status = "tight"
                else:
                    if param_est > 3:
                        status = "tight"

                author = pm.get("author") or ""
                tags_val = pm.get("tags") or []
                downloads_val = pm.get("downloads") or 0
                likes_val = pm.get("likes") or 0
                entry = {
                    "model_id": model_id,
                    "name": (
                        (author + "/" + model_id.split("/")[-1]) if "/" in model_id else model_id
                    ),
                    "params": param_est,
                    "vram_gb": vram_est,
                    "description": f"HF热门模型 (下载量: {downloads_val})",
                    "tags": tags_val[:5] if isinstance(tags_val, list) else [],
                    "status": status,
                    "downloads": downloads_val,
                    "likes": likes_val,
                    "from_hub": True,
                }
                if status != "exceeds":
                    hub_recommendations.append(entry)

    return {
        "system": capability.to_dict(),
        "recommendations": recommendations + hub_recommendations,
        "exceeds_capacity": exceeds,
        "summary": {
            "gpu_mode": capability.has_cuda,
            "total_vram_gb": round(capability.total_vram_gb, 1),
            "available_vram_gb": round(available_vram, 1),
            "can_run_7b": available_vram >= 14
            or (not capability.has_cuda and capability.ram_free_gb > 20),
            "can_run_3b": available_vram >= 6
            or (not capability.has_cuda and capability.ram_free_gb > 10),
            "compatible_count": len(recommendations) + len(hub_recommendations),
        },
    }


def _estimate_vram_from_name(name: str) -> float:
    """从模型名称估算显存需求"""
    param = _estimate_params_from_name(name)
    if param == 0:
        return 0
    return param * 2 / (1024**3) * 1.2


def _estimate_params_from_name(name: str) -> int:
    """从名称估算参数量"""
    name = name.lower()
    patterns = [
        ("405b", 405_000_000_000),
        ("180b", 180_000_000_000),
        ("72b", 72_000_000_000),
        ("70b", 70_000_000_000),
        ("65b", 65_000_000_000),
        ("40b", 40_000_000_000),
        ("34b", 34_000_000_000),
        ("20b", 20_000_000_000),
        ("14b", 14_000_000_000),
        ("13b", 13_000_000_000),
        ("11b", 11_000_000_000),
        ("9.4b", 9_400_000_000),
        ("9b", 9_000_000_000),
        ("8b", 8_000_000_000),
        ("7b", 7_000_000_000),
        ("3.8b", 3_800_000_000),
        ("3b", 3_000_000_000),
        ("2.6b", 2_600_000_000),
        ("2b", 2_000_000_000),
        ("1.5b", 1_500_000_000),
        ("1b", 1_000_000_000),
        ("0.5b", 500_000_000),
    ]
    # Sort by pattern length descending so "1.5b" matches before "1b"
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    for pattern, count in patterns:
        if pattern in name:
            return count
    return 0
