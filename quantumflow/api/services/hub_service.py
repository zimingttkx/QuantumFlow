"""HuggingFace Hub服务 - 模型发现、搜索、验证、下载"""

import json
import os

import structlog
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    list_models,
    model_info,
    snapshot_download,
)
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

logger = structlog.get_logger().bind(component="hub_service")

# 本地模型下载目录
MODELS_CACHE_DIR = os.path.expanduser("~/.cache/quantumflow/models")

_HF_API = HfApi()

# 下载中的模型tracker，避免重复下载
_downloading: dict[str, float] = {}  # model_id -> progress (0-100)


def get_models_dir() -> str:
    """获取模型下载目录"""
    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    return MODELS_CACHE_DIR


def _model_to_dict(m) -> dict:
    """将HF模型信息转为字典"""
    getattr(m, "siblings", None) or []
    return {
        "model_id": m.modelId if hasattr(m, "modelId") else m.id,
        "author": getattr(m, "author", "unknown"),
        "downloads": getattr(m, "downloads", 0) or 0,
        "likes": getattr(m, "likes", 0) or 0,
        "pipeline_tag": getattr(m, "pipeline_tag", "unknown"),
        "tags": getattr(m, "tags", []) or [],
        "last_modified": (
            str(getattr(m, "lastModified", "")) if getattr(m, "lastModified", None) else ""
        ),
        "sha": getattr(m, "sha", ""),
        "created_at": str(getattr(m, "createdAt", "")) if getattr(m, "createdAt", None) else "",
        "private": getattr(m, "private", False),
        "gated": getattr(m, "gated", False),
        "library_name": getattr(m, "library_name", "unknown"),
    }


async def get_trending_models(limit: int = 30, filter_params: dict = None) -> list[dict]:
    """获取HF热门文本生成模型

    Args:
        limit: 返回数量
        filter_params: 额外过滤参数

    Returns:
        模型列表
    """
    filter_params = filter_params or {}
    try:
        models_iter = list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=limit * 3,  # 多取一些做过滤
            full=False,
            **filter_params,
        )

        result = []
        seen = set()
        for m in models_iter:
            model_dict = _model_to_dict(m)
            model_id = model_dict["model_id"]

            # 去重
            if model_id.lower() in seen:
                continue
            seen.add(model_id.lower())

            # 排除gguf/onnx变体等
            if any(skip in model_id.lower() for skip in ["-gguf", "-onnx", "-ggml", "onnx-"]):
                continue

            result.append(model_dict)
            if len(result) >= limit:
                break

        logger.info("trending_models_fetched", count=len(result))
        return result

    except Exception as e:
        logger.error("trending_fetch_error", error=str(e))
        return []


async def search_models(query: str, limit: int = 20) -> list[dict]:
    """搜索HF模型

    Args:
        query: 搜索关键词
        limit: 返回数量

    Returns:
        匹配的模型列表
    """
    try:
        models_iter = list_models(
            search=query,
            sort="downloads",
            limit=limit,
            full=False,
        )

        result = []
        for m in models_iter:
            model_dict = _model_to_dict(m)
            result.append(model_dict)
            if len(result) >= limit:
                break

        logger.info("search_completed", query=query, count=len(result))
        return result

    except Exception as e:
        logger.error("search_error", query=query, error=str(e))
        return []


async def validate_model(model_id: str) -> dict:
    """验证模型是否存在于HF上（不下载）

    Args:
        model_id: HuggingFace模型ID

    Returns:
        {
            "valid": bool,
            "model_id": str,
            "exists": bool,
            "gated": bool,
            "error": str or None,
            "info": dict or None
        }
    """
    try:
        info = model_info(model_id, files_metadata=False)
        return {
            "valid": True,
            "model_id": model_id,
            "exists": True,
            "gated": getattr(info, "gated", False),
            "error": None,
            "info": {
                "author": getattr(info, "author", "unknown"),
                "downloads": getattr(info, "downloads", 0) or 0,
                "likes": getattr(info, "likes", 0) or 0,
                "pipeline_tag": getattr(info, "pipeline_tag", "unknown"),
                "tags": getattr(info, "tags", []) or [],
                "library_name": getattr(info, "library_name", "unknown"),
                "last_modified": str(getattr(info, "lastModified", "")),
                "sha": getattr(info, "sha", ""),
                "private": getattr(info, "private", False),
            },
        }
    except RepositoryNotFoundError:
        return {
            "valid": False,
            "model_id": model_id,
            "exists": False,
            "gated": False,
            "error": f"模型 '{model_id}' 在HuggingFace上不存在",
            "info": None,
        }
    except GatedRepoError:
        return {
            "valid": True,
            "model_id": model_id,
            "exists": True,
            "gated": True,
            "error": f"模型 '{model_id}' 需要授权访问，请先在HuggingFace上申请权限",
            "info": None,
        }
    except HfHubHTTPError as e:
        return {
            "valid": False,
            "model_id": model_id,
            "exists": False,
            "gated": False,
            "error": f"访问错误: {str(e)}",
            "info": None,
        }
    except Exception as e:
        logger.error("validate_model_error", model_id=model_id, error=str(e))
        return {
            "valid": False,
            "model_id": model_id,
            "exists": False,
            "gated": False,
            "error": f"验证失败: {str(e)}",
            "info": None,
        }


async def get_model_detail(model_id: str) -> dict:
    """获取模型详细信息（含真实参数量，优先读取 config.json）"""
    try:
        info = model_info(model_id, files_metadata=True)

        # 优先从 config.json 获取真实参数量
        cfg = _fetch_config_params(model_id)
        if cfg.get("params"):
            param_count = cfg["params"]
            estimated_vram = _estimate_vram(
                param_count, cfg["hidden_size"], cfg["num_hidden_layers"]
            )
            source = "config.json"
        else:
            param_count = _estimate_params(info)
            estimated_vram = _estimate_vram(param_count)
            source = "name_heuristic"

        result = {
            "model_id": model_id,
            "author": getattr(info, "author", "unknown"),
            "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
            "pipeline_tag": getattr(info, "pipeline_tag", "unknown"),
            "tags": getattr(info, "tags", []) or [],
            "library_name": getattr(info, "library_name", "unknown"),
            "last_modified": str(getattr(info, "lastModified", "")),
            "gated": getattr(info, "gated", False),
            "private": getattr(info, "private", False),
            "sha": getattr(info, "sha", ""),
            "estimated_params": param_count,
            "estimated_params_b": round(param_count / 1e9, 1),
            "estimated_vram_gb": estimated_vram,
            "estimation_source": source,
        }
        if cfg:
            result["architecture"] = {
                "hidden_size": cfg["hidden_size"],
                "num_hidden_layers": cfg["num_hidden_layers"],
                "num_attention_heads": cfg["num_attention_heads"],
                "num_key_value_heads": cfg["num_key_value_heads"],
                "intermediate_size": cfg["intermediate_size"],
                "vocab_size": cfg["vocab_size"],
            }
        return result
    except Exception as e:
        logger.error("model_detail_error", model_id=model_id, error=str(e))
        return {"model_id": model_id, "error": str(e)}


def _fetch_config_params(model_id: str) -> dict:
    """从 HF config.json 获取真实架构参数。

    Returns:
        {"params": int, "hidden_size": int, ...} 或空 dict（获取失败时）
    """
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        hidden_size = cfg.get("hidden_size", 0)
        num_hidden_layers = cfg.get("num_hidden_layers", 0)
        num_attention_heads = cfg.get("num_attention_heads", 0)
        num_key_value_heads = cfg.get("num_key_value_heads", num_attention_heads)
        intermediate_size = cfg.get("intermediate_size", 0)
        vocab_size = cfg.get("vocab_size", 0)

        if not all(
            [hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, vocab_size]
        ):
            return {}

        head_dim = hidden_size / num_attention_heads

        # Embedding
        embed = vocab_size * hidden_size

        # Per transformer layer (LLaMA/Qwen style):
        #   Q_proj = hidden_size * hidden_size
        #   K_proj = hidden_size * head_dim * num_key_value_heads
        #   V_proj = hidden_size * head_dim * num_key_value_heads
        #   O_proj = hidden_size * hidden_size
        #   gate_proj = hidden_size * intermediate_size
        #   up_proj   = hidden_size * intermediate_size
        #   down_proj = intermediate_size * hidden_size
        #   input_layernorm = hidden_size
        #   post_attention_layernorm = hidden_size
        per_layer = (
            hidden_size * hidden_size * 2  # Q + O
            + hidden_size * int(head_dim * num_key_value_heads) * 2  # K + V
            + hidden_size * intermediate_size * 3  # gate + up + down
            + hidden_size * 2  # two rmsnorms
        )

        total_params = embed + num_hidden_layers * per_layer + hidden_size  # final layernorm
        # lm_head: shares weights with embedding in most models → 0 extra

        return {
            "params": total_params,
            "params_b": round(total_params / 1e9, 1),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
        }

    except Exception:
        return {}


def _estimate_params(info) -> int:
    """从模型标签等信息估算参数量。优先读 config.json。"""
    model_id = getattr(info, "modelId", getattr(info, "id", ""))

    # 优先从 config.json 获取真实参数量
    if model_id:
        cfg = _fetch_config_params(model_id)
        if cfg.get("params"):
            return cfg["params"]

    # fallback 名称匹配
    tags = [t.lower() for t in (getattr(info, "tags", []) or [])]
    model_id_lower = model_id.lower()

    param_patterns = [
        ("8b", 8_000_000_000),
        ("7b", 7_000_000_000),
        ("13b", 13_000_000_000),
        ("70b", 70_000_000_000),
        ("72b", 72_000_000_000),
        ("34b", 34_000_000_000),
        ("3b", 3_000_000_000),
        ("1b", 1_000_000_000),
        ("1.5b", 1_500_000_000),
        ("0.5b", 500_000_000),
        ("14b", 14_000_000_000),
        ("40b", 40_000_000_000),
        ("65b", 65_000_000_000),
        ("180b", 180_000_000_000),
        ("405b", 405_000_000_000),
        ("20b", 20_000_000_000),
        ("9b", 9_000_000_000),
        ("11b", 11_000_000_000),
    ]

    for pattern, count in param_patterns:
        if pattern in model_id_lower:
            return count

    for tag in tags:
        for pattern, count in param_patterns:
            if pattern in tag:
                return count

    return 0


def _estimate_vram(param_count: int, hidden_size: int = 0, num_hidden_layers: int = 0) -> float:
    """估算 FP16 推理显存需求（GB），含 KV cache 估算。

    KV cache 估算基于: 2 * num_layers * hidden_size * max_seq_len * 2 bytes (FP16)
    默认 max_seq_len = 4096
    """
    if param_count == 0:
        return 0.0
    # FP16 model weights: 2 bytes per param + ~20% overhead
    model_gb = param_count * 2 / (1024**3) * 1.2

    # KV cache 估算
    kv_cache_gb = 0.0
    if hidden_size > 0 and num_hidden_layers > 0:
        max_seq_len = 4096
        kv_cache_gb = 2 * num_hidden_layers * hidden_size * max_seq_len * 2 / (1024**3)

    return round(model_gb + kv_cache_gb, 1)


async def download_model(
    model_id: str,
    progress_callback=None,
    max_retries: int = 3,
) -> dict:
    """从HF下载模型到本地

    Args:
        model_id: HuggingFace模型ID
        progress_callback: 进度回调 async fn(model_id, progress_pct, status)
        max_retries: 最大重试次数

    Returns:
        {"success": bool, "local_path": str, "error": str}
    """
    global _downloading

    # 先验证
    validation = await validate_model(model_id)
    if not validation["valid"]:
        return {"success": False, "local_path": "", "error": validation["error"]}

    if validation["gated"]:
        return {
            "success": False,
            "local_path": "",
            "error": f"模型 '{model_id}' 需要授权访问，请先在HuggingFace上登录并申请权限",
        }

    local_dir = os.path.join(MODELS_CACHE_DIR, model_id.replace("/", "--"))

    if os.path.exists(local_dir) and os.listdir(local_dir):
        logger.info("model_already_downloaded", model_id=model_id, path=local_dir)
        return {"success": True, "local_path": local_dir, "error": None}

    _downloading[model_id] = 0.0

    for attempt in range(max_retries):
        try:
            if progress_callback:
                await progress_callback(model_id, 0, "downloading")

            os.makedirs(local_dir, exist_ok=True)

            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
            )

            _downloading[model_id] = 100.0

            if progress_callback:
                await progress_callback(model_id, 100, "completed")

            logger.info("model_downloaded", model_id=model_id, path=downloaded_path)
            return {"success": True, "local_path": downloaded_path, "error": None}

        except Exception as e:
            logger.error(
                "download_error",
                model_id=model_id,
                attempt=attempt + 1,
                error=str(e),
            )
            if attempt < max_retries - 1:
                import asyncio

                await asyncio.sleep(2**attempt)
            else:
                _downloading.pop(model_id, None)
                return {
                    "success": False,
                    "local_path": "",
                    "error": f"下载失败（重试{max_retries}次后）: {str(e)}",
                }

    _downloading.pop(model_id, None)
    return {"success": False, "local_path": "", "error": "下载失败"}


def get_download_progress(model_id: str) -> float:
    """获取模型下载进度"""
    return _downloading.get(model_id, -1)


def get_downloaded_models() -> list[dict]:
    """获取本地已下载的模型列表"""
    result = []
    models_dir = get_models_dir()

    if not os.path.exists(models_dir):
        return result

    for entry in os.listdir(models_dir):
        entry_path = os.path.join(models_dir, entry)
        if os.path.isdir(entry_path) and os.listdir(entry_path):
            model_id = entry.replace("--", "/")
            # 计算大小
            total_size = 0
            for dirpath, _dirnames, filenames in os.walk(entry_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            result.append(
                {
                    "model_id": model_id,
                    "local_path": entry_path,
                    "size_bytes": total_size,
                    "size_gb": round(total_size / (1024**3), 2),
                }
            )

    return result
