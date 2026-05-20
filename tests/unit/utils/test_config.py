"""配置管理模块 - 严格单元测试

测试覆盖:
1. 所有 Pydantic 配置模型的默认值
2. QuantumFlowConfig.from_file (YAML加载)
3. QuantumFlowConfig.to_dict / to_yaml
4. load_config / get_config / reload_config 全局函数
5. get_default_config LRU缓存
6. InferenceConfig.setup_backends field_validator
7. 边界场景: 文件不存在、空文件、嵌套路径、无效类型
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from quantumflow.utils.config import (
    APIConfig,
    AppConfig,
    ClusterConfig,
    InferenceBackendConfig,
    InferenceConfig,
    ModelConfig,
    MonitoringConfig,
    QuantumFlowConfig,
    RedisConfig,
    SchedulerConfig,
    StorageConfig,
    VLLMBackendConfig,
    WorkerConfig,
    get_config,
    get_default_config,
    load_config,
    reload_config,
)


# ==================== 单独配置模型默认值测试 ====================


class TestAPIConfigDefaults:
    """APIConfig 默认值验证"""

    def test_default_values_all_fields(self):
        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.workers == 4
        assert cfg.timeout == 300
        assert cfg.cors_enabled is True
        assert cfg.cors_origins == ["*"]
        assert cfg.rate_limit_enabled is True
        assert cfg.rate_limit_requests_per_minute == 100
        assert cfg.rate_limit_burst == 20

    def test_custom_all_fields(self):
        cfg = APIConfig(host="127.0.0.1", port=9090, workers=8, timeout=600,
                        cors_enabled=False, cors_origins=["http://local"],
                        rate_limit_enabled=False, rate_limit_requests_per_minute=50,
                        rate_limit_burst=10)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert cfg.workers == 8
        assert cfg.timeout == 600
        assert cfg.cors_enabled is False
        assert cfg.cors_origins == ["http://local"]
        assert cfg.rate_limit_enabled is False
        assert cfg.rate_limit_requests_per_minute == 50
        assert cfg.rate_limit_burst == 10

    def test_zero_timeout(self):
        cfg = APIConfig(timeout=0)
        assert cfg.timeout == 0

    def test_large_port(self):
        cfg = APIConfig(port=65535)
        assert cfg.port == 65535


class TestSchedulerConfigDefaults:
    """SchedulerConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.enabled is True
        assert cfg.loop_interval_ms == 100
        assert cfg.max_concurrent_requests == 1000
        assert cfg.queue_max_size == 10000
        assert cfg.queue_high_priority_threshold == 8
        assert cfg.queue_default_priority == 5
        assert cfg.strategy_default == "adaptive"
        assert cfg.strategy_gang_enabled is True
        assert cfg.strategy_gang_timeout_seconds == 300
        assert cfg.strategy_pack_enabled is True
        assert cfg.strategy_pack_max_batch_size == 32

    def test_custom_values(self):
        cfg = SchedulerConfig(enabled=False, loop_interval_ms=50,
                              max_concurrent_requests=500, queue_max_size=5000,
                              strategy_default="gang", strategy_pack_max_batch_size=16)
        assert cfg.enabled is False
        assert cfg.loop_interval_ms == 50
        assert cfg.max_concurrent_requests == 500
        assert cfg.queue_max_size == 5000
        assert cfg.strategy_default == "gang"
        assert cfg.strategy_pack_max_batch_size == 16

    def test_zero_concurrent(self):
        cfg = SchedulerConfig(max_concurrent_requests=0)
        assert cfg.max_concurrent_requests == 0


class TestClusterConfigDefaults:
    """ClusterConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = ClusterConfig()
        assert cfg.heartbeat_interval_seconds == 5
        assert cfg.heartbeat_timeout_seconds == 30
        assert cfg.node_labels == ["type", "zone", "gpu_type"]

    def test_custom(self):
        cfg = ClusterConfig(heartbeat_interval_seconds=10,
                            heartbeat_timeout_seconds=60, node_labels=["type"])
        assert cfg.heartbeat_interval_seconds == 10
        assert cfg.heartbeat_timeout_seconds == 60
        assert cfg.node_labels == ["type"]


class TestWorkerConfigDefaults:
    """WorkerConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = WorkerConfig()
        assert cfg.heartbeat_interval_seconds == 3
        assert cfg.gpu_monitoring_interval_seconds == 5
        assert cfg.model_cache_size_gb == 100
        assert cfg.max_concurrent_inferences == 10

    def test_custom(self):
        cfg = WorkerConfig(heartbeat_interval_seconds=10,
                           gpu_monitoring_interval_seconds=15,
                           model_cache_size_gb=200,
                           max_concurrent_inferences=50)
        assert cfg.heartbeat_interval_seconds == 10
        assert cfg.gpu_monitoring_interval_seconds == 15
        assert cfg.model_cache_size_gb == 200
        assert cfg.max_concurrent_inferences == 50


class TestInferenceBackendConfigDefaults:
    """InferenceBackendConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = InferenceBackendConfig()
        assert cfg.backend_type == "vllm"
        assert cfg.default_tensor_parallel == 1
        assert cfg.default_pipeline_parallel == 1
        assert cfg.default_gpu_memory_utilization == 0.9
        assert cfg.max_model_len == 8192
        assert cfg.enforce_eager is False
        assert cfg.trust_remote_code is True


class TestVLLMBackendConfigDefaults:
    """VLLMBackendConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = VLLMBackendConfig()
        assert cfg.backend_type == "vllm"
        assert cfg.block_size == 16
        assert cfg.max_num_batched_tokens == 8192
        assert cfg.max_num_seqs == 256
        assert cfg.gpu_memory_utilization == 0.9
        assert cfg.swap_space == 4
        assert cfg.enforce_eager is False
        assert cfg.enable_chunked_prefill is True
        assert cfg.use_queue_for_batch_size is True

    def test_custom(self):
        cfg = VLLMBackendConfig(block_size=32, gpu_memory_utilization=0.8,
                                enable_chunked_prefill=False)
        assert cfg.block_size == 32
        assert cfg.gpu_memory_utilization == 0.8
        assert cfg.enable_chunked_prefill is False


class TestInferenceConfigDefaults:
    """InferenceConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = InferenceConfig()
        assert cfg.default_backend == "vllm"
        assert cfg.defaults["temperature"] == 0.7
        assert cfg.defaults["top_p"] == 0.9
        assert cfg.defaults["top_k"] == 50
        assert cfg.defaults["max_tokens"] == 2048

    def test_setup_backends_empty_dict_returns_default_vllm(self):
        """field_validator: 空字典应返回默认vLLM后端"""
        cfg = InferenceConfig(backends={})
        assert "vllm" in cfg.backends
        assert isinstance(cfg.backends["vllm"], InferenceBackendConfig)

    def test_setup_backends_preserves_explicit_config(self):
        """field_validator: 显式传入后端配置应被保留"""
        custom_backend = InferenceBackendConfig(backend_type="tgi", max_model_len=4096)
        cfg = InferenceConfig(backends={"tgi": custom_backend})
        assert "tgi" in cfg.backends
        assert cfg.backends["tgi"].max_model_len == 4096

    def test_custom_defaults(self):
        cfg = InferenceConfig(defaults={"temperature": 0.5, "top_p": 1.0, "max_tokens": 128})
        assert cfg.defaults["temperature"] == 0.5
        assert cfg.defaults["top_p"] == 1.0
        assert cfg.defaults["max_tokens"] == 128

    def test_custom_backend_name(self):
        cfg = InferenceConfig(default_backend="huggingface")
        assert cfg.default_backend == "huggingface"


class TestRedisConfigDefaults:
    """RedisConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = RedisConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0
        assert cfg.password is None
        assert cfg.pool_size == 20
        assert cfg.socket_timeout == 5
        assert cfg.socket_connect_timeout == 5

    def test_custom(self):
        cfg = RedisConfig(host="redis.prod", port=6380, db=1, password="secret",
                          pool_size=50, socket_timeout=10)
        assert cfg.host == "redis.prod"
        assert cfg.port == 6380
        assert cfg.db == 1
        assert cfg.password == "secret"
        assert cfg.pool_size == 50
        assert cfg.socket_timeout == 10

    def test_password_none_and_empty(self):
        assert RedisConfig(password=None).password is None
        assert RedisConfig(password="").password == ""


class TestStorageConfigDefaults:
    """StorageConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = StorageConfig()
        assert isinstance(cfg.redis, RedisConfig)
        assert cfg.redis.host == "localhost"
        assert cfg.model_registry_type == "filesystem"
        assert cfg.model_registry_base_path == "/models"


class TestMonitoringConfigDefaults:
    """MonitoringConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = MonitoringConfig()
        assert cfg.enabled is True
        assert cfg.metrics_port == 9090
        assert cfg.metrics_path == "/metrics"
        assert cfg.health_check_path == "/health"
        assert cfg.health_check_detailed is True


class TestModelConfigDefaults:
    """ModelConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = ModelConfig()
        assert cfg.allowed_download_sources == ["hf://", "modelscope://"]
        assert cfg.cache_dir == "/root/.cache/huggingface"
        assert cfg.download_timeout_seconds == 3600


class TestAppConfigDefaults:
    """AppConfig 默认值验证"""

    def test_all_defaults(self):
        cfg = AppConfig()
        assert cfg.name == "QuantumFlow"
        assert cfg.version == "1.0.0"
        assert cfg.environment == "development"
        assert cfg.log_level == "INFO"


# ==================== QuantumFlowConfig 根配置测试 ====================


class TestQuantumFlowConfig:
    """QuantumFlowConfig 集成测试"""

    def test_default_config_all_sections_exist(self):
        """[核心功能] 默认配置必须包含所有子配置节"""
        cfg = QuantumFlowConfig()
        assert isinstance(cfg.app, AppConfig)
        assert isinstance(cfg.api, APIConfig)
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert isinstance(cfg.cluster, ClusterConfig)
        assert isinstance(cfg.worker, WorkerConfig)
        assert isinstance(cfg.inference, InferenceConfig)
        assert isinstance(cfg.storage, StorageConfig)
        assert isinstance(cfg.monitoring, MonitoringConfig)
        assert isinstance(cfg.models, ModelConfig)

    def test_to_dict_returns_all_sections(self):
        """[核心功能] to_dict 必须返回所有配置节"""
        cfg = QuantumFlowConfig()
        d = cfg.to_dict()
        for key in ("app", "api", "scheduler", "cluster", "worker",
                    "inference", "storage", "monitoring", "models"):
            assert key in d, f"to_dict 缺少 key: {key}"

    def test_to_yaml_writes_valid_yaml(self):
        """[核心功能] to_yaml 必须写出有效的YAML文件"""
        cfg = QuantumFlowConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp_path = f.name
        try:
            cfg.to_yaml(tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data is not None, "YAML 文件解析结果不应为 None"
            assert "app" in data
            assert "api" in data
        finally:
            os.unlink(tmp_path)

    def test_to_yaml_creates_parent_directories(self):
        """[边界用例] to_yaml 应自动创建父目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "sub", "nested", "config.yaml")
            cfg = QuantumFlowConfig()
            cfg.to_yaml(nested_path)
            assert os.path.exists(nested_path), "嵌套路径应被创建"

    def test_nested_customization(self):
        cfg = QuantumFlowConfig(
            api=APIConfig(port=9090),
            scheduler=SchedulerConfig(strategy_default="gang"),
            storage=StorageConfig(redis=RedisConfig(host="redis.internal")),
        )
        assert cfg.api.port == 9090
        assert cfg.scheduler.strategy_default == "gang"
        assert cfg.storage.redis.host == "redis.internal"


# ==================== from_file 测试 ====================


class TestQuantumFlowConfigFromFile:
    """from_file YAML加载测试"""

    def test_from_file_loads_valid_yaml(self):
        """[核心功能] 从有效YAML文件加载配置"""
        yaml_content = """
app:
  name: TestApp
  version: "2.0.0"
api:
  port: 9999
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            cfg = QuantumFlowConfig.from_file(tmp_path)
            assert cfg.app.name == "TestApp"
            assert cfg.app.version == "2.0.0"
            assert cfg.api.port == 9999
            assert cfg.api.host == "0.0.0.0"  # 未指定字段使用默认值
        finally:
            os.unlink(tmp_path)

    def test_from_file_returns_defaults_when_file_not_found(self):
        """[错误处理] 文件不存在时返回默认配置"""
        cfg = QuantumFlowConfig.from_file("/nonexistent/path/config.yaml")
        assert cfg.app.name == "QuantumFlow"
        assert cfg.api.port == 8000

    def test_from_file_with_empty_file_returns_defaults(self):
        """[边界用例] 空YAML文件返回默认配置"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write("")
            tmp_path = f.name
        try:
            cfg = QuantumFlowConfig.from_file(tmp_path)
            assert cfg.app.name == "QuantumFlow"
        finally:
            os.unlink(tmp_path)

    def test_from_file_with_nested_sections(self):
        """[核心功能] 加载嵌套YAML配置节"""
        yaml_content = """
storage:
  redis:
    host: redis.internal
    port: 6380
    db: 2
inference:
  defaults:
    temperature: 0.3
    max_tokens: 1024
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            cfg = QuantumFlowConfig.from_file(tmp_path)
            assert cfg.storage.redis.host == "redis.internal"
            assert cfg.storage.redis.port == 6380
            assert cfg.storage.redis.db == 2
            assert cfg.inference.defaults["temperature"] == 0.3
            assert cfg.inference.defaults["max_tokens"] == 1024
        finally:
            os.unlink(tmp_path)

    def test_from_file_overrides_only_specified_fields(self):
        """[核心功能] 部分覆盖只影响指定字段"""
        yaml_content = """
scheduler:
  loop_interval_ms: 200
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            cfg = QuantumFlowConfig.from_file(tmp_path)
            assert cfg.scheduler.loop_interval_ms == 200
            assert cfg.scheduler.queue_max_size == 10000
            assert cfg.scheduler.enabled is True
        finally:
            os.unlink(tmp_path)

    def test_from_file_with_invalid_type_raises(self):
        """[异常场景] 无效字段类型应抛出ValidationError"""
        yaml_content = """
api:
  port: "not_an_integer"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            with pytest.raises(Exception):
                QuantumFlowConfig.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_from_file_accepts_pathlib_path(self):
        """[边界用例] 接受 pathlib.Path 类型"""
        yaml_content = """
app:
  environment: staging
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            cfg = QuantumFlowConfig.from_file(Path(tmp_path))
            assert cfg.app.environment == "staging"
        finally:
            os.unlink(tmp_path)

    def test_yaml_roundtrip(self):
        """[核心功能] YAML 写入后读取应保持一致"""
        cfg1 = QuantumFlowConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp_path = f.name
        try:
            cfg1.to_yaml(tmp_path)
            cfg2 = QuantumFlowConfig.from_file(tmp_path)
            assert cfg2.app.name == cfg1.app.name
            assert cfg2.api.port == cfg1.api.port
            assert cfg2.scheduler.loop_interval_ms == cfg1.scheduler.loop_interval_ms
        finally:
            os.unlink(tmp_path)


# ==================== 全局配置函数测试 ====================


class TestGlobalConfigFunctions:
    """load_config / get_config / reload_config / get_default_config 测试"""

    def test_get_config_returns_consistent_instance(self):
        """[核心功能] get_config 应返回相同实例"""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_default_config_returns_defaults(self):
        """[核心功能] get_default_config 返回默认配置"""
        cfg = get_default_config()
        assert cfg.app.name == "QuantumFlow"
        assert cfg.app.version == "1.0.0"

    def test_get_default_config_is_cached(self):
        """[核心功能] get_default_config 使用LRU缓存, 返回同一实例"""
        cfg1 = get_default_config()
        cfg2 = get_default_config()
        assert cfg1 is cfg2

    def test_load_config_with_specific_file(self):
        """[核心功能] load_config 加载指定文件"""
        yaml_content = """
app:
  name: LoadedApp
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            cfg = load_config(config_file=tmp_path)
            assert cfg.app.name == "LoadedApp"
        finally:
            os.unlink(tmp_path)

    def test_load_config_with_nonexistent_file_falls_back(self):
        """[错误处理] 文件不存在时回退到默认配置"""
        cfg = load_config(config_file="/nonexistent/path.yaml")
        assert cfg.app.name == "QuantumFlow"

    def test_load_config_uses_qf_config_file_env(self, monkeypatch):
        """[核心功能] config_file=None 时使用 QF_CONFIG_FILE 环境变量"""
        yaml_content = """
app:
  name: EnvApp
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            monkeypatch.setenv("QF_CONFIG_FILE", tmp_path)
            cfg = load_config()
            assert cfg.app.name == "EnvApp"
        finally:
            os.unlink(tmp_path)

    def test_load_config_with_environment_parameter(self):
        """[核心功能] environment 参数决定配置文件路径"""
        yaml_content = "app:\n  environment: staging\n"
        staging_dir = tempfile.mkdtemp()
        configs_dir = os.path.join(staging_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        staging_file = os.path.join(configs_dir, "staging.yaml")
        try:
            with open(staging_file, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            original_cwd = os.getcwd()
            os.chdir(staging_dir)
            try:
                cfg = load_config(environment="staging")
                assert cfg.app.environment == "staging"
            finally:
                os.chdir(original_cwd)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def test_load_config_no_args_uses_qf_environment_env(self, monkeypatch):
        """[核心功能] 无参数时使用 QF_ENVIRONMENT 决定文件路径"""
        monkeypatch.setenv("QF_ENVIRONMENT", "production")
        monkeypatch.setenv("QF_CONFIG_FILE", "/nonexistent-file.yaml")
        cfg = load_config()
        assert cfg.app.name == "QuantumFlow"

    def test_load_config_no_config_file_env_falls_back(self, monkeypatch):
        """[边界用例] 无 QF_CONFIG_FILE 时使用默认路径"""
        monkeypatch.delenv("QF_CONFIG_FILE", raising=False)
        monkeypatch.setenv("QF_ENVIRONMENT", "development")
        cfg = load_config()
        assert isinstance(cfg, QuantumFlowConfig)

    def test_reload_config_returns_config(self):
        """[核心功能] reload_config 返回 QuantumFlowConfig"""
        cfg = reload_config()
        assert isinstance(cfg, QuantumFlowConfig)


# ==================== from_env 测试 ====================


class TestQuantumFlowConfigFromEnv:
    """from_env 环境变量加载测试"""

    def test_from_env_returns_default_config(self):
        """[核心功能] from_env 当前返回默认配置（stub）"""
        cfg = QuantumFlowConfig.from_env()
        assert isinstance(cfg, QuantumFlowConfig)
        assert cfg.app.name == "QuantumFlow"


# ==================== 类型验证测试 ====================


class TestConfigFieldTypes:
    """字段类型验证"""

    def test_rate_limit_burst_is_int(self):
        assert isinstance(APIConfig().rate_limit_burst, int)

    def test_loop_interval_is_positive_int(self):
        cfg = SchedulerConfig()
        assert isinstance(cfg.loop_interval_ms, int)
        assert cfg.loop_interval_ms > 0

    def test_gpu_memory_utilization_is_float_in_range(self):
        cfg = InferenceBackendConfig()
        assert isinstance(cfg.default_gpu_memory_utilization, float)
        assert 0.0 < cfg.default_gpu_memory_utilization <= 1.0

    def test_cors_origins_is_list_of_str(self):
        cfg = APIConfig()
        assert isinstance(cfg.cors_origins, list)
        assert all(isinstance(o, str) for o in cfg.cors_origins)

    def test_node_labels_is_list_of_str(self):
        cfg = ClusterConfig()
        assert isinstance(cfg.node_labels, list)
        assert all(isinstance(l, str) for l in cfg.node_labels)
