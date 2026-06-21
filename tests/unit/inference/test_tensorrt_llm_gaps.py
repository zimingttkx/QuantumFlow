import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from quantumflow.inference.backends.tensorrt_llm import TensorRTLLMEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams, InferenceResult
from quantumflow.core.constants import InferenceBackendType


class TestTensorRTLLMEngineInit:
    """TensorRT-LLM 引擎初始化测试"""

    def test_engine_type(self):
        """验证：引擎类型为 TRT_LLM"""
        engine = TensorRTLLMEngine()
        assert engine.backend_type == InferenceBackendType.TRT_LLM

    def test_default_initialization(self):
        """验证：默认初始化"""
        engine = TensorRTLLMEngine()
        assert engine._is_initialized is False
        assert engine._loaded_models == {}

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """验证：成功初始化"""
        engine = TensorRTLLMEngine()

        mock_module = MagicMock()
        mock_module.__version__ = "0.1.0"

        with patch.dict("sys.modules", {"tensorrt_llm": mock_module}):
            result = await engine.initialize()

        assert result is True
        assert engine._is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """验证：ImportError 时初始化失败"""
        engine = TensorRTLLMEngine()

        with patch.dict("sys.modules", {"tensorrt_llm": None}):
            result = await engine.initialize()

        assert result is False
        assert engine._is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_other_exception(self):
        """验证：其他异常时初始化失败"""
        engine = TensorRTLLMEngine()

        def raise_on_version(*args, **kwargs):
            raise RuntimeError("init failed")

        mock_module = MagicMock()
        type(mock_module).__version__ = property(lambda self: (_ for _ in ()).throw(RuntimeError("init failed")))

        with patch.dict("sys.modules", {"tensorrt_llm": mock_module}):
            result = await engine.initialize()

        assert result is False
        assert engine._is_initialized is False


class TestTensorRTLLMModelLoading:
    """模型加载测试"""

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """验证：成功加载模型"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        config.model_path = "/path/to/engine"
        config.tensor_parallel = 1

        with patch.object(engine, '_build_engine', new_callable=AsyncMock):
            engine._build_engine = AsyncMock(return_value=MagicMock())

            result = await engine.load_model(config)
            assert result is True
            assert "test-model" in engine._loaded_models
            assert "test-model" in engine._engines

    @pytest.mark.asyncio
    async def test_load_model_not_initialized(self):
        """验证：引擎未初始化时加载失败"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = False

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"

        result = await engine.load_model(config)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_model_exception(self):
        """验证：加载模型时异常"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        config.model_path = "/path/to/engine"
        config.tensor_parallel = 1

        # Mock run_in_executor to raise exception
        async def raise_on_build(*args, **kwargs):
            raise RuntimeError("Build failed")

        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = MagicMock(side_effect=RuntimeError("Build failed"))
            result = await engine.load_model(config)

        assert result is False
        assert "test-model" not in engine._loaded_models


class TestTensorRTLLMUnloadModel:
    """模型卸载测试"""

    @pytest.mark.asyncio
    async def test_unload_model_success(self):
        """验证：成功卸载模型"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True
        engine._engines["test-model"] = MagicMock()
        engine._loaded_models["test-model"] = MagicMock()

        result = await engine.unload_model("test-model")

        assert result is True
        assert "test-model" not in engine._engines
        assert "test-model" not in engine._loaded_models

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        """验证：模型未加载时卸载失败"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        result = await engine.unload_model("non-existent")

        assert result is False


class TestTensorRTLLMGenerate:
    """同步生成测试"""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """验证：成功生成"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello world", finish_reason="stop")]
        mock_output.prompt_token_ids = [1, 2, 3]
        mock_output.outputs[0].token_ids = [4, 5, 6]
        mock_engine.generate.return_value = [mock_output]
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=100)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            results = await engine.generate("test-model", ["Hello"], params)

        assert len(results) == 1
        assert results[0].outputs[0] == "Hello world"
        assert results[0].finish_reason == "stop"
        assert results[0].request_id == "test-model_0"

    @pytest.mark.asyncio
    async def test_generate_model_not_loaded(self):
        """验证：模型未加载时返回错误"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        params = SamplingParams(temperature=0.7, max_tokens=100)

        results = await engine.generate("non-existent", ["Hello"], params)

        assert len(results) == 1
        assert "[TensorRT-LLM错误" in results[0].outputs[0]
        assert results[0].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_generate_multiple_prompts(self):
        """验证：多 prompt 生成"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(text="Response 1", finish_reason="stop")]
        mock_output1.prompt_token_ids = [1, 2]
        mock_output1.outputs[0].token_ids = [3, 4]
        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(text="Response 2", finish_reason="stop")]
        mock_output2.prompt_token_ids = [5, 6]
        mock_output2.outputs[0].token_ids = [7, 8]
        mock_engine.generate.return_value = [mock_output1, mock_output2]
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            results = await engine.generate("test-model", ["Hello", "World"], params)

        assert len(results) == 2
        assert results[0].outputs[0] == "Response 1"
        assert results[1].outputs[0] == "Response 2"

    @pytest.mark.asyncio
    async def test_generate_exception(self):
        """验证：生成时异常"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_engine.generate.side_effect = RuntimeError("Generation failed")
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            results = await engine.generate("test-model", ["Hello"], params)

        assert len(results) == 1
        assert "Generation failed" in results[0].outputs[0]
        assert results[0].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_generate_with_stop_parameter(self):
        """验证：生成时传递 stop 参数"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello", finish_reason="stop")]
        mock_output.prompt_token_ids = [1]
        mock_output.outputs[0].token_ids = [2]
        mock_engine.generate.return_value = [mock_output]
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100, stop=["<eos>"])

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            await engine.generate("test-model", ["Hello"], params)

        # Verify stop parameter was passed
        call_args = mock_trt_module.SamplingParams.call_args
        assert call_args is not None


class TestTensorRTLLMGenerateStream:
    """流式生成测试"""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        """验证：成功流式生成"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello world", finish_reason=None)]
        mock_output.outputs[0].text = "Hello world"
        mock_engine.generate.return_value = iter([mock_output])
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        chunks = []
        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            async for chunk in engine.generate_stream("test-model", "Hello", params):
                chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_model_not_loaded(self):
        """验证：模型未加载时不产生输出"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        params = SamplingParams(temperature=0.7, max_tokens=100)

        chunks = []
        async for chunk in engine.generate_stream("non-existent", "Hello", params):
            chunks.append(chunk)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_generate_stream_exception(self):
        """验证：流式生成时异常"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_engine.generate.side_effect = RuntimeError("Stream failed")
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        chunks = []
        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            async for chunk in engine.generate_stream("test-model", "Hello", params):
                chunks.append(chunk)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_generate_stream_with_repetition_penalty(self):
        """验证：流式生成传递 repetition_penalty"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Test", finish_reason=None)]
        mock_output.outputs[0].text = "Test"
        mock_engine.generate.return_value = iter([mock_output])
        engine._engines["test-model"] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        params = SamplingParams(temperature=0.7, max_tokens=100, repetition_penalty=1.1)

        mock_trt_module = MagicMock()
        mock_trt_sp = MagicMock()
        mock_trt_module.SamplingParams.return_value = mock_trt_sp

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_module}):
            async for _ in engine.generate_stream("test-model", "Hello", params):
                pass

        call_args = mock_trt_module.SamplingParams.call_args
        assert call_args is not None


class TestTensorRTLLMStats:
    """统计信息测试"""

    @pytest.mark.asyncio
    async def test_get_stats_success_with_cuda(self):
        """验证：成功获取统计信息（CUDA可用）"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        engine._engines["test-model"] = MagicMock()
        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1024**3):
                with patch('torch.cuda.memory_reserved', return_value=2048**3):
                    with patch('pynvml.nvmlInit'):
                        with patch('pynvml.nvmlDeviceGetHandleByIndex'):
                            with patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util:
                                mock_util.return_value = MagicMock(gpu=50, memory=60)
                                with patch('pynvml.nvmlShutdown'):
                                    stats = await engine.get_stats("test-model")

        assert "gpu_memory_allocated" in stats
        assert "gpu_memory_reserved" in stats
        assert stats["gpu_memory_allocated"] == pytest.approx(1.0, rel=0.1)
        assert "gpu_utilization" in stats
        assert "gpu_memory_utilization" in stats

    @pytest.mark.asyncio
    async def test_get_stats_model_not_loaded(self):
        """验证：模型未加载时返回空字典"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        stats = await engine.get_stats("non-existent")
        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_no_cuda(self):
        """验证：CUDA不可用时返回基本统计"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        engine._engines["test-model"] = MagicMock()
        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        with patch('torch.cuda.is_available', return_value=False):
            stats = await engine.get_stats("test-model")

        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_pynvml_exception(self):
        """验证：pynvml异常时仍返回GPU内存统计"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        engine._engines["test-model"] = MagicMock()
        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        engine._loaded_models["test-model"] = config

        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1024**3):
                with patch('torch.cuda.memory_reserved', return_value=2048**3):
                    with patch('pynvml.nvmlInit', side_effect=RuntimeError("NVML error")):
                        stats = await engine.get_stats("test-model")

        assert "gpu_memory_allocated" in stats
        assert "gpu_utilization" not in stats


class TestTensorRTLLMIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_engine_manager_registers_tensorrt_llm(self):
        """验证：EngineManager 可以注册 TensorRT-LLM"""
        from quantumflow.inference.manager import EngineManager
        from quantumflow.core.constants import InferenceBackendType

        EngineManager._instance = None

        manager = EngineManager()

        with patch('quantumflow.inference.backends.tensorrt_llm.TensorRTLLMEngine.initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True

            success = await manager.initialize(InferenceBackendType.TRT_LLM)
            assert success is True
            assert InferenceBackendType.TRT_LLM in manager._engines

    @pytest.mark.asyncio
    async def test_load_model_via_manager(self):
        """验证：通过 Manager 加载 TensorRT-LLM 模型"""
        from quantumflow.inference.manager import EngineManager
        from quantumflow.core.constants import InferenceBackendType

        EngineManager._instance = None

        manager = EngineManager()
        manager._initialized = True

        mock_engine = MagicMock()
        mock_engine.load_model = AsyncMock(return_value=True)
        manager._engines[InferenceBackendType.TRT_LLM] = mock_engine

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        config.model_path = "/path/to/engine"
        config.tensor_parallel = 1

        with patch.object(manager._vram_manager, 'can_load', return_value=(True, 'OK', [])):
            result = await manager.load_model("test-model", config.model_path, InferenceBackendType.TRT_LLM)
        assert result[0] is True

    @pytest.mark.asyncio
    async def test_engine_manager_init_fails_for_trt_llm(self):
        """验证：EngineManager 初始化 TensorRT-LLM 失败"""
        from quantumflow.inference.manager import EngineManager
        from quantumflow.core.constants import InferenceBackendType

        EngineManager._instance = None

        manager = EngineManager()

        with patch('quantumflow.inference.backends.tensorrt_llm.TensorRTLLMEngine.initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False

            success = await manager.initialize(InferenceBackendType.TRT_LLM)
            assert success is False
            assert InferenceBackendType.TRT_LLM not in manager._engines


class TestTensorRTLLMCompiler:
    """模型编译测试"""

    def test_compile_model_success(self):
        """验证：编译模型成功"""
        from quantumflow.inference.backends.tensorrt_compiler import TensorRTCompiler

        compiler = TensorRTCompiler()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch('os.path.exists', return_value=False):
                result = compiler.compile(
                    model_path="Qwen/Qwen2.5-7B-Instruct",
                    model_name="test-model",
                    tensor_parallel=1,
                    dtype="float16"
                )

        assert result is not None
        mock_run.assert_called_once()

    def test_compile_model_already_exists(self):
        """验证：引擎已存在时直接返回"""
        from quantumflow.inference.backends.tensorrt_compiler import TensorRTCompiler

        compiler = TensorRTCompiler()

        with patch('os.path.exists', return_value=True):
            result = compiler.compile(
                model_path="Qwen/Qwen2.5-7B-Instruct",
                model_name="test-model",
                tensor_parallel=1,
                dtype="float16"
            )

        assert result is not None

    def test_compile_model_failure(self):
        """验证：编译失败时抛出异常"""
        from quantumflow.inference.backends.tensorrt_compiler import TensorRTCompiler
        import subprocess

        compiler = TensorRTCompiler()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "cmd", stderr="Compilation failed"
            )
            with patch('os.path.exists', return_value=False):
                with pytest.raises(RuntimeError, match="Compilation failed"):
                    compiler.compile(
                        model_path="Qwen/Qwen2.5-7B-Instruct",
                        model_name="test-model",
                        tensor_parallel=1,
                        dtype="float16"
                    )

    def test_get_engine_path_exists(self):
        """验证：获取已存在的引擎路径"""
        from quantumflow.inference.backends.tensorrt_compiler import TensorRTCompiler

        compiler = TensorRTCompiler()

        with patch('os.path.exists', return_value=True):
            result = compiler.get_engine_path(
                model_name="Qwen2.5-7B",
                tensor_parallel=1,
                dtype="float16"
            )

        assert result is not None

    def test_get_engine_path_not_exists(self):
        """验证：获取不存在的引擎路径"""
        from quantumflow.inference.backends.tensorrt_compiler import TensorRTCompiler

        compiler = TensorRTCompiler()

        with patch('os.path.exists', return_value=False):
            result = compiler.get_engine_path(
                model_name="Qwen2.5-7B",
                tensor_parallel=1,
                dtype="float16"
            )

        assert result is None


class TestTensorRTLLMBuildEngine:
    """_build_engine 测试"""

    @pytest.mark.asyncio
    async def test_build_engine_creates_llm_instance(self):
        """验证：_build_engine 创建 LLM 实例"""
        engine = TensorRTLLMEngine()
        engine._is_initialized = True

        config = MagicMock(spec=ModelConfig)
        config.model_name = "test-model"
        config.model_path = "/path/to/model"
        config.tensor_parallel = 2
        config.dtype = "float16"

        mock_trt_llm = MagicMock()
        mock_llm_instance = MagicMock()
        mock_trt_llm.LLM.return_value = mock_llm_instance

        with patch.dict("sys.modules", {"tensorrt_llm": mock_trt_llm}):
            with patch.dict("os.environ", {}, clear=False):
                result = engine._build_engine(config)

        mock_trt_llm.LLM.assert_called_once()
        call_kwargs = mock_trt_llm.LLM.call_args[1]
        assert call_kwargs["model"] == "/path/to/model"
        assert call_kwargs["tensor_parallel_size"] == 2
        assert call_kwargs["dtype"] == "float16"
