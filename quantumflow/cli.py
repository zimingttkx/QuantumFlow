"""CLI命令行工具"""

import asyncio
import time

import click
import httpx
import structlog
from rich.console import Console
from rich.table import Table

from quantumflow import __version__
from quantumflow.utils.config import load_config
from quantumflow.utils.logging import setup_logging

logger = structlog.get_logger()
console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", type=click.Path(), help="配置文件路径")
@click.option("--log-level", "-l", default="INFO", help="日志级别")
@click.pass_context
def cli(ctx, config, log_level):
    """QuantumFlow - 分布式大模型推理平台"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config) if config else {}
    setup_logging(log_level=log_level)


@cli.command()
def version():
    """显示版本信息"""
    console.print(f"[bold blue]QuantumFlow[/bold blue] [green]{__version__}[/green]")


@cli.command()
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8000, help="监听端口")
@click.option("--workers", default=1, help="工作进程数")
@click.option("--reload", is_flag=True, help="启用热重载")
@click.option("--enable-grpc", is_flag=True, help="同时启动gRPC服务器")
@click.option("--grpc-port", default=50051, help="gRPC服务器端口")
def serve(host, port, workers, reload, enable_grpc, grpc_port):
    """启动API服务器"""
    import uvicorn

    from quantumflow.api.server import create_app

    console.print("[bold green]启动QuantumFlow API服务器[/bold green]")
    console.print(f"  地址: {host}:{port}")
    console.print(f"  工作进程: {workers}")
    if enable_grpc:
        console.print(f"  gRPC: {host}:{grpc_port} (已启用)")
    else:
        console.print(f"  gRPC: {grpc_port} (默认禁用，使用 --enable-grpc 启用)")

    app = create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


@cli.command()
@click.option("--controller-url", default="http://localhost:8000", help="Controller URL")
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8080, help="监听端口")
@click.option(
    "--backend", default="vllm", type=click.Choice(["vllm", "tgi", "sglang"]), help="推理后端"
)
@click.option("--tgi-url", default="http://localhost:8080", help="TGI服务URL")
@click.option("--sglang-url", default="http://localhost:30000", help="SGLang服务URL")
def worker(controller_url, host, port, backend, tgi_url, sglang_url):
    """启动Worker节点"""
    from quantumflow.inference.backends import SGLangEngine, TGIEngine, VLLMEngine
    from quantumflow.worker import WorkerConfig, WorkerNode

    console.print("[bold green]启动QuantumFlow Worker节点[/bold green]")
    console.print(f"  Controller: {controller_url}")
    console.print(f"  地址: {host}:{port}")
    console.print(f"  后端: {backend}")

    # 创建引擎
    if backend == "vllm":
        engine = VLLMEngine()
    elif backend == "tgi":
        engine = TGIEngine(base_url=tgi_url)
    elif backend == "sglang":
        engine = SGLangEngine(base_url=sglang_url)
    else:
        engine = None

    # 创建Worker
    config = WorkerConfig(
        node_id=f"worker-{port}",
        host=host,
        port=port,
    )
    worker = WorkerNode(config=config, engine=engine)

    async def run():
        await worker.start(controller_url=controller_url)
        console.print("[green]Worker已启动，按Ctrl+C停止[/green]")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]正在停止Worker...[/yellow]")
            await worker.stop()

    asyncio.run(run())


@cli.command()
@click.argument("model")
@click.option(
    "--backend",
    "-b",
    default="huggingface",
    type=click.Choice(["huggingface", "vllm", "tgi", "sglang"]),
    help="推理后端",
)
@click.option("--tensor-parallel", "-tp", default=1, help="张量并行度")
@click.option("--gpu-memory", "-m", default=0.8, help="GPU显存利用率")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def load(model, backend, tensor_parallel, gpu_memory, url):
    """加载模型"""

    async def do_load():
        async with httpx.AsyncClient() as client:
            try:
                console.print(f"[cyan]正在加载模型 {model}...[/cyan]")
                response = await client.post(
                    f"{url}/api/v1/models/load",
                    json={
                        "model": model,
                        "backend": backend,
                        "tensor_parallel": tensor_parallel,
                        "gpu_memory_utilization": gpu_memory,
                    },
                    timeout=300.0,
                )
                if response.status_code in [200, 201]:
                    data = response.json()
                    console.print(f"[green]✓ 模型 {model} 加载成功[/green]")
                    console.print(f"  状态: {data.get('status')}")
                else:
                    console.print(f"[red]✗ 加载失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_load())


@cli.command()
@click.argument("model")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def unload(model, url):
    """卸载模型"""

    async def do_unload():
        async with httpx.AsyncClient() as client:
            try:
                console.print(f"[cyan]正在卸载模型 {model}...[/cyan]")
                response = await client.post(
                    f"{url}/api/v1/models/unload",
                    json={"model": model},
                    timeout=30.0,
                )
                if response.status_code == 200:
                    response.json()
                    console.print(f"[green]✓ 模型 {model} 卸载成功[/green]")
                else:
                    console.print(f"[red]✗ 卸载失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_unload())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
@click.option("--watch", "-w", is_flag=True, help="持续监控")
@click.option("--interval", default=5, help="刷新间隔（秒）")
def status(url, watch, interval):
    """查看集群状态"""

    async def do_status():
        while True:
            async with httpx.AsyncClient() as client:
                try:
                    # 获取集群状态
                    response = await client.get(f"{url}/api/v1/cluster/status")
                    if response.status_code == 200:
                        data = response.json()

                        console.clear()
                        console.print("[bold blue]QuantumFlow 集群状态[/bold blue]")
                        console.print()

                        # 创建统计表
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("指标", style="cyan")
                        table.add_column("值", style="green")

                        table.add_row("总节点数", str(data.get("total_nodes", 0)))
                        table.add_row("健康节点", str(data.get("healthy_nodes", 0)))
                        table.add_row("总GPU数", str(data.get("total_gpus", 0)))
                        table.add_row("可用GPU", str(data.get("available_gpus", 0)))
                        table.add_row("已加载模型", str(data.get("active_models", 0)))

                        console.print(table)
                        console.print()

                        if not watch:
                            break

                        await asyncio.sleep(interval)

                except Exception as e:
                    console.print(f"[red]✗ 连接失败: {e}[/red]")
                    break

    asyncio.run(do_status())


@cli.command()
@click.argument("model")
@click.option("--prompt", "-p", default="Hello, world!", help="输入提示")
@click.option("--max-tokens", "-t", default=100, help="最大生成token数")
@click.option("--temperature", default=0.7, help="温度参数")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def generate(model, prompt, max_tokens, temperature, url):
    """测试生成"""

    async def do_generate():
        start_time = time.time()

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                console.print("[cyan]正在生成...[/cyan]")
                console.print(f"  模型: {model}")
                console.print(f"  提示: {prompt}")

                response = await client.post(
                    f"{url}/api/v1/inference/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "sampling_params": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )

                (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    console.print()
                    console.print("[bold green]生成结果:[/bold green]")
                    console.print(data.get("generated_text", ""))
                    console.print()
                    console.print(f"[dim]延迟: {data.get('latency_ms', 0):.2f}ms[/dim]")
                else:
                    console.print(f"[red]✗ 生成失败: {response.text}[/red]")

            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_generate())


@cli.command()
@click.argument("model")
@click.option("--prompt", "-p", default="Hello", help="输入提示")
@click.option("--max-tokens", "-t", default=100, help="最大生成token数")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def chat(model, prompt, max_tokens, url):
    """测试对话"""

    async def do_chat():
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                console.print("[cyan]对话中...[/cyan]")

                response = await client.post(
                    f"{url}/api/v1/inference/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        "sampling_params": {
                            "max_tokens": max_tokens,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    console.print()
                    console.print("[bold green]助手:[/bold green]")
                    console.print(data.get("generated_text", ""))
                else:
                    console.print(f"[red]✗ 对话失败: {response.text}[/red]")

            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_chat())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def models(url):
    """列出可用模型和已加载模型"""

    async def do_list():
        async with httpx.AsyncClient() as client:
            try:
                # 获取可用模型列表
                resp_available = await client.get(f"{url}/api/v1/models/list")
                # 获取已加载模型
                resp_loaded = await client.get(f"{url}/api/v1/models/status")

                if resp_available.status_code == 200:
                    data = resp_available.json()
                    available = data.get("available_models", [])
                    mappings = data.get("mappings", {})

                    loaded = []
                    if resp_loaded.status_code == 200:
                        loaded = resp_loaded.json().get("loaded_models", [])

                    console.print("[bold blue]QuantumFlow 模型[/bold blue]")
                    console.print()

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("模型名称", style="cyan")
                    table.add_column("HF Hub ID", style="dim")
                    table.add_column("状态", style="green")

                    for name in available:
                        status = (
                            "[green]已加载[/green]" if name in loaded else "[yellow]未加载[/yellow]"
                        )
                        table.add_row(name, mappings.get(name, "N/A"), status)

                    console.print(table)
                else:
                    console.print("[red]✗ 获取模型列表失败[/red]")

            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_list())


@cli.group()
def queue():
    """分布式队列管理命令"""
    pass


@queue.command("stats")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def queue_stats(url):
    """查看分布式队列统计信息"""

    async def do_stats():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{url}/api/v1/inference/queue/stats")
                if response.status_code == 200:
                    data = response.json()

                    console.print("[bold blue]分布式队列状态[/bold blue]")
                    console.print()

                    connected = data.get("connected", False)
                    if not connected:
                        console.print("[red]✗ 未连接到 Redis[/red]")
                        console.print(f"  错误: {data.get('error', 'Unknown')}")
                        return

                    queue_stats = data.get("queue_stats", {})
                    metrics = data.get("metrics", {})

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("指标", style="cyan")
                    table.add_column("值", style="green")

                    table.add_row("队列大小", str(queue_stats.get("size", 0)))
                    table.add_row("等待请求数", str(queue_stats.get("pending", 0)))
                    table.add_row("处理中请求数", str(queue_stats.get("processing", 0)))
                    table.add_row("已完成请求数", str(queue_stats.get("completed", 0)))
                    table.add_row("失败请求数", str(queue_stats.get("failed", 0)))

                    console.print(table)
                    console.print()

                    # 显示性能指标
                    perf_table = Table(show_header=True, header_style="bold magenta")
                    perf_table.add_column("指标", style="cyan")
                    perf_table.add_column("值", style="yellow")

                    perf_table.add_row("入队速率", f"{metrics.get('enqueue_rate', 0):.2f}/s")
                    perf_table.add_row("出队速率", f"{metrics.get('dequeue_rate', 0):.2f}/s")
                    perf_table.add_row(
                        "平均等待时间", f"{metrics.get('avg_wait_time_ms', 0):.0f}ms"
                    )
                    perf_table.add_row("成功率", f"{metrics.get('success_rate', 0):.1%}")

                    console.print(perf_table)
                else:
                    console.print(f"[red]✗ 获取队列状态失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_stats())


@queue.command("submit")
@click.argument("model")
@click.option("--prompt", "-p", required=True, help="输入提示")
@click.option("--max-tokens", "-t", default=100, help="最大生成token数")
@click.option("--temperature", default=0.7, help="温度参数")
@click.option("--priority", default=5, type=int, help="优先级 (1-10, 越高越优先)")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
@click.option("--wait", "-w", is_flag=True, help="等待结果")
@click.option("--timeout", default=30000, help="等待超时(毫秒)")
def queue_submit(model, prompt, max_tokens, temperature, priority, url, wait, timeout):
    """提交推理请求到分布式队列"""

    async def do_submit():
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                console.print("[cyan]正在提交请求到队列...[/cyan]")
                console.print(f"  模型: {model}")
                console.print(f"  提示: {prompt[:50]}...")

                response = await client.post(
                    f"{url}/api/v1/inference/submit",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "sampling_params": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        "priority": priority,
                    },
                    params={"wait_for_result": wait, "timeout_ms": timeout},
                )

                if response.status_code == 200:
                    data = response.json()
                    if wait:
                        result = data.get("result", {})
                        console.print()
                        console.print("[bold green]生成结果:[/bold green]")
                        console.print(result.get("result", {}).get("generated_text", ""))
                        console.print()
                        console.print(f"[dim]状态: {data.get('status')}[/dim]")
                    else:
                        console.print()
                        console.print(f"[green]✓ 请求已提交: {data.get('request_id')}[/green]")
                        console.print(f"[dim]状态: {data.get('status')}[/dim]")
                        console.print(
                            f"[dim]使用 'qf queue result {data.get('request_id')}' 查询结果[/dim]"
                        )
                else:
                    console.print(f"[red]✗ 提交失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_submit())


@queue.command("result")
@click.argument("request_id")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def queue_result(request_id, url):
    """查询分布式队列请求结果"""

    async def do_result():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{url}/api/v1/inference/result/{request_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")

                    console.print(f"[bold]请求ID:[/bold] {request_id}")
                    console.print(f"[bold]状态:[/bold] {status}")

                    if status == "success":
                        result = data.get("result", {})
                        console.print()
                        console.print("[bold green]结果:[/bold green]")
                        console.print(result.get("result", {}).get("generated_text", ""))
                    elif status == "pending":
                        console.print("[yellow]请求正在处理中...[/yellow]")
                    elif status == "error":
                        console.print(
                            f"[red]错误: {data.get('result', {}).get('reason', 'Unknown')}[/red]"
                        )
                    elif status == "timeout":
                        console.print("[yellow]等待结果超时[/yellow]")
                elif response.status_code == 404:
                    console.print(f"[red]✗ 请求不存在: {request_id}[/red]")
                else:
                    console.print(f"[red]✗ 查询失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_result())


@cli.group()
def worker():
    """Worker节点管理命令"""
    pass


@worker.command("register")
@click.argument("node_id")
@click.option("--host", default="localhost", help="Worker主机地址")
@click.option("--port", default=8080, type=int, help="Worker端口")
@click.option("--url", default="http://localhost:8000", help="Controller URL")
def worker_register(node_id, host, port, url):
    """注册Worker节点到Controller"""

    async def do_register():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                console.print(f"[cyan]正在注册 Worker {node_id}...[/cyan]")
                console.print(f"  地址: {host}:{port}")

                response = await client.post(
                    f"{url}/api/v1/cluster/heartbeat",
                    json={
                        "node_id": node_id,
                        "hostname": host,
                        "ip": host,
                        "port": port,
                        "status": "healthy",
                    },
                )

                if response.status_code == 200:
                    console.print(f"[green]✓ Worker {node_id} 注册成功[/green]")
                else:
                    console.print(f"[red]✗ 注册失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_register())


@worker.command("unregister")
@click.argument("node_id")
@click.option("--url", default="http://localhost:8000", help="Controller URL")
def worker_unregister(node_id, url):
    """从Controller注销Worker节点"""

    async def do_unregister():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                console.print(f"[cyan]正在注销 Worker {node_id}...[/cyan]")

                response = await client.delete(f"{url}/api/v1/cluster/nodes/{node_id}")

                if response.status_code == 200:
                    console.print(f"[green]✓ Worker {node_id} 注销成功[/green]")
                elif response.status_code == 404:
                    console.print(f"[yellow]Worker {node_id} 不存在[/yellow]")
                else:
                    console.print(f"[red]✗ 注销失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_unregister())


@cli.group()
def grpc():
    """gRPC服务管理命令"""
    pass


@grpc.command("serve")
@click.option("--port", default=50051, help="gRPC监听端口")
@click.option("--max-workers", default=10, help="最大工作线程数")
@click.option("--reflection/--no-reflection", default=True, help="启用gRPC Reflection")
@click.option("--enable-auth", is_flag=True, help="启用认证")
@click.option("--api-key", default="", help="API密钥(启用认证时必填)")
def grpc_serve(port, max_workers, reflection, enable_auth, api_key):
    """启动独立gRPC服务器(需配合API服务器使用)"""
    from quantumflow.grpc.server import GrpcServer

    console.print("[bold green]启动QuantumFlow gRPC服务器[/bold green]")
    console.print(f"  端口: {port}")
    console.print(f"  最大工作线程: {max_workers}")
    console.print(f"  Reflection: {'启用' if reflection else '禁用'}")

    # 创建gRPC服务器
    server = GrpcServer(
        port=port,
        max_workers=max_workers,
        reflection_enabled=reflection,
    )

    # 添加所有服务
    server.add_inference_service()
    server.add_cluster_service()
    server.add_scheduler_service()
    server.add_model_management_service()
    server.add_health_service()
    server.add_metrics_service()

    # 添加拦截器
    server.add_logging_interceptor()
    server.add_metrics_interceptor()
    server.add_rate_limit_interceptor()

    if enable_auth:
        if not api_key:
            console.print("[red]✗ 启用认证时必须提供 --api-key[/red]")
            return
        server.add_auth_interceptor(
            allowed_tokens={api_key: "cli_user"},
            bypass_methods={"/quantumflow.v1.HealthService/Check", "/quantumflow.v1.HealthService/Watch"},
        )
        console.print(f"  认证: 启用 (api_key: {api_key[:4]}***)")

    server.start()
    console.print(f"[green]✓ gRPC服务器已启动 on port {port}[/green]")
    console.print("[dim]按 Ctrl+C 停止[/dim]")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[yellow]正在停止gRPC服务器...[/yellow]")
        server.stop()
        console.print("[green]✓ gRPC服务器已停止[/green]")


@grpc.command("invoke")
@click.argument("method")
@click.argument("model")
@click.option("--prompt", "-p", default="Hello, world!", help="输入提示")
@click.option("--max-tokens", "-t", default=100, help="最大生成token数")
@click.option("--temperature", default=0.7, help="温度参数")
@click.option("--host", default="localhost", help="gRPC服务器主机")
@click.option("--port", default=50051, help="gRPC服务器端口")
def grpc_invoke(method, model, prompt, max_tokens, temperature, host, port):
    """通过gRPC调用推理服务

    METHOD: inference(同步推理) | inference_stream(流式推理) | batch(批量推理)
    MODEL: 模型名称
    """
    import grpc

    from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc

    console.print(f"[cyan]连接gRPC服务器 {host}:{port}...[/cyan]")

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = quantumflow_pb2_grpc.InferenceServiceStub(channel)

    if method == "inference":
        console.print(f"[cyan]同步推理: {model}[/cyan]")
        request = quantumflow_pb2.InferenceRequest(
            request_id=f"cli-{int(time.time())}",
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = stub.Inference(request)
        console.print()
        console.print("[bold green]结果:[/bold green]")
        console.print(response.text)
        console.print(f"[dim]状态: {quantumflow_pb2.Status.Name(response.status)}[/dim]")
        console.print(f"[dim]Token数: {response.tokens_generated}[/dim]")

    elif method == "inference_stream":
        console.print(f"[cyan]流式推理: {model}[/cyan]")
        request = quantumflow_pb2.InferenceRequest(
            request_id=f"cli-{int(time.time())}",
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print("[bold green]结果:[/bold green] ", end="")
        for chunk in stub.InferenceStream(request):
            console.print(chunk.text, end="", markup=False)
        console.print()

    elif method == "batch":
        prompts = [prompt, f"{prompt} (2)", f"{prompt} (3)"]
        console.print(f"[cyan]批量推理: {len(prompts)} 条请求[/cyan]")
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=f"cli-batch-{int(time.time())}",
            model_name=model,
            prompts=prompts,
            max_tokens=max_tokens,
        )
        response = stub.BatchInference(request)
        console.print(f"[green]批量推理完成: {len(response.results)} 条结果[/green]")
        for i, result in enumerate(response.results):
            console.print(f"  [{i+1}] {result.text[:50]}...")

    else:
        console.print(f"[red]✗ 未知方法: {method}[/red]")
        console.print("[dim]可用方法: inference | inference_stream | batch[/dim]")

    channel.close()


@grpc.command("health")
@click.option("--host", default="localhost", help="gRPC服务器主机")
@click.option("--port", default=50051, help="gRPC服务器端口")
def grpc_health(host, port):
    """检查gRPC服务器健康状态"""
    import grpc

    from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc

    console.print(f"[cyan]检查gRPC服务器健康状态 {host}:{port}...[/cyan]")

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = quantumflow_pb2_grpc.HealthServiceStub(channel)

    try:
        request = quantumflow_pb2.HealthCheckRequest()
        response = stub.Check(request)
        if response.healthy:
            console.print(f"[green]✓ 服务器健康: {response.status}[/green]")
        else:
            console.print(f"[yellow]⚠ 服务器状态: {response.status}[/yellow]")
    except grpc.RpcError as e:
        console.print(f"[red]✗ 连接失败: {e.code().name}: {e.details()}[/red]")
    finally:
        channel.close()


@grpc.command("status")
@click.option("--host", default="localhost", help="gRPC服务器主机")
@click.option("--port", default=50051, help="gRPC服务器端口")
def grpc_status(host, port):
    """获取gRPC服务器集群状态"""
    import grpc

    from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc

    console.print(f"[cyan]获取集群状态 {host}:{port}...[/cyan]")

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = quantumflow_pb2_grpc.ClusterServiceStub(channel)

    try:
        request = quantumflow_pb2.ListNodesRequest()
        response = stub.ListNodes(request)
        console.print(f"[green]集群节点数: {len(response.nodes)}[/green]")
        if response.nodes:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("节点ID", style="cyan")
            table.add_column("主机", style="yellow")
            table.add_column("端口", style="magenta")
            table.add_column("状态", style="green")
            for node in response.nodes:
                table.add_row(node.node_id, node.host, str(node.port), quantumflow_pb2.NodeStatus.Name(node.status))
            console.print(table)
    except grpc.RpcError as e:
        console.print(f"[red]✗ 连接失败: {e.code().name}: {e.details()}[/red]")
    finally:
        channel.close()


@worker.command("start")
@click.option("--node-id", default="local-worker", help="节点ID")
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8080, type=int, help="监听端口")
@click.option(
    "--backend", default="huggingface", type=click.Choice(["huggingface", "vllm"]), help="推理后端"
)
@click.option("--controller-url", default="http://localhost:8000", help="Controller URL")
def worker_start(node_id, host, port, backend, controller_url):
    """启动Worker节点（连接到Controller接收任务）"""
    from quantumflow.inference.backends import HuggingFaceEngine, VLLMEngine
    from quantumflow.worker import WorkerConfig, WorkerNode

    console.print("[bold green]启动QuantumFlow Worker节点[/bold green]")
    console.print(f"  节点ID: {node_id}")
    console.print(f"  地址: {host}:{port}")
    console.print(f"  后端: {backend}")
    console.print(f"  Controller: {controller_url}")

    # 创建推理引擎
    if backend == "vllm":
        engine = VLLMEngine()
    else:
        engine = HuggingFaceEngine()

    # 创建Worker配置
    config = WorkerConfig(
        node_id=node_id,
        host=host,
        port=port,
    )

    # 创建Worker
    worker = WorkerNode(config=config, engine=engine)

    async def run():
        await worker.start(controller_url=controller_url)
        console.print("[green]Worker已启动，按Ctrl+C停止[/green]")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]正在停止Worker...[/yellow]")
            await worker.stop()

    asyncio.run(run())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def workers(url):
    """列出已注册的Worker节点"""

    async def do_list():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{url}/api/v1/cluster/nodes")
                if response.status_code == 200:
                    nodes = response.json()

                    if not nodes:
                        console.print("[yellow]没有已注册的节点[/yellow]")
                        return

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("节点ID", style="cyan")
                    table.add_column("主机", style="yellow")
                    table.add_column("端口", style="magenta")
                    table.add_column("状态", style="green")
                    table.add_column("GPU数", style="blue")
                    table.add_column("已加载模型", style="dim")

                    for node in nodes:
                        status_color = {
                            "healthy": "[green]healthy[/green]",
                            "unhealthy": "[red]unhealthy[/red]",
                            "offline": "[dim]offline[/dim]",
                            "draining": "[yellow]draining[/yellow]",
                        }.get(node.get("status", ""), "[yellow]unknown[/yellow]")

                        models = ", ".join(node.get("loaded_models", [])) or "无"
                        if len(models) > 30:
                            models = models[:30] + "..."

                        table.add_row(
                            node.get("node_id", ""),
                            node.get("hostname", ""),
                            str(node.get("port", "")),
                            status_color,
                            str(node.get("gpu_count", 0)),
                            models,
                        )

                    console.print(table)
                else:
                    console.print("[red]✗ 获取节点列表失败[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_list())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
@click.option("--watch", "-w", is_flag=True, help="持续监控")
@click.option("--interval", default=5, help="刷新间隔（秒）")
def monitor(url, watch, interval):
    """监控集群和GPU状态"""

    async def do_monitor():
        while True:
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    # 获取集群状态
                    cluster_resp = await client.get(f"{url}/api/v1/cluster/status")
                    scheduler_resp = await client.get(f"{url}/api/v1/scheduler/status")

                    if cluster_resp.status_code == 200:
                        cluster = cluster_resp.json()
                        scheduler = (
                            scheduler_resp.json() if scheduler_resp.status_code == 200 else {}
                        )

                        console.clear()
                        console.print("[bold blue]QuantumFlow 集群监控[/bold blue]")
                        console.print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        console.print()

                        # 集群概览
                        cluster_table = Table(show_header=True, header_style="bold magenta")
                        cluster_table.add_column("指标", style="cyan")
                        cluster_table.add_column("值", style="green")

                        cluster_table.add_row("总节点数", str(cluster.get("total_nodes", 0)))
                        cluster_table.add_row("健康节点", str(cluster.get("healthy_nodes", 0)))
                        cluster_table.add_row("总GPU数", str(cluster.get("total_gpus", 0)))
                        cluster_table.add_row("可用GPU", str(cluster.get("available_gpus", 0)))
                        cluster_table.add_row("已加载模型", str(cluster.get("active_models", 0)))

                        console.print(cluster_table)
                        console.print()

                        # GPU状态
                        gpu_table = Table(show_header=True, header_style="bold cyan")
                        gpu_table.add_column("GPU", style="cyan")
                        gpu_table.add_column("利用率", style="green")
                        gpu_table.add_column("显存使用", style="yellow")
                        gpu_table.add_column("温度", style="magenta")

                        gpu_status = scheduler.get("gpu", {})
                        if gpu_status:
                            for gpu in gpu_status if isinstance(gpu_status, list) else []:
                                util = gpu.get("utilization", 0)
                                mem_used = gpu.get("memory_used_gb", 0)
                                mem_total = gpu.get("memory_total_gb", 0)
                                temp = gpu.get("temperature", 0)

                                util_str = f"{util*100:.0f}%" if util else "N/A"
                                mem_str = (
                                    f"{mem_used:.1f}/{mem_total:.1f} GB" if mem_total else "N/A"
                                )
                                temp_str = f"{temp:.0f}°C" if temp else "N/A"

                                gpu_table.add_row(
                                    gpu.get("name", f"GPU {gpu.get('gpu_id', 0)}"),
                                    util_str,
                                    mem_str,
                                    temp_str,
                                )

                        if gpu_status:
                            console.print(gpu_table)
                            console.print()

                        # VRAM状态
                        vram = scheduler.get("vram", {})
                        if vram:
                            console.print(
                                f"[bold]VRAM 可用:[/bold] {vram.get('available_vram_gb', 0):.1f} GB"
                            )
                            console.print(f"[bold]已加载模型:[/bold] {vram.get('loaded_count', 0)}")

                        if not watch:
                            break

                        await asyncio.sleep(interval)

                except Exception as e:
                    console.print(f"[red]✗ 连接失败: {e}[/red]")
                    if not watch:
                        break
                    await asyncio.sleep(interval)

    asyncio.run(do_monitor())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
@click.option("--limit", default=20, help="返回数量")
def hub(url, limit):
    """浏览 HuggingFace 热门模型"""

    async def do_hub():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{url}/api/v1/hub/trending", params={"limit": limit})
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("models", [])

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("#", style="dim")
                    table.add_column("模型ID", style="cyan")
                    table.add_column("下载量", style="green")
                    table.add_column("类型", style="yellow")

                    for i, m in enumerate(models_list, 1):
                        downloads = m.get("downloads", 0) or 0
                        dl_str = (
                            f"{downloads/1_000_000:.1f}M"
                            if downloads >= 1_000_000
                            else f"{downloads/1_000:.0f}K" if downloads >= 1_000 else str(downloads)
                        )
                        table.add_row(
                            str(i),
                            m.get("model_id", "")[:60],
                            dl_str,
                            m.get("pipeline_tag", "unknown"),
                        )

                    console.print(table)
                else:
                    console.print("[red]获取失败[/red]")
            except Exception as e:
                console.print(f"[red]连接失败: {e}[/red]")

    asyncio.run(do_hub())


@cli.command()
@click.argument("query")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
@click.option("--limit", default=15, help="返回数量")
def search(query, url, limit):
    """搜索 HuggingFace 模型"""

    async def do_search():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                console.print(f"[cyan]搜索 '{query}'...[/cyan]")
                resp = await client.get(
                    f"{url}/api/v1/hub/search", params={"q": query, "limit": limit}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("models", [])

                    if not models_list:
                        console.print(f"[yellow]未找到匹配 '{query}' 的模型[/yellow]")
                        return

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("#", style="dim")
                    table.add_column("模型ID", style="cyan")
                    table.add_column("下载量", style="green")
                    table.add_column("作者", style="dim")

                    for i, m in enumerate(models_list, 1):
                        downloads = m.get("downloads", 0) or 0
                        dl_str = (
                            f"{downloads/1_000_000:.1f}M"
                            if downloads >= 1_000_000
                            else f"{downloads/1_000:.0f}K" if downloads >= 1_000 else str(downloads)
                        )
                        table.add_row(
                            str(i), m.get("model_id", ""), dl_str, m.get("author", "unknown")
                        )

                    console.print(table)
                    console.print(
                        "\n[dim]使用 'python -m quantumflow.cli download <model_id>' 下载[/dim]"
                    )
                else:
                    console.print("[red]搜索失败[/red]")
            except Exception as e:
                console.print(f"[red]连接失败: {e}[/red]")

    asyncio.run(do_search())


@cli.command()
@click.argument("model_id")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def download(model_id, url):
    """从 HuggingFace 下载模型"""

    async def do_download():
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                # 验证
                console.print(f"[cyan]验证模型 '{model_id}'...[/cyan]")
                resp = await client.get(f"{url}/api/v1/hub/validate", params={"model_id": model_id})
                if resp.status_code == 200:
                    val_data = resp.json()
                    if not val_data.get("valid"):
                        console.print(f"[red]✗ {val_data.get('error', '模型不存在')}[/red]")
                        return
                    if val_data.get("gated"):
                        console.print("[yellow]⚠ 该模型需要授权访问[/yellow]")
                    console.print("[green]✓ 模型存在[/green]")
                else:
                    console.print("[red]✗ 验证失败[/red]")
                    return

                # 触发下载
                console.print("[cyan]开始下载...[/cyan]")
                # fire and forget — 后端会异步下载
                asyncio.create_task(
                    client.post(f"{url}/api/v1/hub/download", json={"model_id": model_id})
                )

                # 轮询进度
                from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TextColumn("{task.percentage:3.0f}%"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"[cyan]下载 {model_id}", total=100)
                    last_pct = 0
                    stall_count = 0
                    for _ in range(600):  # max 10 minutes
                        await asyncio.sleep(1)
                        try:
                            pr = await client.get(
                                f"{url}/api/v1/hub/download/progress", params={"model_id": model_id}
                            )
                            if pr.status_code == 200:
                                pd = pr.json()
                                pct = max(0.0, pd.get("progress", -1))
                                if pct < 0:
                                    # error / removed from tracker
                                    progress.update(task, description=f"[red]下载失败: {model_id}")
                                    console.print("[red]✗ 下载失败[/red]")
                                    return
                                progress.update(task, completed=pct)
                                last_pct = pct
                                if pct >= 100:
                                    progress.update(task, description=f"[green]✓ {model_id} 完成")
                                    console.print("[green]✓ 下载完成[/green]")
                                    return
                            # detect stall
                            if pct == last_pct:
                                stall_count += 1
                                if stall_count > 120:
                                    console.print("[yellow]⚠ 下载似乎卡住了，仍在等待...[/yellow]")
                                    stall_count = 0
                        except Exception:
                            pass
                    console.print("[yellow]⚠ 下载超时，请检查服务器状态[/yellow]")

            except Exception as e:
                console.print(f"[red]连接失败: {e}[/red]")

    asyncio.run(do_download())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def recommend(url):
    """基于系统配置推荐模型"""

    async def do_recommend():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                console.print("[cyan]检测系统配置并生成推荐...[/cyan]")
                resp = await client.get(f"{url}/api/v1/hub/recommendations")
                if resp.status_code == 200:
                    data = resp.json()
                    sys_info = data.get("system", {})
                    recs = data.get("recommendations", [])
                    summary = data.get("summary", {})

                    console.print("\n[bold]系统配置:[/bold]")
                    console.print(
                        f"  GPU: {sys_info.get('gpu_names', ['无'])[0]} × {sys_info.get('gpu_count', 0)}"
                    )
                    console.print(
                        f"  显存: {sys_info.get('total_vram_gb', 0)} GB (可用: {sys_info.get('free_vram_gb', 0)} GB)"
                    )
                    console.print(f"  内存: {sys_info.get('ram_total_gb', 0)} GB")

                    if recs:
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("状态", style="dim")
                        table.add_column("模型", style="cyan")
                        table.add_column("参数", style="yellow")
                        table.add_column("显存需求", style="green")
                        table.add_column("说明", style="dim")

                        for m in recs[:15]:
                            badge = (
                                "[green]✓[/green]"
                                if m["status"] == "compatible"
                                else "[yellow]⚠[/yellow]"
                            )
                            table.add_row(
                                badge,
                                m["name"],
                                f"{m['params']}B",
                                f"~{m['vram_gb']}GB",
                                m.get("description", ""),
                            )

                        console.print(table)

                    console.print(
                        f"\n[dim]兼容: {summary.get('compatible_count', 0)}个 | 可跑7B: {'是' if summary.get('can_run_7b') else '否'}[/dim]"
                    )
                else:
                    console.print("[red]获取失败[/red]")
            except Exception as e:
                console.print(f"[red]连接失败: {e}[/red]")

    asyncio.run(do_recommend())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def interactive(url):
    """进入交互式终端"""
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    console.clear()
    console.print(
        Panel.fit(
            "[bold blue]⚡ QuantumFlow 交互式终端[/bold blue]\n" f"[dim]服务地址: {url}[/dim]",
            border_style="blue",
        )
    )

    async def check_connection():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/v1/models/status")
                return resp.status_code == 200
        except Exception:
            return False

    async def show_status():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/v1/cluster/status")
                if resp.status_code == 200:
                    data = resp.json()
                    console.print()
                    console.print("[bold]集群状态[/bold]")
                    console.print(
                        f"  节点: {data['total_nodes']} | 健康: {data['healthy_nodes']} | GPU: {data['total_gpus']}"
                    )
                    console.print(
                        f"  已加载模型: {data['active_models']} | CPU: {data['system_metrics']['cpu_usage']*100:.0f}% | GPU: {data['system_metrics']['gpu_usage']*100:.0f}%"
                    )
                    console.print()
        except Exception:
            pass

    async def do_load_model():
        resp_available = httpx.get(f"{url}/api/v1/models/list", timeout=5.0)
        available = (
            resp_available.json().get("available_models", [])
            if resp_available.status_code == 200
            else []
        )
        resp_loaded = httpx.get(f"{url}/api/v1/models/status", timeout=5.0)
        loaded = (
            resp_loaded.json().get("loaded_models", []) if resp_loaded.status_code == 200 else []
        )

        console.print("\n[bold]可用模型:[/bold]")
        for i, m in enumerate(available, 1):
            mark = "[green]✓[/green]" if m in loaded else "[dim]○[/dim]"
            console.print(f"  {mark} {i}. {m}")

        choice = Prompt.ask("选择模型编号 (Q返回)", default="Q")
        if choice.upper() == "Q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                model = available[idx]
                backend = Prompt.ask("后端", default="huggingface", choices=["huggingface", "vllm"])
                console.print(f"[cyan]正在加载 {model}...[/cyan]")
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(
                        f"{url}/api/v1/models/load", json={"model": model, "backend": backend}
                    )
                    if resp.status_code in [200, 201]:
                        console.print(f"[green]✓ {model} 加载成功[/green]")
                    else:
                        console.print(f"[red]✗ 加载失败: {resp.text}[/red]")
        except (ValueError, IndexError):
            console.print("[red]无效选择[/red]")

    async def do_unload_model():
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/v1/models/status")
            loaded = resp.json().get("loaded_models", []) if resp.status_code == 200 else []

        if not loaded:
            console.print("\n[yellow]没有已加载的模型[/yellow]")
            return

        console.print("\n[bold]已加载模型:[/bold]")
        for i, m in enumerate(loaded, 1):
            console.print(f"  {i}. {m}")

        choice = Prompt.ask("选择要卸载的模型编号 (Q返回)", default="Q")
        if choice.upper() == "Q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(loaded):
                model = loaded[idx]
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(f"{url}/api/v1/models/unload", json={"model": model})
                    if resp.status_code == 200:
                        console.print(f"[green]✓ {model} 卸载成功[/green]")
                    else:
                        console.print(f"[red]✗ 卸载失败: {resp.text}[/red]")
        except (ValueError, IndexError):
            console.print("[red]无效选择[/red]")

    async def do_chat():
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/v1/models/status")
            loaded = resp.json().get("loaded_models", []) if resp.status_code == 200 else []

        if not loaded:
            console.print("\n[yellow]请先加载模型[/yellow]")
            return

        console.print("\n[bold]选择模型:[/bold]")
        for i, m in enumerate(loaded, 1):
            console.print(f"  {i}. {m}")
        choice = Prompt.ask("模型编号", default="1")
        try:
            model = loaded[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("[red]无效选择[/red]")
            return

        console.print(f"\n[bold green]开始对话 (模型: {model})[/bold green]")
        console.print("[dim]输入 'exit' 或 'quit' 退出[/dim]\n")

        messages = []
        while True:
            user_input = Prompt.ask("[bold blue]你[/bold blue]")
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]对话结束[/dim]")
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            console.print("[bold green]助手[/bold green] ", end="")
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{url}/api/v1/inference/chat",
                        json={
                            "model": model,
                            "messages": messages,
                            "sampling_params": {
                                "temperature": 0.7,
                                "max_tokens": 500,
                                "repetition_penalty": 1.1,
                            },
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        reply = data.get("generated_text", "[无响应]")
                        console.print(reply)
                        messages.append({"role": "assistant", "content": reply})
                    else:
                        console.print(f"[red]错误: {resp.text}[/red]")
                        messages.pop()
            except Exception as e:
                console.print(f"[red]连接失败: {e}[/red]")
                messages.pop()

    async def do_generate():
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/v1/models/status")
            loaded = resp.json().get("loaded_models", []) if resp.status_code == 200 else []

        if not loaded:
            console.print("\n[yellow]请先加载模型[/yellow]")
            return

        console.print("\n[bold]选择模型:[/bold]")
        for i, m in enumerate(loaded, 1):
            console.print(f"  {i}. {m}")
        choice = Prompt.ask("模型编号", default="1")
        try:
            model = loaded[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("[red]无效选择[/red]")
            return

        prompt = Prompt.ask("[bold]输入 Prompt[/bold]")
        if not prompt.strip():
            return

        max_tokens = Prompt.ask("Max Tokens", default="200")
        temperature = Prompt.ask("Temperature", default="0.7")

        console.print("[cyan]生成中...[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{url}/api/v1/inference/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "sampling_params": {
                            "max_tokens": int(max_tokens),
                            "temperature": float(temperature),
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    console.print("\n[bold green]结果:[/bold green]")
                    console.print(data.get("generated_text", "[无响应]"))
                    console.print(f"[dim]延迟: {data.get('latency_ms', 0):.0f}ms[/dim]")
                else:
                    console.print(f"[red]错误: {resp.text}[/red]")
        except Exception as e:
            console.print(f"[red]连接失败: {e}[/red]")

    async def do_hub_trending():
        """显示热门模型"""
        console.print("\n[cyan]正在获取 HuggingFace 热门模型...[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{url}/api/v1/hub/trending?limit=15")
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("models", [])
                    if not models_list:
                        console.print("[yellow]无法获取热门模型[/yellow]")
                        return

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("#", style="dim")
                    table.add_column("模型ID", style="cyan")
                    table.add_column("下载量", style="green")
                    table.add_column("类型", style="yellow")

                    for i, m in enumerate(models_list[:15], 1):
                        downloads = m.get("downloads", 0) or 0
                        if downloads >= 1_000_000:
                            dl_str = f"{downloads/1_000_000:.1f}M"
                        elif downloads >= 1_000:
                            dl_str = f"{downloads/1_000:.0f}K"
                        else:
                            dl_str = str(downloads)
                        table.add_row(
                            str(i),
                            m.get("model_id", "")[:60],
                            dl_str,
                            m.get("pipeline_tag", "unknown"),
                        )

                    console.print(table)
                    console.print(f"\n[dim]共 {len(models_list)} 个模型[/dim]")
                    console.print(
                        "[dim]使用 'python -m quantumflow.cli download <model_id>' 下载模型[/dim]"
                    )
                else:
                    console.print(f"[red]获取失败: {resp.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]连接失败: {e}[/red]")

    async def do_hub_search():
        """搜索模型"""
        query = Prompt.ask("[bold]输入搜索关键词[/bold]")
        if not query.strip():
            return

        console.print(f"\n[cyan]正在搜索 '{query}'...[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{url}/api/v1/hub/search", params={"q": query, "limit": 15}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("models", [])
                    if not models_list:
                        console.print(f"[yellow]未找到与 '{query}' 匹配的模型[/yellow]")
                        return

                    console.print(f"\n[bold]搜索结果 ({len(models_list)} 个):[/bold]")
                    for i, m in enumerate(models_list, 1):
                        downloads = m.get("downloads", 0) or 0
                        dl_str = (
                            f"{downloads/1_000_000:.1f}M"
                            if downloads >= 1_000_000
                            else f"{downloads/1_000:.0f}K" if downloads >= 1_000 else str(downloads)
                        )
                        console.print(
                            f"  {i}. [cyan]{m.get('model_id', '')}[/cyan] [dim]↓{dl_str}[/dim]"
                        )

                    choice = Prompt.ask("输入编号下载 (Q返回)", default="Q")
                    if choice.upper() == "Q":
                        return
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(models_list):
                            await do_download_model(models_list[idx]["model_id"])
                    except (ValueError, IndexError):
                        console.print("[red]无效选择[/red]")
                else:
                    console.print(f"[red]搜索失败: {resp.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]连接失败: {e}[/red]")

    async def do_hub_recommend():
        """显示推荐模型"""
        console.print("[cyan]正在检测系统配置...[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{url}/api/v1/hub/recommendations")
                if resp.status_code == 200:
                    data = resp.json()
                    sys_info = data.get("system", {})
                    recs = data.get("recommendations", [])
                    summary = data.get("summary", {})

                    console.print("\n[bold]系统配置:[/bold]")
                    console.print(
                        f"  GPU: {sys_info.get('gpu_names', ['无'])[0]} × {sys_info.get('gpu_count', 0)}"
                    )
                    console.print(
                        f"  总显存: {sys_info.get('total_vram_gb', 0)} GB | 可用: {sys_info.get('free_vram_gb', 0)} GB"
                    )
                    console.print(f"  系统内存: {sys_info.get('ram_total_gb', 0)} GB")

                    if recs:
                        console.print(f"\n[bold green]推荐模型 ({len(recs)} 个):[/bold green]")
                        for i, m in enumerate(recs[:10], 1):
                            badge = (
                                "[green]✓[/green]"
                                if m["status"] == "compatible"
                                else "[yellow]⚠[/yellow]"
                            )
                            console.print(
                                f"  {badge} {i}. [cyan]{m['name']}[/cyan] - {m['description']} [dim]({m['params']}B, ~{m['vram_gb']}GB)[/dim]"
                            )

                    console.print(
                        f"\n[dim]兼容模型: {summary.get('compatible_count', 0)} 个 | 支持7B: {'是' if summary.get('can_run_7b') else '否'}[/dim]"
                    )
                else:
                    console.print("[red]获取推荐失败[/red]")
        except Exception as e:
            console.print(f"[red]连接失败: {e}[/red]")

    async def do_download_model(model_id: str = None):
        """下载模型"""
        if not model_id:
            model_id = Prompt.ask(
                "[bold]输入 HuggingFace 模型ID[/bold] (如 Qwen/Qwen2.5-1.5B-Instruct)"
            )
        if not model_id or not model_id.strip():
            return

        console.print(f"[cyan]正在验证模型 '{model_id}'...[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # 先验证
                resp = await client.get(f"{url}/api/v1/hub/validate", params={"model_id": model_id})
                if resp.status_code == 200:
                    val_data = resp.json()
                    if not val_data.get("valid"):
                        console.print(f"[red]✗ {val_data.get('error', '模型不存在')}[/red]")
                        return
                    if val_data.get("gated"):
                        console.print("[yellow]⚠ 该模型需要授权访问[/yellow]")
                        if not Confirm.ask("继续尝试下载?", default=False):
                            return

                # 触发下载 (fire and forget)
                console.print(f"[cyan]开始下载 {model_id}...[/cyan]")
                asyncio.create_task(
                    client.post(f"{url}/api/v1/hub/download", json={"model_id": model_id})
                )

                # 轮询进度
                from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=30),
                    TextColumn("{task.percentage:3.0f}%"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"[cyan]{model_id.split('/')[-1]}", total=100)
                    for _ in range(600):
                        await asyncio.sleep(1)
                        try:
                            pr = await client.get(
                                f"{url}/api/v1/hub/download/progress", params={"model_id": model_id}
                            )
                            if pr.status_code == 200:
                                pd = pr.json()
                                pct = max(0.0, pd.get("progress", -1))
                                if pct < 0:
                                    console.print("[red]✗ 下载失败[/red]")
                                    return
                                progress.update(task, completed=pct)
                                if pct >= 100:
                                    progress.update(
                                        task, description=f"[green]✓ {model_id.split('/')[-1]} 完成"
                                    )
                                    console.print("[green]✓ 下载成功[/green]")
                                    # 询问是否加载
                                    if Confirm.ask("是否立即加载模型?", default=True):
                                        short_name = model_id.split("/")[-1]
                                        resp = await client.post(
                                            f"{url}/api/v1/models/load",
                                            json={"model": short_name, "model_path": model_id},
                                        )
                                        if resp.status_code in [200, 201]:
                                            console.print(f"[green]✓ {short_name} 加载成功[/green]")
                                        else:
                                            console.print(f"[red]加载失败: {resp.text}[/red]")
                                    return
                        except Exception:
                            pass
                    console.print("[yellow]⚠ 下载超时[/yellow]")
        except Exception as e:
            console.print(f"[red]连接失败: {e}[/red]")

    async def main_loop():
        if not await check_connection():
            console.print(f"\n[red]✗ 无法连接到 {url}[/red]")
            console.print("[yellow]请先启动服务器: python -m quantumflow.cli serve[/yellow]")
            return

        await show_status()

        MENU = {
            "1": ("对话 (Chat)", do_chat),
            "2": ("文本生成 (Generate)", do_generate),
            "3": ("加载模型", do_load_model),
            "4": ("卸载模型", do_unload_model),
            "5": ("模型中心 (Hub)", do_hub_trending),
            "6": ("搜索模型", do_hub_search),
            "7": ("智能推荐", do_hub_recommend),
            "8": ("下载模型", do_download_model),
            "9": ("查看状态", lambda: show_status()),
            "0": ("退出", None),
        }

        while True:
            console.print()
            console.print("[bold]菜单:[/bold]")
            for k, (label, _) in MENU.items():
                console.print(f"  {k}. {label}")

            choice = Prompt.ask("选择", default="1", choices=list(MENU.keys()))

            _, handler = MENU[choice]
            if handler is None:
                console.print("[dim]再见![/dim]")
                break
            await handler()

    asyncio.run(main_loop())


def main():
    cli()


if __name__ == "__main__":
    main()
