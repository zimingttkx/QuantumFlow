"""CLI命令行工具"""

import asyncio
import sys
import json
import time
import click
import structlog
import httpx
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
def serve(host, port, workers, reload):
    """启动API服务器"""
    import uvicorn
    from quantumflow.api.server import create_app

    console.print(f"[bold green]启动QuantumFlow API服务器[/bold green]")
    console.print(f"  地址: {host}:{port}")
    console.print(f"  工作进程: {workers}")

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
@click.option("--backend", default="vllm", type=click.Choice(["vllm", "tgi", "sglang"]), help="推理后端")
@click.option("--tgi-url", default="http://localhost:8080", help="TGI服务URL")
@click.option("--sglang-url", default="http://localhost:30000", help="SGLang服务URL")
def worker(controller_url, host, port, backend, tgi_url, sglang_url):
    """启动Worker节点"""
    from quantumflow.worker import WorkerNode, WorkerConfig
    from quantumflow.inference.backends import VLLMEngine, TGIEngine, SGLangEngine

    console.print(f"[bold green]启动QuantumFlow Worker节点[/bold green]")
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
        console.print(f"[green]Worker已启动，按Ctrl+C停止[/green]")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]正在停止Worker...[/yellow]")
            await worker.stop()

    asyncio.run(run())


@cli.command()
@click.argument("model")
@click.option("--tensor-parallel", "-tp", default=1, help="张量并行度")
@click.option("--gpus", "-g", default="0", help="GPU IDs (逗号分隔)")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def deploy(model, tensor_parallel, gpus, url):
    """部署模型"""
    async def do_deploy():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{url}/api/v1/models/deploy",
                    json={
                        "model": model,
                        "tensor_parallel": tensor_parallel,
                        "gpus": [int(g) for g in gpus.split(",")],
                    },
                    timeout=60.0,
                )
                if response.status_code == 201:
                    data = response.json()
                    console.print(f"[green]✓ 模型 {model} 部署成功[/green]")
                    console.print(f"  模型ID: {data.get('model_id')}")
                    console.print(f"  状态: {data.get('status')}")
                else:
                    console.print(f"[red]✗ 部署失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_deploy())


@cli.command()
@click.argument("model")
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def undeploy(model, url):
    """卸载模型"""
    async def do_undeploy():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(f"{url}/api/v1/models/{model}")
                if response.status_code == 200:
                    console.print(f"[green]✓ 模型 {model} 卸载成功[/green]")
                else:
                    console.print(f"[red]✗ 卸载失败: {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_undeploy())


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
                        console.print(f"[bold blue]QuantumFlow 集群状态[/bold blue]")
                        console.print()

                        # 创建统计表
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("指标", style="cyan")
                        table.add_column("值", style="green")

                        table.add_row("总节点数", str(data.get("total_nodes", 0)))
                        table.add_row("健康节点", str(data.get("healthy_nodes", 0)))
                        table.add_row("总GPU数", str(data.get("total_gpus", 0)))
                        table.add_row("可用GPU", str(data.get("available_gpus", 0)))
                        table.add_row("总模型数", str(data.get("total_models", 0)))

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
                console.print(f"[cyan]正在生成...[/cyan]")
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

                elapsed = (time.time() - start_time) * 1000

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
                console.print(f"[cyan]对话中...[/cyan]")

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
    """列出可用模型"""
    async def do_list():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{url}/api/v1/models")
                if response.status_code == 200:
                    data = response.json()

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("模型名称", style="cyan")
                    table.add_column("状态", style="green")
                    table.add_column("参数量", style="yellow")

                    for model in data:
                        status_color = "[green]ready[/green]" if model.get("status") == "ready" else "[yellow]loading[/yellow]"
                        table.add_row(
                            model.get("name", ""),
                            status_color,
                            str(model.get("parameter_count", "")),
                        )

                    console.print(table)
                else:
                    console.print(f"[red]✗ 获取模型列表失败[/red]")

            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_list())


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def nodes(url):
    """列出计算节点"""
    async def do_list():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{url}/api/v1/cluster/nodes")
                if response.status_code == 200:
                    data = response.json()

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("节点ID", style="cyan")
                    table.add_column("主机名", style="yellow")
                    table.add_column("状态", style="green")
                    table.add_column("GPU数", style="magenta")

                    for node in data:
                        status_color = {
                            "healthy": "[green]healthy[/green]",
                            "unhealthy": "[red]unhealthy[/red]",
                            "offline": "[dim]offline[/dim]",
                        }.get(node.get("status", ""), "[yellow]unknown[/yellow]")

                        table.add_row(
                            node.get("node_id", ""),
                            node.get("hostname", ""),
                            status_color,
                            str(node.get("gpu_count", 0)),
                        )

                    console.print(table)
                else:
                    console.print(f"[red]✗ 获取节点列表失败[/red]")

            except Exception as e:
                console.print(f"[red]✗ 连接失败: {e}[/red]")

    asyncio.run(do_list())


def main():
    cli()


if __name__ == "__main__":
    main()
