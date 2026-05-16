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
@click.option("--backend", "-b", default="huggingface", type=click.Choice(["huggingface", "vllm", "tgi", "sglang"]), help="推理后端")
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
                    data = response.json()
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

                    console.print(f"[bold blue]QuantumFlow 模型[/bold blue]")
                    console.print()

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("模型名称", style="cyan")
                    table.add_column("HF Hub ID", style="dim")
                    table.add_column("状态", style="green")

                    for name in available:
                        status = "[green]已加载[/green]" if name in loaded else "[yellow]未加载[/yellow]"
                        table.add_row(name, mappings.get(name, "N/A"), status)

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


@cli.command()
@click.option("--url", default="http://localhost:8000", help="API服务器地址")
def interactive(url):
    """进入交互式终端"""
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown

    console.clear()
    console.print(Panel.fit(
        "[bold blue]⚡ QuantumFlow 交互式终端[/bold blue]\n"
        f"[dim]服务地址: {url}[/dim]",
        border_style="blue"
    ))

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
                    console.print(f"  节点: {data['total_nodes']} | 健康: {data['healthy_nodes']} | GPU: {data['total_gpus']}")
                    console.print(f"  已加载模型: {data['active_models']} | CPU: {data['system_metrics']['cpu_usage']*100:.0f}% | GPU: {data['system_metrics']['gpu_usage']*100:.0f}%")
                    console.print()
        except Exception:
            pass

    async def do_load_model():
        resp_available = httpx.get(f"{url}/api/v1/models/list", timeout=5.0)
        available = resp_available.json().get("available_models", []) if resp_available.status_code == 200 else []
        resp_loaded = httpx.get(f"{url}/api/v1/models/status", timeout=5.0)
        loaded = resp_loaded.json().get("loaded_models", []) if resp_loaded.status_code == 200 else []

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
                    resp = await client.post(f"{url}/api/v1/models/load",
                        json={"model": model, "backend": backend})
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
                    resp = await client.post(f"{url}/api/v1/inference/chat",
                        json={
                            "model": model,
                            "messages": messages,
                            "sampling_params": {"temperature": 0.7, "max_tokens": 500, "repetition_penalty": 1.1},
                        })
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
                resp = await client.post(f"{url}/api/v1/inference/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "sampling_params": {
                            "max_tokens": int(max_tokens),
                            "temperature": float(temperature),
                        },
                    })
                if resp.status_code == 200:
                    data = resp.json()
                    console.print(f"\n[bold green]结果:[/bold green]")
                    console.print(data.get("generated_text", "[无响应]"))
                    console.print(f"[dim]延迟: {data.get('latency_ms', 0):.0f}ms[/dim]")
                else:
                    console.print(f"[red]错误: {resp.text}[/red]")
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
            "5": ("查看状态", lambda: show_status()),
            "6": ("退出", None),
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
