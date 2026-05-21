"""SDK 数据模型"""
from typing import Any


class SamplingParams:
    """采样参数"""

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.stop = stop

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop,
        }


class InferenceRequest:
    """推理请求"""

    def __init__(
        self,
        model: str,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        stream: bool = False,
        session_id: str | None = None,
        priority: int = 5,
    ):
        self.model = model
        self.prompt = prompt
        self.sampling_params = sampling_params or SamplingParams()
        self.stream = stream
        self.session_id = session_id
        self.priority = priority

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "sampling_params": self.sampling_params.to_dict(),
            "stream": self.stream,
        }
        if self.session_id:
            data["session_id"] = self.session_id
        data["priority"] = self.priority
        return data


class ChatMessage:
    """对话消息"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class InferenceResponse:
    """推理响应"""

    def __init__(
        self,
        request_id: str,
        model: str,
        generated_text: str,
        finish_reason: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ):
        self.request_id = request_id
        self.model = model
        self.generated_text = generated_text
        self.finish_reason = finish_reason
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens