"""QuantumFlow SDK 异常类"""


class QuantumFlowError(Exception):
    """SDK 基础异常类"""
    pass


class APIError(QuantumFlowError):
    """API 调用错误"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


class RateLimitError(APIError):
    """限流错误"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(status_code=429, message=message)


class TimeoutError(QuantumFlowError):
    """请求超时错误"""
    pass


class ValidationError(QuantumFlowError):
    """参数验证错误"""
    pass
