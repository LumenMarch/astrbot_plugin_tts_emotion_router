"""Per-task sanitized provider failures, safe across concurrent synthesis."""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisFailure:
    code: str
    message: str
    retryable: bool = False


failure = ContextVar("tts_failure", default=None)


def set_failure(code, message, retryable=False):
    failure.set(SynthesisFailure(code, message, retryable))


def http_failure(status):
    if status in (401, 403):
        set_failure("auth", "鉴权失败，请检查 TTS 引擎中的 API Key 和接口权限")
    elif status == 402:
        set_failure("balance", "余额不足，请到服务商控制台充值")
    elif status == 429:
        set_failure("rate_limit", "请求限流或额度不足，请检查配额并稍后重试", True)
    elif status >= 500:
        set_failure("upstream", "服务商暂时异常，请稍后重试", True)
    else:
        set_failure("request", "请求参数错误，请检查服务地址、模型和对应音色")
