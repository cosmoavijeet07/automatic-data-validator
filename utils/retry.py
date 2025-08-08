from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.errors import CodeExecutionError, LLMError, ProfilingError

retry_llm = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    retry=retry_if_exception_type(LLMError),
)

retry_exec = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    retry=retry_if_exception_type((CodeExecutionError, ProfilingError)),
)
