"""Helpers shared by every tool module.

``safe_tool`` is what makes "tools never crash the graph" true: any exception
becomes a short ``ERROR: ...`` string the LLM reads as an ordinary tool result.
"""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from core.errors import DomainError

logger = logging.getLogger(__name__)

GENERIC_FAILURE = (
    "ERROR: the tool failed due to an internal problem. "
    "Do not retry more than once; tell the user the action is temporarily unavailable."
)


class ToolInputError(DomainError):
    """Raised for caller mistakes the LLM can fix by changing its arguments."""


def _fixable(name: str, exc: Exception) -> str:
    logger.warning("Tool %s rejected input: %s", name, exc)
    return f"ERROR: invalid input - {exc}"


def safe_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync or async tool function so failures become clean strings.

    * ``DomainError`` / ``ValidationError``: expected and fixable; the message is
      returned verbatim so the model can correct its arguments.
    * Anything else: logged with a stack trace; only a generic message is returned
      so internals (SQL, hostnames, stack frames) never reach the conversation.
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (DomainError, ValidationError) as exc:
                return _fixable(func.__name__, exc)
            except Exception:  # noqa: BLE001 - last line of defence by design
                logger.exception("Tool %s failed unexpectedly", func.__name__)
                return GENERIC_FAILURE

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (DomainError, ValidationError) as exc:
            return _fixable(func.__name__, exc)
        except Exception:  # noqa: BLE001 - last line of defence by design
            logger.exception("Tool %s failed unexpectedly", func.__name__)
            return GENERIC_FAILURE

    return wrapper
