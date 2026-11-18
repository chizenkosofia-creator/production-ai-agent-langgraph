"""Shared domain error type.

``DomainError`` marks failures that are the *caller's* fault and fixable (unknown
ticket, inactive operator, ...). Services raise it; the tool layer converts it
into an ``ERROR: ...`` string the LLM can act on, and API code can map it to a
4xx. Anything else is an unexpected bug and is logged with a stack trace.
"""


class DomainError(ValueError):
    """A business-rule violation with a message safe to show to the caller."""
