"""Provider-neutral token usage contracts."""

from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

TokenCount: TypeAlias = Annotated[int, Field(ge=0)]


class TokenUsage(BaseModel):
    """Token counts reported for one or more provider responses."""

    model_config = ConfigDict(frozen=True)

    input_tokens: TokenCount
    output_tokens: TokenCount
    total_tokens: TokenCount
    cached_input_tokens: TokenCount = 0
    reasoning_output_tokens: TokenCount = 0
