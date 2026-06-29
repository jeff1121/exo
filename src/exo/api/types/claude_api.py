"""此說明已翻譯為繁體中文。"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import ModelId

# 工具定義型別
ClaudeToolInputSchema = dict[str, Any]


class ClaudeToolDefinition(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    name: str
    description: str | None = None
    input_schema: ClaudeToolInputSchema


# 型別別名
ClaudeRole = Literal["user", "assistant"]
ClaudeStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


# 內容區塊型別
class ClaudeTextBlock(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["text"] = "text"
    text: str


class ClaudeImageSource(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ClaudeImageBlock(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["image"] = "image"
    source: ClaudeImageSource


class ClaudeThinkingBlock(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ClaudeToolUseBlock(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ClaudeToolResultBlock(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[ClaudeTextBlock | ClaudeImageBlock] | None = None
    is_error: bool | None = None
    cache_control: dict[str, str] | None = None


ClaudeContentBlock = (
    ClaudeTextBlock | ClaudeImageBlock | ClaudeThinkingBlock | ClaudeToolUseBlock
)

# 已翻譯註解。
ClaudeInputContentBlock = (
    ClaudeTextBlock
    | ClaudeImageBlock
    | ClaudeThinkingBlock
    | ClaudeToolUseBlock
    | ClaudeToolResultBlock
)


# 請求型別
class ClaudeMessage(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    role: ClaudeRole
    content: str | list[ClaudeInputContentBlock]


class ClaudeThinkingConfig(BaseModel, frozen=True):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = None


class ClaudeMessagesRequest(BaseModel):
    """此說明已翻譯為繁體中文。"""

    model: ModelId
    max_tokens: int
    messages: list[ClaudeMessage]
    system: str | list[ClaudeTextBlock] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[ClaudeToolDefinition] | None = None
    metadata: dict[str, str] | None = None
    thinking: ClaudeThinkingConfig | None = None


# 回應型別
class ClaudeUsage(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    input_tokens: int
    output_tokens: int


class ClaudeMessagesResponse(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ClaudeContentBlock]
    model: str
    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


# 串流事件型別
class ClaudeMessageStart(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ClaudeTextBlock] = Field(default_factory=list)
    model: str
    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


class ClaudeMessageStartEvent(BaseModel, frozen=True):
    """在訊息串流開始時送出的事件。"""

    type: Literal["message_start"] = "message_start"
    message: ClaudeMessageStart


class ClaudeContentBlockStartEvent(BaseModel, frozen=True):
    """在內容區塊開始時送出的事件。"""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ClaudeTextBlock | ClaudeThinkingBlock | ClaudeToolUseBlock


class ClaudeTextDelta(BaseModel, frozen=True):
    """文字內容區塊的增量。"""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ClaudeThinkingDelta(BaseModel, frozen=True):
    """思考內容區塊的增量。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class ClaudeInputJsonDelta(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ClaudeContentBlockDeltaEvent(BaseModel, frozen=True):
    """送出內容區塊增量時的事件。"""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ClaudeTextDelta | ClaudeThinkingDelta | ClaudeInputJsonDelta


class ClaudeContentBlockStopEvent(BaseModel, frozen=True):
    """在內容區塊結束時送出的事件。"""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class ClaudeMessageDeltaUsage(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    output_tokens: int


class ClaudeMessageDelta(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None


class ClaudeMessageDeltaEvent(BaseModel, frozen=True):
    """附帶最終訊息增量送出的事件。"""

    type: Literal["message_delta"] = "message_delta"
    delta: ClaudeMessageDelta
    usage: ClaudeMessageDeltaUsage


class ClaudeMessageStopEvent(BaseModel, frozen=True):
    """在訊息串流結束時送出的事件。"""

    type: Literal["message_stop"] = "message_stop"


ClaudeStreamEvent = (
    ClaudeMessageStartEvent
    | ClaudeContentBlockStartEvent
    | ClaudeContentBlockDeltaEvent
    | ClaudeContentBlockStopEvent
    | ClaudeMessageDeltaEvent
    | ClaudeMessageStopEvent
)
