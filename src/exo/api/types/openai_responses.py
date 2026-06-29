"""此說明已翻譯為繁體中文。

此說明已翻譯為繁體中文。
此說明已翻譯為繁體中文。
任務參數型別，請參考
此說明已翻譯為繁體中文。
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import ReasoningEffort

# 型別別名
ResponseStatus = Literal["completed", "failed", "in_progress", "incomplete"]
ResponseRole = Literal["user", "assistant", "system", "developer"]


# 請求輸入內容分段型別
class ResponseInputTextPart(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImagePart(BaseModel, frozen=True):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: str | None = None


class ResponseOutputTextPart(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["output_text"] = "output_text"
    text: str


ResponseContentPart = (
    ResponseInputTextPart | ResponseInputImagePart | ResponseOutputTextPart
)


# 請求輸入項目型別
class ResponseInputMessage(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    role: ResponseRole
    content: str | list[ResponseContentPart]
    type: Literal["message"] = "message"


class FunctionCallInputItem(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: ResponseStatus | None = None


class FunctionCallOutputInputItem(BaseModel, frozen=True):
    """輸入中的函式呼叫輸出項目（使用者提供工具結果）。"""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    id: str | None = None
    status: ResponseStatus | None = None


class ReasoningInputItem(BaseModel, frozen=True):
    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    summary: list[dict[str, Any]] | None = None
    encrypted_content: str | None = None
    content: list[dict[str, Any]] | None = None
    status: ResponseStatus | None = None


class ComputerCallInputItem(BaseModel, frozen=True):
    type: Literal["computer_call"] = "computer_call"
    id: str | None = None
    call_id: str = ""
    action: dict[str, Any] = {}
    status: ResponseStatus | None = None


class ComputerCallOutputInputItem(BaseModel, frozen=True):
    type: Literal["computer_call_output"] = "computer_call_output"
    id: str | None = None
    call_id: str = ""
    output: dict[str, Any] | str = ""
    status: ResponseStatus | None = None


class WebSearchCallInputItem(BaseModel, frozen=True):
    type: Literal["web_search_call"] = "web_search_call"
    id: str | None = None
    call_id: str = ""
    query: str = ""
    status: ResponseStatus | None = None


class FileSearchCallInputItem(BaseModel, frozen=True):
    type: Literal["file_search_call"] = "file_search_call"
    id: str | None = None
    call_id: str = ""
    queries: list[str] = []
    results: list[dict[str, Any]] | None = None
    status: ResponseStatus | None = None


class CodeInterpreterCallInputItem(BaseModel, frozen=True):
    type: Literal["code_interpreter_call"] = "code_interpreter_call"
    id: str | None = None
    call_id: str = ""
    code: str = ""
    results: list[dict[str, Any]] | None = None
    status: ResponseStatus | None = None


class ImageGenerationCallInputItem(BaseModel, frozen=True):
    type: Literal["image_generation_call"] = "image_generation_call"
    id: str | None = None
    call_id: str = ""
    prompt: str = ""
    result: dict[str, Any] | None = None
    status: ResponseStatus | None = None


class LocalShellCallInputItem(BaseModel, frozen=True):
    type: Literal["local_shell_call"] = "local_shell_call"
    id: str | None = None
    call_id: str = ""
    action: dict[str, Any] = {}
    status: ResponseStatus | None = None


class LocalShellCallOutputInputItem(BaseModel, frozen=True):
    type: Literal["local_shell_call_output"] = "local_shell_call_output"
    id: str | None = None
    call_id: str = ""
    output: str = ""
    status: ResponseStatus | None = None


class ShellCallInputItem(BaseModel, frozen=True):
    type: Literal["shell_call"] = "shell_call"
    id: str | None = None
    call_id: str = ""
    action: dict[str, Any] = {}
    status: ResponseStatus | None = None


class ShellCallOutputInputItem(BaseModel, frozen=True):
    type: Literal["shell_call_output"] = "shell_call_output"
    id: str | None = None
    call_id: str = ""
    output: str = ""
    status: ResponseStatus | None = None


class ApplyPatchCallInputItem(BaseModel, frozen=True):
    type: Literal["apply_patch_call"] = "apply_patch_call"
    id: str | None = None
    call_id: str = ""
    patch: str = ""
    status: ResponseStatus | None = None


class ApplyPatchCallOutputInputItem(BaseModel, frozen=True):
    type: Literal["apply_patch_call_output"] = "apply_patch_call_output"
    id: str | None = None
    call_id: str = ""
    output: str = ""
    status: ResponseStatus | None = None


class ToolSearchCallInputItem(BaseModel, frozen=True):
    type: Literal["tool_search_call"] = "tool_search_call"
    id: str | None = None
    call_id: str = ""
    query: str = ""
    status: ResponseStatus | None = None


class ToolSearchOutputInputItem(BaseModel, frozen=True):
    type: Literal["tool_search_output"] = "tool_search_output"
    id: str | None = None
    call_id: str = ""
    output: str = ""
    status: ResponseStatus | None = None


class McpCallInputItem(BaseModel, frozen=True):
    type: Literal["mcp_call"] = "mcp_call"
    id: str | None = None
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    server_label: str = ""
    approval_request_id: str | None = None
    error: str | None = None
    output: str | None = None
    status: ResponseStatus | None = None


class McpListToolsInputItem(BaseModel, frozen=True):
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    id: str | None = None
    server_label: str = ""
    tools: list[dict[str, Any]] = []
    error: str | None = None
    status: ResponseStatus | None = None


class McpApprovalRequestInputItem(BaseModel, frozen=True):
    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    id: str | None = None
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    server_label: str = ""
    status: ResponseStatus | None = None


class McpApprovalResponseInputItem(BaseModel, frozen=True):
    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    id: str | None = None
    approval_request_id: str = ""
    approve: bool = True
    reason: str = ""
    status: ResponseStatus | None = None


class CustomToolCallInputItem(BaseModel, frozen=True):
    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: str | None = None
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    status: ResponseStatus | None = None


class CustomToolCallOutputInputItem(BaseModel, frozen=True):
    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    id: str | None = None
    call_id: str = ""
    output: str = ""
    status: ResponseStatus | None = None


class CompactionInputItem(BaseModel, frozen=True):
    type: Literal["compaction"] = "compaction"
    id: str | None = None
    summary: str = ""
    encrypted_content: str | None = None
    status: ResponseStatus | None = None


class ItemReferenceInputItem(BaseModel, frozen=True):
    type: Literal["item_reference"] = "item_reference"
    id: str | None = None


ResponseInputItem = (
    ResponseInputMessage
    | FunctionCallInputItem
    | FunctionCallOutputInputItem
    | ReasoningInputItem
    | ComputerCallInputItem
    | ComputerCallOutputInputItem
    | WebSearchCallInputItem
    | FileSearchCallInputItem
    | CodeInterpreterCallInputItem
    | ImageGenerationCallInputItem
    | LocalShellCallInputItem
    | LocalShellCallOutputInputItem
    | ShellCallInputItem
    | ShellCallOutputInputItem
    | ApplyPatchCallInputItem
    | ApplyPatchCallOutputInputItem
    | ToolSearchCallInputItem
    | ToolSearchOutputInputItem
    | McpCallInputItem
    | McpListToolsInputItem
    | McpApprovalRequestInputItem
    | McpApprovalResponseInputItem
    | CustomToolCallInputItem
    | CustomToolCallOutputInputItem
    | CompactionInputItem
    | ItemReferenceInputItem
)


class Reasoning(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    effort: ReasoningEffort | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class ResponsesRequest(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    # 已翻譯註解。
    model: ModelId
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, str] | None = None
    reasoning: Reasoning | None = None

    # 已翻譯註解。
    enable_thinking: bool | None = Field(
        default=None,
        description="[exo extension] Boolean thinking toggle. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )

    top_k: int | None = Field(
        default=None,
        description="[exo extension] Top-k sampling parameter. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="[exo extension] Stop sequence(s). Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )
    seed: int | None = Field(
        default=None,
        description="[exo extension] Seed for deterministic sampling. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )

    # 已翻譯註解。
    chat_template_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Internal: pre-formatted messages for tokenizer chat template. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-internal": True},
    )


# 回應型別
class ResponseOutputText(BaseModel, frozen=True):
    """回應輸出中的文字內容。"""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, str]] = Field(default_factory=list)


class ResponseMessageItem(BaseModel, frozen=True):
    """回應輸出陣列中的訊息項目。"""

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]
    status: ResponseStatus = "completed"


class ResponseFunctionCallItem(BaseModel, frozen=True):
    """回應輸出陣列中的函式呼叫項目。"""

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: ResponseStatus = "completed"


class ResponseReasoningSummaryText(BaseModel, frozen=True):
    """推理輸出項目中的摘要文字分段。"""

    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningItem(BaseModel, frozen=True):
    """回應輸出陣列中的推理輸出項目。"""

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponseReasoningSummaryText] = Field(default_factory=list)
    status: ResponseStatus = "completed"


ResponseItem = ResponseMessageItem | ResponseFunctionCallItem | ResponseReasoningItem


class InputTokensDetails(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    cached_tokens: int = 0


class OutputTokensDetails(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    reasoning_tokens: int = 0


class ResponseUsage(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class ResponsesResponse(BaseModel, frozen=True):
    """此說明已翻譯為繁體中文。"""

    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus = "completed"
    model: str
    output: list[ResponseItem]
    output_text: str
    usage: ResponseUsage | None = None


# 串流事件型別
class ResponseCreatedEvent(BaseModel, frozen=True):
    """回應建立時送出的事件。"""

    type: Literal["response.created"] = "response.created"
    sequence_number: int
    response: ResponsesResponse


class ResponseInProgressEvent(BaseModel, frozen=True):
    """回應開始處理時送出的事件。"""

    type: Literal["response.in_progress"] = "response.in_progress"
    sequence_number: int
    response: ResponsesResponse


class ResponseOutputItemAddedEvent(BaseModel, frozen=True):
    """新增輸出項目時送出的事件。"""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    sequence_number: int
    output_index: int
    item: ResponseItem


class ResponseContentPartAddedEvent(BaseModel, frozen=True):
    """新增內容分段時送出的事件。"""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseTextDeltaEvent(BaseModel, frozen=True):
    """串流期間送出文字增量時的事件。"""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(BaseModel, frozen=True):
    """文字內容完成時送出的事件。"""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseModel, frozen=True):
    """內容分段完成時送出的事件。"""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseOutputItemDoneEvent(BaseModel, frozen=True):
    """輸出項目完成時送出的事件。"""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    sequence_number: int
    output_index: int
    item: ResponseItem


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel, frozen=True):
    """串流期間送出函式呼叫參數增量時的事件。"""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(BaseModel, frozen=True):
    """函式呼叫參數完成時送出的事件。"""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    name: str
    arguments: str


class ResponseReasoningSummaryPartAddedEvent(BaseModel, frozen=True):
    """新增推理摘要分段時送出的事件。"""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    part: ResponseReasoningSummaryText


class ResponseReasoningSummaryTextDeltaEvent(BaseModel, frozen=True):
    """串流期間送出推理摘要文字增量時的事件。"""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    delta: str


class ResponseReasoningSummaryTextDoneEvent(BaseModel, frozen=True):
    """推理摘要文字完成時送出的事件。"""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    text: str


class ResponseReasoningSummaryPartDoneEvent(BaseModel, frozen=True):
    """推理摘要分段完成時送出的事件。"""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    part: ResponseReasoningSummaryText


class ResponseCompletedEvent(BaseModel, frozen=True):
    """回應完成時送出的事件。"""

    type: Literal["response.completed"] = "response.completed"
    sequence_number: int
    response: ResponsesResponse


ResponsesStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseOutputItemAddedEvent
    | ResponseContentPartAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputItemDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseCompletedEvent
)
