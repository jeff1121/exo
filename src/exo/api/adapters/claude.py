"""用於轉換請求/回應的 Claude Messages API 轉接器。"""

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from exo.api.adapters.chat_completions import fetch_image_url
from exo.api.types import FinishReason, Usage
from exo.api.types.claude_api import (
    ClaudeContentBlock,
    ClaudeContentBlockDeltaEvent,
    ClaudeContentBlockStartEvent,
    ClaudeContentBlockStopEvent,
    ClaudeImageBlock,
    ClaudeInputJsonDelta,
    ClaudeMessageDelta,
    ClaudeMessageDeltaEvent,
    ClaudeMessageDeltaUsage,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessageStart,
    ClaudeMessageStartEvent,
    ClaudeMessageStopEvent,
    ClaudeStopReason,
    ClaudeTextBlock,
    ClaudeTextDelta,
    ClaudeThinkingBlock,
    ClaudeThinkingDelta,
    ClaudeToolResultBlock,
    ClaudeToolUseBlock,
    ClaudeUsage,
)
from exo.shared.logging import logger
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import (
    Base64Image,
    ChatTemplateValue,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)


def finish_reason_to_claude_stop_reason(
    finish_reason: FinishReason | None,
) -> ClaudeStopReason | None:
    """將 OpenAI finish_reason 對應到 Claude stop_reason。"""
    if finish_reason is None:
        return None
    mapping: dict[FinishReason, ClaudeStopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    return mapping.get(finish_reason, "end_turn")


def _extract_tool_result_text(block: ClaudeToolResultBlock) -> str:
    """從 tool_result 的 content 欄位擷取純文字。"""
    if block.content is None:
        return ""
    if isinstance(block.content, str):
        return block.content
    return "".join(
        sub.text for sub in block.content if isinstance(sub, ClaudeTextBlock)
    )


# 比對「x-anthropic-billing-header: ...;」（可含可選的結尾換行）
# 或其他每次請求都會改變、破壞 KV 前綴快取的遙測標頭。
_VOLATILE_HEADER_RE = re.compile(r"^x-anthropic-[^\n]*;\n?", re.MULTILINE)


def _strip_volatile_headers(text: str) -> str:
    """從 system prompt 文字中移除 Anthropic 計費/遙測標頭。

    Claude Code 會在前面加上如 'x-anthropic-billing-header: cc_version=...;
    cc_entrypoint=...; cch=...;' 這類包含每次請求內容雜湊的標頭。它們
    每次請求都會改變，並破壞 KV 前綴快取（前綴約在 20 個 token 處即分歧，
    而非可匹配數千個對話 token）。
    """
    return _VOLATILE_HEADER_RE.sub("", text)


async def handle_image_block(block: ClaudeImageBlock) -> Base64Image | None:
    if block.source.type == "base64" and block.source.data:
        return Base64Image(block.source.data)
    elif block.source.type == "url" and block.source.url:
        try:
            return await fetch_image_url(block.source.url)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to fetch image at {block.source.url}"
            )

    return None


async def claude_request_to_text_generation(
    request: ClaudeMessagesRequest,
) -> TextGenerationTaskParams:
    # 處理 system 訊息
    instructions: str | None = None
    chat_template_messages: list[dict[str, ChatTemplateValue]] = []
    images: list[Base64Image] = []

    if request.system:
        if isinstance(request.system, str):
            instructions = request.system
        else:
            instructions = "".join(block.text for block in request.system)

        instructions = _strip_volatile_headers(instructions)
        chat_template_messages.append(
            {"role": "system", "content": InputMessageContent(instructions)}
        )

    # 將訊息轉為 input
    input_messages: list[InputMessage] = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            input_messages.append(
                InputMessage(role=msg.role, content=InputMessageContent(msg.content))
            )
            chat_template_messages.append(
                {"role": msg.role, "content": InputMessageContent(msg.content)}
            )
            continue

        # 處理結構化內容區塊
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[ClaudeToolResultBlock] = []
        has_images = False

        for block in msg.content:
            if isinstance(block, ClaudeTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ClaudeImageBlock):
                if (img := await handle_image_block(block)) is not None:
                    has_images = True
                    images.append(img)
            elif isinstance(block, ClaudeThinkingBlock):
                thinking_parts.append(block.thinking)
            elif isinstance(block, ClaudeToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            else:
                tool_results.append(block)
                if isinstance(block.content, list):
                    for sub in block.content:
                        if (
                            isinstance(sub, ClaudeImageBlock)
                            and (img := await handle_image_block(sub)) is not None
                        ):
                            has_images = True
                            images.append(img)

        content = "".join(text_parts)
        reasoning_content = "".join(thinking_parts) if thinking_parts else None

        # 由文字內容建立 InputMessage
        if msg.role in ("user", "assistant"):
            input_messages.append(
                InputMessage(role=msg.role, content=InputMessageContent(content))
            )

        # 建立並保留工具結構的 chat_template_messages
        if tool_calls:
            chat_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            if reasoning_content:
                chat_msg["reasoning_content"] = reasoning_content
            chat_template_messages.append(chat_msg)
        elif tool_results:
            for tr in tool_results:
                chat_template_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": _extract_tool_result_text(tr),
                    }
                )
        elif has_images:
            multimodal_content: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, ClaudeTextBlock):
                    multimodal_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ClaudeImageBlock):
                    multimodal_content.append({"type": "image"})
            chat_msg = {"role": msg.role, "content": multimodal_content}
            if reasoning_content:
                chat_msg["reasoning_content"] = reasoning_content
            chat_template_messages.append(chat_msg)
        else:
            chat_msg = {"role": msg.role, "content": content}
            if reasoning_content:
                chat_msg["reasoning_content"] = reasoning_content
            chat_template_messages.append(chat_msg)

    # 將 Claude 工具定義轉為 OpenAI 風格的 function tools
    tools: list[dict[str, Any]] | None = None
    if request.tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]

    enable_thinking: bool | None = None
    if request.thinking is not None:
        enable_thinking = request.thinking.type in ("enabled", "adaptive")

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content=InputMessageContent(""))],
        instructions=InputMessageContent(instructions) if instructions else None,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop_sequences,
        stream=request.stream,
        tools=tools,
        enable_thinking=enable_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        images=images,
    )


async def collect_claude_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # 這裡使用 AsyncGenerator[str] 而不是直接回傳 ChatCompletionReponse，因為
    # FastAPI 對取消處理較佳，但不知為何不會自動序列化
    """收集所有 token chunks 並回傳單一 ClaudeMessagesResponse。"""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_use_blocks: list[ClaudeToolUseBlock] = []
    stop_reason: ClaudeStopReason | None = None
    last_usage: Usage | None = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if isinstance(chunk, PrefillProgressChunk):
            continue

        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
            for tool in chunk.tool_calls:
                tool_use_blocks.append(
                    ClaudeToolUseBlock(
                        id=f"toolu_{tool.id}",
                        name=tool.name,
                        input=json.loads(tool.arguments),  # pyright: ignore[reportAny]
                    )
                )
            stop_reason = "tool_use"
            continue

        if chunk.is_thinking:
            thinking_parts.append(chunk.text)
        else:
            text_parts.append(chunk.text)

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts)

    # 建立內容區塊
    content: list[ClaudeContentBlock] = []
    if combined_thinking:
        content.append(ClaudeThinkingBlock(thinking=combined_thinking))
    if combined_text:
        content.append(ClaudeTextBlock(text=combined_text))
    content.extend(tool_use_blocks)

    # 若完全沒有內容，加入空文字區塊
    if not content:
        content.append(ClaudeTextBlock(text=""))

    # 若有可用資料則使用實際 usage 資料
    input_tokens = last_usage.prompt_tokens if last_usage else 0
    output_tokens = last_usage.completion_tokens if last_usage else 0

    yield ClaudeMessagesResponse(
        id=f"msg_{command_id}",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    ).model_dump_json()
    return


async def generate_claude_stream(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """由 TokenChunks 產生 Claude Messages API 串流事件。"""
    # 初始 message_start 事件
    initial_message = ClaudeMessageStart(
        id=f"msg_{command_id}",
        model=model,
        content=[],
        stop_reason=None,
        usage=ClaudeUsage(input_tokens=0, output_tokens=0),
    )
    start_event = ClaudeMessageStartEvent(message=initial_message)
    yield f"event: message_start\ndata: {start_event.model_dump_json()}\n\n"

    output_tokens = 0
    stop_reason: ClaudeStopReason | None = None
    last_usage: Usage | None = None
    next_block_index = 0

    # 追蹤是否已開始 thinking/text 區塊
    thinking_block_started = False
    thinking_block_index = -1
    text_block_started = False
    text_block_index = -1

    async for chunk in chunk_stream:
        if isinstance(chunk, PrefillProgressChunk):
            continue

        if isinstance(chunk, ErrorChunk):
            # 關閉文字區塊並結束
            break

        last_usage = chunk.usage or last_usage

        if isinstance(chunk, ToolCallChunk):
            stop_reason = "tool_use"

            # 輸出 tool_use 內容區塊
            for tool in chunk.tool_calls:
                tool_id = f"toolu_{tool.id}"
                tool_input_json = tool.arguments

                # tool_use 的 content_block_start
                tool_block_start = ClaudeContentBlockStartEvent(
                    index=next_block_index,
                    content_block=ClaudeToolUseBlock(
                        id=tool_id, name=tool.name, input={}
                    ),
                )
                yield f"event: content_block_start\ndata: {tool_block_start.model_dump_json()}\n\n"

                # 含 input_json_delta 的 content_block_delta
                tool_delta_event = ClaudeContentBlockDeltaEvent(
                    index=next_block_index,
                    delta=ClaudeInputJsonDelta(partial_json=tool_input_json),
                )
                yield f"event: content_block_delta\ndata: {tool_delta_event.model_dump_json()}\n\n"

                # content_block_stop
                tool_block_stop = ClaudeContentBlockStopEvent(index=next_block_index)
                yield f"event: content_block_stop\ndata: {tool_block_stop.model_dump_json()}\n\n"

                next_block_index += 1
            continue

        output_tokens += 1  # 每個 chunk 計為一個 token

        if chunk.is_thinking:
            # 在第一個 thinking token 開始時建立 thinking 區塊
            if not thinking_block_started:
                thinking_block_started = True
                thinking_block_index = next_block_index
                next_block_index += 1
                block_start = ClaudeContentBlockStartEvent(
                    index=thinking_block_index,
                    content_block=ClaudeThinkingBlock(thinking=""),
                )
                yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

            delta_event = ClaudeContentBlockDeltaEvent(
                index=thinking_block_index,
                delta=ClaudeThinkingDelta(thinking=chunk.text),
            )
            yield f"event: content_block_delta\ndata: {delta_event.model_dump_json()}\n\n"
        else:
            # 轉換到文字時關閉 thinking 區塊
            if thinking_block_started and text_block_index == -1:
                block_stop = ClaudeContentBlockStopEvent(index=thinking_block_index)
                yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

            # 在第一個文字 token 開始時建立文字區塊
            if not text_block_started:
                text_block_started = True
                text_block_index = next_block_index
                next_block_index += 1
                block_start = ClaudeContentBlockStartEvent(
                    index=text_block_index,
                    content_block=ClaudeTextBlock(text=""),
                )
                yield f"event: content_block_start\ndata: {block_start.model_dump_json()}\n\n"

            delta_event = ClaudeContentBlockDeltaEvent(
                index=text_block_index,
                delta=ClaudeTextDelta(text=chunk.text),
            )
            yield f"event: content_block_delta\ndata: {delta_event.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            stop_reason = finish_reason_to_claude_stop_reason(chunk.finish_reason)

    # 若可用，使用 usage 中的實際 token 數
    if last_usage is not None:
        output_tokens = last_usage.completion_tokens

    # 關閉所有仍開啟的區塊
    if thinking_block_started and text_block_index == -1:
        block_stop = ClaudeContentBlockStopEvent(index=thinking_block_index)
        yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

    if text_block_started:
        block_stop = ClaudeContentBlockStopEvent(index=text_block_index)
        yield f"event: content_block_stop\ndata: {block_stop.model_dump_json()}\n\n"

    if not thinking_block_started and not text_block_started:
        empty_start = ClaudeContentBlockStartEvent(
            index=0, content_block=ClaudeTextBlock(text="")
        )
        yield f"event: content_block_start\ndata: {empty_start.model_dump_json()}\n\n"
        empty_stop = ClaudeContentBlockStopEvent(index=0)
        yield f"event: content_block_stop\ndata: {empty_stop.model_dump_json()}\n\n"

    # message_delta
    message_delta = ClaudeMessageDeltaEvent(
        delta=ClaudeMessageDelta(stop_reason=stop_reason),
        usage=ClaudeMessageDeltaUsage(output_tokens=output_tokens),
    )
    yield f"event: message_delta\ndata: {message_delta.model_dump_json()}\n\n"

    # message_stop
    message_stop = ClaudeMessageStopEvent()
    yield f"event: message_stop\ndata: {message_stop.model_dump_json()}\n\n"
