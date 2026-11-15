// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The upstream wires this entry speaks (anthropic and gemini), and the answers
// it rebuilds for the client. The entry buffers every upstream answer, then
// re-emits it in the client's wire: the responses are a few hundred bytes, so
// buffering costs nothing and guarantees the exact text, usage and stop reason
// the provider answered, in the client's own event order. Nothing here reads
// the clock or any randomness: the same upstream answer gives the same bytes.
// ---------------------------------------------------------------------------

/** One provider answer, normalized: the whole text, the usage it reported, and its stop token in its own wire's spelling. */
export interface UpstreamAnswer {
  text: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  stop: string | null;
}

function usageOf(value: unknown): UpstreamAnswer['usage'] {
  if (!isRecord(value)) return null;
  const input = numberOrZero(value.prompt_tokens);
  const output = numberOrZero(value.completion_tokens);
  const total = typeof value.total_tokens === 'number' ? value.total_tokens : input + output;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: total };
}

function usageOfAnthropic(value: unknown): UpstreamAnswer['usage'] {
  if (!isRecord(value)) return null;
  const input = numberOrZero(value.input_tokens);
  const output = numberOrZero(value.output_tokens);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

function usageOfGemini(value: unknown): UpstreamAnswer['usage'] {
  if (!isRecord(value)) return null;
  const input = numberOrZero(value.promptTokenCount);
  const output = numberOrZero(value.candidatesTokenCount);
  const total = typeof value.totalTokenCount === 'number' ? value.totalTokenCount : input + output;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: total };
}

/** The text in fixed-size pieces, the way the providers chunk their streams. */
function splitText(text: string, size = 13): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

/** One SSE frame's `data:` payloads, in order; `[DONE]` and unparsable lines are skipped. */
function ssePayloads(text: string): unknown[] {
  const out: unknown[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        out.push(JSON.parse(payload));
      } catch {
        // Not JSON: not an answer frame.
      }
    }
  }
  return out;
}

/** A Chat Completions SSE stream, parsed: the concatenated deltas, the usage chunk, the finish reason. */
export function parseOpenAISSE(text: string): UpstreamAnswer {
  let answer = '';
  let usage: UpstreamAnswer['usage'] = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usage)) usage = usageOf(payload.usage);
    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answer += delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
  }
  return { text: answer, usage, stop };
}

/** A Chat Completions JSON body, parsed. */
export function parseOpenAIJson(body: Json): UpstreamAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  return {
    text,
    usage: usageOf(body.usage),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
  };
}

/** A Messages SSE stream, parsed: the concatenated text deltas, the usage, the stop reason. */
export function parseAnthropicSSE(text: string): UpstreamAnswer {
  let answer = '';
  let usage: UpstreamAnswer['usage'] = null;
  let stop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    if (payload.type === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
    }
    if (payload.type === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
      answer += payload.delta.text;
    }
    if (payload.type === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') stop = payload.delta.stop_reason;
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
    }
  }
  if (inputTokens !== 0 || outputTokens !== 0) usage = { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
  return { text: answer, usage, stop };
}

/** A Messages JSON body, parsed. */
export function parseAnthropicJson(body: Json): UpstreamAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let answer = '';
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') answer += block.text;
  }
  return {
    text: answer,
    usage: usageOfAnthropic(body.usage),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
  };
}

/** A Gemini SSE stream, parsed: the concatenated candidate text, the usage metadata, the finish reason. */
export function parseGeminiSSE(text: string): UpstreamAnswer {
  let answer = '';
  let usage: UpstreamAnswer['usage'] = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usageMetadata)) usage = usageOfGemini(payload.usageMetadata);
    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (typeof candidate.finishReason === 'string' && candidate.finishReason !== '') stop = candidate.finishReason;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') answer += part.text;
    }
  }
  return { text: answer, usage, stop };
}

/** A Gemini JSON body, parsed. */
export function parseGeminiJson(body: Json): UpstreamAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let answer = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') answer += part.text;
  }
  return {
    text: answer,
    usage: usageOfGemini(body.usageMetadata),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
  };
}

/** A fixed id for rebuilt Chat Completions answers: nothing in the reply reads the clock. */
const FIXED_CHAT_ID = 'chatcmpl-aggregate';
/** A fixed id for rebuilt Messages answers. */
const FIXED_MESSAGE_ID = 'msg_aggregate';

/** The normalized answer as a Chat Completions JSON object naming `model`. */
export function openAIJsonAnswer(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: answer.stop ?? 'stop',
        logprobs: null,
      },
    ],
    usage: answer.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

/** The normalized answer as a Chat Completions SSE stream naming `model`: role, content, finish, usage, [DONE]. */
export function openAISSE(model: string, answer: UpstreamAnswer): string {
  const out: string[] = [];
  const base = { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model };
  out.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (const piece of splitText(answer.text)) {
    out.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  out.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: answer.stop ?? 'stop' }] }));
  if (answer.usage !== null) {
    out.push(sseEvent(null, { ...base, choices: [], usage: answer.usage }));
  }
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** The normalized answer as a Messages JSON object naming `model`. */
export function anthropicJsonAnswer(model: string, answer: UpstreamAnswer): Json {
  const usage = answer.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: answer.stop ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** The normalized answer as the Messages event sequence naming `model`. */
export function anthropicSSE(model: string, answer: UpstreamAnswer): string {
  const out: string[] = [];
  const usage = answer.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(answer.text)) {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: answer.stop ?? 'end_turn', stop_sequence: null },
      usage: { output_tokens: usage.completion_tokens },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}

/** A Gemini finish reason as an OpenAI finish_reason. */
export function openAIFinishOfGemini(finish: string | null): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** A Gemini finish reason as an Anthropic stop_reason. */
export function anthropicStopOfGemini(finish: string | null): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function openAIFinishOfAnthropic(stop: string | null): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** One OpenAI message's image_url value as a Gemini part: inline data for data URLs, a file reference otherwise. */
function geminiImagePart(url: string): Json {
  const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } };
  return { fileData: { mimeType: 'image/png', fileUri: url } };
}

/** One OpenAI content part as Gemini parts: text or an image. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
  }
  return parts;
}

/** One Anthropic content block as Gemini parts: text or an image; null for the rest. */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: source.url } });
      }
    } else if (block.type === 'tool_result') {
      const text = toolResultText(block.content);
      if (text !== '') parts.push({ text });
    }
  }
  return parts;
}

/** One OpenAI tool definition as a Gemini function declaration. */
function geminiFunctionOfOpenAI(tool: unknown): Json | null {
  if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') return null;
  return {
    name: tool.function.name,
    description: typeof tool.function.description === 'string' ? tool.function.description : '',
    parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
  };
}

/** One Anthropic tool definition as a Gemini function declaration. */
function geminiFunctionOfAnthropic(tool: unknown): Json | null {
  if (!isRecord(tool) || typeof tool.name !== 'string') return null;
  return {
    name: tool.name,
    description: typeof tool.description === 'string' ? tool.description : '',
    parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
  };
}

/** The generation config for a Gemini call: the output cap, stops, JSON mode, sampling. */
function geminiGenerationConfig(source: Json): Json {
  const config: Json = {};
  if (typeof source.max_tokens === 'number') config.maxOutputTokens = source.max_tokens;
  const stops = Array.isArray(source.stop) ? source.stop : Array.isArray(source.stop_sequences) ? source.stop_sequences : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(source.response_format) && (source.response_format.type === 'json_object' || source.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (typeof source.temperature === 'number') config.temperature = source.temperature;
  if (typeof source.top_p === 'number') config.topP = source.top_p;
  return config;
}

/**
 * A Chat Completions request as a Gemini generate call naming `model`: the system prompt becomes the
 * system instruction, user and assistant messages become contents, image_url parts become inline data,
 * tools become function declarations, and the output cap, stops, JSON mode and sampling pass through.
 */
export function openAIToGeminiBody(body: Json, model: string): { path: string; body: Json } {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemTexts: string[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfOpenAI(raw.content);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message, or a tool message carried as user text: the provider reads the last user text.
    const parts = geminiPartsOfOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemTexts.length > 0) out.systemInstruction = { parts: [{ text: systemTexts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((tool) => {
      const fn = geminiFunctionOfOpenAI(tool);
      return fn === null ? [] : [fn];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = geminiGenerationConfig(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const stream = body.stream === true;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`, body: out };
}

/**
 * A Messages request as a Gemini generate call naming `model`: the system prompt becomes the system
 * instruction, user and assistant messages become contents, image blocks become inline data,
 * tool_result blocks become text, and the output cap, stops and sampling pass through.
 */
export function anthropicToGeminiBody(body: Json, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfAnthropic(raw.content);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts = geminiPartsOfAnthropic(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((tool) => {
      const fn = geminiFunctionOfAnthropic(tool);
      return fn === null ? [] : [fn];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = geminiGenerationConfig(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const stream = body.stream === true;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`, body: out };
}

/**
 * A Chat Completions request as a Messages request naming `model`: the system messages become one system
 * prompt, image_url parts become image blocks, tool definitions and tool choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. JSON mode has no Anthropic
 * spelling and is dropped: never route a JSON-mode request to an anthropic-wire provider.
 */
export function openAIToAnthropicBody(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemTexts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    const content: Json[] = [];
    const parts = Array.isArray(raw.content) ? raw.content : typeof raw.content === 'string' ? [{ type: 'text', text: raw.content }] : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') {
        content.push({ type: 'text', text: part.text });
        continue;
      }
      if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
        if (dataUrl !== null) content.push({ type: 'image', source: { type: 'base64', media_type: dataUrl[1], data: dataUrl[2] } });
        else content.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }

  const out: Json = { model, messages };
  if (systemTexts.length > 0) out.system = systemTexts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'any' || choice.type === 'required') out.tool_choice = { type: 'any' };
    else if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      out.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }
  if (body.stream === true) out.stream = true;
  return out;
}

