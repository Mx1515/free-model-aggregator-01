// The three wires: what the client speaks, what the providers speak, and the answers rebuilt between them.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The providers speak one of three
// wires — OpenAI, Anthropic or Gemini — so every request is translated into the provider's wire and every
// answer is rebuilt in the client's wire:
//
//   - a Chat Completions request passes through to an OpenAI provider almost unchanged, and an Anthropic
//     request is rewritten into one (system prompt, image blocks, tool definitions, tool results and tool
//     choice included);
//   - a Messages request is rewritten from a Chat Completions request (system messages, image_url parts,
//     tool calls and tool choice), and a Gemini call is built from either client wire (system instruction,
//     contents, function declarations, generation config);
//   - every upstream answer is buffered whole and re-emitted in the client's wire — JSON or the event
//     sequence the client asked for — with the provider's text whole, its usage counts as that wire
//     reported them, and its stop reason mapped. Buffering costs nothing at these sizes and guarantees
//     the exact text, usage and stop reason the provider answered.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The model listings and the rebuilt answers
// use fixed ids and timestamps for the same reason: a `created` that moved between runs would be a
// nondeterministic document.

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
/** A fixed id for rebuilt Chat Completions answers: nothing in the reply reads the clock. */
const FIXED_CHAT_ID = 'chatcmpl-aggregate';
/** A fixed id for rebuilt Messages answers. */
const FIXED_MESSAGE_ID = 'msg_aggregate';

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

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// --- small readers ---

/** A finite number, or 0 when the value is not one. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
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

// --- stop reasons, mapped between the three wires ---

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
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

// --- usage, in each wire's spelling, normalized to prompt/completion/total ---

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

/** One provider answer, normalized: the whole text, the usage it reported, and its stop token in its own wire's spelling. */
export interface UpstreamAnswer {
  text: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  stop: string | null;
}

// --- requests: one client wire rewritten into a provider wire ---

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

/** One Anthropic content block as Gemini parts: text or an image; tool results become text. */
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

// --- answers: one provider answer, in the client's wire ---

/** The first choice of a Chat Completions answer: its text, usage, and finish reason. */
export function parseOpenAIJson(body: unknown): UpstreamAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stop: null };
  const choice = Array.isArray(body.choices) && body.choices.length > 0 && isRecord(body.choices[0]) ? body.choices[0] : null;
  const message = choice !== null && isRecord(choice.message) ? choice.message : null;
  const text = message !== null ? textOf(message.content) : '';
  const stop = choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
  return { text, usage: usageOf(body.usage), stop };
}

/** One Chat Completions stream: the delta text joined, the usage it carried, and its finish reason. */
export function parseOpenAISSE(body: string): UpstreamAnswer {
  let text = '';
  let usage: UpstreamAnswer['usage'] = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(body)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usage)) usage = usageOf(payload.usage);
    const choice = Array.isArray(payload.choices) && payload.choices.length > 0 && isRecord(payload.choices[0]) ? payload.choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
  }
  return { text, usage, stop };
}

/** One Messages answer: its text blocks joined, its usage, and its stop reason. */
export function parseAnthropicJson(body: unknown): UpstreamAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stop: null };
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : null;
  return { text: textOf(body.content), usage: usageOfAnthropic(body.usage), stop };
}

/** One Messages stream: text deltas joined, the usage it carried, and its stop reason. */
export function parseAnthropicSSE(body: string): UpstreamAnswer {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  for (const payload of ssePayloads(body)) {
    if (!isRecord(payload)) continue;
    if (payload.type === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
    }
    if (payload.type === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
      text += payload.delta.text;
    }
    if (payload.type === 'message_delta' && isRecord(payload.delta)) {
      if (typeof payload.delta.stop_reason === 'string') stop = payload.delta.stop_reason;
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
    }
  }
  const usage = inputTokens === 0 && outputTokens === 0 ? null : { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
  return { text, usage, stop };
}

/** One Gemini answer: its first candidate's text, its usage, and its finish reason. */
export function parseGeminiJson(body: unknown): UpstreamAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stop: null };
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  const parts = first !== null && isRecord(first.content) ? first.content.parts : undefined;
  const text = Array.isArray(parts) ? parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n') : '';
  const finish = first !== null && typeof first.finishReason === 'string' ? first.finishReason : null;
  return { text, usage: usageOfGemini(body.usageMetadata), stop: finish };
}

/** One Gemini stream: every candidate's text joined, the usage it carried, and its finish reason. */
export function parseGeminiSSE(body: string): UpstreamAnswer {
  let text = '';
  let usage: UpstreamAnswer['usage'] = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(body)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usageMetadata)) usage = usageOfGemini(payload.usageMetadata);
    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
    if (first === null) continue;
    if (typeof first.finishReason === 'string' && first.finishReason !== '') stop = first.finishReason;
    const parts = isRecord(first.content) ? first.content.parts : undefined;
    if (Array.isArray(parts)) text += parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
  }
  return { text, usage, stop };
}

/** One Chat Completions answer in OpenAI's wire, from an answer in any provider wire. */
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

/** One Chat Completions answer as an SSE stream: a role chunk, the text in 13-character pieces, a finish chunk, then the usage when one was reported. */
export function openAISSE(model: string, answer: UpstreamAnswer): string {
  const chunks: string[] = [
    sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  ];
  for (const piece of splitText(answer.text)) {
    chunks.push(
      sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }),
    );
  }
  chunks.push(
    sseEvent(null, {
      id: FIXED_CHAT_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: answer.stop ?? 'stop' }],
    }),
  );
  if (answer.usage !== null) {
    chunks.push(
      sseEvent(null, {
        id: FIXED_CHAT_ID,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [],
        usage: answer.usage,
      }),
    );
  }
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

/** One Messages answer in Anthropic's wire, from an answer in any provider wire. */
export function anthropicJsonAnswer(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: answer.stop ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.usage?.prompt_tokens ?? 0, output_tokens: answer.usage?.completion_tokens ?? 0 },
  };
}

/** One Messages answer as an Anthropic event stream: message_start, one text block with 13-character deltas, then the stop and the usage. */
export function anthropicSSE(model: string, answer: UpstreamAnswer): string {
  const input = answer.usage?.prompt_tokens ?? 0;
  const output = answer.usage?.completion_tokens ?? 0;
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  events.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: answer.stop ?? 'end_turn', stop_sequence: null },
      usage: { output_tokens: output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}

// --- streams: one Chat Completions stream, translated into a Messages stream, event by event ---

/** One Chat Completions stream as a Messages event stream: the role and text deltas, the tool calls as tool_use blocks, and the stop reason as a message delta. */
export async function* anthropicEventsFromOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
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
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };

  const closeEvents = (): string[] => {
    const events: string[] = [];
    if (textIndex !== -1) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return events;
  };

  const finishEvents = (stop: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const events: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      if (payload === '[DONE]') {
        events.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      if (isRecord(parsed.usage)) {
        inputTokens = numberOrZero(parsed.usage.prompt_tokens);
        outputTokens = numberOrZero(parsed.usage.completion_tokens);
      }
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      events.push(...startEvents());
      const delta = isRecord(choice.delta) ? choice.delta : null;
      if (delta !== null && typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          events.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta?.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : null;
          events.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: fn !== null && typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : null;
        if (fn !== null && typeof fn.arguments === 'string' && fn.arguments !== '') {
          events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return events;
  };

  try {
    for (;;) {
      const read = await reader.read();
      if (read.done) break;
      buffer += decoder.decode(read.value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';
      for (const frame of frames) {
        for (const event of eventsOfFrame(frame)) yield event;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents().concat(finishEvents(pendingStop ?? 'end_turn'))) yield event;
  }
}

// --- one more answer translation: a Chat Completions answer, in Messages shape ---

/** One Chat Completions answer as a Messages answer: its text, its tool calls as tool_use blocks, its stop reason, its usage. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choice = Array.isArray(body.choices) && body.choices.length > 0 && isRecord(body.choices[0]) ? body.choices[0] : null;
  const message = choice !== null && isRecord(choice.message) ? choice.message : null;
  const content: Json[] = [];
  const text = message !== null ? textOf(message.content) : '';
  if (text !== '') content.push({ type: 'text', text });
  if (message !== null && Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      content.push({
        type: 'tool_use',
        id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        input: argumentsOf(call.function.arguments),
      });
    }
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: choice !== null && typeof choice.finish_reason === 'string' ? stopReasonOf(choice.finish_reason) : 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: isRecord(body.usage) ? numberOrZero(body.usage.prompt_tokens) : 0,
      output_tokens: isRecord(body.usage) ? numberOrZero(body.usage.completion_tokens) : 0,
    },
  };
}
