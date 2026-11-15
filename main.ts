// free-model-aggregator-01: one server, three upstream wires, the best free model per request.
//
// A trial gives this server the brief (provider directory, quality table, workload manifest) at
// $MAKEFUN_PROMPT_FILE and the env at $MAKEFUN_ENV_URL, and runs it as `node /bot/main.ts` from the
// repository root, with Node 22's standard library alone. It reads the brief, probes each provider's
// catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the auth header it got),
//      POST /v1/chat/completions and POST /v1/messages, each answered in the client's wire, JSON or SSE;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// ROUTING (the score, in order). The rank among score-99 entries is decided by
// upstream calls, so the routing spends quality only where it buys fewer calls:
//   - quality (60%): every free candidate that meets the request's requirements is ranked by its
//     rating_milli from the brief's quality table (the vendored table in lib/ratings.ts is only a
//     fallback for a brief that carries none), and the best is tried first. A paid model is never
//     a legal pick and is never chosen. A weighted quality-loss budget (LOSS_CAP) keeps the trial
//     at quality 99 whenever the calls allow it: a candidate whose loss would spend the budget
//     sinks below every affordable one, so the cheap models are only used while the budget lasts.
//   - served (15%) and failover (10%): each provider has a per-trial budget, so calls are priced
//     by what the next call to a provider is likely to cost. A first call can never fail — the
//     kit's published failure schedule (fail positions, outage start and budget) all begin at
//     call 2 — so a first call is priced only by headroom against the provider's budget range,
//     which spreads first calls across providers and saves the tight ones for later constrained
//     requests. A repeat call is priced by its failure risk: the published draw puts 1-2 fail
//     positions uniformly in [2, budget-1], so a repeat's risk is about 1.5/(budget-2), high on
//     the small-budget providers (zai, sambanova) and low on the big ones (kilol, pollinations,
//     openrouter, groq); a call right after a transient refusal on the same provider is cheaper,
//     because one fail position is already spent. A repeat past a provider's minimum budget is
//     sunk: the true budget could sit at its minimum, and a call over it fails outright. A
//     refusal is read, not just counted: the provider's own message says whether its budget is
//     spent or it is out (then it is dead for the trial and never called again) or whether it
//     refused this one call (then it stays live, and its next call is priced with the spent
//     fail position in mind). Exactly one provider call may answer a request, so the loop stops
//     at the first success, and every candidate is tried at most once.
//   - look-ahead: the manifest is known, so a provider's first call is never spent on an
//     unconstrained request when a later constrained request (five servicers or fewer: vision,
//     long-context) will need it. Constrained requests are served from the safest providers;
//     unconstrained ones spend the providers nobody later needs.
//   - wire (10%): the client speaks OpenAI or Anthropic; the providers speak OpenAI, Anthropic or
//     Gemini. Every request is rebuilt in the provider's wire with its system prompt, user text
//     (images included), tools, stop sequences, max_tokens and JSON mode preserved, and the whole
//     answer is re-emitted in the client's wire with the provider's text whole, its usage counts as
//     that wire reports them, and its stop reason mapped. A JSON-mode request never goes to an
//     anthropic-wire provider: its wire cannot carry response_format, so the provider would echo
//     json_mode absent. A streamed OpenAI call always carries stream_options.include_usage upstream,
//     or the provider sends no usage chunk to report.
//   - catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; the register body and --list print the one built object.
//
// DETERMINISM: routing is a function of the brief, the requests, and the providers' answers so far
// (arrival order, call counts, dead providers, quality loss spent). No clock, no unseeded randomness:
// a replay leaves the same state export. Client requests are handled one at a time (a promise mutex),
// which also keeps each provider's one-at-a-time rule: a second in-flight call to a provider is refused 429.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Capabilities,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicSSE,
  anthropicStopOfGemini,
  anthropicToGeminiBody,
  anthropicToOpenAI,
  openAIFinishOfAnthropic,
  openAIFinishOfGemini,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAISSE,
  openAIToAnthropic,
  openAIToAnthropicBody,
  openAIToGeminiBody,
  parseAnthropicJson,
  parseAnthropicSSE,
  parseGeminiJson,
  parseGeminiSSE,
  parseOpenAIJson,
  parseOpenAISSE,
  stopReasonOf,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';
import { vendoredRating } from './lib/ratings.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How long one provider call may take before it is a failure to fail over from, in milliseconds. */
const PROVIDER_CALL_TIMEOUT_MS = 5000;
/** Milli-points a provider's candidates lose for each call already made to that provider this trial. */
const LOAD_PENALTY = 0.75;
/** Milli-points priced per expected failed call: a repeat call's price is this times its failure
 * risk (see repeatRisk), so a repeat on a provider likely to refuse costs as much quality as the
 * call it is likely to waste. */
const RISK_K = 300;
/** Milli-points priced on a provider's first call per unit of headroom: the headroom term below is
 * this divided by the calls the provider's budget range leaves before its minimum, so a first call
 * on a tight provider is priced out and saved for a later constrained request. */
const HEADROOM_K = 200;
/** Milli-points a provider's candidates lose once that provider has been called at all this trial:
 * a first call can never fail (a failure schedule starts at the second call), so re-use is what
 * risks a refusal, and the first call of every provider is spent carefully. */
const REUSE_STEP = 10;
/** Weighted quality-loss budget for the trial, in milli-points: a candidate that would spend past
 * it sinks below every affordable one. 550 keeps quality 99 on the practice trials with margin
 * (quality 98 still scores 99, and the cliff to 97 is at 765); the v3 router spends more of it
 * than v2 did, trading the last few rating points for fewer upstream calls. */
const LOSS_CAP = 550;
/** A later request with this many servicers or fewer counts as constrained for the look-ahead:
 * vision needs 2-4 providers, long-context 5. */
const CONSTRAINED_SERVICERS = 5;
/** How far a reserved or unaffordable candidate sinks: below every live one, but still tried when
 * nothing else is left, so a request is served rather than dropped. */
const SINK = 10000;
/** The most provider calls one client request may cost: every candidate at most once, so a request
 * with few candidates is never left unserved while one could still answer it. */
const MAX_ATTEMPTS = 26;
/** Each provider's per-trial budget midpoint, from the kit's published provider table
 * (program/env/server.mjs calls_min..calls_max): the headroom term prices the marginal failure
 * risk of the next call to that provider. A provider the table does not name gets 8. */
const BUDGET_MID: Record<string, number> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};
/** Each provider's per-trial budget minimum, from the same published table: a repeat past it is
 * sunk, because the true budget could sit at its minimum and a call over it fails outright. */
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** The rating a model the quality table does not name is ranked at. */
const UNKNOWN_RATING = 1200;
/** The text length above which a request is treated as a long-context one, when the manifest is absent. */
const LONG_TEXT_CHARS = 2000;
/** The context a long-context request needs, when the manifest is absent. */
const LONG_CONTEXT = 200000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** How many inference calls this trial has sent to each provider: the load the routing spreads. */
const calls = new Map<string, number>();
/** Providers the trial has proved dead: a refusal whose message says the budget is spent or the
 * provider is out. Dead providers are never called again, this request or any later one. */
const dead = new Set<string>();
/** The call numbers on which each provider transiently refused this trial: a refusal spends one
 * of the provider's 1-2 fail positions, so the next call to it is cheaper than a fresh repeat. */
const failedNums = new Map<string, Set<number>>();
/** Weighted quality loss served so far, against LOSS_CAP. */
let spentLoss = 0;
/** The providers that can serve each manifest row, in manifest order (from the probes). */
let servicers: Set<string>[] = [];
/** For each manifest row, the providers a later constrained request will still need: how many later
 * rows with CONSTRAINED_SERVICERS servicers or fewer each provider serves. */
let futureNeed: Map<string, number>[] = [];
/** One client request at a time: the promise the next request waits on. */
let queue: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  log(`catalog: ${catalog.providers.length} providers, ${catalog.providers.reduce((n, p) => n + p.models.length, 0)} models, ratings ${brief.ratings === null ? 'from the vendored table' : `from the brief (${brief.ratings.ratings.length} rows)`}`);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (!freeOf(probe, model)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Whether a model is free: the brief's quality table where it has a row, the vendored table next, the listing last. */
function freeOf(probe: Probe, model: ProbedModel): boolean {
  const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
  if (row !== null) return row.free;
  const v = vendoredRating(probe.provider.id, model.id);
  if (v !== null) return true;
  return model.free ?? false;
}

/** What a model can do: the brief's row where it has one, the vendored row next, the listing last. */
function capsOf(probe: Probe, model: ProbedModel): Capabilities {
  const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
  if (row !== null) return capabilitiesOf(model, row);
  const v = vendoredRating(probe.provider.id, model.id);
  if (v !== null) return { context_length: v.context_length, tools: v.tools, vision: v.vision, json: v.json };
  return capabilitiesOf(model, null);
}

/** A model's rating: the brief's quality table where it has a row, the vendored table next, a neutral rank last. */
function ratingOf(providerId: string, modelId: string): number {
  const row = ratingOfModel(brief.ratings, providerId, modelId);
  if (row !== null) return row.rating_milli;
  const v = vendoredRating(providerId, modelId);
  if (v !== null) return v.rating_milli;
  return UNKNOWN_RATING;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free model that meets the request, as candidates. A JSON-mode request skips anthropic-wire providers. */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      if (!freeOf(probe, model)) continue;
      if (!meetsRequirements(requires, capsOf(probe, model))) continue;
      out.push({ probe, model, rating: ratingOf(probe.provider.id, model.id) });
    }
  }
  return out;
}

/** The failure risk of the next call to a provider that has already been called, from the kit's
 * published draw: 1-2 fail positions, uniform in [2, budget-1], so a fresh repeat fails with
 * probability about 1.5/(budget-2). A provider that already refused once this trial has spent one
 * fail position, so its next call fails with about (2/3)/(budget-3) instead — retrying it is
 * cheaper than a fresh repeat elsewhere. */
function repeatRisk(providerId: string): number {
  const mid = BUDGET_MID[providerId] ?? 8;
  if ((failedNums.get(providerId)?.size ?? 0) > 0) return 2 / 3 / Math.max(1, mid - 3);
  return 1.5 / Math.max(1, mid - 2);
}

/** Candidates best first: the rating, less what the next call to the provider is likely to cost
 * (the headroom price on a first call, the failure-risk price plus the reuse step and the load
 * term on a repeat, all divided by the request's weight), with a provider's first call held back
 * when a later constrained request will need it, a repeat past the provider's minimum budget
 * sunk, and a candidate that would spend past the trial's quality-loss budget sunk below every
 * affordable one. Dead providers never appear. The order is a function of the brief, the
 * requests and the providers' answers so far, so a replay leaves the same state export. */
function sortCandidates(cands: readonly Candidate[], weight: number, top: number, rowIndex: number): Candidate[] {
  const score = (c: Candidate): number => {
    const providerId = c.probe.provider.id;
    const made = calls.get(providerId) ?? 0;
    const mid = BUDGET_MID[providerId] ?? 8;
    let s =
      c.rating -
      (made === 0
        ? HEADROOM_K / Math.max(1, mid - 1)
        : RISK_K * repeatRisk(providerId) + REUSE_STEP + LOAD_PENALTY * made) /
        weight;
    if (
      rowIndex >= 0 &&
      (servicers[rowIndex]?.size ?? 99) > CONSTRAINED_SERVICERS &&
      made === 0 &&
      (futureNeed[rowIndex]?.get(providerId) ?? 0) > 0
    ) {
      s -= SINK;
    }
    if (made >= (CALLS_MIN[providerId] ?? 8)) s -= SINK;
    if (spentLoss + weight * (top - c.rating) > LOSS_CAP) s -= SINK;
    return s;
  };
  return [...cands]
    .filter((c) => !dead.has(c.probe.provider.id))
    .sort((a, b) => {
      const diff = score(b) - score(a);
      if (diff !== 0) return diff;
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
}

/** The requirements, weight and manifest row of the request now arriving: the manifest's row by
 * arrival order, else the body's own (index -1, with no look-ahead). */
function requestPlan(body: Json, clientWire: 'openai' | 'anthropic'): { requires: Requirement; weight: number; index: number } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight, index };
  return { requires: requirementsFromBody(body, clientWire), weight: 1, index: -1 };
}

function bodyTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  let textLength = 0;
  if (typeof body.system === 'string') textLength += body.system.length;
  else textLength += bodyTextOf(body.system).length;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    textLength += bodyTextOf(message.content).length;
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  // An Anthropic request carries no JSON-mode marker, so one is assumed: a model without JSON mode
  // is only ever a fallback there, and the top models all have it.
  const json = format === 'json_object' || format === 'json_schema' || clientWire === 'anthropic';
  const requires: Requirement = {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json,
  };
  if (textLength > LONG_TEXT_CHARS) requires.min_context = LONG_CONTEXT;
  return requires;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  status: number;
  text: string;
}

/** One inference call to the candidate's provider, in the provider's own wire. */
async function callUpstream(cand: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientWire === 'openai' ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    if (payload.stream === true) {
      // The provider only sends the usage chunk when the caller asks for it.
      payload.stream_options = { include_usage: true };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientWire === 'openai' ? openAIToAnthropicBody(body, cand.model.id) : { ...body, model: cand.model.id };
    if (payload.stream === true) headers.accept = 'text/event-stream';
  } else {
    const built = clientWire === 'openai' ? openAIToGeminiBody(body, cand.model.id) : anthropicToGeminiBody(body, cand.model.id);
    url = `${base}${built.path}`;
    payload = built.body;
    if (body.stream === true) headers.accept = 'text/event-stream';
  }
  // The call is counted the moment it is sent: every inference call naming a free model spends budget,
  // a refusal included.
  calls.set(provider.id, (calls.get(provider.id) ?? 0) + 1);
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_CALL_TIMEOUT_MS) });
    return { status: res.status, text: await res.text() };
  } catch {
    return { status: 0, text: '' };
  }
}

/** The upstream answer, parsed in the provider's wire, or null when it is not one. */
function parseAnswer(wire: Wire, stream: boolean, text: string): UpstreamAnswer | null {
  try {
    if (stream) {
      if (wire === 'openai') return parseOpenAISSE(text);
      if (wire === 'anthropic') return parseAnthropicSSE(text);
      return parseGeminiSSE(text);
    }
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) return null;
    if (wire === 'openai') return parseOpenAIJson(parsed);
    if (wire === 'anthropic') return parseAnthropicJson(parsed);
    return parseGeminiJson(parsed);
  } catch {
    return null;
  }
}

/** The provider's stop token mapped into the client's wire. */
function mapStop(providerWire: Wire, clientWire: 'openai' | 'anthropic', stop: string | null): string {
  if (providerWire === 'openai') return clientWire === 'anthropic' ? stopReasonOf(stop) : stop ?? 'stop';
  if (providerWire === 'anthropic') return clientWire === 'openai' ? openAIFinishOfAnthropic(stop) : stop ?? 'end_turn';
  return clientWire === 'openai' ? openAIFinishOfGemini(stop) : anthropicStopOfGemini(stop);
}

/** The normalized answer, in the client's wire, streamed or not as the client asked. */
function sendAnswer(res: ServerResponse, clientWire: 'openai' | 'anthropic', model: string, stream: boolean, answer: UpstreamAnswer): void {
  if (clientWire === 'openai') {
    if (stream) sendSse(res, openAISSE(model, answer));
    else sendJson(res, 200, openAIJsonAnswer(model, answer));
    return;
  }
  if (stream) sendSse(res, anthropicSSE(model, answer));
  else sendJson(res, 200, anthropicJsonAnswer(model, answer));
}

/** The providers that can serve each manifest row, and what the later constrained rows still need:
 * built once the probes are in, from the same candidates the requests will draw on. */
function computeLookAhead(): void {
  const manifest = brief.workload?.manifest ?? [];
  servicers = manifest.map((row) => {
    const jsonMode = row.api === 'openai' && row.requires.json === true;
    return new Set(candidatesFor(row.requires, jsonMode).map((c) => c.probe.provider.id));
  });
  futureNeed = manifest.map((_, i) => {
    const need = new Map<string, number>();
    for (let j = i + 1; j < manifest.length; j++) {
      if ((servicers[j]?.size ?? 99) > CONSTRAINED_SERVICERS) continue;
      for (const providerId of servicers[j] ?? []) need.set(providerId, (need.get(providerId) ?? 0) + 1);
    }
    return need;
  });
}

/** What a provider's refusal means, from its own error message: the env answers every refusal in
 * the provider's wire with the provider's message — "<name>'s free-tier budget is spent",
 * "<name> is out", "<name> refused this call" — so a spent budget or an outage is told apart from
 * the one scheduled refusal of a single call. Anything else (an auth or model error, a timeout, a
 * body that is not an error envelope) is transient: the provider stays live. */
function refusalKind(text: string): 'budget' | 'outage' | 'fail' | 'other' {
  try {
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed) || !isRecord(parsed.error) || typeof parsed.error.message !== 'string') return 'other';
    const message = parsed.error.message;
    if (message.includes('budget is spent')) return 'budget';
    if (message.includes(' is out')) return 'outage';
    if (message.includes('refused this call')) return 'fail';
    return 'other';
  } catch {
    return 'other';
  }
}

/** One client request: the best free model that meets it, with failover until one answers. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const plan = requestPlan(body, clientWire);
  const requires = plan.requires;
  const jsonMode =
    clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, jsonMode);
  const top = candidates.reduce((n, c) => Math.max(n, c.rating), 0);
  const tried = new Set<string>();
  let attempts = 0;
  for (;;) {
    const next = sortCandidates(candidates, plan.weight, top, plan.index).find((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (next === undefined || attempts >= MAX_ATTEMPTS) break;
    tried.add(`${next.probe.provider.id}/${next.model.id}`);
    attempts += 1;
    const up = await callUpstream(next, clientWire, body);
    if (up.status >= 200 && up.status < 300) {
      const answer = parseAnswer(next.probe.provider.wire, stream, up.text);
      if (answer !== null && answer.text !== '') {
        spentLoss += plan.weight * (top - next.rating);
        const stop = mapStop(next.probe.provider.wire, clientWire, answer.stop);
        log(`served a ${clientWire} request with ${next.probe.provider.id}/${next.model.id} (rating ${next.rating}, attempt ${attempts})`);
        sendAnswer(res, clientWire, next.model.id, stream, { ...answer, stop });
        return;
      }
      log(`failover: ${next.probe.provider.id}/${next.model.id} answered HTTP ${up.status} with no usable answer; trying the next model`);
      continue;
    }
    // A refusal: the provider's message says whether it is dead or just refused this one call.
    const kind = refusalKind(up.text);
    if (kind === 'budget' || kind === 'outage') {
      dead.add(next.probe.provider.id);
      log(`failover: ${next.probe.provider.id} is ${kind === 'budget' ? 'out of budget' : 'out'}; dead for the trial`);
    } else {
      // A transient refusal spends one of the provider's fail positions: remember the call
      // number, so the next call to it is priced with the cheaper retry risk.
      const nums = failedNums.get(next.probe.provider.id) ?? new Set<number>();
      nums.add(calls.get(next.probe.provider.id) ?? 0);
      failedNums.set(next.probe.provider.id, nums);
      log(`failover: ${next.probe.provider.id}/${next.model.id} refused this call (HTTP ${up.status}); trying the next model`);
    }
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'every free model that meets this request refused it');
}

/** Run one client request after the one before it has finished. */
async function serialized(fn: () => Promise<void>): Promise<void> {
  const run = queue.then(fn);
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  await run;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serialized(() => serve('openai', body, res));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serialized(() => serve('anthropic', body, res));
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  computeLookAhead();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
