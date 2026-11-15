// The free models' public ratings, as vendored data from the kit's own sealed quality table
// (practice/trial-0/reference/ratings.json; the table is the same in every trial). This is only a
// fallback: when the brief at $MAKEFUN_PROMPT_FILE carries its own ratings document, main.ts uses
// that one (it is the trial's own copy) and never consults this table. A model the table does not
// name still routes: it gets a neutral rating and the probe's own capability columns.

export interface VendoredRating {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** The 26 free rows of the sealed table, best first. The paid row is deliberately absent: a paid model is never a legal pick. */
export const VENDORED_RATINGS: VendoredRating[] = [
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free', rating_milli: 1475, context_length: 204800, tools: true, vision: false, json: true },
  { provider: 'zai', model: 'glm-4.7-flash', rating_milli: 1470, context_length: 204800, tools: true, vision: false, json: true },
  { provider: 'cloudflare', model: 'glm-4.7-flash', rating_milli: 1470, context_length: 204800, tools: true, vision: false, json: true },
  { provider: 'sambanova', model: 'deepseek-v3.2', rating_milli: 1465, context_length: 163840, tools: true, vision: false, json: true },
  { provider: 'openrouter', model: 'deepseek/deepseek-v4.1:free', rating_milli: 1462, context_length: 163840, tools: true, vision: false, json: true },
  { provider: 'gemini', model: 'gemini-3.8-flash', rating_milli: 1458, context_length: 1048576, tools: true, vision: true, json: true },
  { provider: 'gemini', model: 'gemma-4-31b', rating_milli: 1452, context_length: 131072, tools: true, vision: true, json: false },
  { provider: 'sambanova', model: 'gemma-4-31b', rating_milli: 1452, context_length: 131072, tools: false, vision: true, json: false },
  { provider: 'ovh', model: 'qwen3.5-397b', rating_milli: 1440, context_length: 262144, tools: true, vision: false, json: false },
  { provider: 'openrouter', model: 'minimax/m3:free', rating_milli: 1436, context_length: 196608, tools: true, vision: false, json: true },
  { provider: 'groq', model: 'qwen3.8-27b', rating_milli: 1434, context_length: 131072, tools: true, vision: false, json: true },
  { provider: 'cerebras', model: 'qwen-3.8-27b', rating_milli: 1434, context_length: 131072, tools: true, vision: false, json: true },
  { provider: 'zai', model: 'glm-4.6v-flash', rating_milli: 1432, context_length: 204800, tools: true, vision: true, json: false },
  { provider: 'kilol', model: 'kilo-auto/free', rating_milli: 1420, context_length: 131072, tools: true, vision: false, json: false },
  { provider: 'cloudflare', model: 'nemotron-3.5-super-vl', rating_milli: 1414, context_length: 131072, tools: false, vision: true, json: false },
  { provider: 'gemini', model: 'gemma-4-26b', rating_milli: 1410, context_length: 32768, tools: false, vision: true, json: false },
  { provider: 'pollinations', model: 'openai-fast', rating_milli: 1390, context_length: 32768, tools: true, vision: false, json: false },
  { provider: 'groq', model: 'gpt-oss-120b', rating_milli: 1352, context_length: 131072, tools: true, vision: false, json: true },
  { provider: 'cerebras', model: 'gpt-oss-120b', rating_milli: 1352, context_length: 131072, tools: true, vision: false, json: true },
  { provider: 'cloudflare', model: 'llama-3.3-70b', rating_milli: 1330, context_length: 131072, tools: true, vision: false, json: false },
  { provider: 'ovh', model: 'llama-3.3-70b', rating_milli: 1330, context_length: 131072, tools: true, vision: false, json: false },
  { provider: 'groq', model: 'gpt-oss-20b', rating_milli: 1318, context_length: 131072, tools: true, vision: false, json: false },
  { provider: 'cerebras', model: 'llama-4-scout-17b', rating_milli: 1250, context_length: 131072, tools: true, vision: false, json: false },
  { provider: 'groq', model: 'llama-3.1-8b-instant', rating_milli: 1210, context_length: 131072, tools: true, vision: false, json: true },
  { provider: 'pollinations', model: 'mistral-nemo', rating_milli: 1180, context_length: 32768, tools: true, vision: false, json: false },
  { provider: 'kilol', model: 'kilo-fast', rating_milli: 1140, context_length: 131072, tools: true, vision: false, json: false },
];

/** The vendored row for one provider's model, or null when the table does not name it. */
export function vendoredRating(providerId: string, modelId: string): VendoredRating | null {
  return VENDORED_RATINGS.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}
