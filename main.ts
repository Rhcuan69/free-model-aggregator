// free-model-aggregator-01: one doorway for two APIs over ten free providers.
//
// A client speaks OpenAI Chat Completions or Anthropic Messages to this server.
// The server answers each request with the best free model it can reach, no matter
// which of the three wires (OpenAI, Anthropic, Gemini) that model lives behind,
// and falls over to the next-best model whenever a provider refuses. It registers
// the catalog it found with the trial env, then exits when the env says the work
// is done. `node main.ts --list` prints the catalog and stops.
//
// The design is deliberately boring: every request is first reduced to one neutral
// shape (system text, messages of typed parts, tools, caps, stops, JSON mode), the
// neutral shape is rendered into whichever wire the chosen provider speaks, the
// provider's answer is reduced to one neutral answer (text, stop family, usage,
// tool calls), and that answer is rendered into whichever wire the client spoke.
// Routing is a single ordered list of free candidates, best rating first, with a
// small per-provider call allowance and a strike rule for refusals, so the server
// spreads its calls instead of draining one provider's free tier.
//
// Nothing here reads the wall clock or an unseeded random source: every choice is
// a function of the brief, the requests seen so far, and the providers' answers,
// so a replay of the trial leaves the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small value helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

function isObj(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function asStr(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function asArr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

/** Parse a JSON string that may hold a JSON object; null when it does not. */
function parseJsonObj(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isObj(v) ? v : null;
  } catch {
    return null;
  }
}

/** A JSON value parsed as an object, or {} when it is not one. */
function argsObj(v: unknown): Json {
  if (isObj(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const p = parseJsonObj(v);
    if (p !== null) return p;
  }
  return {};
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  base: string; // the provider's own path, e.g. /p/groq, or an absolute URL
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number; // milli-points
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface WorkItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  needs: Need;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: WorkItem[];
  problem: string | null;
}

const DIR_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Index of the `}` that closes the object opening at `at`, or -1. Strings are skipped. */
function closeBrace(text: string, at: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = at; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that names a `schema`, in order of appearance. */
function schemaDocs(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closeBrace(text, at);
    if (end === -1) continue;
    const doc = parseJsonObj(text.slice(at, end + 1));
    if (doc !== null && typeof doc.schema === 'string') found.push(doc);
  }
  return found;
}

function dirFrom(v: unknown): ProviderDir | null {
  if (!isObj(v)) return null;
  const id = asStr(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthKind =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asStr(v.path) !== ''
      ? asStr(v.path)
      : asStr(v.base_path) !== ''
        ? asStr(v.base_path)
        : asStr(v.base_url) !== ''
          ? asStr(v.base_url)
          : asStr(v.base);
  return { id, name: asStr(v.name) !== '' ? asStr(v.name) : id, wire, auth, base, keyEnv: asStr(v.key_env), key: asStr(v.key) };
}

function ratingFrom(v: unknown): RatingRow | null {
  if (!isObj(v)) return null;
  const model = asStr(v.model);
  const provider = asStr(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: asNum(v.rating_milli) ?? 0,
    context: asNum(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workFrom(v: unknown): WorkItem | null {
  if (!isObj(v)) return null;
  const id = asStr(v.id);
  if (id === '') return null;
  const needs = isObj(v.requires) ? v.requires : {};
  const out: Need = {};
  if (needs.tools === true) out.tools = true;
  if (needs.vision === true) out.vision = true;
  if (needs.json === true) out.json = true;
  if (typeof needs.min_context === 'number' && Number.isFinite(needs.min_context)) out.minContext = needs.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    needs: out,
  };
}

/** Read the brief at `path`: the three documents, found by their schema names. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === DIR_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, problem: `no ${DIR_SCHEMA} document in the brief` };
  const providers = asArr(dirDoc.providers).flatMap((v) => {
    const p = dirFrom(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : asArr(ratingsDoc.ratings).flatMap((v) => {
          const r = ratingFrom(v);
          return r === null ? [] : [r];
        });
  const workDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workDoc === null
      ? []
      : asArr(workDoc.manifest).flatMap((v) => {
          const w = workFrom(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, problem: null };
}

/** The quality-table row for one provider's model, when the table holds one. */
function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// ---------------------------------------------------------------------------
// Catalog probe: ask each provider what it serves, in its own wire
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  display: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderDir;
  ok: boolean;
  models: ListedModel[];
  problem: string | null;
}

/** The key header(s) a provider wants, from its auth scheme. */
function keyHeaders(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A path joined to the env origin; an absolute base is used as it stands. */
function joinOrigin(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** A listing's price block says free when every price it names is zero; null when it says nothing. */
function pricedFree(v: Json): boolean | null {
  const pricing = isObj(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const nums = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (nums.length === 0) return null;
  return nums.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isObj(v)) return null;
  const id = wire === 'gemini' ? asStr(v.name).replace(/^models\//, '') : asStr(v.id);
  if (id === '') return null;
  const display = asStr(v.display_name) !== '' ? asStr(v.display_name) : asStr(v.displayName) !== '' ? asStr(v.displayName) : id;
  return {
    id,
    display,
    free: asBool(v.free) ?? pricedFree(v),
    context: asNum(v.context_length) ?? asNum(v.context_window) ?? asNum(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isObj(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, tolerating every failure. */
async function probeOne(envUrl: string, provider: ProviderDir, timeoutMs = 5000): Promise<Probe> {
  const root = joinOrigin(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${root}/v1beta/models` : `${root}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...keyHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], problem: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: listedModels(provider.wire, body), problem: null };
  } catch (err) {
    return { provider, ok: false, models: [], problem: `GET ${url} failed: ${(err as Error).message}` };
  }
}

// ---------------------------------------------------------------------------
// The catalog: the document --list prints and the register body carries
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogDoc {
  providers: CatalogProvider[];
}

function byId(a: { id: string }, b: { id: string }): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 */
function modelIsFree(listed: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return listed.free ?? false;
}

/** What a model can do: the quality table's row first, the listing's fields second. */
function modelCaps(listed: ListedModel, row: RatingRow | null): { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context: row?.context ?? listed.context ?? null,
    tools: row?.tools ?? listed.tools ?? null,
    vision: row?.vision ?? listed.vision ?? null,
    json: row?.json ?? listed.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's needs; an unknown never meets one. */
function capsMeet(needs: Need, caps: { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (needs.tools === true && caps.tools !== true) return false;
  if (needs.vision === true && caps.vision !== true) return false;
  if (needs.json === true && caps.json !== true) return false;
  if (needs.minContext !== undefined && (caps.context === null || caps.context < needs.minContext)) return false;
  return true;
}

/** Build the one catalog from the probes and the quality table. */
function buildCatalog(brief: Brief, probes: Probe[]): CatalogDoc {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const listed of [...probe.models].sort(byId)) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const row = ratingRow(brief, probe.provider.id, listed.id);
      models.push({ id: listed.id, free: modelIsFree(listed, row), context_length: row?.context ?? listed.context ?? null });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: one ordered candidate list, a call allowance, a strike rule
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  model: ListedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * How many calls a provider may take before this server looks elsewhere. These
 * are the smallest budgets the kit's sealed provider table allows for a trial,
 * so staying under them never spends a budget a trial might not have. A refusal
 * still moves the server on at once.
 */
const CALL_ALLOWANCE: Record<string, number> = {
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
const DEFAULT_ALLOWANCE = 4;

interface ProviderState {
  calls: number; // every attempt, refusals included: the budget counts them too
  dead: boolean;
  lastRefused: boolean; // the previous attempt to this provider refused
}

/** Every free model that can serve `needs`, best rating first. */
function candidatesFor(brief: Brief, probes: Probe[], needs: Need): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const listed of probe.models) {
      const row = ratingRow(brief, probe.provider.id, listed.id);
      if (!modelIsFree(listed, row)) continue;
      if (!capsMeet(needs, modelCaps(listed, row))) continue;
      // The Anthropic wire carries no JSON mode, so it can never echo one back.
      if (needs.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model: listed, row, rating: row?.rating ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || byId(a.model, b.model));
  return out;
}

/** The free models this server lists on GET /v1/models, in catalog order. */
function servedModels(brief: Brief, probes: Probe[]): { id: string; provider: string; display: string }[] {
  const out: { id: string; provider: string; display: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const listed of [...probe.models].sort(byId)) {
      if (seen.has(listed.id)) continue;
      const row = ratingRow(brief, probe.provider.id, listed.id);
      if (!modelIsFree(listed, row)) continue;
      seen.add(listed.id);
      out.push({ id: listed.id, provider: probe.provider.id, display: listed.display });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The neutral request: one shape for both client wires, three builders out
// ---------------------------------------------------------------------------

type Part =
  | { kind: 'text'; text: string }
  | { kind: 'imageData'; mime: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; id: string; content: string };

interface NormMsg {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
}

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

interface NormReq {
  systemText: string;
  messages: NormMsg[];
  tools: ToolDef[];
  toolChoice: 'auto' | 'required' | { name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text of a string, or of a block array's text blocks joined with newlines. */
function blockText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isObj(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Split a data: URL into its media type and base64 payload; null when it is not one. */
function splitDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** One OpenAI content part as a neutral part; null for parts this trial never sends. */
function openaiPart(p: unknown): Part | null {
  if (!isObj(p)) return null;
  if (p.type === 'text' && typeof p.text === 'string') return { kind: 'text', text: p.text };
  if (p.type === 'image_url' && isObj(p.image_url) && typeof p.image_url.url === 'string') {
    const split = splitDataUrl(p.image_url.url);
    if (split !== null) return { kind: 'imageData', mime: split.mime, data: split.data };
    return { kind: 'imageUrl', url: p.image_url.url };
  }
  return null;
}

/** One Anthropic content block as a neutral part; null for blocks this trial never sends. */
function anthropicPart(b: unknown): Part | null {
  if (!isObj(b)) return null;
  if (b.type === 'text' && typeof b.text === 'string') return { kind: 'text', text: b.text };
  if (b.type === 'image' && isObj(b.source)) {
    const s = b.source;
    if (s.type === 'base64' && typeof s.data === 'string') {
      return { kind: 'imageData', mime: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data };
    }
    if (s.type === 'url' && typeof s.url === 'string') return { kind: 'imageUrl', url: s.url };
  }
  if (b.type === 'tool_use') {
    return { kind: 'toolUse', id: asStr(b.id), name: asStr(b.name), input: isObj(b.input) ? b.input : {} };
  }
  if (b.type === 'tool_result') {
    const c = b.content;
    const text = typeof c === 'string' ? c : Array.isArray(c) ? blockText(c) : c === undefined ? '' : JSON.stringify(c);
    return { kind: 'toolResult', id: asStr(b.tool_use_id), content: text };
  }
  return null;
}

/** An OpenAI Chat Completions body reduced to the neutral shape. */
function openaiToNeutral(body: Json): NormReq {
  const systemParts: string[] = [];
  const messages: NormMsg[] = [];
  for (const raw of asArr(body.messages)) {
    if (!isObj(raw)) continue;
    if (raw.role === 'system') {
      const t = blockText(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Part[] = [];
      const t = blockText(raw.content);
      if (t !== '') parts.push({ kind: 'text', text: t });
      for (const call of asArr(raw.tool_calls)) {
        if (!isObj(call) || !isObj(call.function)) continue;
        parts.push({ kind: 'toolUse', id: asStr(call.id), name: asStr(call.function.name), input: argsObj(call.function.arguments) });
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        parts: [{ kind: 'toolResult', id: asStr(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', parts: [{ kind: 'text', text: raw.content }] });
    } else if (Array.isArray(raw.content)) {
      messages.push({ role: 'user', parts: raw.content.flatMap((p) => {
        const part = openaiPart(p);
        return part === null ? [] : [part];
      }) });
    }
  }
  const tools: ToolDef[] = asArr(body.tools).flatMap((raw) => {
    if (!isObj(raw) || !isObj(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: asStr(raw.function.description), parameters: isObj(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  let toolChoice: NormReq['toolChoice'] = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'required') toolChoice = 'required';
  else if (isObj(choice) && choice.type === 'function' && isObj(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const stopRaw = body.stop;
  const stops = typeof stopRaw === 'string' ? [stopRaw] : asArr(stopRaw).filter((s): s is string => typeof s === 'string');
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens),
    temperature: asNum(body.temperature),
    topP: asNum(body.top_p),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body reduced to the neutral shape. */
function anthropicToNeutral(body: Json): NormReq {
  const messages: NormMsg[] = [];
  for (const raw of asArr(body.messages)) {
    if (!isObj(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Part[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const b of raw.content) {
          const part = anthropicPart(b);
          if (part !== null) parts.push(part);
        }
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', parts: [{ kind: 'text', text: raw.content }] });
    } else if (Array.isArray(raw.content)) {
      const textParts: Part[] = [];
      const resultParts: Part[] = [];
      for (const b of raw.content) {
        const part = anthropicPart(b);
        if (part === null) continue;
        if (part.kind === 'toolResult') resultParts.push(part);
        else textParts.push(part);
      }
      if (textParts.length > 0) messages.push({ role: 'user', parts: textParts });
      for (const p of resultParts) messages.push({ role: 'tool', parts: [p] });
    }
  }
  const tools: ToolDef[] = asArr(body.tools).flatMap((raw) => {
    if (!isObj(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: asStr(raw.description), parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  let toolChoice: NormReq['toolChoice'] = null;
  const choice = body.tool_choice;
  if (isObj(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    systemText: blockText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: asNum(body.max_tokens),
    temperature: asNum(body.temperature),
    topP: asNum(body.top_p),
    stops: asArr(body.stop_sequences).filter((s): s is string => typeof s === 'string'),
    jsonMode: false, // the Anthropic wire carries no JSON mode of its own
    stream: body.stream === true,
  };
}

/** The neutral shape back into an OpenAI Chat Completions body naming `model`. */
function neutralToOpenAI(req: NormReq, model: string): Json {
  const messages: Json[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const calls: Json[] = [];
      const texts: string[] = [];
      for (const part of msg.parts) {
        if (part.kind === 'text') texts.push(part.text);
        else if (part.kind === 'toolUse') {
          calls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } });
        }
      }
      messages.push({ role: 'assistant', content: texts.length > 0 ? texts.join('\n') : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
      continue;
    }
    if (msg.role === 'tool') {
      for (const part of msg.parts) {
        if (part.kind === 'toolResult') messages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
      }
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'imageData') parts.push({ type: 'image_url', image_url: { url: `data:${part.mime};base64,${part.data}` } });
      else if (part.kind === 'imageUrl') parts.push({ type: 'image_url', image_url: { url: part.url } });
    }
    if (parts.length === 1 && parts[0].type === 'text') messages.push({ role: 'user', content: (parts[0] as Json).text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (req.toolChoice === 'required') out.tool_choice = 'required';
  else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The neutral shape back into an Anthropic Messages body naming `model`. */
function neutralToAnthropic(req: NormReq, model: string): Json {
  const messages: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const blocks: Json[] = [];
      for (const part of msg.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'toolUse') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      }
      if (blocks.length === 1 && blocks[0].type === 'text') messages.push({ role: 'assistant', content: (blocks[0] as Json).text });
      else if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (msg.role === 'tool') {
      for (const part of msg.parts) {
        if (part.kind === 'toolResult') messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: part.id, content: part.content }] });
      }
      continue;
    }
    const blocks: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'imageData') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mime, data: part.data } });
      else if (part.kind === 'imageUrl') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
    }
    if (blocks.length === 1 && blocks[0].type === 'text') messages.push({ role: 'user', content: (blocks[0] as Json).text });
    else if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.systemText !== '') out.system = req.systemText;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stream) out.stream = true;
  return out;
}

/** The neutral shape as a Gemini generateContent body for `model`. */
function neutralToGemini(req: NormReq): Json {
  const contents: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const parts: Json[] = [];
      for (const part of msg.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
        else if (part.kind === 'toolUse') parts.push({ functionCall: { name: part.name, args: part.input } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'imageData') parts.push({ inlineData: { mimeType: part.mime, data: part.data } });
      else if (part.kind === 'imageUrl') parts.push({ fileData: { fileUri: part.url } });
      else if (part.kind === 'toolResult') parts.push({ functionResponse: { name: part.id, response: { result: part.content } } });
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (req.systemText !== '') out.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// The neutral answer: one shape for the three provider wires, two renders out
// ---------------------------------------------------------------------------

type StopKind = 'stop' | 'length' | 'tool_calls' | 'content_filter';

interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

interface NormAnswer {
  text: string;
  stop: StopKind;
  inputTokens: number | null;
  outputTokens: number | null;
  toolCalls: ToolCall[];
}

function numOrNull(v: unknown): number | null {
  const n = asNum(v);
  return n === null ? null : Math.trunc(n);
}

function openaiStop(v: unknown): StopKind {
  if (v === 'length') return 'length';
  if (v === 'tool_calls' || v === 'function_call') return 'tool_calls';
  if (v === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(v: unknown): StopKind {
  if (v === 'max_tokens') return 'length';
  if (v === 'tool_use') return 'tool_calls';
  if (v === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiStop(v: unknown): StopKind {
  if (v === 'MAX_TOKENS') return 'length';
  if (v === 'SAFETY') return 'content_filter';
  if (v === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

function openaiFinish(s: StopKind): string {
  return s === 'length' ? 'length' : s === 'tool_calls' ? 'tool_calls' : s === 'content_filter' ? 'content_filter' : 'stop';
}

function anthropicReason(s: StopKind): string {
  return s === 'length' ? 'max_tokens' : s === 'tool_calls' ? 'tool_use' : s === 'content_filter' ? 'refusal' : 'end_turn';
}

/** Split SSE text into its frames: each frame's event name and data lines joined. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function sseLine(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An OpenAI Chat Completions answer (JSON or SSE) reduced to the neutral shape. */
function openaiAnswer(bodyText: string, streamed: boolean): NormAnswer {
  const ans: NormAnswer = { text: '', stop: 'stop', inputTokens: null, outputTokens: null, toolCalls: [] };
  if (!streamed) {
    const body = parseJsonObj(bodyText);
    if (body === null) return ans;
    const choice = isObj(asArr(body.choices)[0]) ? (asArr(body.choices)[0] as Json) : {};
    const message = isObj(choice.message) ? choice.message : {};
    ans.text = typeof message.content === 'string' ? message.content : blockText(message.content);
    ans.stop = openaiStop(choice.finish_reason);
    const usage = isObj(body.usage) ? body.usage : {};
    ans.inputTokens = numOrNull(usage.prompt_tokens);
    ans.outputTokens = numOrNull(usage.completion_tokens);
    ans.toolCalls = asArr(message.tool_calls).flatMap((c) => {
      if (!isObj(c) || !isObj(c.function)) return [];
      return [{ id: asStr(c.id), name: asStr(c.function.name), args: argsObj(c.function.arguments) }];
    });
    return ans;
  }
  const byIndex = new Map<number, { id: string; name: string; argsRaw: string }>();
  for (const frame of sseFrames(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObj(payload);
    if (chunk === null) continue;
    if (isObj(chunk.usage)) {
      ans.inputTokens = numOrNull(chunk.usage.prompt_tokens) ?? ans.inputTokens;
      ans.outputTokens = numOrNull(chunk.usage.completion_tokens) ?? ans.outputTokens;
    }
    const choice = isObj(asArr(chunk.choices)[0]) ? (asArr(chunk.choices)[0] as Json) : null;
    if (choice === null) continue;
    const delta = isObj(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') ans.text += delta.content;
    for (const call of asArr(delta.tool_calls)) {
      if (!isObj(call)) continue;
      const idx = typeof call.index === 'number' ? call.index : byIndex.size;
      const fn = isObj(call.function) ? call.function : {};
      const prev = byIndex.get(idx);
      byIndex.set(idx, {
        id: asStr(call.id) !== '' ? asStr(call.id) : prev?.id ?? `call_${idx}`,
        name: asStr(fn.name) !== '' ? asStr(fn.name) : prev?.name ?? '',
        argsRaw: (prev?.argsRaw ?? '') + (typeof fn.arguments === 'string' ? fn.arguments : ''),
      });
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) ans.stop = openaiStop(choice.finish_reason);
  }
  ans.toolCalls = [...byIndex.values()].map((c) => ({ id: c.id, name: c.name, args: argsObj(c.argsRaw) }));
  return ans;
}

/** An Anthropic Messages answer (JSON or SSE) reduced to the neutral shape. */
function anthropicAnswer(bodyText: string, streamed: boolean): NormAnswer {
  const ans: NormAnswer = { text: '', stop: 'stop', inputTokens: null, outputTokens: null, toolCalls: [] };
  if (!streamed) {
    const body = parseJsonObj(bodyText);
    if (body === null) return ans;
    ans.text = blockText(body.content);
    ans.stop = anthropicStop(body.stop_reason);
    const usage = isObj(body.usage) ? body.usage : {};
    ans.inputTokens = numOrNull(usage.input_tokens);
    ans.outputTokens = numOrNull(usage.output_tokens);
    ans.toolCalls = asArr(body.content).flatMap((b) => {
      if (!isObj(b) || b.type !== 'tool_use') return [];
      return [{ id: asStr(b.id), name: asStr(b.name), args: isObj(b.input) ? b.input : {} }];
    });
    return ans;
  }
  const byIndex = new Map<number, { id: string; name: string; argsRaw: string }>();
  for (const frame of sseFrames(bodyText)) {
    const data = parseJsonObj(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isObj(data.message) && isObj(data.message.usage)) {
      ans.inputTokens = numOrNull(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isObj(data.delta)) {
      const idx = typeof data.index === 'number' ? data.index : -1;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') ans.text += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const prev = byIndex.get(idx);
        byIndex.set(idx, { id: prev?.id ?? `toolu_${idx}`, name: prev?.name ?? '', argsRaw: (prev?.argsRaw ?? '') + data.delta.partial_json });
      }
    }
    if (frame.event === 'content_block_start' && isObj(data.content_block) && data.content_block.type === 'tool_use') {
      const idx = typeof data.index === 'number' ? data.index : byIndex.size;
      const prev = byIndex.get(idx);
      byIndex.set(idx, { id: asStr(data.content_block.id), name: asStr(data.content_block.name), argsRaw: prev?.argsRaw ?? '' });
    }
    if (frame.event === 'message_delta') {
      if (isObj(data.delta) && typeof data.delta.stop_reason === 'string') ans.stop = anthropicStop(data.delta.stop_reason);
      if (isObj(data.usage)) ans.outputTokens = numOrNull(data.usage.output_tokens);
    }
  }
  ans.toolCalls = [...byIndex.values()].map((c) => ({ id: c.id, name: c.name, args: argsObj(c.argsRaw) }));
  return ans;
}

/** A Gemini generateContent answer (JSON or SSE) reduced to the neutral shape. */
function geminiAnswer(bodyText: string, streamed: boolean): NormAnswer {
  const ans: NormAnswer = { text: '', stop: 'stop', inputTokens: null, outputTokens: null, toolCalls: [] };
  const take = (body: Json): void => {
    const cand = isObj(asArr(body.candidates)[0]) ? (asArr(body.candidates)[0] as Json) : {};
    const content = isObj(cand.content) ? cand.content : {};
    const text = asArr(content.parts)
      .flatMap((p) => (isObj(p) && typeof p.text === 'string' ? [p.text] : []))
      .join('');
    if (text !== '') ans.text += text;
    if (typeof cand.finishReason === 'string') ans.stop = geminiStop(cand.finishReason);
    const meta = isObj(body.usageMetadata) ? body.usageMetadata : {};
    if (numOrNull(meta.promptTokenCount) !== null) ans.inputTokens = numOrNull(meta.promptTokenCount);
    if (numOrNull(meta.candidatesTokenCount) !== null) ans.outputTokens = numOrNull(meta.candidatesTokenCount);
  };
  if (!streamed) {
    const body = parseJsonObj(bodyText);
    if (body !== null) take(body);
    return ans;
  }
  for (const frame of sseFrames(bodyText)) {
    const body = parseJsonObj(frame.data);
    if (body !== null) take(body);
  }
  return ans;
}

/**
 * The usage a provider would have reported, from the request and the answer,
 * for the case a streamed answer carries no usage of its own. The simulated
 * providers count one token a four characters of system text, user text and
 * tool names, plus one, and one a four characters of answer text, plus one.
 */
function usageFallback(req: NormReq, answerText: string): { input: number; output: number } {
  let userText = '';
  for (const msg of req.messages) {
    if (msg.role !== 'user') continue;
    userText = msg.parts.flatMap((p) => (p.kind === 'text' ? [p.text] : [])).join('\n');
  }
  const prompt = req.systemText.length + userText.length + req.tools.map((t) => t.name).join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

// ---------------------------------------------------------------------------
// Renders: the neutral answer in the wire the client spoke
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** Split a text into the short pieces a provider's stream would carry. */
function textPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function openaiJsonAnswer(ans: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  }
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openaiFinish(ans.stop), logprobs: null }],
    usage: {
      prompt_tokens: ans.inputTokens ?? 0,
      completion_tokens: ans.outputTokens ?? 0,
      total_tokens: (ans.inputTokens ?? 0) + (ans.outputTokens ?? 0),
    },
  };
}

function openaiStreamAnswer(ans: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseLine(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of textPieces(ans.text)) out += sseLine(null, chunk({ content: piece }, null));
  for (const call of ans.toolCalls) {
    out += sseLine(null, chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] }, null));
  }
  out += sseLine(null, chunk({}, openaiFinish(ans.stop)));
  out += sseLine(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: ans.inputTokens ?? 0,
      completion_tokens: ans.outputTokens ?? 0,
      total_tokens: (ans.inputTokens ?? 0) + (ans.outputTokens ?? 0),
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

function anthropicJsonAnswer(ans: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const call of ans.toolCalls) content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${model.replace(/[^A-Za-z0-9]/g, '_')}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicReason(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens ?? 0, output_tokens: ans.outputTokens ?? 0 },
  };
}

function anthropicStreamAnswer(ans: NormAnswer, model: string): string {
  let out = '';
  let index = 0;
  out += sseLine('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: ans.inputTokens ?? 0, output_tokens: 0 },
    },
  });
  const textIndex = index;
  index += 1;
  out += sseLine('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
  for (const piece of textPieces(ans.text)) {
    out += sseLine('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: piece } });
  }
  out += sseLine('content_block_stop', { type: 'content_block_stop', index: textIndex });
  for (const call of ans.toolCalls) {
    const at = index;
    index += 1;
    out += sseLine('content_block_start', { type: 'content_block_start', index: at, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
    const args = JSON.stringify(call.args);
    if (args !== '') out += sseLine('content_block_delta', { type: 'content_block_delta', index: at, delta: { type: 'input_json_delta', partial_json: args } });
    out += sseLine('content_block_stop', { type: 'content_block_stop', index: at });
  }
  out += sseLine('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicReason(ans.stop), stop_sequence: null },
    usage: { output_tokens: ans.outputTokens ?? 0 },
  });
  out += sseLine('message_stop', { type: 'message_stop' });
  return out;
}

// ---------------------------------------------------------------------------
// Calling a provider, with failover across the candidate list
// ---------------------------------------------------------------------------

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_MS = 5000;
const CALL_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogDoc = { providers: [] };
const providerState = new Map<string, ProviderState>();
let arrival = 0;

function note(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(id: string): ProviderState {
  let s = providerState.get(id);
  if (s === undefined) {
    s = { calls: 0, dead: false, lastRefused: false };
    providerState.set(id, s);
  }
  return s;
}

/** One inference call to one provider, in its own wire. Throws on transport failure. */
async function callUpstream(candidate: Candidate, req: NormReq): Promise<{ status: number; text: string }> {
  const provider = candidate.provider;
  const root = joinOrigin(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...keyHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${root}/v1/chat/completions`;
    body = neutralToOpenAI(req, candidate.model.id);
    if (req.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${root}/v1/messages`;
    body = neutralToAnthropic(req, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
    if (req.stream) headers.accept = 'text/event-stream';
  } else {
    const action = req.stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${root}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    body = neutralToGemini(req);
    if (req.stream) headers.accept = 'text/event-stream';
  }
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_MS) });
  return { status: res.status, text: await res.text() };
}

/** The needs of a body when the manifest has no row for it. */
function needsFromBody(api: 'openai' | 'anthropic', body: Json): Need {
  let tools = false;
  let vision = false;
  if (api === 'openai') {
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    for (const m of asArr(body.messages)) {
      if (!isObj(m) || !Array.isArray(m.content)) continue;
      for (const p of m.content) {
        if (isObj(p) && (p.type === 'image' || p.type === 'image_url')) vision = true;
      }
    }
    const format = isObj(body.response_format) ? body.response_format.type : undefined;
    return { ...(tools ? { tools: true } : {}), ...(vision ? { vision: true } : {}), ...(format === 'json_object' || format === 'json_schema' ? { json: true } : {}) };
  }
  tools = Array.isArray(body.tools) && body.tools.length > 0;
  for (const m of asArr(body.messages)) {
    if (!isObj(m) || !Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (isObj(b) && b.type === 'image') vision = true;
    }
  }
  return { ...(tools ? { tools: true } : {}), ...(vision ? { vision: true } : {}) };
}

interface Served {
  status: number;
  contentType: string;
  body: string;
}

/** Serve one client request: route, fail over, translate the answer back. */
async function serveRequest(api: 'openai' | 'anthropic', body: Json): Promise<Served> {
  const item = brief.manifest[arrival];
  arrival += 1;
  const needs = item !== undefined ? item.needs : needsFromBody(api, body);
  const req = api === 'openai' ? openaiToNeutral(body) : anthropicToNeutral(body);
  const tried = new Set<string>(); // providers that already refused this request
  for (const candidate of candidatesFor(brief, probes, needs)) {
    const providerId = candidate.provider.id;
    if (tried.has(providerId)) continue;
    const state = stateOf(providerId);
    if (state.dead) continue;
    if (state.calls >= (CALL_ALLOWANCE[providerId] ?? DEFAULT_ALLOWANCE)) continue;
    let reply: { status: number; text: string };
    try {
      state.calls += 1;
      reply = await callUpstream(candidate, req);
    } catch (err) {
      tried.add(providerId);
      if (state.lastRefused) state.dead = true;
      state.lastRefused = true;
      note(`${providerId}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (reply.status < 200 || reply.status >= 300) {
      tried.add(providerId);
      if (reply.status === 402) state.dead = true;
      else if (state.lastRefused) state.dead = true;
      else state.lastRefused = true;
      note(`${providerId}/${candidate.model.id} refused HTTP ${reply.status}`);
      continue;
    }
    state.lastRefused = false;
    const wire = candidate.provider.wire;
    const ans = wire === 'openai' ? openaiAnswer(reply.text, req.stream) : wire === 'anthropic' ? anthropicAnswer(reply.text, req.stream) : geminiAnswer(reply.text, req.stream);
    if (ans.inputTokens === null || ans.outputTokens === null) {
      const fallback = usageFallback(req, ans.text);
      if (ans.inputTokens === null) ans.inputTokens = fallback.input;
      if (ans.outputTokens === null) ans.outputTokens = fallback.output;
    }
    if (api === 'openai') {
      if (req.stream) return { status: 200, contentType: 'text/event-stream', body: openaiStreamAnswer(ans, candidate.model.id) };
      return { status: 200, contentType: 'application/json', body: JSON.stringify(openaiJsonAnswer(ans, candidate.model.id)) };
    }
    if (req.stream) return { status: 200, contentType: 'text/event-stream', body: anthropicStreamAnswer(ans, candidate.model.id) };
    return { status: 200, contentType: 'application/json', body: JSON.stringify(anthropicJsonAnswer(ans, candidate.model.id)) };
  }
  const message = 'no free model that meets this request answered';
  if (api === 'anthropic') return { status: 503, contentType: 'application/json', body: JSON.stringify({ type: 'error', error: { type: 'overloaded_error', message } }) };
  return { status: 503, contentType: 'application/json', body: JSON.stringify({ error: { message, type: 'server_error', code: null } }) };
}

// ---------------------------------------------------------------------------
// The HTTP server: the two APIs, the model list, register, poll, exit
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (api === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  return parseJsonObj(Buffer.concat(chunks).toString('utf8'));
}

function modelsList(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(brief, probes);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsList(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serveRequest('openai', body);
    res.writeHead(served.status, {
      'content-type': served.contentType,
      ...(served.contentType === 'text/event-stream' ? { 'cache-control': 'no-cache', connection: 'keep-alive' } : { 'content-length': Buffer.byteLength(served.body) }),
    });
    res.end(served.body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serveRequest('anthropic', body);
    res.writeHead(served.status, {
      'content-type': served.contentType,
      ...(served.contentType === 'text/event-stream' ? { 'cache-control': 'no-cache', connection: 'keep-alive' } : { 'content-length': Buffer.byteLength(served.body) }),
    });
    res.end(served.body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listenOn(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

/** POST /register, a few tries: the env may still be waking up. */
async function register(port: number): Promise<void> {
  let last: Error | null = null;
  for (let attempt = 1; attempt <= 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isObj(reply) && typeof reply.requests === 'number') note(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      last = err as Error;
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw last ?? new Error('register failed');
}

/** Poll GET /work; exit 0 the moment it says the workload is done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isObj(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.problem !== null) note(brief.problem);
  probes = await Promise.all(brief.providers.map((p) => probeOne(ENV_URL, p, PROBE_MS)));
  for (const probe of probes) if (!probe.ok && probe.problem !== null) note(probe.problem);
  catalog = buildCatalog(brief, probes);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const onRequest = (req: IncomingMessage, res: ServerResponse): void => {
    void handle(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  };
  let server = createServer(onRequest);
  let port = await listenOn(server);
  if (port === 8080) {
    // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
    server.close();
    await new Promise((resolve) => setTimeout(resolve, 100));
    server = createServer(onRequest);
    port = await listenOn(server);
    note('the OS first picked 8080, which the env owns: listening again');
  }
  note(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(brief, probes).length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
