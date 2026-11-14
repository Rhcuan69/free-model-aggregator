# free-model-aggregator

Entry for make.fun `free-model-aggregator-01`: one server that serves an
OpenAI- and Anthropic-compatible API over ten simulated free-tier model
providers, routing each of 20 requests to the best free model able to serve
it, failing over when a provider refuses, and reporting the catalog it read.

## How it works

Every request is reduced to one neutral shape (system text, typed message
parts, tools, caps, stops, JSON mode), rendered into the chosen provider's
wire (OpenAI / Anthropic / Gemini), and the answer reduced to a neutral
answer (text, stop family, usage, tool calls) and rendered back into the
client's wire. Routing ranks every free model meeting the request by the
quality table's rating, with a per-provider call allowance and a strike rule:
a 402 kills a provider at once, two consecutive refusals kill it, otherwise
it stays eligible. Anthropic-wire providers are excluded for JSON-mode
requests, since that wire cannot echo JSON mode.

## Run

```sh
node main.ts --list   # print the registered catalog and stop
node main.ts          # serve the APIs (expects MAKEFUN_ENV_URL)
```

Node 22, standard library only. Deterministic: no wall clock, no unseeded
randomness.

## Practice results (sealed scorer, kit practice trials)

| Trial | Score | Served | Valid |
|-------|-------|--------|-------|
| trial-0 | 99 | 100 | 20/20 |
| trial-1 | 99 | 100 | 20/20 |
| trial-2 | 99 | 100 | 20/20 |

Replay matched byte for byte on all three trials.
