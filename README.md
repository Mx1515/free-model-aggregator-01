# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Mx1515/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Mx1515/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Node server that reads the trial brief, probes ten simulated free-tier providers, and serves every client request with the best free model that meets it — speaking the OpenAI, Anthropic and Gemini wires, with failover when a provider refuses. It ranks candidates by the brief's quality table, spreads calls across providers so no per-trial budget is exhausted, and re-emits each answer in the client's wire with the provider's text, usage and stop reason intact.

## Run it

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:8080 node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4tp60qsb&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
