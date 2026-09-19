# Researcher-Writer-Agents-reinforced-by-RL

A local (cloud version in progress), file-based, end-to-end **Research → Writing**
agentic system whose exploratory-research-planning step is fine-tuned with **offline GRPO**.
The researcher agent, implemented using FastMCP, is built with clear and detailed prompts for both the individual tools and the instruction for research orchestration, a rich suite of tools, and read-only resources, all exposed on a Model-Context-Protocol(MCP) server one can connect to in in-memory, stdio or HTTP modes. The writer agent consists of writing and editing workflows built with LangGraph functional API and can also be exposed as tools on any MCP server. A combined server mounting both researcher and writer agents is also built to help facilitate more convenient usage of the full end-to-end research + writing workflow.

For the exploration phase targeting depth and/or breadth enhancements during research, a small Qwen3-4B + LoRA policy reads a structured *research digest* of an article topic and selects one of four **exploration presets** (skip / light / standard / deep) that drives how the research agent explores the web before the writing agent drafts the article.

**Input and output:** The research workflow takes an `article_guideline.md`
(plus any golden sources it names — see below) and produces a single
`research.md` containing every gathered source, tagged by where it came from.
The writing workflow then takes that `research.md` plus the same guideline
and produces the final `article.md`. The exploration preset is decided
in between these two phases, right after research's exploitation phase and
before its exploration phase (see [Pipeline at a glance](#pipeline-at-a-glance)).

**The three tiers of research sources used to generate the final article:** 
Every source the research agent gathers is one of:

- **Golden sources** — URLs, local files, code, or video explicitly named in
  `article_guideline.md`. Ingested first and unconditionally, before any
  query generation runs; never gated by the RL policy.
- **Exploitation-phase sources** — found via up to 3 rounds of
  guideline-driven query generation + web search that *always* run, directly
  following the structure the guideline lays out and filling the gaps of the core topics 
  in the guideline.
- **Exploration-phase sources** — found via 0-3 *additional* rounds of
  complementary query generation + search that run only *after* the preset
  decision, filling gaps the guideline didn't anticipate by further exploring sources
  targeting **depth enhancements** (motivation, theoretical foundations, technical nuances,
  latest advancements, limitations/failure modes, implementation challenges, real-world case studies,
  future implications) and/or **breadth enhancements** (adjacent concepts, cross-domain
  analogies, historical context, enabling/disrupting technologies, applications in other industries,
  emerging trends in adjacent fields) of the core topics in the guideline.

The chosen **exploration preset** maps to a fixed exploration-phase recipe — this exact mapping is
what the RL model's own training data was generated with:

| preset | exploration-phase rounds |
|---|---|
| `skip` (P0) | 0 rounds — the exploration phase is not run at all |
| `light` (P1) | 1 round, `balanced` focus (~50% depth / 50% breadth) |
| `standard` (P2) | 2 rounds — round 1 `depth`, round 2 `breadth` |
| `deep` (P3) | 3 rounds — round 1 `depth`, round 2 `breadth`, round 3 `depth` |

on top of the golden and exploitation-phase material that always gets gathered. Choosing well matters
because exploration is not free — under-exploring risks a shallow article, over-exploring wastes API
calls and can dilute a section with tangential material. **This is precisely the
decision the Reinforcement Learning+guards pipeline is trained to make well; the** [Results](#results-rlguards-vs-baselines)
**section quantifies how well.**

The repository is a fork of the [Towards AI](https://academy.towardsai.net/courses/agent-engineering)
*Agentic AI Engineering* course materials, extended with but not limited to:

- a full RL data-generation pipeline (research + writing + grading),
- offline GRPO + QLoRA training of an exploration-preset selector,
- a production inference pipeline (RL policy + a deterministic policy guard —
  no LLM call), plus a separate eval-only harness that can benchmark the RL
  policy against an LLM planner (Grok, Claude, or any other model),
- a composed MCP server that fronts the research and writing agents over HTTP.

---

## Contents

- [Pipeline at a glance](#pipeline-at-a-glance)
- [Key prompts](#key-prompts)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quickstart: use the agentic system](#quickstart-use-the-agentic-system)
- [How the RL+guards policy is evaluated](#how-the-rlguards-policy-is-evaluated)
- [Results: RL+guards vs. baselines](#results-rlguards-vs-baselines)
- [Train your own GRPO policy](#train-your-own-grpo-policy)
- [Credits](#credits)

---

## Pipeline at a glance

```
                 article_guideline.md  +  golden & other sources
                    (URLs / local files named in the guideline)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Research MCP (Nova) — setup & golden-source ingestion               │
│  • extract_guidelines_urls        • process_local_files             │
│  • scrape golden web / YouTube / code sources                       │
└────────────────────────────────┬────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ Exploitation phase (3 rounds, guideline-driven)                     │
│  generate_next_queries → dedupe queries → Tavily search             │
│  → select & scrape sources                                          │
└────────────────────────────────┬────────────────────────────────────┘
                                   ▼
                      ┌───────────────────────────┐
                      │ generate_digests.py        │
                      │  → research_digest.md      │  (exploitation-phase
                      └─────────────┬───────────────┘   content so far)
                                    ▼
                      ┌───────────────────────────┐
                      │ RL preset selector         │
                      │  Qwen3-4B + LoRA            │
                      │  + deterministic guard      │
                      │  → skip / light / standard  │
                      │    / deep                    │
                      └─────────────┬───────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Exploration phase (0-3 rounds, preset-driven —                      │
│ entirely skipped when the chosen preset is "skip")                  │
│  generate_next_complementary_queries → dedupe → Tavily search        │
│  → scrape                                                            │
└────────────────────────────────┬────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ Content dedup (optional) + write final research.md                  │
│  XML-tagged source structure: golden / exploitation / exploration    │
└────────────────────────────────┬────────────────────────────────────┘
                                   ▼
                              research.md
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Writing MCP (Brown)                                                  │
│  • generate article (LangGraph) → integrate exploration               │
│  • review / edit iterations        • Mermaid media tools              │
└────────────────────────────────┬────────────────────────────────────┘
                                   ▼
                               article.md
```

The RL policy is trained **offline** on 24 training article-variants (8 base
lessons × 3 guideline-depth variants), each rolled out across all 4 presets,
plus 16 held-out test articles used for evaluation only. At inference time the
policy runs right after the exploitation phase — before any exploration
happens — and emits a per-section preset distribution from the exploitation-only
digest; a deterministic policy guard (not an LLM) clamps the result when the
article's external-evidence policy requires it (e.g. forbidden → skip,
required → at least light). That preset then determines how many exploration
rounds (0-3) actually run. A separate eval-only harness can additionally route
the decision through an LLM planner (Grok, Claude, or any other model) to
benchmark it against the RL policy, but that stage is disabled by default and
is not part of production inference.

---

## Key prompts

The workflow and utilization of the tools of both agents is driven by a handful of prompt files rather than
scattered inline strings — these are the highest-leverage files to read if
you want to understand *why* the agents act the way they do:

| Prompt | What it drives | File |
|---|---|---|
| Research workflow instructions | The full numbered research workflow (setup → exploitation → RL preset decision → exploration → source filtering → write) that the research MCP client follows turn-by-turn | [`research_instructions_prompt.py`](RL_researcher_writer_ymaxing/research_agent_local/mcp_server/src/prompts/research_instructions_prompt.py) |
| Research tool prompts | Per-tool prompts used by the research MCP server's individual tools (YouTube transcription, arXiv cleanup, query generation, source selection, content dedup, etc.) | [`config/prompts.py`](RL_researcher_writer_ymaxing/research_agent_local/mcp_server/src/config/prompts.py) |
| Article writer | Brown's system prompt for drafting the article (and later integrating exploration content) from research + guideline + style profiles | [`nodes/article_writer.py`](RL_researcher_writer_ymaxing/writing_workflow/src/brown/nodes/article_writer.py) |
| Graders (Phase 2b reward labeling) | The two LLM-as-judge metrics that grade every `(article × preset)` episode on 9 dimensions to produce both human-readable scores and reviews on article quality and the GRPO reward signal | [`FollowsGTMetric`](RL_researcher_writer_ymaxing/writing_workflow/src/brown/evals/metrics/new_follows_gt/prompts.py) · [`UserIntentMetric`](RL_researcher_writer_ymaxing/writing_workflow/src/brown/evals/metrics/new_user_intent/prompts.py) |

---

## Repository layout

```
Reinsearch_agent/
├── README.md                          # ← you are here
└── RL_researcher_writer_ymaxing/      # All code lives here
    ├── README_PACKAGE.md              # Subpackage map
    ├── research_agent_local/          # Research MCP server + client + RL data gen + training
    │   ├── mcp_server/                # FastMCP research tools (Tavily, Firecrawl, arXiv, …)
    │   ├── mcp_client/                # Interactive REPL + batch runner
    │   ├── rl_inference_service/      # Production RL inference: infer.py, generate_digests.py,
    │   │                             #   and the production LoRA checkpoint
    │   ├── evaluation/                # Eval-only harness: RL-vs-LLM-planner benchmarking
    │   │                             #   (test_planner.py; --planner-model for Grok/Claude/etc.)
    │   ├── rl_data_generator.py       # Shim: Phase 1, produce research.md per (article × preset)
    │   └── training/                  # GRPO + QLoRA trainer + offline analysis scripts
    │       ├── pipeline/              # train_grpo.py, generate_episode_oracles.py, etc.
    │       ├── maintenance/           # reusable data-repair scripts
    │       └── analysis/              # reward-formula sweeps, noise/signal-quality audits
    ├── writing_workflow/              # Brown writing agent
    │   └── rl_pipeline/               # RL writing/grading generators (Phase 0 / 2a / 2b)
    ├── agents_integration_local/      # Composed MCP server (research + writing over HTTP)
    │   ├── composed_server_script.py  # Launcher: starts both backends + composed server + client
    │   ├── mcp_server/                # Composed server implementation
    │   └── mcp_client/                # Client that talks to the composed server
    ├── models/Qwen3-4B/               # Local base-model weights (downloaded separately)
    ├── rl_training_data/              # Offline RL artifacts: bases/, episodes/, test_episodes/,
    │                                 #   checkpoints/ (historical experiment runs), oracle_review/
    └── utils/                         # env loader + pretty-print helpers
```

---

## Prerequisites

- **Python 3.12** (subprojects pin `3.12.11` via `pyproject.toml`).
- **[uv](https://github.com/astral-sh/uv)** package manager — each subproject
  has its own `uv` environment; nothing is installed at the repo root.
- **GNU Make** (used by the writing workflow Makefile).
- A POSIX shell. On Windows, use **WSL**.
- For training and local inference: an **NVIDIA GPU** with bitsandbytes-NF4
  support (≥ 16 GB VRAM recommended for Qwen3-4B in 4-bit).
- API keys for the providers used by each agent (see the per-subproject
  READMEs):
  - `GOOGLE_API_KEY` — Gemini, used by the writing agent and graders
  - `TAVILY_API_KEY` — web search for the research agent
  - `FIRECRAWL_API_KEY` (and optionally a second one for round-robin) — scraping
  - `OPIK_API_KEY` — optional, observability
  - `XAI_API_KEY` — optional; used only by the eval-only harness's LLM-planner
    baseline (`evaluation/test_planner.py --llm-only`, default model
    `grok-4.6`). NOT required for production inference (RL policy +
    deterministic guard only, no LLM call).
  - `ANTHROPIC_API_KEY` — optional; same eval-only harness, when
    `--planner-model` is set to a Claude model (e.g. `claude-opus-4-5`,
    `claude-sonnet-5`).

---

## Quickstart: use the agentic system

The fastest end-to-end path is the **composed MCP server**, which boots the
research and writing agents as HTTP services and exposes them through one
client.

```bash
# 1. Install dependencies for each subproject (one-time).
cd RL_researcher_writer_ymaxing/research_agent_local/mcp_server && uv sync && cd -
cd RL_researcher_writer_ymaxing/research_agent_local/mcp_client && uv sync && cd -
cd RL_researcher_writer_ymaxing/writing_workflow                   && uv sync && cd -
cd RL_researcher_writer_ymaxing/agents_integration_local/mcp_server && uv sync && cd -
cd RL_researcher_writer_ymaxing/agents_integration_local/mcp_client && uv sync && cd -

# 2. Configure API keys in each .env (copy from .env.example, then edit).
#    At minimum: GOOGLE_API_KEY, TAVILY_API_KEY, FIRECRAWL_API_KEY.

# 3. Launch the composed server + client.
cd RL_researcher_writer_ymaxing/agents_integration_local
uv run python composed_server_script.py
```

The launcher starts the research server on `:8001`, the writing server on
`:8002`, the composed server on `:8003`, and drops you into the client REPL.
From there you can drive a full *research → write* run end to end.

If you only want one half of the system, see:

- [research_agent_local/README.md](RL_researcher_writer_ymaxing/research_agent_local/README.md)
  — run the research agent standalone and produce a `research.md`.
- [writing_workflow/README.md](RL_researcher_writer_ymaxing/writing_workflow/README.md)
  — run the writing agent on an existing `research.md` + guideline.

---

## How the RL+guards policy is evaluated

Every number in the [Results](#results-rlguards-vs-baselines) section below rests on one specific
measurement: for a given article, does the exploration preset RL+guards *would choose* match the preset
that offline grading shows *actually produced the best article* for that topic? That ground-truth answer
— the **oracle preset** — is computed once per article by rolling out the full research → write → grade
pipeline under all 4 presets and comparing the graded reward each one earns; the policy never sees it at
inference time.

To make this a genuine test of generalization rather than memorization, the corpus is split in two:
**24 TRAIN article-variants** (8 base lessons × 3 guideline-depth variants, used during GRPO training)
and **16 held-out TEST articles** (never seen during training, no guideline-variant expansion). Unless
stated otherwise, every number in [Results](#results-rlguards-vs-baselines) is measured on the TEST split only.

The RL policy's production checkpoint (`rl_inference_service/checkpoints/production/`)
is evaluated against those 16 held-out test articles (plus the 24 training article-variants for
reference), using the eval-only harness under `research_agent_local/evaluation/`. No API keys are
required for the default RL-only / RL+guards modes — everything needed (digests, oracle
rewards) is already checked into `rl_training_data/`.

### 1. Install dependencies

```bash
cd RL_researcher_writer_ymaxing/research_agent_local/mcp_server && uv sync && cd -
cd RL_researcher_writer_ymaxing/research_agent_local/rl_inference_service && uv sync && cd -
```

`mcp_server` drives the eval harness and the MCP tools; `rl_inference_service`
(spawned as a subprocess) actually loads Qwen3-4B + the LoRA adapter, pulling
in `transformers`, `peft`, `bitsandbytes`, `torch` (CUDA).

### 2. Make sure the base model is present

`infer.py` (`research_agent_local/rl_inference_service/infer.py`) looks first
at `RL_researcher_writer_ymaxing/models/Qwen3-4B/` for the base weights
(falls back to downloading `Qwen/Qwen3-4B` from the Hugging Face Hub). The
production LoRA adapter is already checked into
`research_agent_local/rl_inference_service/checkpoints/production/` — no
`--adapter-dir` flag needed for the default run.

The adapter is also published on the Hugging Face Hub as
[**xintelligence/qwen3-4b-research-planner-lora**](https://huggingface.co/xintelligence/qwen3-4b-research-planner-lora),
and the full offline RL dataset (`bases/`, `episodes/`, `test_episodes/`) as
[**xintelligence/research-agent-rl-episodes**](https://huggingface.co/datasets/xintelligence/research-agent-rl-episodes)
— useful if you cloned the repo shallowly or just want the artifacts without the code.

### 3. Run the eval harness

From `RL_researcher_writer_ymaxing/research_agent_local/`:

```bash
# RL policy only (fastest -- no LLM call, no policy guard applied)
uv run --project mcp_server python evaluation/test_planner.py --rl-only --test-only

# Production-equivalent: RL policy + the deterministic policy guard actually
# used in production (forbidden→skip, required→≥light, capped→≤light)
uv run --project mcp_server python evaluation/test_planner.py --rl-guards-only --test-only
```

For each article this:

1. Reads `rl_training_data/bases/<article>/research_digest.md`.
2. Runs section-level inference with the LoRA policy and aggregates a
   per-article preset recommendation (confidence, entropy, floor-correction flag).
3. Compares against the **oracle preset** in
   `rl_training_data/bases/<article>/article_oracle.json` (derived offline
   from graded episode rewards).
4. Reports each article as `EXACT`, `NEAR` (±1 preset), or `MISS`, plus
   reward-regret, a confusion matrix, and majority/random baselines.

Useful flags:

```bash
uv run --project mcp_server python evaluation/test_planner.py --articles 09_RAG,04_structured_outputs
uv run --project mcp_server python evaluation/test_planner.py --save-json     # persist per-article JSON
uv run --project mcp_server python evaluation/test_planner.py --llm-only --planner-model claude-opus-4-5
```

The last line is the LLM-planner ablation baseline (requires `XAI_API_KEY` or
`ANTHROPIC_API_KEY`, matching `--planner-model`) — it measures the RL policy's
marginal contribution and is not part of the production pipeline.

---

## Results: RL+guards vs. baselines

This section backs up the project's central claim: that an offline-GRPO-trained preset selector, plus a
deterministic policy guard, plans exploration research (the three source tiers described at the top of
this README) measurably better than doing no learned planning at all — and specifically, better than simply
asking an LLM to make the same call. Every number below comes from the same 16 held-out TEST articles and
the same eval harness described in [How the RL+guards policy is evaluated](#how-the-rlguards-policy-is-evaluated),
cross-checked against five baselines (majority-class, uniform-random, weighted-random, and two frontier
LLM-only planners) and backed by exact significance tests (McNemar, Poisson-binomial) rather than raw
accuracy alone — 16 articles is a small enough sample that raw accuracy gaps can be misleading on their own.

**Headline (held-out TEST, n=16, production config — RL policy + deterministic
guards, no LLM call):**

| metric | RL + guards (production) |
|---|---:|
| exact match | **13/16 (81.2%)** |
| near (±1 preset) | 3/16 (19%) |
| miss | **0/16 (0%)** |
| ordinal MAE | 0.188 |
| reward-regret (mean / max) | 0.0083 / 0.0508 |

The pipeline never lands more than one preset away from the graded-optimal
choice on any held-out article — the zero-miss result is the single most
robust finding across every checkpoint and reward formula this project has
tried.

**Against baselines and two frontier LLM-only planners** (same 16 TEST
articles, same eval harness, `--llm-only`):

| predictor | exact | ordinal MAE |
|---|---:|---:|
| **RL + guards (production)** | **81.2%** | **0.188** |
| Grok 4.6, LLM-only (no RL) | 37.5% | 0.688 |
| Claude Opus 5, LLM-only (no RL) | 37.5% | 0.625 |
| majority-class baseline (always "light") | 50.0% | — |
| weighted-random baseline | 41.0% | — |
| uniform-random baseline | 35.9% | — |

**Statistical significance** (McNemar exact test for paired comparisons,
Poisson-binomial exact test vs. per-article chance level):

| comparison | split | result |
|---|---|---|
| RL+guards vs. majority-class baseline | TEST (n=16) | b=5, c=0 → p=0.031 — **strong** |
| RL+guards vs. per-article chance level | TEST (n=16) | p=0.0002 — **strong** |
| RL+guards vs. Grok 4.6 (paired, same articles) | TEST / COMBINED (n=40) | p=0.0078 / p=0.0009 — **strong** |
| RL+guards vs. Claude Opus 5 (paired, same articles) | TEST / COMBINED (n=40) | p=0.0195 / p=0.0007 — **strong** |

The RL policy is not redundant with LLM reasoning: on the same articles, an
LLM-only planner (Grok 4.6 or Claude Opus 5, same harness, no RL) lands almost
exactly at the majority-class baseline's own accuracy and clears neither the
chance nor the majority-class significance bar. RL+guards beats both
frontier-model baselines at conventional significance on every split except
one (TRAIN vs. Grok 4.6, p=0.063 — still directionally favorable, just
underpowered at n=24).

**Beyond McNemar: rank-based, permutation, and bootstrap tests** (same 16 TEST
articles, paired against the toughest constant baseline, guarded-`light`;
exact one-sided tests throughout, since `n` is too small for a normal
approximation):

| test | quantity | n | statistic | p (one-sided) |
|---|---|---:|---:|---:|
| Wilcoxon signed-rank | MAE/dist reduction | 5 | W+=15.0 | 0.0312 |
| Wilcoxon signed-rank | regret reduction | 6 | W+=19.0 | 0.0469 |
| Sign-flip permutation | MAE/dist reduction | 5 | sum=7.0000 | 0.0312 |
| Sign-flip permutation | regret reduction | 6 | sum=0.2692 | 0.0469 |
| Paired bootstrap (B=100,000) | MAE/dist reduction | 16 | mean=0.4375, CI=[0.1250, 0.8125] | 0.0023 |
| Paired bootstrap (B=100,000) | regret reduction | 16 | mean=0.0168, CI=[0.0024, 0.0346] | 0.0063 |

All six clear the conventional 0.05 threshold. These two families ask
different questions of the same 16 articles — Wilcoxon and the sign-flip
permutation test use only the *direction* of each article's win or loss
(discarding magnitude, hence the smaller `n` of discordant pairs), while the
paired bootstrap uses the full *magnitude* of every article's difference
across all 16 — and they corroborate rather than contradict each other here.
The RL+guards exact-match rate itself carries a Wilson 95% confidence
interval of **[57.0%, 93.4%]** on TEST (n=16).

**Full derivation, TRAIN-split numbers, confusion matrices, and the
multi-week investigation behind these results** (reward-formula calibration,
label-noise measurement, entropy-collapse diagnosis, tie-aware oracle
scoring, etc.) live in
[rl_training_data/rl_planner_test_results/](RL_researcher_writer_ymaxing/rl_training_data/rl_planner_test_results/README.md)
— see `analysis_document.md` §A.23 (production baseline, refreshed) and §A.24
(LLM-only baseline comparisons) for the latest, most rigorous cut.

---

## Train your own GRPO policy

To re-run training (24 training article-variants, section-level granularity):

```bash
cd RL_researcher_writer_ymaxing/research_agent_local/training
uv run python pipeline/train_grpo.py --dry-run                  # sanity-check setup
uv run python pipeline/train_grpo.py --task-id my_run            # full training
uv run python pipeline/train_grpo.py --task-id my_run --epochs 200 --lr 5e-5 --beta 0.15
```

Outputs land under `RL_researcher_writer_ymaxing/rl_training_data/checkpoints/tasks/<task-id>/`
(an auto-generated `task_<timestamp>/` name is used if `--task-id` is
omitted), with `best/`, `epoch_*/`, `latest/`, and a TensorBoard log under
`runs/`. Evaluate a new checkpoint with
`evaluation/test_planner.py --adapter-dir tasks/<task-id>/best --rl-guards-only --test-only`
(see [How the RL+guards policy is evaluated](#how-the-rlguards-policy-is-evaluated)),
or copy it to `rl_inference_service/checkpoints/production/` to make it the
default for production inference.

The full data-generation pipeline (Phase 1 research → Phase 2a writing →
Phase 2b grading) is documented in the per-subproject READMEs.

---

## Credits

Built on top of the [Agentic AI Engineering course](https://github.com/towardsai/agentic-ai-engineering-course)
by **Towards AI** and **Decoding AI**. The Nova research agent and Brown
writing agent originated in that course; the RL policy, training pipeline,
composed MCP server, and held-out evaluation are extensions in this fork.

Licensed under Apache-2.0 — see [LICENSE](RL_researcher_writer_ymaxing/LICENSE).
