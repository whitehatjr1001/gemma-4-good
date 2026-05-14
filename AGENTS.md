# Global Agent Instructions

@LEAN-CTX.md

## Default Operating Mode

- Use `caveman` style by default: compact, direct, low-token, technically precise.
- Expand only for planning, architecture tradeoffs, debugging evidence, user-facing docs, or review findings.
- Treat the skills below as high-priority agent superpowers. Use them whenever their trigger fits.
- Before creating any new docs, ask where the user wants them saved.
- Do not create or update agent memory files unless the user explicitly asks for memory work.

## Project Context

This repo is a Python `uv` project for Gemma health model training, evaluation, scaling experiments, and Android serving export.

The product vision lives in:

- `PRD.md`

Code package:

- `src/gemma_health/`

Single config source:

- `config.yaml`

Main workflows:

- `scripts/build_data.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/sweep.py`
- `scripts/export.py`
- `scripts/package_android.py`

## Primary Repo Rules

Code wins over docs when they disagree, but do not ignore docs silently. If code and docs conflict, call out the conflict and follow the safer existing runtime pattern.

Core rules:

- Use one root `config.yaml` as the control plane unless the user explicitly asks to split config.
- Load config through `src/gemma_health/config.py`; do not scatter `yaml.safe_load`, environment reads, or hardcoded paths across modules.
- Keep object reuse explicit. Prefer typed config objects and reusable instances over duplicated literals.
- Do not use vague module names like `facade.py`, `manager.py`, `handler.py`, or `utils.py`.
- Name workflow modules after what they do: `training/run.py`, `scaling/run.py`, `serving/export.py`, `data/mixture.py`.
- Keep app branding out of Python package names. Use `gemma_health`, not product/app names.
- Use factory/registry patterns only at real variation points: datasets, trainers, rewards, evaluators, exporters.
- Avoid speculative abstractions. Add interfaces only when multiple concrete implementations exist or are imminent.
- Training logic must be reproducible from `uv run ... --config config.yaml`, not only notebooks.
- Notebooks are for inspection and exploration, not the source of truth.
- Medical safety evals matter more than raw validation loss.
- For triage, track unsafe under-triage and emergency recall explicitly.
- For serving, Python owns export/package artifacts for Android; Android runtime code should live outside this package unless user asks otherwise.
- Serving export must preserve offline-first assumptions: LiteRT/AI Edge SDK target, MediaPipe preprocessing metadata, no internet requirement in Android manifest checks.

## Python Rules

Use `python-best-practices` whenever reading or writing Python.

- Prefer `@dataclass(frozen=True)` for config/domain values.
- Use `Protocol` for pluggable dataset/trainer/reward interfaces.
- Use `NewType` or `Literal` for important domain primitives when useful.
- Validate inputs at boundaries: config loading, dataset loading, manifest creation.
- Raise descriptive exceptions. Do not silently skip bad datasets, bad weights, or missing fields.
- Keep modules small and focused.
- Add or update tests when touching logic.
- Prefer pure functions for formatting, rewards, scoring, and scaling math.
- Use `pathlib.Path`, explicit encodings, and typed function signatures.

## Hugging Face / Unsloth Training Rules

Use these skills when work touches model training, datasets, or deployment:

- `hugging-face:llm-trainer` - use for Hugging Face datasets, TRL, SFT, GRPO, DPO, PEFT/LoRA/QLoRA, Hugging Face Jobs, Hub persistence, and GGUF/export-oriented training flows.
- Unsloth training guidance - use when implementing or changing local/Colab/Kaggle GPU training with Unsloth, especially Gemma fine-tuning, low-VRAM training, fast SFT, GRPO, LoRA/QLoRA, and notebook-to-script conversion.

Training rules:

- Unsloth is the default trainer implementation.
- Hugging Face is the ecosystem layer: datasets, model hub, TRL APIs, Jobs, persistence.
- Prefer Unsloth for this repo's main training path unless there is a clear reason to use plain TRL.
- Use Hugging Face `datasets` for loading and transforming public datasets.
- Use TRL concepts directly: SFT, GRPO, DPO, reward modeling.
- Use Unsloth implementations for the actual trainer wiring where supported.
- Keep SFT and GRPO as separate execution paths, both driven by `config.yaml`.
- Do cheap smoke runs before expensive runs.
- Use mini-experiments to choose dataset mixture, token budget, reward weights, sequence length, LoRA rank, and compute size before H100.
- Do not optimize only for loss. Track safety, Telugu quality, prescription accuracy, runtime, and cost.
- Push important training outputs to persistent storage or Hugging Face Hub when running on ephemeral compute.
- Never start paid/cloud training without explicit user confirmation.

## Data Rules

Dataset modules live under:

- `src/gemma_health/datasets/`

Data mixture logic lives under:

- `src/gemma_health/data/`

Rules:

- Raw data goes in `data/raw/`.
- Staged/intermediate data goes in `data/staged/`.
- Final training-ready data goes in `data/processed/`.
- Dataset manifests go in `data/manifests/`.
- Small committed examples may go in `data/samples/`.
- Do not commit large datasets, model checkpoints, or private medical data unless user explicitly confirms.
- Dataset weights in `config.yaml` must be validated.
- Data loaders should return typed examples or Hugging Face Dataset-compatible structures.
- Keep Telugu normalization and chat formatting separate from dataset loading.
- Do not mix medical labeling assumptions into generic dataset loader code.

## Rewards And Evaluation Rules

Reward code lives under:

- `src/gemma_health/rewards/`

Evaluation code lives under:

- `src/gemma_health/evals/`

Rules:

- Rewards must be small, testable functions.
- Combined rewards should call named component rewards.
- Safety reward must be asymmetric: missing emergencies is worse than over-referral.
- Eval reports should separate:
  - validation loss
  - emergency recall
  - unsafe under-triage rate
  - Telugu quality
  - prescription reading accuracy
  - cost/runtime
- Do not claim clinical validity from toy evals.
- If eval data is synthetic or tiny, state that clearly.

## Scaling Experiment Rules

Scaling logic lives under:

- `src/gemma_health/scaling/`

Rules:

- Chinchilla-style scaling is a decision aid, not a promise.
- Mini-experiments should answer what to run next, not just produce charts.
- Track examples, tokens, GPU type, runtime, loss, safety score, and cost.
- Prefer 3-5 useful probe points over broad unfocused sweeps.
- Final H100 run should be justified by prior smaller runs.

## Serving / Android Export Rules

Serving code lives under:

- `src/gemma_health/serving/`

Rules:

- Python serving code means export/package support, not backend hosting.
- Target runtime is Android on-device AI Edge SDK / LiteRT.
- MediaPipe support should describe preprocessing/metadata needed by Android.
- Export artifacts go under `artifacts/android/` or `artifacts/exports/`.
- Serving manifest must include model asset name, runtime, quantization, and no-internet requirement.
- Keep Android-specific package metadata separate from model training code.
- Do not imply cloud serving unless the user explicitly asks for it.

## Superpowers First

These skills exist to reduce common coding-agent failure modes: misalignment, verbosity, weak feedback loops, and codebase entropy.

### Alignment Before Building

Use these when the work is ambiguous, strategic, product-facing, or likely to branch into multiple interpretations.

- `grill-me` - interview the user until the plan or design is clear. Use for non-code and general planning.
- `grill-with-docs` - same alignment loop, but grounded in the repo's domain model. Use when terminology, ADRs, `CONTEXT.md`, or architecture decisions matter.
- `to-prd` - turn existing conversation context into a PRD. Ask where to save docs before creating files.
- `to-issues` - break a plan, spec, or PRD into independently-grabbable vertical slices.

### Feedback Loops

Use these when code behavior needs proof.

- `tdd` - red-green-refactor for features and bug fixes. Write or identify the failing check first, then implement the smallest passing change.
- `diagnose` - disciplined debugging loop: reproduce, minimize, hypothesize, instrument, fix, regression-test.
- `prototype` - throwaway exploration for uncertain state machines, business logic, or UI directions.

### Architecture And Entropy Control

Use these when changes touch shared modules, boundaries, lifecycle behavior, or confusing areas.

- `zoom-out` - explain unfamiliar code in the context of the whole system.
- `improve-codebase-architecture` - find deepening opportunities and rescue muddy design. Use on active areas before large changes.
- `triage` - move issues through a role/state driven triage workflow.
- `setup-matt-pocock-skills` - run once per repo before first use of `to-issues`, `to-prd`, `triage`, `diagnose`, `tdd`, `improve-codebase-architecture`, or `zoom-out` if repo config is missing.

### Productivity And Misc

- `caveman` - default compact communication mode.
- `handoff` - compact the current conversation so another agent can continue.
- `write-a-skill` - create new skills with progressive disclosure and bundled resources.
- `git-guardrails-claude-code` - set up hooks that block dangerous git commands.

## Look-Back Loop

Use this bounded self-improvement loop on non-trivial edits, bug fixes, reviews, docs, and architecture work. Keep it short; do not turn it into a separate project.

1. Define success criteria before editing.
   - What user-visible behavior changes?
   - What repo rules apply?
   - What test, build, lint, script run, or manual check will verify it?

2. Implement the smallest useful change.
   - Match existing style.
   - Avoid speculative abstractions.
   - Touch only files directly tied to the request.

3. Verify against objective feedback.
   - Prefer tests and type checks.
   - For training code, run smoke scripts or unit tests.
   - For architecture, re-check nearby patterns and `config.yaml`.

4. Look back before final response.
   - Did the change bypass `config.yaml`?
   - Did it duplicate dataset/training/export logic?
   - Did it create vague modules or dead imports?
   - Did it weaken medical safety evaluation?
   - Did it make paid/cloud training possible without confirmation?
   - Is there a simpler version that preserves behavior?
   - Are assumptions, skipped checks, and residual risks clear to the user?

5. Refine once if the look-back finds a concrete issue.
   - Fix real issues found by the checklist.
   - Do not churn code just to make it look different.
   - Stop after the criteria pass or after naming the blocker.

## Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.

License: MIT

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask: would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it instead of deleting it.
- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

Test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let the agent loop independently. Weak criteria like "make it work" require clarification.

## Repo Navigation And Code Search

Use low-token, high-signal navigation before broad file reads.

### RTK

Always prefix shell commands with `rtk` where practical.

```bash
rtk git status
rtk rg "load_config|TrainingExample|combined_triage_reward" src tests
rtk uv run python -m pytest
rtk uv run python scripts/build_data.py --config config.yaml
```

Use `rtk proxy <cmd>` when raw command output is required.

### Lean Context

Use the `lean-ctx` MCP server for continuity and compact repo inspection.

Configured Codex entry:

```toml
[mcp_servers.lean-ctx]
command = "lean-ctx"
args = []
```

Notes:

- `lean-ctx` is installed locally at `/opt/homebrew/bin/lean-ctx`.
- If `PATH` lookup is stale, call `/opt/homebrew/bin/lean-ctx` explicitly.
- Keep the MCP server enabled in `~/.codex/config.toml`.
- Use `full` for files you expect to edit, `map` for architecture, `signatures` for API surface, and `diff` or `lines:N-M` when revisiting known files.

Suggested commands:

```bash
rtk lean-ctx read LEAN-CTX.md -m full
rtk lean-ctx read PRD.md -m full
rtk lean-ctx read config.yaml -m full
rtk lean-ctx grep "load_config|NotImplementedError|combined_triage_reward|ServingManifest" src tests
```

### Graphify

Use `graphify` when code search needs a map, not just matches:

- Build a dependency or knowledge graph for a subsystem.
- Identify clusters, coupled modules, and architectural communities.
- Produce an HTML/JSON/audit view when the user asks for deeper codebase understanding.

Default search order:

1. `lean-ctx` map/signatures for known repo areas.
2. `rtk rg` for precise lexical search.
3. `graphify` for relationship-heavy exploration.
4. Full file reads only after narrowing the target.
