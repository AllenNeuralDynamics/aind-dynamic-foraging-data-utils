# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions (see `CLAUDE.local.md`) as needed.

The four-principle backbone (Think Before Coding, Simplicity First, Surgical
Changes, Goal-Driven Execution) is adapted from
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills),
which distills [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876)
on LLM coding pitfalls. The commit-message convention below is a local addition.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks,
use judgment.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If the solution is overcomplicated, simplify it.

## 3. Surgical Changes

Touch only what is needed. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style.
- If you find unrelated dead code, mention it but do not delete it.

When your changes create orphans:
- Remove imports, variables, or functions made unused by your change.
- Do not remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria and verify.

Transform tasks into verifiable goals:
- Add validation -> write failing tests for invalid inputs, then make them pass.
- Fix a bug -> write a reproducing test, then make it pass.
- Refactor -> ensure tests pass before and after.

For multi-step tasks, use a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs contain fewer unnecessary changes,
solutions are simpler, and clarifications happen before implementation.

## 5. Semantic Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format for every commit:

```text
<type>(<optional scope>): <short imperative summary>

<optional body explaining what and why, wrapped at ~72 chars>

<optional footer, e.g. "Refs #123" or "BREAKING CHANGE: ...">
```

Allowed `<type>` values:

- `feat` — new user-visible feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — add or fix tests
- `build` — build system, dependencies, or packaging
- `ci` — CI configuration or scripts
- `chore` — maintenance, tooling, or housekeeping with no src/test impact
- `revert` — revert a prior commit

Rules:

- Summary line in the imperative mood, no trailing period, <= 72 chars.
- One logical change per commit; split unrelated changes into separate commits.
- Use `<scope>` for the affected area when helpful (e.g. `feat(launcher): ...`, `docs(readme): ...`).
- Mark breaking changes with `!` after the type/scope (e.g. `feat(api)!: ...`) and a `BREAKING CHANGE:` footer.
- Body explains the motivation and any non-obvious consequences; don't restate the diff.
