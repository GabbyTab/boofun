# AI Usage Log

This project uses generative AI tools during development. This document
discloses where and how, per [JOSS policy](https://joss.readthedocs.io/en/latest/policies.html),
and is kept current as development continues. Reconstructed historical
entries are marked as such; where exact model versions were not recorded at
the time, that limitation is stated rather than guessed.

## Disclosure statement

Generative AI tools were used during development and manuscript preparation.
Cursor's AI coding assistant was used with the model versions recorded in
the log below. Assistance included code search and explanation, draft code
and test scaffolding, refactoring suggestions, static analysis remediation,
documentation drafting, literature and venue discovery, and copy-editing.
AI-generated suggestions were not accepted automatically. Gabriel Taboada
reviewed and edited all AI-assisted code and prose, ran the relevant
automated tests and mathematical cross-validation checks, verified factual
and bibliographic claims against primary sources, and remains responsible
for accuracy, originality, licensing, and ethical compliance. The problem
framing, mathematical conventions, architecture, feature scope, validation
strategy, and publication decisions were made by the human author. AI tools
will not be used to conduct conversational interactions with JOSS editors
or reviewers, except if permitted for translation under JOSS policy.

## How AI-assisted work is validated

Every change, AI-assisted or not, passes the same gates before merging:

- The full test suite (3,800+ tests), including mathematical-identity and
  cross-validation tests against closed-form results and independent
  implementations
- Strict mypy (zero errors), a zero-warning Ruff profile, and an enforced
  line-and-branch coverage floor in CI
- Human review of the diff by the maintainer

## Log

Entries record: date, tool, model/version (if known), work affected, type of
assistance, and validation performed.

| Date | Tool | Model | Work affected | Assistance | Validation |
|------|------|-------|---------------|------------|------------|
| 2025 – early 2026 (reconstructed) | Cursor AI assistant, ChatGPT | Exact model versions not recorded at the time | Initial library development: core representations, analysis modules, tests, notebooks, documentation | Code drafting, refactoring suggestions, docstring drafting, debugging | Test suite, manual review, cross-validation tests against closed-form results and Tal's scripts; noted in CONTRIBUTING.md ("partially AI-assisted") since early development |
| 2026-07 (weeks 3–4) | Cursor AI assistant (agent mode) | GPT-5.6 Sol (planning); Cursor agent models per session, including Claude (Fable) | Strict-typing/lint/coverage phase: mypy strict compliance across 86 modules, Ruff profile, ~600 new tests, complexity refactors (issues #45–#50) | Static-analysis remediation, type-annotation codemods, test scaffolding, refactoring, commit/PR drafting | Full test suite + new CI gates (strict mypy, Ruff, coverage floor) on every commit; human review of all diffs |
| 2026-07-25 | Cursor AI assistant (agent mode) | Claude (Fable 5) | v1.3.0 release engineering; JOSS groundwork: CITATION.cff, SUPPORT.md, statement-of-need page, paper skeleton, this log (issues #53–#56) | Document drafting, release automation, CI workflow authoring | cffconvert validation, docs build, CI checks on each PR; human review |

## Maintenance

- Add a row when a meaningful unit of AI-assisted work lands (a phase, a
  feature, a release — not every commit).
- The pull request template includes a reminder checkbox.
- Before JOSS submission, the disclosure statement above is finalized with
  the complete tool/model list from this log.
