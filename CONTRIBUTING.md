# Contributing to Cola2

Thank you for helping improve Cola2. Contributions should keep the released
inference contract reproducible and the English and Chinese documentation
aligned.

## Before opening an issue

- Search existing issues and documentation.
- Use public issues for reproducible bugs, feature requests, and documentation
  problems.
- Do not report suspected vulnerabilities publicly. Follow
  [SECURITY.md](SECURITY.md).
- Do not include private, proprietary, personal, or access-controlled data.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The package supports Python 3.10 and newer. Unit tests use
`Cola2Config.tiny()` and do not require released weights or a GPU.

## Tests

Run the complete CPU test suite:

```bash
python -m pytest
```

The suite covers model/configuration behavior, benchmark contracts, resumable
artifacts, deterministic per-record seeds, repeated-run aggregation,
documentation links, and license headers.

Before submitting a packaging change, also run:

```bash
python -m build
python -m twine check dist/*
```

Do not commit `dist/`, local checkpoints, benchmark artifacts, downloaded
JSONL files, caches, or the private paper-source workspace.

## Source conventions

- Keep public APIs typed and document non-obvious invariants.
- Preserve deterministic benchmark behavior across resume and sharding.
- Reject incompatible artifacts instead of silently mixing configurations.
- Avoid adding runtime dependencies unless they are necessary for inference.
- Add this header to every human-authored `.py` and `.sh` source file (after a
  shebang, when present):

  ```text
  Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
  SPDX-License-Identifier: Apache-2.0
  ```

## Documentation

- Update English and Chinese counterparts together when behavior changes.
- Use repository-relative links and verify that local targets exist.
- Distinguish reported paper results from newly reproduced results.
- Do not invent unreleased paper, model, benchmark, or repository URLs.
- The normalized benchmark data are external artifacts. Update
  `eval_data/manifest.json` and the release documentation if their contract
  changes; do not commit the JSONL files.

## Benchmark changes

Changes to task sizes, prompt templates, scoring, seed derivation, or output
schema alter the public evaluation contract. Such changes must:

1. bump `CONTRACT_VERSION`;
2. add or update focused tests;
3. document compatibility and migration behavior; and
4. avoid rewriting the reported paper result files unless the paper source
   itself changes.

Use small synthetic fixtures in tests. Full benchmark files and checkpoints
must remain outside the source repository.

## Pull requests

Keep each pull request focused. Include:

- the problem and proposed behavior;
- affected public interfaces or artifact formats;
- tests executed;
- documentation updates; and
- any compatibility, security, licensing, or data-provenance considerations.

By contributing, you agree that your contribution is licensed under the
repository's [Apache License 2.0](LICENSE) and that you have the right to submit
it.
