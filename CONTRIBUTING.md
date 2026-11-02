# Contributing

Please open an issue for questions or submit a pull request with a short
description of the change and the checks performed. Maintained documentation
and code comments use English.

Run the checks before submitting:

```sh
uv sync --locked --extra test
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_docs.py --repo .
uv run --no-sync python scripts/check_reproduction.py --repo .
```

Keep recorded numerical data, source snapshots, and fixed dataset releases
unchanged. Publish corrections as explicitly versioned derivatives, retaining
the original values, source identifiers, missingness, and comparison conditions.
Reference sources are for inspection; use the documented reproduction commands
instead of launching historical acquisition scripts.

Do not commit credentials, private correspondence, generated environments,
download caches, or local development instructions. Automated checks establish
software and saved-data consistency, not independent scientific validation.
