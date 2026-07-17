# Scripts

This directory contains project execution scripts and script-specific references.

- `run_first_demo.py`
- `first_demo_data.md`

Run the first local end-to-end demo:

```bash
uv run python scripts/run_first_demo.py --policy round_robin
```

The demo writes generated inputs, tool outputs, and JSONL traces under `data/`.
See `first_demo_data.md` for the data layout and trace fields.
