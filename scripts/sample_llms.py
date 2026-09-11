"""Measure configured LLM endpoints or import a traceable benchmark without credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler.llm_sampling import (
    import_benchmark,
    run_llm_sampling,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/llm_sampling_v1.json"))
    parser.add_argument("--profiles", type=Path, default=Path("configs/llm_profiles.toml"))
    parser.add_argument("--llm-id", action="append", help="Repeat to select multiple deployments")
    parser.add_argument("--output-dir", type=Path, default=Path("data/llm_sampling"))
    parser.add_argument("--import-benchmark", type=Path)
    args = parser.parse_args()
    if args.import_benchmark:
        output = import_benchmark(args.import_benchmark, args.output_dir)
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        output = run_llm_sampling(config, args.profiles, args.output_dir, llm_ids=args.llm_id)
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
