"""Isolated PDF extraction entry point, terminated by the parent's timeout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter


def main() -> None:
    import pypdf

    started = perf_counter()
    reader = pypdf.PdfReader(sys.argv[1])
    if reader.is_encrypted:
        raise ValueError("encrypted PDFs are not supported")
    texts = []
    image_count = 0
    for page in reader.pages:
        texts.append(page.extract_text() or "")
        resources = page.get("/Resources")
        objects = resources.get_object().get("/XObject", {}) if resources else {}
        for reference in objects.get_object().values() if objects else ():
            if reference.get_object().get("/Subtype") == "/Image":
                image_count += 1
    Path(sys.argv[2]).write_text("\n\f\n".join(texts), encoding="utf-8")
    print(
        json.dumps(
            {
                "page_count": len(reader.pages),
                "top_level_image_count": image_count,
                "empty_text_pages": sum(not text.strip() for text in texts),
                "backend_time_sec": perf_counter() - started,
                "backend_version": pypdf.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
