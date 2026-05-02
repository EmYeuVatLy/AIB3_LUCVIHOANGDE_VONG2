"""
Text parser for interactive plain-text ESG inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from core.cache import CacheManager


class TextParser:
    CACHE_DIR = "outputs/cache"
    CACHE_SCHEMA = "text_parser_v1"

    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.cache_manager = CacheManager(run_key="document_cache")

    def _file_hash(self) -> str:
        digest = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cache_key(self) -> str:
        basename = os.path.basename(self.file_path)
        return os.path.join(self.CACHE_DIR, f"{basename}_{self._file_hash()[:12]}_text.json")

    def _cache_fingerprint(self) -> str:
        return CacheManager.file_fingerprint(
            self.file_path,
            extra={"schema": self.CACHE_SCHEMA, "mode": "text"},
        )

    def extract_text(self) -> list[dict]:
        cache_path = self._cache_key()
        fingerprint = self._cache_fingerprint()
        forced = CacheManager.is_forced("ocr")
        if os.path.exists(cache_path) and not forced:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if (
                    isinstance(payload, dict)
                    and payload.get("schema_version") == self.CACHE_SCHEMA
                    and payload.get("input_fingerprint") == fingerprint
                    and isinstance(payload.get("pages"), list)
                ):
                    self.cache_manager.record(
                        "ocr",
                        "hit",
                        self.CACHE_SCHEMA,
                        fingerprint,
                        path=cache_path,
                    )
                    return payload["pages"]
            except Exception:
                pass

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        pages = self._chunk_text(text)
        CacheManager.atomic_write_json(
            cache_path,
            {
                "schema_version": self.CACHE_SCHEMA,
                "input_fingerprint": fingerprint,
                "file_path": os.path.abspath(self.file_path),
                "pages": pages,
            },
            indent=2,
        )
        self.cache_manager.record(
            "ocr",
            "rebuilt",
            self.CACHE_SCHEMA,
            fingerprint,
            path=cache_path,
            reason="interactive_text",
        )
        return pages

    def get_full_text(self) -> str:
        return "\n\n".join(page["text"] for page in self.extract_text() if page.get("text"))

    def _chunk_text(self, text: str, target_chars: int = 3200) -> list[dict]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
        if not paragraphs:
            paragraphs = [(text or "").strip()]

        pages = []
        current = []
        current_length = 0
        page_number = 1

        for paragraph in paragraphs:
            para_length = len(paragraph)
            if current and current_length + para_length > target_chars:
                pages.append(self._build_page(page_number, "\n\n".join(current)))
                page_number += 1
                current = [paragraph]
                current_length = para_length
            else:
                current.append(paragraph)
                current_length += para_length

        if current:
            pages.append(self._build_page(page_number, "\n\n".join(current)))
        return pages

    def _build_page(self, page_number: int, text: str) -> dict:
        normalized = (text or "").strip()
        readable = [char for char in normalized if not char.isspace()]
        alpha_numeric = sum(1 for char in readable if char.isalnum() or char in ".,:;!?%/-()[]{}\"'")
        quality = (alpha_numeric / len(readable)) if readable else 0.0
        return {
            "page": page_number,
            "text": normalized,
            "extraction_method": "text",
            "ocr_quality_score": round(min(1.0, quality), 3),
            "char_count": len(normalized),
            "word_count": len(normalized.split()) if normalized else 0,
        }
