#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Persian OCR Semantic Chunker v14 - full-LLM semantic-boundary edition
===============================================================================

Production design for the full-LLM construction run over a heterogeneous Persian OCR corpus:
- BookPlan, Atomizer, Planner, Chunk Review, and one Repair pass are LLM-driven in the default
  ``sft_dpo_quality`` preset; strict Full-LLM BookPlan/Atomizer/Planner with no post-planner semantic review/repair/filtering.
- The chunker owns ingestion, source ordering, lossless AtomicUnit coverage, semantic boundaries,
  OCR cleanup, hard-size limits, repair provenance, and final chunk provenance.
- The primary quality target is a self-contained source chunk suitable for grounded SFT generation.
- Semantic dependency is stronger than punctuation or target length: claim/reason/conclusion,
  objection/answer, definition/conditions, and enumeration header/items should remain together
  whenever the downstream hard limit permits it.
- Neighbor-aware repair can safely merge/reboundary one adjacent chunk and replaces all originals
  only when the provided AtomicUnits are covered exactly once.
- Planner/repair output is normalized and audited so no AtomicUnit is silently lost or duplicated.
- No LLM rewriting/paraphrasing of source text is performed.

Main output schema per chunk keeps the old fields and adds audit metadata:
{
  "id": "...__000001",
  "doc": "...",
  "src": "book folder name",
  "ps": 20,
  "pe": 21,
  "start": "p0020_f000012_b003_l000",
  "end": "p0021_f000013_b004_l002",
  "sec": ["..."],
  "text": "...",

  "file_start": 12,
  "file_end": 13,
  "page_labels": [20, 21],
  "unit_start": "u000123",
  "unit_end": "u000129",
  "unit_ids": ["u000123", "u000124"],
  "unit_types": ["numbered_item"],
  "item_numbers": [16, 17, 18],
  "char_count": 1234
}

Example (full LLM):
  python -u persian_ocr_chunker_full_llm_v14.py \
    --input-folders "/path/to/book" \
    --output-root "/path/to/chunk_outputs_v14" \
    --workers 1 \
    --llm-url "http://HOST:PORT/v1/chat/completions" \
    --llm-model "MODEL_NAME" \
    --preset sft_dpo_quality \
    --output-mode minimal \
    --force

The default preset in this v14 file is ``sft_dpo_quality`` (Full-LLM chunk construction, no Review/Repair quality gate).

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import threading
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOG_LOCK:
        print(msg, flush=True)


def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


# -----------------------------------------------------------------------------
# General structural regexes only. No book-specific title/theme regex here.
# -----------------------------------------------------------------------------

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

# A line that starts with a numbered marker. General, not book-specific.
NUMBERED_LINE_RE = re.compile(
    r"^\s*[\(\[\{（]?([۰-۹0-9٠-٩]{1,5})[\)\]\}）]?[\.:：\-ـ\s]+\S+"
)

# A much stricter independent-item marker.  Academic prose often begins with
# hierarchical labels such as ``3-2-1`` or repeats small clause numbers inside
# every section.  Those lines must not make the whole book look like a sequence
# of independent items.
STRICT_ITEM_LINE_RE = re.compile(
    r"^\s*[\(\[\{（]?([۰-۹0-9٠-٩]{1,5})[\)\]\}）]\s+\S+"
)
HIERARCHICAL_NUMBER_RE = re.compile(
    r"^\s*[۰-۹0-9٠-٩]{1,3}(?:\s*[-ـ.]\s*[۰-۹0-9٠-٩]{1,3}){1,4}(?:\s+|[:：])"
)
ACADEMIC_STRUCTURE_RE = re.compile(
    r"(?:^|\s)(?:فصل|بخش|گفتار|مبحث|جدول|نمودار|شکل|پیوست|فرضیه|روش(?:‌| )?شناسی|"
    r"پرسش(?:‌| )?نامه|جامعه آماری|نتایج تحقیق|منابع و مآخذ)(?:\s|$|[:：۰-۹0-9])"
)
SCENE_BREAK_RE = re.compile(
    r"^\s*(?:فصل|بخش|باب)\s+(?:[۰-۹0-9٠-٩]+|[یکدو‌سهچهارپنجشششتهفتوهشتنهصد]+)\s*$|"
    r"^\s*(?:صبح|ظهر|عصر|شب|فردای آن روز|چند روز بعد|سال‌ها بعد)\s*$"
)
PROSE_TERMINAL_RE = re.compile(r"[\.؟?!！؛;…»”\"\)\]\}]\s*$")
ARGUMENT_TRANSITION_RE = re.compile(
    r"^\s*(?:اما|با این حال|از سوی دیگر|در مقابل|بنابراین|ازاین‌رو|در نتیجه|"
    r"به عبارت دیگر|برای مثال|در واقع|نخست|دوم|سوم)\b"
)

# Semantic dependency signals.  These are deliberately used as anti-cut signals:
# a clean sentence boundary is not necessarily a clean semantic boundary.
OBJECTION_START_RE = re.compile(
    r"^\s*(?:إن\s*قلت|ان\s*قلت|اشکال|اعتراض|اگر\s+(?:گفته\s+شود|بگویی)|"
    r"ممکن\s+است\s+(?:گفته\s+شود|اشکال\s+شود)|ممکن\s+است\s+کسی\s+بگوید)\b"
)
ANSWER_START_RE = re.compile(
    r"^\s*(?:قلت|پاسخ|جواب|در\s+پاسخ|در\s+جواب|می[‌ ]?گوییم|گوییم|جواب\s+آن)\b"
)
DEPENDENT_CONTINUATION_RE = re.compile(
    r"^\s*(?:و|اما|ولی|لیکن|زیرا|چراکه|چون|همچنین|بنابراین|ازاین‌رو|از\s+این\s+رو|"
    r"در\s+نتیجه|پس|لذا|در\s+واقع|به\s+عبارت\s+دیگر|برای\s+مثال|مثلاً|مثلا|"
    r"به\s+عنوان\s+مثال|با\s+این\s+حال|البته|هرچند|با\s+وجود\s+این|از\s+سوی\s+دیگر)\b"
)
ENUM_HEADER_RE = re.compile(
    r"(?:عبارت(?:ند|‌اند| اند)\s+از|به\s+شرح\s+زیر|موارد\s+زیر|اقسام(?:\s+[^\n:：]{0,50})?\s*(?:عبارت(?:ند|‌اند| اند)\s+از|چنین\s+است)|"
    r"انواع(?:\s+[^\n:：]{0,50})?\s*(?:عبارت(?:ند|‌اند| اند)\s+از|چنین\s+است)|به\s+چند\s+(?:قسم|دسته|نوع)|"
    r"(?:به|دارای)\s+(?:[۰-۹0-9٠-٩]+|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|چند)\s+(?:قسم|نوع|دسته|بخش)|"
    r"تقسیم\s+می[‌ ]?(?:شود|گردد)\s+به)"
)
ENUM_ITEM_RE = re.compile(
    r"^\s*(?:(?:[۰-۹0-9٠-٩]{1,2})[\)\]\.،:\-ـ\s]+|(?:اول|نخست|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)[\s:،.-]+|(?:الف|ب|ج|د|هـ|ه|و)[\)\]\.،:\-ـ\s]+)\S+"
)
CONCLUSION_START_RE = re.compile(
    r"^\s*(?:بنابراین|در\s+نتیجه|ازاین‌رو|از\s+این\s+رو|پس|لذا|حاصل(?:\s+آن)?|نتیجه(?:\s+این)?\s+(?:است|می[‌ ]?شود))\b"
)
REASON_START_RE = re.compile(
    r"^\s*(?:زیرا|چراکه|چون|به\s+این\s+دلیل|دلیل\s+آن\s+این\s+است|بدین\s+سبب|از\s+آنجا\s+که)\b"
)

# Date-only line. Used as a structural signal, not as a universal assumption.
DATE_ONLY_RE = re.compile(
    r"^\s*[۰-۹0-9٠-٩]{2,4}\s*/\s*[۰-۹0-9٠-٩]{1,2}\s*/\s*[۰-۹0-9٠-٩]{1,2}\s*$"
)

# Generic Q/A markers.
QA_Q_RE = re.compile(r"^\s*(?:سؤال|سوال|پرسش|پرسش‌|س)\s*[:：\-]")
QA_A_RE = re.compile(r"^\s*(?:جواب|پاسخ|پاسخ‌|ج)\s*[:：\-]")

YEAR_HEADER_RE = re.compile(r"\bسال\s+[۰-۹0-9٠-٩]{3,4}\b")
PAGE_NO_RE = re.compile(r"^\s*[۰-۹0-9٠-٩]{1,5}\s*$")
SEPARATOR_RE = re.compile(r"^\s*[\*★☆•●◦▪▫_ـ\-–—=\.·…]{3,}\s*$")
HTML_TABLE_RE = re.compile(r"<\s*/?\s*(?:table|tr|td|th)\b", re.IGNORECASE)
DOTTED_LINE_RE = re.compile(r"^\s*[۰-۹0-9٠-٩]{1,4}\s+.{1,80}[\.·…]{5,}\s*$")
# OCR book footer / TOC artifacts that commonly survive page-level cleanup.
# Examples: "هزار نکته .... ۵۸", "۱۰۰ .... هزار نکته".
BOOK_FOOTER_DOTTED_RE = re.compile(
    r"^\s*(?:[۰-۹0-9٠-٩]{1,5}\s*)?(?:[\u0600-\u06FFA-Za-z][\u0600-\u06FFA-Za-z\s‌ـ]{1,60})[\.·…]{4,}\s*(?:[۰-۹0-9٠-٩]{1,5})?\s*$"
)
BOOK_FOOTER_DOTTED_RE_2 = re.compile(
    r"^\s*[۰-۹0-9٠-٩]{1,5}\s*[\.·…]{4,}\s*(?:[\u0600-\u06FFA-Za-z][\u0600-\u06FFA-Za-z\s‌ـ]{1,60})\s*$"
)
GENERIC_FLOATING_HEADING_RE = re.compile(r"^\s*(?:نکته|نكته)\s*(?:ها|های|هاى)\s*$")

# Generic possible speaker/label line: short line ending with colon.
# This is deliberately broad; it is only a weak structural hint.
COLON_LABEL_RE = re.compile(r"^\s*.{1,90}[:：]\s*$")

# The downstream SFT project currently slices source text at these exact limits.
# Keep one shared contract here so prompts, policies, review, and finalization do
# not silently optimize for a different consumer.
SFT_ASSESSOR_VISIBLE_CHARS = 3000
SFT_GENERATOR_VISIBLE_CHARS = 4000
SAFE_GENERATOR_HARD_MAX_CHARS = 3800

# Semantic shapes returned by BookPlan. Rule profiling may be coarser, but the
# LLM strategy should map every heterogeneous book into one of these families.
SEMANTIC_BOOK_SHAPES = {
    "independent_items",
    "numbered_analytical_prose",
    "qa_book",
    "narrative",
    "poetry",
    "argumentative_philosophy",
    "section_prose",
    "dense_expository",
    "dialogue",
    "procedural",
    "reference",
    "table_heavy",
    "mixed",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class Block:
    block_id: str
    category: str
    bbox: List[int]
    text: str
    file_index: int
    page_label: int


@dataclass
class Page:
    file_index: int
    page_label: int
    json_path: Path
    blocks: List[Block]


@dataclass
class LineRef:
    lid: str
    global_index: int
    file_index: int
    page_label: int
    block_id: str
    block_category: str
    line_index: int
    text: str


@dataclass
class AtomicUnit:
    uid: str
    idx: int
    unit_type: str
    text: str
    lines: List[LineRef]
    section_path: List[str]
    title: Optional[str] = None
    confidence: float = 1.0
    source: str = "rule"  # rule|llm|fallback

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def page_start(self) -> int:
        return min((ln.page_label for ln in self.lines), default=0)

    @property
    def page_end(self) -> int:
        return max((ln.page_label for ln in self.lines), default=0)

    @property
    def file_start(self) -> int:
        return min((ln.file_index for ln in self.lines), default=0)

    @property
    def file_end(self) -> int:
        return max((ln.file_index for ln in self.lines), default=0)

    @property
    def start_ref(self) -> Optional[str]:
        return self.lines[0].lid if self.lines else None

    @property
    def end_ref(self) -> Optional[str]:
        return self.lines[-1].lid if self.lines else None


@dataclass
class BookProfile:
    detected_shape: str
    confidence: float
    page_count: int
    line_count: int
    avg_chars_per_page: float
    median_chars_per_page: float
    numbered_line_rate: float
    qa_line_rate: float
    section_header_rate: float
    table_rate: float
    short_line_rate: float
    repeated_noise_lines: int
    strict_item_rate: float = 0.0
    hierarchical_number_rate: float = 0.0
    sequence_continuity: float = 0.0
    sequence_uniqueness: float = 0.0
    sequence_coverage: float = 0.0
    strict_item_count: int = 0
    academic_marker_rate: float = 0.0


@dataclass
class ChunkPolicy:
    min_chars: int
    target_chars: int
    max_chars: int
    max_pages: int
    max_units: int
    min_units: int
    target_units: int
    mode: str
    allow_cross_section: bool = False


@dataclass
class GeneratorContract:
    """Per-book limits derived from the actual downstream SFT/DPO consumers."""

    min_chars: int
    ideal_min_chars: int
    target_chars: int
    assessor_visible_chars: int
    generator_visible_chars: int
    hard_max_chars: int
    min_units: int
    semantic_shape: str


@dataclass
class LLMConfig:
    url: str
    model: str
    api_key: Optional[str] = None
    timeout_sec: int = 180
    max_retries: int = 1
    temperature: float = 0.0
    max_output_tokens: int = 1024
    response_format_json: bool = False
    urls: List[str] = field(default_factory=list)
    models_by_url: Dict[str, str] = field(default_factory=dict)
    cache_dir: Optional[Path] = None
    cache_enabled: bool = False
    _cache_lock: Any = field(default_factory=threading.Lock, repr=False)
    _rr_index: int = 0
    _rr_lock: Any = field(default_factory=threading.Lock, repr=False)

    def endpoint_urls(self) -> List[str]:
        urls = self.urls or [self.url]
        return [u.strip() for u in urls if str(u).strip()] or [self.url]

    def next_url(self) -> str:
        urls = self.endpoint_urls()
        with self._rr_lock:
            u = urls[self._rr_index % len(urls)]
            self._rr_index += 1
        return u

    def model_for_url(self, url: str) -> str:
        return self.models_by_url.get(url) or self.model


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in (text or "").splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_digits(text: str) -> str:
    return (text or "").translate(PERSIAN_DIGITS)


def to_persian_digits(value: Any) -> str:
    return str(value).translate(TO_PERSIAN_DIGITS)


def estimate_tokens(text_or_obj: Any) -> int:
    if not isinstance(text_or_obj, str):
        text_or_obj = json.dumps(text_or_obj, ensure_ascii=False, separators=(",", ":"))
    return max(1, len(text_or_obj) // 3)


def stable_doc_id(path: Path) -> str:
    name = path.name.strip()
    simple = re.sub(r"[^\w\u0600-\u06FF]+", "_", name, flags=re.UNICODE).strip("_")
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{simple or 'doc'}_{h}"


def extract_file_page_index(path: Path) -> int:
    m = re.search(r"page[_\-]?(\d+)", path.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


# -----------------------------------------------------------------------------
# General noise detection
# -----------------------------------------------------------------------------


def is_chunk_noise_line(line: str) -> bool:
    """Noise that should not appear inside final generator chunks.

    Conservative for body text, but aggressive against page footers/TOC dotted
    lines that poison SFT/DPO generation.
    """
    ln = clean_text(line)
    if not ln:
        return True
    if BOOK_FOOTER_DOTTED_RE.match(ln) or BOOK_FOOTER_DOTTED_RE_2.match(ln):
        return True
    if GENERIC_FLOATING_HEADING_RE.match(ln):
        return True
    norm = normalize_digits(ln)
    # Book-title/page-number footer variants without dots, e.g. "هزار نکته 58".
    if re.fullmatch(r"[\u0600-\u06FF][\u0600-\u06FF\s‌ـ]{2,45}\s+\d{1,4}", norm):
        words = re.sub(r"\s+\d{1,4}$", "", norm).split()
        if 1 <= len(words) <= 5 and not re.search(r"[.!؟?؛:،]", norm):
            return True
    return False


def is_general_noise_line(line: str) -> bool:
    ln = clean_text(line)
    if not ln:
        return True
    if PAGE_NO_RE.match(ln):
        return True
    if SEPARATOR_RE.match(ln):
        return True
    if HTML_TABLE_RE.search(ln):
        return True
    if DOTTED_LINE_RE.match(ln):
        return True
    if is_chunk_noise_line(ln):
        return True
    if is_probable_ocr_artifact_line(ln):
        return True
    return False


def is_probable_ocr_artifact_line(line: str) -> bool:
    """General OCR/header/footer noise; deliberately not tied to any one book title."""
    ln = clean_text(line)
    if not ln:
        return True
    if is_chunk_noise_line(ln):
        return True
    # Export artifacts like A671, B12, OCR034.
    if re.fullmatch(r"[A-Za-z]{1,5}[-_ ]?[0-9]{2,6}", ln):
        return True
    # Stray bare numbers with punctuation, often page/footnote debris.
    if re.fullmatch(r"[۰-۹0-9٠-٩]{1,5}[\.)،،]?", ln):
        return True
    # Running header form: page number + very short title phrase, without item punctuation.
    # Example: "۱۷ هزار نکته". Avoid normal numbered items like "(۱۷) ..." or "۱۷) ...".
    norm = normalize_digits(ln)
    if re.fullmatch(r"\d{1,4}\s+[؀-ۿ][؀-ۿ\s‌]{2,45}", norm):
        rest = re.sub(r"^\d{1,4}\s+", "", norm).strip()
        if len(rest.split()) <= 5 and not re.search(r"[.!؟?؛:،]", rest):
            return True
    return False

def alpha_len(text: str) -> int:
    return len(re.sub(r"[^\w\u0600-\u06FF]+", "", text or "", flags=re.UNICODE))


def looks_like_colon_label(line: str) -> bool:
    ln = clean_text(line)
    if not COLON_LABEL_RE.match(ln):
        return False
    if len(ln) > 90:
        return False
    # Avoid treating complete long sentences as speaker labels.
    body = ln.rstrip(":：").strip()
    if len(body.split()) > 12:
        return False
    return True


STRUCTURAL_COLON_LABEL_RE = re.compile(
    r"^\s*(?:پی\s*نوشت|پانوشت|منابع|مآخذ|فهرست|مقدمه|نتیجه(?:‌| )?گیری|جمع(?:‌| )?بندی|"
    r"اهداف|عوامل|شرایط|دلایل|روش|جدول|نمودار|شکل|فصل|بخش|موضوع)\s*[:：]\s*$"
)
SPEAKER_HINT_RE = re.compile(
    r"(?:آیت\s*الله|حجت\s*الاسلام|امام|دکتر|استاد|مهندس|خانم|آقا|راوی|"
    r"گفت|فرمود|پرسید|پاسخ\s*داد|اظهار\s*کرد)"
)


def looks_like_speaker_label(line: str) -> bool:
    """A conservative speaker label, not every short heading ending in colon."""
    ln = clean_text(line)
    if not looks_like_colon_label(ln) or STRUCTURAL_COLON_LABEL_RE.match(ln):
        return False
    body = ln.rstrip(":：").strip()
    if SPEAKER_HINT_RE.search(body):
        return True
    # A short person-like name (two to five words) is useful in dialogue, but
    # generic one-word headings are deliberately excluded.
    words = body.split()
    return 2 <= len(words) <= 5 and not ACADEMIC_STRUCTURE_RE.search(body)


def is_section_like_line(line: str, category: str = "") -> bool:
    ln = clean_text(line)
    if not ln:
        return False
    if category == "Section-header":
        # OCR layout models sometimes label a whole paragraph as a header.
        # Accept the signal only when the text itself is heading-sized.
        if len(ln) <= 160 and len(ln.split()) <= 20 and ln[-1:] not in ".؟?!؛;،":
            return True
    if YEAR_HEADER_RE.search(ln):
        return True
    if len(ln) <= 80 and not NUMBERED_LINE_RE.match(ln) and not DATE_ONLY_RE.match(ln):
        # A broad heading guess; used softly, not as a hard domain-specific rule.
        if ln.startswith("#") or ln.endswith(":") or ln.endswith("："):
            return True
    return False


# -----------------------------------------------------------------------------
# IO and OCR page loading
# -----------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


NON_OCR_JSON_NAMES = {
    "manifest.json",
    "run_manifest.json",
    "profile.json",
    "quality_report.json",
    "book_plan.json",
    "hosting.json",
}


def find_page_jsons(folder: Path) -> List[Path]:
    """Return JSON candidates without assuming a filename convention.

    Older versions silently ignored valid OCR exports such as ``0001.json`` or
    ``result.json`` because their basename did not contain ``page``.  Discovery
    is now broad; ``load_pages_sorted`` performs the actual schema check.
    """
    paths: List[Path] = []
    for p in folder.rglob("*.json"):
        name = p.name.lower()
        lowered_parts = {part.lower() for part in p.parts}
        if name in NON_OCR_JSON_NAMES:
            continue
        if name.endswith(".decision.json"):
            continue
        if "debug" in lowered_parts or "_llm_cache" in lowered_parts:
            continue
        paths.append(p)
    return sorted(paths, key=lambda x: (extract_file_page_index(x), str(x.parent), x.name))


def extract_blocks_raw(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("blocks", "items", "layout", "regions", "elements"):
            val = raw.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        for wrapper_key in ("result", "data", "document", "output"):
            wrapped = raw.get(wrapper_key)
            if isinstance(wrapped, (dict, list)):
                blocks = extract_blocks_raw(wrapped)
                if blocks:
                    return blocks
        pages = raw.get("pages")
        if isinstance(pages, list) and pages:
            blocks: List[Dict[str, Any]] = []
            for pg in pages:
                if isinstance(pg, dict) and isinstance(pg.get("blocks"), list):
                    blocks.extend([x for x in pg["blocks"] if isinstance(x, dict)])
            return blocks
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def extract_page_records(raw: Any) -> List[Any]:
    """Split a JSON object into page-level records.

    A list of block dictionaries is one page; a list/dict containing page
    containers is many pages. Wrapper objects used by common OCR exporters are
    traversed conservatively.
    """
    if isinstance(raw, dict):
        pages = raw.get("pages")
        if isinstance(pages, list) and pages:
            page_records = [pg for pg in pages if isinstance(pg, (dict, list))]
            if page_records and any(extract_blocks_raw(pg) for pg in page_records):
                return page_records
        for wrapper_key in ("result", "data", "document", "output"):
            wrapped = raw.get(wrapper_key)
            if isinstance(wrapped, (dict, list)):
                records = extract_page_records(wrapped)
                if records:
                    return records
        return [raw] if extract_blocks_raw(raw) else []

    if isinstance(raw, list):
        dict_rows = [x for x in raw if isinstance(x, dict)]
        page_container_keys = {"blocks", "items", "layout", "regions", "elements"}
        if dict_rows and all(any(k in x for k in page_container_keys) for x in dict_rows):
            return dict_rows
        return [raw] if extract_blocks_raw(raw) else []
    return []


def parse_int_value(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = normalize_digits(str(value or "")).strip()
    if re.fullmatch(r"\d{1,6}", text):
        return int(text)
    return None


def page_label_from_record(record: Any) -> Optional[int]:
    if not isinstance(record, dict):
        return None
    for key in ("page_label", "page_number", "page_no", "page_num", "page"):
        value = parse_int_value(record.get(key))
        if value is not None and value >= 0:
            return value
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("page_label", "page_number", "page_no", "page_num", "page"):
            value = parse_int_value(metadata.get(key))
            if value is not None and value >= 0:
                return value
    return None


def parse_page_label(blocks_raw: List[Dict[str, Any]], fallback: int, record: Any = None) -> int:
    explicit = page_label_from_record(record)
    if explicit is not None:
        return explicit

    for b in blocks_raw:
        cat = str(b.get("category", ""))
        if cat.lower().replace("_", "-") not in {"page-header", "pageheader", "header"}:
            continue
        text = normalize_digits(str(b.get("text", "") or ""))
        # A date such as 1391/06/10 is metadata, not page 1391.
        text_without_dates = re.sub(r"\b\d{2,4}\s*/\s*\d{1,2}\s*/\s*\d{1,2}\b", " ", text)
        m = re.search(r"(?:صفحه|page)\s*[:：\-]?\s*(\d{1,5})\b", text_without_dates, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.fullmatch(r"\s*(\d{1,5})\s*(?:/\s*\d{1,5})?\s*", text_without_dates)
        if m:
            return int(m.group(1))
    return fallback


def pages_from_json(json_path: Path, raw: Any, source_order: int) -> List[Page]:
    records = extract_page_records(raw)
    pages: List[Page] = []
    filename_index = extract_file_page_index(json_path)
    for local_index, record in enumerate(records):
        blocks_raw = extract_blocks_raw(record)
        if not blocks_raw:
            continue
        provisional_index = filename_index + local_index if len(records) > 1 else filename_index
        if filename_index == 0 and len(records) > 1:
            provisional_index = source_order * 1_000_000 + local_index
        page_label = parse_page_label(blocks_raw, fallback=provisional_index, record=record)

        blocks: List[Block] = []
        for i, b in enumerate(blocks_raw):
            text = str(b.get("text", "") or "")
            bbox = b.get("bbox") or [0, 0, 0, 0]
            category = str(b.get("category", "Text") or "Text")
            block_id = f"p{page_label:04d}_f{provisional_index:06d}_b{i:03d}"
            blocks.append(Block(
                block_id=block_id,
                category=category,
                bbox=list(bbox) if isinstance(bbox, list) else [0, 0, 0, 0],
                text=text,
                file_index=provisional_index,
                page_label=page_label,
            ))
        if blocks:
            pages.append(Page(
                file_index=provisional_index,
                page_label=page_label,
                json_path=json_path,
                blocks=blocks,
            ))
    return pages


def load_page(json_path: Path) -> Page:
    """Backward-compatible single-page loader used by external callers/tests."""
    pages = pages_from_json(json_path, load_json(json_path), source_order=0)
    if not pages:
        raise ValueError(f"JSON has no supported OCR page schema: {json_path}")
    return pages[0]


def page_content_hash(page: Page) -> str:
    parts = []
    for b in page.blocks:
        if b.category in {"Page-header", "Page-footer"}:
            continue
        if b.text.strip():
            parts.append(clean_text(b.text))
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def deduplicate_pages(pages: List[Page]) -> Tuple[List[Page], int]:
    seen_source: set[Tuple[str, int, str]] = set()
    seen_label: set[Tuple[int, str]] = set()
    out: List[Page] = []
    dropped = 0
    for p in pages:
        content_hash = page_content_hash(p)
        source_key = (str(p.json_path), p.file_index, content_hash)
        label_key = (p.page_label, content_hash)
        # Remove repeated exports of the same page, but do not collapse two
        # different pages merely because both are blank or share a short footer.
        if source_key in seen_source or (content_hash and label_key in seen_label):
            dropped += 1
            continue
        seen_source.add(source_key)
        if content_hash:
            seen_label.add(label_key)
        out.append(p)
    return out, dropped


def normalize_loaded_page_indices(pages: List[Page]) -> None:
    """Make internal file indices contiguous and repair implausible labels."""
    total = len(pages)
    max_plausible_label = max(1000, total * 5)
    previous_label: Optional[int] = None
    for new_index, page in enumerate(pages):
        label = int(page.page_label)
        implausible = label < 0 or label > max_plausible_label
        if previous_label is not None and label - previous_label > max(1000, total * 3):
            implausible = True
        if implausible:
            label = new_index
        page.file_index = new_index
        page.page_label = label
        for block_index, block in enumerate(page.blocks):
            block.file_index = new_index
            block.page_label = label
            block.block_id = f"p{label:04d}_f{new_index:06d}_b{block_index:03d}"
        previous_label = label


def load_pages_sorted(folder: Path) -> Tuple[List[Page], int, Dict[str, Any]]:
    candidates = find_page_jsons(folder)
    stats: Dict[str, Any] = {
        "json_candidates": len(candidates),
        "json_parsed": 0,
        "json_invalid": 0,
        "json_non_ocr": 0,
        "files_with_pages": 0,
        "whole_book_json_files": 0,
        "page_records_loaded": 0,
        "invalid_samples": [],
    }
    loaded_with_key: List[Tuple[Tuple[int, str, int], Page]] = []

    for source_order, path in enumerate(candidates):
        try:
            raw = load_json(path)
            stats["json_parsed"] += 1
        except Exception as exc:
            stats["json_invalid"] += 1
            if len(stats["invalid_samples"]) < 20:
                stats["invalid_samples"].append({"path": str(path), "error": str(exc)})
            continue

        records = extract_page_records(raw)
        if not records:
            stats["json_non_ocr"] += 1
            continue
        pages = pages_from_json(path, raw, source_order)
        if not pages:
            stats["json_non_ocr"] += 1
            continue
        stats["files_with_pages"] += 1
        if len(pages) > 1:
            stats["whole_book_json_files"] += 1
        stats["page_records_loaded"] += len(pages)

        filename_index = extract_file_page_index(path)
        for local_index, page in enumerate(pages):
            loaded_with_key.append(((filename_index, str(path), local_index), page))

    loaded_with_key.sort(key=lambda item: item[0])
    pages = [page for _key, page in loaded_with_key]
    pages, dropped = deduplicate_pages(pages)
    normalize_loaded_page_indices(pages)
    stats["pages_after_dedup"] = len(pages)
    stats["duplicate_pages_dropped"] = dropped
    stats["blocks_loaded"] = sum(len(page.blocks) for page in pages)
    stats["text_chars_loaded"] = sum(len(block.text or "") for page in pages for block in page.blocks)
    return pages, dropped, stats


# -----------------------------------------------------------------------------
# Lines, repeated noise, profiling
# -----------------------------------------------------------------------------


def infer_repeated_noise_lines(pages: List[Page]) -> set[str]:
    """General running header/footer detector based on repetition across pages."""
    line_pages: Dict[str, set[int]] = defaultdict(set)
    for pi, page in enumerate(pages):
        seen_on_page: set[str] = set()
        for b in page.blocks:
            if b.category in {"Page-header", "Page-footer"}:
                continue
            for raw in b.text.splitlines():
                ln = clean_text(raw)
                if not ln or len(ln) > 90:
                    continue
                if is_general_noise_line(ln):
                    continue
                # Avoid removing real numbered items, Q/A markers, and date-only content.
                if NUMBERED_LINE_RE.match(ln) or QA_Q_RE.match(ln) or QA_A_RE.match(ln) or DATE_ONLY_RE.match(ln):
                    continue
                seen_on_page.add(ln)
        for ln in seen_on_page:
            line_pages[ln].add(pi)
    min_pages = max(4, int(len(pages) * 0.04))
    return {ln for ln, pset in line_pages.items() if len(pset) >= min_pages}


def build_clean_lines(pages: List[Page], repeated_noise_lines: set[str]) -> List[LineRef]:
    lines: List[LineRef] = []
    gi = 0
    for page in pages:
        for b in page.blocks:
            if b.category in {"Page-header", "Page-footer"}:
                continue
            for li, raw in enumerate(b.text.splitlines()):
                text = clean_text(raw)
                if not text:
                    continue
                if text in repeated_noise_lines:
                    continue
                if is_general_noise_line(text):
                    continue
                lid = f"l{gi:08d}"
                lines.append(LineRef(
                    lid=lid,
                    global_index=gi,
                    file_index=page.file_index,
                    page_label=page.page_label,
                    block_id=b.block_id,
                    block_category=b.category,
                    line_index=li,
                    text=text,
                ))
                gi += 1
    return lines


def sample_indices(n: int, first: int = 120, middle: int = 240, last: int = 120) -> List[int]:
    if n <= first + middle + last:
        return list(range(n))
    selected: set[int] = set()
    selected.update(range(min(first, n)))
    mid_start = max(0, n // 2 - middle // 2)
    selected.update(range(mid_start, min(n, mid_start + middle)))
    selected.update(range(max(0, n - last), n))
    return sorted(selected)


def profile_book(pages: List[Page], lines: List[LineRef], repeated_noise_lines: set[str]) -> BookProfile:
    page_chars: List[int] = []
    for p in pages:
        txt = "\n".join(
            clean_text(raw)
            for b in p.blocks
            if b.category not in {"Page-header", "Page-footer"}
            for raw in b.text.splitlines()
            if clean_text(raw)
        )
        page_chars.append(len(txt))

    sample = [lines[i] for i in sample_indices(len(lines))]
    total = max(1, len(sample))
    numbered = sum(1 for ln in sample if NUMBERED_LINE_RE.match(ln.text))
    qa = sum(1 for ln in sample if QA_Q_RE.match(ln.text) or QA_A_RE.match(ln.text))
    section = sum(1 for ln in sample if is_section_like_line(ln.text, ln.block_category))
    table = sum(1 for ln in sample if HTML_TABLE_RE.search(ln.text) or ln.block_category.lower() in {"table", "table-cell"})
    short = sum(1 for ln in sample if len(ln.text) < 45)
    strict_numbers: List[int] = []
    for ln in lines:
        m = STRICT_ITEM_LINE_RE.match(ln.text)
        if m:
            try:
                strict_numbers.append(int(normalize_digits(m.group(1))))
            except Exception:
                pass
    hierarchical = sum(1 for ln in sample if HIERARCHICAL_NUMBER_RE.match(ln.text))
    academic = sum(1 for ln in sample if ACADEMIC_STRUCTURE_RE.search(ln.text))

    numbered_rate = numbered / total
    qa_rate = qa / total
    section_rate = section / total
    table_rate = table / total
    short_rate = short / total
    strict_item_rate = len(strict_numbers) / max(1, len(lines))
    hierarchical_rate = hierarchical / total
    academic_rate = academic / total
    if len(strict_numbers) >= 2:
        adjacent = [
            1 for a, b in zip(strict_numbers, strict_numbers[1:])
            if b > a and (b - a) <= 3
        ]
        sequence_continuity = sum(adjacent) / max(1, len(strict_numbers) - 1)
        sequence_uniqueness = len(set(strict_numbers)) / len(strict_numbers)
        lo, hi = min(strict_numbers), max(strict_numbers)
        sequence_coverage = len(set(strict_numbers)) / max(1, hi - lo + 1)
    else:
        sequence_continuity = sequence_uniqueness = sequence_coverage = 0.0

    shape = "dense_prose"
    confidence = 0.55
    strong_item_sequence = bool(
        len(strict_numbers) >= 20
        and sequence_continuity >= 0.72
        and sequence_uniqueness >= 0.82
        and sequence_coverage >= 0.55
        and hierarchical_rate < 0.025
    )
    analytical_numbering = bool(
        numbered_rate > 0.07
        and (
            hierarchical_rate >= 0.015
            or academic_rate >= 0.025
            or (
                len(strict_numbers) >= 20
                and (sequence_uniqueness < 0.65 or sequence_coverage < 0.35)
            )
        )
    )
    if qa_rate > 0.045:
        shape = "qa_book"
        confidence = min(0.95, 0.65 + qa_rate * 4)
    elif strong_item_sequence:
        shape = "item_book"
        confidence = min(0.98, 0.72 + sequence_continuity * 0.12 + sequence_uniqueness * 0.12)
    elif analytical_numbering:
        shape = "numbered_analytical_prose"
        confidence = min(0.94, 0.68 + hierarchical_rate * 3 + academic_rate * 2)
    elif table_rate > 0.15:
        shape = "table_heavy"
        confidence = min(0.9, 0.65 + table_rate)
    elif (
        short_rate > 0.60
        and numbered_rate < 0.04
        and qa_rate < 0.02
        and table_rate < 0.05
        and (page_chars and median(page_chars) < 1400)
    ):
        # Poetry books often contain section headings; check stanza-like short
        # line density before the broad section-prose rule.
        shape = "poetry_or_aphorism"
        confidence = min(0.9, 0.55 + short_rate / 2)
    elif section_rate > 0.04:
        shape = "section_prose"
        confidence = min(0.9, 0.58 + section_rate * 4)

    return BookProfile(
        detected_shape=shape,
        confidence=round(confidence, 3),
        page_count=len(pages),
        line_count=len(lines),
        avg_chars_per_page=round(sum(page_chars) / max(1, len(page_chars)), 1),
        median_chars_per_page=round(float(median(page_chars)) if page_chars else 0.0, 1),
        numbered_line_rate=round(numbered_rate, 4),
        qa_line_rate=round(qa_rate, 4),
        section_header_rate=round(section_rate, 4),
        table_rate=round(table_rate, 4),
        short_line_rate=round(short_rate, 4),
        repeated_noise_lines=len(repeated_noise_lines),
        strict_item_rate=round(strict_item_rate, 4),
        hierarchical_number_rate=round(hierarchical_rate, 4),
        sequence_continuity=round(sequence_continuity, 4),
        sequence_uniqueness=round(sequence_uniqueness, 4),
        sequence_coverage=round(sequence_coverage, 4),
        strict_item_count=len(strict_numbers),
        academic_marker_rate=round(academic_rate, 4),
    )


def choose_policy(profile: BookProfile, args: argparse.Namespace) -> ChunkPolicy:
    if args.fixed_policy:
        return ChunkPolicy(
            min_chars=args.min_chars,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            max_pages=args.max_pages,
            max_units=args.max_units,
            min_units=args.min_units,
            target_units=args.target_units,
            mode="fixed",
            allow_cross_section=args.allow_cross_section,
        )

    shape = profile.detected_shape
    # Generator-quality defaults are intentionally below the downstream 3k/4k
    # visibility limits. BookPlan may refine them, but cannot exceed the shared
    # generator contract.
    if shape == "item_book":
        return ChunkPolicy(900, 2100, 3200, 4, 14, 2, 7, shape, False)
    if shape == "numbered_analytical_prose":
        return ChunkPolicy(1300, 2600, 3600, 5, 18, 1, 7, shape, False)
    if shape == "qa_book":
        return ChunkPolicy(700, 1900, 3000, 4, 12, 1, 4, shape, False)
    if shape == "section_prose":
        return ChunkPolicy(1300, 2500, 3400, 5, 14, 1, 5, shape, False)
    if shape == "poetry_or_aphorism":
        return ChunkPolicy(600, 1400, 2400, 4, 14, 2, 6, shape, False)
    if shape == "table_heavy":
        return ChunkPolicy(800, 1700, 2800, 3, 10, 1, 4, shape, False)
    return ChunkPolicy(1300, 2500, 3400, 5, 16, 1, 6, shape, False)



def make_book_plan_samples(lines: List[LineRef], max_samples: int = 24, text_limit: int = 120) -> List[Dict[str, Any]]:
    idxs = sample_indices(len(lines), first=max_samples//3, middle=max_samples//3, last=max_samples//3)
    samples = []
    for i in idxs[:max_samples]:
        ln = lines[i]
        samples.append({
            "lid": ln.lid,
            "page": ln.page_label,
            "category": ln.block_category,
            "text": ln.text[:text_limit],
        })
    return samples


def clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return max(lo, min(hi, x))


def llm_book_plan(profile: BookProfile, lines: List[LineRef], cfg: LLMConfig, args: argparse.Namespace, out_path: Optional[Path] = None) -> Dict[str, Any]:
    """Ask the LLM for a per-book strategy, with adaptive payload shrinking.

    vLLM can return HTTP 400 when a planning request is too large for the
    served context window or request parser. BookPlan is useful but optional,
    so we retry with smaller representative samples before falling back to
    the rule profile. This keeps the same production behavior over large
    corpora without relying on a single oversized BookPlan request.
    """
    requested_n = max(4, int(getattr(args, "book_plan_sample_lines", 24)))
    requested_limit = max(40, int(getattr(args, "book_plan_text_limit", 120)))

    # Adaptive plan sizes. Keep them small; BookPlan needs representative
    # signals, not 90 full lines. Duplicate values are removed while preserving order.
    candidates_raw = [
        (requested_n, requested_limit),
        (min(requested_n, 24), min(requested_limit, 120)),
        (min(requested_n, 16), min(requested_limit, 100)),
        (min(requested_n, 10), min(requested_limit, 80)),
        (min(requested_n, 6), min(requested_limit, 60)),
    ]
    candidates: List[Tuple[int, int]] = []
    seen = set()
    for n, lim in candidates_raw:
        n = max(4, int(n))
        lim = max(40, int(lim))
        key = (n, lim)
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    last_error: Optional[Exception] = None
    last_payload: Dict[str, Any] = {}
    for attempt_i, (n_samples, text_limit) in enumerate(candidates, 1):
        payload = {
            "language": "fa",
            "rule_profile": asdict(profile),
            "samples": make_book_plan_samples(lines, max_samples=n_samples, text_limit=text_limit),
            "constraints": {
                "output": "compact_json",
                "goal": "long coherent generator source chunks",
                "avoid_short_retrieval_snippets": True,
            },
        }
        last_payload = payload
        try:
            log(f"[BOOK_PLAN_LLM_CALL] attempt={attempt_i}/{len(candidates)} samples={len(payload['samples'])} text_limit={text_limit} est_tok={estimate_tokens(payload)}")
            obj = call_llm_json(BOOK_PLAN_PROMPT, payload, cfg)
            if not isinstance(obj, dict):
                raise ValueError("book plan is not an object")
            obj.setdefault("book_shape", profile.detected_shape)
            raw_shape = clean_text(str(obj.get("book_shape") or "")).lower()
            if raw_shape not in SEMANTIC_BOOK_SHAPES:
                raw_shape = semantic_book_shape(profile, None)
            obj["book_shape"] = raw_shape
            # Reconcile the model with high-precision sequence evidence.  A
            # model can easily mistake hierarchical academic numbering for
            # independent items from short samples; conversely a clean 1..N
            # sequence is stronger evidence than a topical guess.
            if (
                profile.detected_shape == "numbered_analytical_prose"
                and raw_shape in {"independent_items", "reference", "mixed"}
            ):
                obj["llm_book_shape"] = raw_shape
                obj["book_shape"] = "numbered_analytical_prose"
                obj["shape_override_reason"] = "hierarchical_or_repeated_academic_numbering"
            elif (
                profile.detected_shape == "item_book"
                and profile.sequence_continuity >= 0.85
                and profile.sequence_uniqueness >= 0.90
                and raw_shape in {"numbered_analytical_prose", "section_prose", "mixed"}
            ):
                obj["llm_book_shape"] = raw_shape
                obj["book_shape"] = "independent_items"
                obj["shape_override_reason"] = "strong_unique_monotonic_item_sequence"
            obj.setdefault("confidence", profile.confidence)
            obj.setdefault("unit_strategy", {})
            obj.setdefault("chunk_strategy", {})
            obj.setdefault("planner_notes", [])
            obj.setdefault("review_notes", [])
            obj.setdefault("risks", [])
            obj["source"] = "llm"
            obj["book_plan_attempt"] = {"samples": len(payload["samples"]), "text_limit": text_limit, "attempt": attempt_i}
            if out_path:
                write_json(out_path, {"payload": payload, "book_plan": obj})
            log(f"[BOOK_PLAN_LLM_DONE] shape={obj.get('book_shape')} conf={obj.get('confidence')} samples={len(payload['samples'])}")
            return obj
        except Exception as e:
            last_error = e
            log(f"[BOOK_PLAN_LLM_RETRY] attempt={attempt_i}/{len(candidates)} failed={e}")
            continue

    if out_path:
        write_json(out_path, {"payload": last_payload, "book_plan_error": str(last_error), "source": "llm_failed_no_fallback"})
    log(f"[BOOK_PLAN_LLM_ERROR] {last_error} no_fallback=strict")
    raise RuntimeError(f"STRICT_BOOK_PLAN_LLM_FAILED after adaptive shrinking: {last_error}")


def policy_from_book_plan(base: ChunkPolicy, book_plan: Dict[str, Any], args: argparse.Namespace) -> ChunkPolicy:
    if args.fixed_policy:
        return base
    cs = book_plan.get("chunk_strategy") or {}
    # The downstream SFT assessor sees 3k chars and the generator sees 4k.
    # The LLM may adapt within these bounds, but it cannot expand past them.
    min_chars = clamp_int(cs.get("soft_min_chars"), 500, 1800, base.min_chars)
    target_chars = clamp_int(cs.get("target_chars"), 1000, SFT_ASSESSOR_VISIBLE_CHARS, base.target_chars)
    max_chars = clamp_int(
        cs.get("hard_max_chars") or cs.get("soft_max_chars"),
        1800,
        SAFE_GENERATOR_HARD_MAX_CHARS,
        min(base.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS),
    )
    shape = clean_text(str(book_plan.get("book_shape") or "")).lower()
    shape_caps: Dict[str, Tuple[int, int, int]] = {
        # (largest useful soft minimum, target cap, hard cap)
        "poetry": (900, 1800, 2600),
        "qa_book": (1000, 2200, 3000),
        "independent_items": (1300, 2500, 3300),
        "dialogue": (1200, 2400, 3300),
        "reference": (1200, 2300, 3200),
        "table_heavy": (1100, 2200, 3000),
        "narrative": (1700, 2900, 3600),
        "argumentative_philosophy": (1700, 2900, 3600),
        "numbered_analytical_prose": (1700, 2900, 3600),
    }
    if shape in shape_caps:
        min_cap, target_cap, max_cap = shape_caps[shape]
        min_chars = min(min_chars, min_cap)
        target_chars = min(target_chars, target_cap)
        max_chars = min(max_chars, max_cap)
    # Repair inconsistent LLM numbers deterministically.
    target_chars = min(max(target_chars, min_chars), max_chars)
    min_chars = min(min_chars, target_chars)
    max_chars = max(max_chars, target_chars)
    max_pages = clamp_int(cs.get("max_page_span"), 1, 6, base.max_pages)
    target_units = clamp_int(cs.get("target_units"), 1, 16, base.target_units)
    max_units = clamp_int(cs.get("max_units"), max(target_units, 4), 24, base.max_units)
    allow_cross = bool(cs.get("allow_cross_section", base.allow_cross_section))
    if shape in {
        "independent_items", "numbered_analytical_prose", "qa_book",
        "narrative", "poetry", "argumentative_philosophy", "dialogue",
    }:
        allow_cross = False
    mode = f"bookplan:{book_plan.get('book_shape') or base.mode}"
    return ChunkPolicy(min_chars, target_chars, max_chars, max_pages, max_units, base.min_units, target_units, mode, allow_cross)


def semantic_book_shape(profile: BookProfile, book_plan: Optional[Dict[str, Any]] = None) -> str:
    raw = clean_text(str((book_plan or {}).get("book_shape") or "")).lower()
    if raw in SEMANTIC_BOOK_SHAPES:
        return raw
    mapping = {
        "item_book": "independent_items",
        "qa_book": "qa_book",
        "poetry_or_aphorism": "poetry",
        "section_prose": "section_prose",
        "numbered_analytical_prose": "numbered_analytical_prose",
        "dense_prose": "dense_expository",
        "table_heavy": "table_heavy",
    }
    return mapping.get(profile.detected_shape, "mixed")


def generator_contract_for_book(
    profile: BookProfile,
    policy: ChunkPolicy,
    book_plan: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> GeneratorContract:
    """Return adaptive final limits without mutating shared CLI args.

    A single global 1,500-char floor damages complete poems, QA pairs, and
    independent items. Conversely, narrative and philosophical prose need more
    context. These limits optimize the common source file for both SFT and DPO.
    """
    shape = semantic_book_shape(profile, book_plan)
    presets: Dict[str, Tuple[int, int, int, int]] = {
        "independent_items": (850, 1400, 2200, 2),
        "numbered_analytical_prose": (1200, 1900, 2600, 1),
        "qa_book": (650, 1100, 1900, 1),
        "narrative": (1400, 2100, 2700, 1),
        "poetry": (500, 900, 1500, 2),
        "argumentative_philosophy": (1400, 2100, 2700, 1),
        "section_prose": (1200, 1900, 2500, 1),
        "dense_expository": (1200, 1900, 2500, 1),
        "dialogue": (900, 1500, 2200, 2),
        "procedural": (1100, 1700, 2400, 1),
        "reference": (800, 1400, 2100, 1),
        "table_heavy": (800, 1400, 2000, 1),
        "mixed": (1200, 1800, 2400, 1),
    }
    min_chars, ideal_min, target, min_units = presets.get(shape, presets["mixed"])
    if not bool(getattr(args, "adaptive_final_limits", True)):
        min_chars = int(getattr(args, "final_min_chars", min_chars))
        ideal_min = int(getattr(args, "final_soft_min_chars", ideal_min))
        target = policy.target_chars
        min_units = int(getattr(args, "final_min_units", min_units))

    hard_max = min(
        int(policy.max_chars),
        int(getattr(args, "generator_hard_max_chars", SAFE_GENERATOR_HARD_MAX_CHARS)),
        SFT_GENERATOR_VISIBLE_CHARS,
    )
    hard_max = max(500, hard_max)
    min_chars = min(min_chars, hard_max)
    target = min(max(target, min_chars), hard_max)
    ideal_min = min(max(ideal_min, min_chars), target)
    return GeneratorContract(
        min_chars=min_chars,
        ideal_min_chars=ideal_min,
        target_chars=target,
        assessor_visible_chars=SFT_ASSESSOR_VISIBLE_CHARS,
        generator_visible_chars=SFT_GENERATOR_VISIBLE_CHARS,
        hard_max_chars=hard_max,
        min_units=min_units,
        semantic_shape=shape,
    )



# -----------------------------------------------------------------------------
# LLM JSON calls
# -----------------------------------------------------------------------------

BOOK_PLAN_PROMPT = """
You are the book-level strategy planner for a 20,000-book heterogeneous Persian OCR corpus.

You receive rule-based profile statistics plus representative text samples from the beginning, middle, and end of one OCR book.
Your job is to infer the book's structural family and return a GENERAL, book-specific strategy for producing source chunks consumed by BOTH an SFT generator and a DPO generator.

DOWNSTREAM CONTRACT:
- The SFT quality assessor reads at most the first 3000 characters.
- The SFT pair generator reads at most the first 4000 characters.
- Prefer complete chunks between roughly 1400 and 3000 characters.
- hard_max_chars must never exceed 3800.
- A complete short poem, QA pair, dialogue exchange, aphorism cluster, or independent item may be shorter than prose.
- Semantic completeness is more important than mechanically reaching a length target.
- The same chunk should support useful grounded tasks: QA/explanation and, when the source permits, summarization, comparison, analysis, extraction, application, or controlled DPO defects.

STRUCTURAL REASONING:
- Use the rule profile as evidence, not as truth.
- Distinguish independent numbered items from numbered clauses that form one argument.
- A real independent-item book normally has a mostly unique, monotonic sequence
  across the whole book (for example 1..1000). Repeated small numbers, labels
  such as 3-2-1, and numbers attached to فصل/جدول/شکل/نمودار are hierarchical
  analytical prose, not independent items.
- Detect narrative scenes/events, poetry/stanzas, philosophical or analytical argument chains, dialogue, procedural material, QA, reference/list material, expository prose, tables, and mixed books.
- Do not infer a subject domain from a few keywords. Plan structure, not content.
- Never rewrite, summarize, correct, or normalize source text.

Return compact JSON only:
{
  "book_shape": "independent_items|numbered_analytical_prose|qa_book|narrative|poetry|argumentative_philosophy|section_prose|dense_expository|dialogue|procedural|reference|table_heavy|mixed",
  "confidence": 0.0,
  "reason": "short reason",
  "unit_strategy": {
    "primary_unit": "numbered_item|paragraph|qa_pair|scene_beat|poetry_stanza|argument_block|dialogue_turn|procedure_step|section_block|mixed",
    "preserve_numbered_items": true,
    "merge_short_continuations": true,
    "speaker_or_section_must_be_in_text": true,
    "atomic_boundary_notes": ["short structural instruction"]
  },
  "chunk_strategy": {
    "target_chars": 2400,
    "soft_min_chars": 1200,
    "soft_max_chars": 3000,
    "hard_max_chars": 3600,
    "target_units": 4,
    "max_units": 14,
    "max_page_span": 4,
    "allow_cross_section": false
  },
  "planner_notes": ["how to preserve this book's semantic continuity"],
  "boundary_risks": ["likely bad starts/ends for this book"],
  "review_notes": ["short practical instruction"],
  "risks": ["short risk"]
}
""".strip()

ATOMIZER_PROMPT = """
You are a general Persian OCR structure atomizer for a heterogeneous 20,000-book corpus.

You receive exact OCR lines. Each line has a stable line_id.
Your job is to mark contiguous atomic text units by line_id ranges.

Atomic unit means the smallest complete source unit that should not be split later:
- section_title: a heading/title
- paragraph: a coherent prose paragraph or paragraph group
- numbered_item: a complete numbered item, including its attached label/date if present
- qa_pair: one complete question/answer pair
- list_item: one list item
- poetry_stanza: one poem/stanza block
- dialogue_turn: one speaker turn, with the speaker label attached
- argument_block: one claim/reason/evidence step that should remain intact
- table: table-like content
- caption: image/table caption

Return only valid compact JSON:
{
  "atomic_units": [
    {"start_line":"l00000001", "end_line":"l00000005", "unit_type":"paragraph", "title":null, "confidence":0.92}
  ],
  "carry_start_line": "l00000006 or null"
}

Rules:
- Use only line_ids from the input.
- Ranges must be contiguous in reading order, non-overlapping, and not reordered.
- Do not rewrite, fix, summarize, or output source text.
- Skip obvious page noise if any remains.
- Preserve speaker/label + numbered item + following date together if they form one item.
- Preserve question and answer together when possible.
- In narrative prose, do not join unrelated scene beats merely because they are on one page.
- In poetry, preserve stanza/poem boundaries; never mix the end of one poem with the start of another.
- In philosophical/analytical prose, keep a claim with its immediate reason, condition, example, qualification, and conclusion.
- Treat semantic dependency as stronger than paragraph or sentence punctuation. A full stop does NOT justify a split if the next lines complete the same reasoning chain.
- NEVER split objection/answer pairs such as «إن قلت ... قلت ...», «اشکال ... پاسخ ...», «اگر گفته شود ... در پاسخ ...», or equivalent Persian/Arabic forms.
- NEVER split an enumeration header from its members, and avoid splitting sibling members of one enumeration when they jointly define the concept (e.g. «اقسام ... عبارت‌اند از: ۱... ۲... ۳...»).
- Words such as «زیرا»، «چون»، «بنابراین»، «در نتیجه»، «پس»، «برای مثال»، «به عبارت دیگر»، «اما» usually signal continuation of the previous semantic unit, not a new unit.
- For analytical/philosophical text, prefer one complete argument block even if it is somewhat longer than the normal target. Semantic completeness is more important than a 1600-character preference; only hard downstream limits are absolute.
- Prefer several safe units over one very large unit, but never create a smaller unit by breaking a required dependency chain.
- If the last lines may continue in the next window, do not commit them; set carry_start_line.
- If this is the final window, carry_start_line must be null and all useful lines should be committed.
- When unsure, still mark a safe unit and lower confidence.
JSON only. No prose outside JSON.
""".strip()

PLANNER_PROMPT = """
You are the semantic boundary planner for Persian OCR source chunks used by BOTH an SFT and a DPO data generator.

You receive ordered ATOMIC UNITS from one book. Your task is ONLY to choose chunk boundary END unit IDs.
Python will assemble the chunks deterministically. You must not rewrite or summarize source text.

CRITICAL RULES:
- Return ONLY valid compact JSON.
- Do NOT output source text.
- Do NOT decide keep/drop. The reviewer handles quality later.
- Do NOT skip any unit inside the committed part.
- Do NOT reorder units.
- Do NOT output start ids. Starts are implicit and continuous.
- Every end_uid must be one of the input uid values.
- end_uids must be in the same order as input.
- If a tail should continue into the next batch, set carry_start to the first uid of that tail.
- If is_final_batch is true, carry_start must be null.
- Units before carry_start must be completely covered by end_uids.
- If there is only one input unit and no carry, return that unit uid as the only end_uid.

DOWNSTREAM CONTRACT:
- The SFT assessor sees at most 3000 source characters.
- The SFT generator sees at most 4000 source characters.
- Prefer the supplied policy target, normally 1400-3000 characters.
- Never intentionally exceed policy.hard_max_chars.
- A semantically complete short poem, QA pair, dialogue exchange, aphorism cluster, or independent item is better than merging unrelated material to satisfy a minimum.
- Do not create tiny fragments when a safe adjacent continuation belongs to the same semantic unit.

BOUNDARY RULES BY BOOK TYPE:
- independent_items: cluster only adjacent items with the same speaker, time frame, theme, or educational purpose; do not mix unrelated quotations.
- numbered_analytical_prose / argumentative_philosophy: preserve complete claim -> reason/evidence -> qualification/example -> conclusion chains. A sentence-ending punctuation mark alone is NOT a safe cut.
- In analytical/philosophical prose, NEVER cut between «إن قلت» and «قلت», objection and answer, question/objection and its direct resolution, a reason and its conclusion, or an enumeration header and the items that complete it.
- Treat «زیرا/چون/چراکه»، «بنابراین/در نتیجه/پس/لذا»، «برای مثال/مثلاً»، «به عبارت دیگر»، «اما/با این حال» as dependency signals when they continue the same argument. Prefer keeping the dependency together even if the chunk is longer than the soft target; only policy.hard_max_chars is absolute.
- Numbered clauses in analytical prose are not independent items by default. Keep sibling numbered clauses together when they are parts, premises, cases, or members of one division.
- qa_book: never split a question from its answer; group adjacent QA only when they share a topic.
- narrative: preserve a scene, event, viewpoint, time, and causal continuity; close at scene/time/location/viewpoint shifts.
- poetry: preserve complete poems or coherent stanza groups; do not mix different poems just to make a long chunk.
- dialogue: keep speaker labels in text and preserve complete exchanges; avoid chunks ending on a speaker label or unanswered question.
- procedural: preserve prerequisites, ordered steps, warnings, and result together when possible.
- section_prose / dense_expository / reference: preserve subsection and topic continuity; avoid mixing separate entries.
- mixed/table-heavy: use conservative boundaries and prefer quarantine-ready small coherent regions over contaminated mixtures.

The source text must remain untouched. You only return ordered boundary unit IDs.

Return exactly this JSON shape:
{
  "end_uids": ["u000004", "u000009", "u000017"],
  "carry_start": null,
  "notes": "very short optional note"
}

Examples:
Input units: u000000,u000001,u000002,u000003
Good: {"end_uids":["u000001","u000003"],"carry_start":null}
This means chunks are u000000-u000001 and u000002-u000003.

Input units: u000010,u000011,u000012 and u000012 probably continues next batch
Good: {"end_uids":["u000011"],"carry_start":"u000012"}

JSON only. No prose outside JSON.
"""

CHUNK_REVIEW_PROMPT = """
You are a strict quality reviewer for Persian OCR source chunks used later by a separate SFT/DPO generation service.

Your task is NOT to decide separate SFT or DPO files. Your task is to judge whether each chunk is a coherent, self-contained, high-quality source evidence chunk with enough context and reasonable length for downstream generation.

Score each chunk from 1 to 5:
- coherence_score: one clear topic/argument/claim cluster; no unrelated mixing.
- self_contained_score: speaker/section/date/context needed to understand the chunk is present in the chunk text or explicit metadata.
- evidence_density_score: enough concrete evidence for grounded QA, reasoning, or preference generation; short quote-only chunks should score low unless unusually complete.
- sft_value_score: usable for a precise grounded instruction/answer.
- dpo_value_score: usable for meaningful rejected answers such as source distortion, wrong attribution, wrong date, overgeneralization, missing condition, or vague answer.
- boundary_quality_score: starts/ends at good natural boundaries.
- noise_score: 1 means clean, 5 means noisy/header/footer/OCR-garbage.

Decision values:
- keep: coherent, self-contained, and useful as a source chunk for downstream generation.
- repair: fixable by split/reboundary/merge; must not be final until repaired and re-reviewed.
- drop: low-value/noisy/metadata/OCR-garbage or not useful.
- quarantine: uncertain, OCR-damaged, table-heavy, or needs human/safer pass.

Repair actions:
- none
- split: mixed-topic or too broad; split inside its own unit range.
- merge_prev: too short/incomplete and likely needs previous chunk.
- merge_next: too short/incomplete and likely needs next chunk.
- reboundary: start/end boundary is bad; nearby units may fix it.
- drop_noise: mostly noise/metadata/header/footer/table junk.

Important quality rules:
- Do NOT keep a chunk merely because it is adjacent ordered text. It must be semantically useful.
- The SFT assessor sees 3000 characters and the generator sees 4000. Penalize chunks whose useful meaning is pushed beyond those limits.
- Do not penalize a complete short poem, QA pair, dialogue exchange, aphorism cluster, or independent item merely for being shorter than prose.
- Penalize short fragments that are incomplete, contextless, or too thin for even one strong grounded task.
- For numbered/item books, prefer coherent clusters by speaker/time/theme/educational purpose. A compact chronological/source cluster is better than many tiny quote-only fragments.
- For narrative, require a coherent scene/event and avoid unexplained scene or viewpoint jumps.
- For philosophical/analytical prose, require a complete argument dependency chain, not merely a grammatically complete paragraph. Penalize cuts between claim/support, reason/conclusion, objection/answer, definition/conditions, and enumeration header/items.
- A chunk that starts with a dependent continuation (e.g. «زیرا»، «بنابراین»، «در نتیجه»، «برای مثال»، «قلت/پاسخ») or ends with an unresolved objection/list header should normally be repaired by merging/reboundary with the relevant neighbor.
- For poetry, preserve poem/stanza integrity and do not demand prose-like evidence density.
- If the chunk text lacks required speaker/section context and metadata cannot safely supply it, lower self_contained_score and use repair/merge/reboundary.
- If you mark repairability high/medium, decision should normally be repair unless the chunk is already good enough.
- If decision is repair, choose a concrete repair_action other than none.

Return exactly this JSON shape:
{
  "reviews": [
    {
      "chunk_id": "...",
      "coherence_score": 4,
      "self_contained_score": 4,
      "evidence_density_score": 4,
      "sft_value_score": 4,
      "dpo_value_score": 3,
      "boundary_quality_score": 4,
      "noise_score": 1,
      "final_score": 4.0,
      "decision": "keep",
      "repair_action": "none",
      "error_type": "good|mixed_topics|too_short|too_long|not_self_contained|bad_boundary|low_information|noisy_ocr|metadata_or_toc|table_or_layout|uncertain",
      "repairability": "none|low|medium|high",
      "reason": "short explanation"
    }
  ]
}
JSON only. No prose outside JSON.
""".strip()

CHUNK_REPAIR_PROMPT = """
You are a Persian OCR chunk repair planner for SFT/DPO training chunks.

You receive the atomic units that formed one bad chunk, plus the reviewer error.
Your task is ONLY to propose better contiguous chunk ranges within the provided units.
Do not rewrite text. Do not output source text. Use uid ranges only.

Goal:
- Fix mixed-topic, too-long, incomplete, or bad-boundary chunks by splitting, merging, or rebounding within the PROVIDED REPAIR WINDOW.
- The repair window may include the original chunk plus one adjacent chunk when the reviewer requested merge_prev/merge_next.
- Preserve complete semantic dependency chains: claim/reason/conclusion, objection/answer, definition/conditions, and enumeration header/items.
- Do not create tiny fragments. Prefer fewer, longer, coherent chunks that preserve enough evidence for downstream generation.
- If the area cannot be repaired without producing short/weak chunks, say quarantine or drop.

Rules:
- Use only uid values in the input.
- Ranges must be contiguous, ordered, non-overlapping, AND collectively cover every provided uid exactly once whenever repair_decision is split or keep.
- Never reorder or split atomic units.
- Never separate «إن قلت» from «قلت», an objection from its answer, or a list/division header from the members needed to complete it.
- Prefer semantically complete chunks over mechanically balanced sizes.
- Return only chunks with at least medium coherence and training value.

Return exactly this JSON shape:
{
  "repair_decision": "split|keep|drop|quarantine",
  "chunks": [
    {"start":"u000001", "end":"u000003", "reason":"coherent subtopic"}
  ],
  "reason": "short explanation"
}
JSON only. No prose outside JSON.
""".strip()



def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM response is not a JSON object")



def models_url_from_chat_url(chat_url: str) -> str:
    """Convert an OpenAI-compatible /v1/chat/completions URL to /v1/models."""
    u = str(chat_url).rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if u.endswith(suffix):
            return u[: -len(suffix)] + "/v1/models"
    if u.endswith("/v1"):
        return u + "/models"
    return u + "/v1/models"


def discover_model_for_endpoint(chat_url: str, api_key: Optional[str], timeout: int = 10) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(models_url_from_chat_url(chat_url), headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    data = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(data, list) and data:
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    return None


def discover_models_for_endpoints(urls: List[str], api_key: Optional[str], timeout: int = 10) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for u in urls:
        try:
            m = discover_model_for_endpoint(u, api_key, timeout=timeout)
            if m:
                out[u] = m
                log(f"[LLM_MODEL_DISCOVERED] {u} model={m}")
            else:
                log(f"[LLM_MODEL_DISCOVERED] {u} model=<none>")
        except Exception as e:
            log(f"[LLM_MODEL_DISCOVERY_ERROR] {u} {e}")
    return out

def call_llm_json(prompt: str, payload: Dict[str, Any], cfg: LLMConfig, max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Call one endpoint from the pool; on refusal/timeout/JSON error, try another.

    Optimizations that do not change semantic logic:
    - endpoint round-robin is thread-safe;
    - max_retries is per endpoint, not global;
    - optional disk cache makes resume/re-runs cheap for identical prompts;
    - HTTP error body prefixes are included for diagnosability.
    """
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    effective_max_tokens = int(max_output_tokens or cfg.max_output_tokens)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload_json},
    ]

    cache_path: Optional[Path] = None
    if cfg.cache_enabled and cfg.cache_dir:
        cache_material = json.dumps({
            "prompt": prompt,
            "payload": payload,
            "model": cfg.model,
            "models_by_url": cfg.models_by_url,
            "temperature": cfg.temperature,
            "max_tokens": effective_max_tokens,
            "response_format_json": cfg.response_format_json,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha1(cache_material.encode("utf-8")).hexdigest()
        cache_path = cfg.cache_dir / key[:2] / f"{key}.json"
        if cache_path.exists():
            try:
                with cfg._cache_lock:
                    cached = load_json(cache_path)
                if isinstance(cached, dict) and isinstance(cached.get("result"), dict):
                    log(f"[LLM_CACHE_HIT] {key[:10]}")
                    return cached["result"]
            except Exception as e:
                log(f"[LLM_CACHE_READ_ERROR] {cache_path} {e}")

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    urls = cfg.endpoint_urls()
    attempts = max(1, cfg.max_retries) * max(1, len(urls))
    last_error: Optional[Exception] = None
    tried: List[str] = []
    # Reserve a stable local starting point. This guarantees each call cycles
    # through the whole endpoint pool even when many threads share cfg.
    with cfg._rr_lock:
        start_index = cfg._rr_index % max(1, len(urls))
        cfg._rr_index += 1
    for attempt in range(attempts):
        endpoint = urls[(start_index + attempt) % len(urls)]
        tried.append(endpoint)
        model = cfg.model_for_url(endpoint)
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": effective_max_tokens,
        }
        if cfg.response_format_json:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            obj = extract_json_object(content)
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache_path.with_suffix(".tmp")
                    with cfg._cache_lock:
                        with tmp.open("w", encoding="utf-8") as f:
                            json.dump({"result": obj}, f, ensure_ascii=False, separators=(",", ":"))
                        os.replace(tmp, cache_path)
                except Exception as e:
                    log(f"[LLM_CACHE_WRITE_ERROR] {cache_path} {e}")
            return obj
        except Exception as e:
            # Include HTTP status/body prefix when vLLM rejects the request; this
            # makes 400 errors diagnosable without storing huge prompts.
            if isinstance(e, urllib.error.HTTPError):
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:800]
                except Exception:
                    err_body = ""
                last_error = RuntimeError(f"HTTP {e.code} {e.reason}: {err_body}")
            else:
                last_error = e
            # Tiny backoff. We still rotate to the next endpoint on the next attempt.
            time.sleep(min(0.5 * (attempt + 1), 3.0))
    tried_short = ",".join(dict.fromkeys(tried))
    raise RuntimeError(f"LLM call failed after {attempts} pooled attempt(s); endpoints={tried_short}; last_error={last_error}")

# -----------------------------------------------------------------------------
# Atomic unit construction
# -----------------------------------------------------------------------------


def line_range_text(lines: Sequence[LineRef]) -> str:
    return clean_text("\n".join(ln.text for ln in lines if ln.text.strip()))


def make_atomic_unit(
    idx: int,
    unit_type: str,
    lines: List[LineRef],
    section_path: Optional[List[str]] = None,
    title: Optional[str] = None,
    confidence: float = 1.0,
    source: str = "rule",
) -> Optional[AtomicUnit]:
    if not lines:
        return None
    text = line_range_text(lines)
    if not text or alpha_len(text) < 3:
        return None
    return AtomicUnit(
        uid=f"u{idx:06d}",
        idx=idx,
        unit_type=unit_type,
        text=text,
        lines=lines,
        section_path=section_path or [],
        title=title,
        confidence=confidence,
        source=source,
    )


def assign_sections(units: List[AtomicUnit]) -> None:
    """Assign a simple rolling section path from section_title units."""
    section_path: List[str] = []
    for u in units:
        if u.unit_type == "section_title":
            title = clean_text(u.title or u.text).replace("#", "").strip()
            if title:
                if not section_path or section_path[-1] != title:
                    if len(section_path) >= 3:
                        section_path = section_path[-2:]
                    section_path.append(title)
            u.section_path = list(section_path)
        else:
            if not u.section_path:
                u.section_path = list(section_path)


def fallback_paragraph_units(lines: List[LineRef], source: str = "fallback") -> List[AtomicUnit]:
    """Safe generic fallback units from lines; groups by page/block/soft size."""
    units: List[AtomicUnit] = []
    cur: List[LineRef] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        ut = "paragraph"
        if len(cur) == 1 and is_section_like_line(cur[0].text, cur[0].block_category):
            ut = "section_title"
        elif any(QA_Q_RE.match(x.text) or QA_A_RE.match(x.text) for x in cur):
            ut = "qa_pair"
        elif any(NUMBERED_LINE_RE.match(x.text) for x in cur):
            ut = "numbered_item"
        u = make_atomic_unit(len(units), ut, cur, source=source)
        if u:
            units.append(u)
        cur = []

    for ln in lines:
        if not cur:
            cur = [ln]
            continue
        same_block = cur[-1].block_id == ln.block_id
        hard_start = is_section_like_line(ln.text, ln.block_category) or QA_Q_RE.match(ln.text) or NUMBERED_LINE_RE.match(ln.text)
        too_big = len(line_range_text(cur + [ln])) > 1300
        page_jump = ln.file_index != cur[-1].file_index
        if hard_start or too_big or (page_jump and not same_block):
            flush()
            cur = [ln]
        else:
            cur.append(ln)
    flush()
    assign_sections(units)
    for i, u in enumerate(units):
        u.idx = i
        u.uid = f"u{i:06d}"
    return units


def rule_atomize(
    lines: List[LineRef],
    profile: BookProfile,
    semantic_shape: Optional[str] = None,
) -> List[AtomicUnit]:
    """General rule atomizer. This is a fallback/base, not a domain-specific final judge."""
    units: List[AtomicUnit] = []
    cur: List[LineRef] = []
    pending_label: List[LineRef] = []
    shape = semantic_shape or profile.detected_shape

    def add_unit(unit_type: str, group: List[LineRef], conf: float = 1.0) -> None:
        if not group:
            return
        u = make_atomic_unit(len(units), unit_type, group, confidence=conf, source="rule")
        if u:
            units.append(u)

    def flush(default_type: str = "paragraph") -> None:
        nonlocal cur
        if not cur:
            return
        ut = default_type
        if any(QA_Q_RE.match(x.text) or QA_A_RE.match(x.text) for x in cur):
            ut = "qa_pair"
        elif any(NUMBERED_LINE_RE.match(x.text) for x in cur):
            ut = "numbered_item"
        elif shape in {"poetry", "poetry_or_aphorism"}:
            ut = "poetry_stanza"
        elif shape in {"dialogue", "narrative"} and any(looks_like_speaker_label(x.text) for x in cur):
            ut = "dialogue_turn"
        elif shape in {"argumentative_philosophy", "numbered_analytical_prose"}:
            ut = "argument_block"
        add_unit(ut, cur)
        cur = []

    for ln in lines:
        txt = ln.text

        if is_section_like_line(txt, ln.block_category) or SCENE_BREAK_RE.match(txt):
            flush()
            add_unit("section_title", [ln])
            pending_label = []
            continue

        if QA_Q_RE.match(txt):
            flush()
            cur = [ln]
            pending_label = []
            continue

        if NUMBERED_LINE_RE.match(txt):
            flush()
            cur = pending_label + [ln]
            pending_label = []
            continue

        if DATE_ONLY_RE.match(txt) and cur:
            cur.append(ln)
            continue

        # Poetry/stanza layout is often represented as multiple short OCR
        # blocks. Preserve compact stanza groups and never grow a page-sized
        # prose unit merely because punctuation is sparse.
        if shape in {"poetry", "poetry_or_aphorism"}:
            block_changed = bool(cur and cur[-1].block_id != ln.block_id)
            page_changed = bool(cur and cur[-1].file_index != ln.file_index)
            candidate_len = len(line_range_text(cur + [ln])) if cur else len(txt)
            if cur and (
                page_changed
                or candidate_len > 900
                or (block_changed and len(cur) >= 2 and len(line_range_text(cur)) >= 160)
            ):
                flush("poetry_stanza")
            cur.append(ln)
            continue

        # A true speaker label attaches to its following turn.  Generic
        # structural labels remain headings/body text and are not treated as a
        # person merely because they end in a colon.
        if looks_like_speaker_label(txt) and not cur:
            pending_label = [ln]
            continue
        if looks_like_speaker_label(txt) and cur and shape in {"item_book", "independent_items", "dialogue", "narrative"}:
            # Likely next item's label, not current item's content.
            flush()
            pending_label = [ln]
            continue

        if pending_label and not cur:
            cur = pending_label + [ln]
            pending_label = []
        else:
            # In prose, prefer a complete sentence/argument/scene beat near the
            # target instead of cutting mechanically at exactly 1400 chars.
            # The oversized-atom pass remains the hard safety net.
            if cur and shape not in {"item_book", "independent_items", "qa_book"}:
                current_len = len(line_range_text(cur))
                candidate_len = len(line_range_text(cur + [ln]))
                natural_end = bool(PROSE_TERMINAL_RE.search(cur[-1].text))
                argument_shift = bool(
                    shape in {"argumentative_philosophy", "numbered_analytical_prose"}
                    and ARGUMENT_TRANSITION_RE.match(txt)
                )
                narrative_shift = bool(
                    shape in {"narrative", "dialogue"}
                    and (
                        SCENE_BREAK_RE.match(txt)
                        or (looks_like_speaker_label(txt) and current_len >= 350)
                    )
                )
                if (
                    candidate_len > 1550
                    or (current_len >= 900 and natural_end and (argument_shift or narrative_shift))
                    or (current_len >= 1250 and natural_end)
                ):
                    flush("paragraph")
            cur.append(ln)

    flush()
    # Preserve a final dangling label as a risk-flagged source unit.  Lossless
    # downstream-trust mode must not silently discard readable OCR text.
    if pending_label:
        add_unit("dialogue_turn", pending_label, conf=0.55)
    assign_sections(units)
    for i, u in enumerate(units):
        u.idx = i
        u.uid = f"u{i:06d}"
    return units


# -----------------------------------------------------------------------------
# LLM atomizer
# -----------------------------------------------------------------------------


def compact_line_for_llm(ln: LineRef, text_limit: int = 500) -> Dict[str, Any]:
    t = ln.text
    if len(t) > text_limit:
        t = t[:text_limit].rstrip() + " …"
    return {
        "line_id": ln.lid,
        "p": ln.page_label,
        "f": ln.file_index,
        "block": ln.block_id,
        "cat": ln.block_category,
        "text": t,
    }


def make_line_batches(lines: List[LineRef], max_tokens: int, max_lines: int) -> List[List[LineRef]]:
    batches: List[List[LineRef]] = []
    cur: List[LineRef] = []
    cur_tok = 0
    for ln in lines:
        tok = estimate_tokens(compact_line_for_llm(ln)) + 4
        if cur and (cur_tok + tok > max_tokens or len(cur) >= max_lines):
            batches.append(cur)
            cur = []
            cur_tok = 0
        cur.append(ln)
        cur_tok += tok
    if cur:
        batches.append(cur)
    return batches


def validate_atomizer_plan(plan: Dict[str, Any], candidate: List[LineRef]) -> Tuple[List[Tuple[int, int, str, Optional[str], float]], Optional[str]]:
    id_to_pos = {ln.lid: i for i, ln in enumerate(candidate)}
    ranges: List[Tuple[int, int, str, Optional[str], float]] = []
    last_end = -1
    for item in plan.get("atomic_units") or []:
        if not isinstance(item, dict):
            continue
        s = item.get("start_line")
        e = item.get("end_line")
        if s not in id_to_pos or e not in id_to_pos:
            continue
        a, b = id_to_pos[s], id_to_pos[e]
        if a > b or a <= last_end:
            continue
        unit_type = str(item.get("unit_type") or "paragraph")
        if unit_type not in {
            "section_title", "paragraph", "numbered_item", "qa_pair",
            "list_item", "poetry_stanza", "dialogue_turn", "argument_block",
            "table", "caption",
        }:
            unit_type = "paragraph"
        title = item.get("title") if isinstance(item.get("title"), str) else None
        try:
            conf = float(item.get("confidence", 0.8))
        except Exception:
            conf = 0.8
        conf = max(0.0, min(1.0, conf))
        ranges.append((a, b, unit_type, title, conf))
        last_end = b
    carry = plan.get("carry_start_line")
    if carry is not None and carry not in id_to_pos:
        carry = None
    return ranges, carry


def build_units_from_ranges_with_gap_fill(
    ranges: List[Tuple[int, int, str, Optional[str], float]],
    candidate: List[LineRef],
    commit_stop: int,
    source: str,
) -> List[AtomicUnit]:
    """Create units from validated LLM ranges and fill uncovered committed gaps."""
    out: List[AtomicUnit] = []
    cursor = 0
    local_ranges = [r for r in ranges if r[0] <= commit_stop]
    for a, b, unit_type, title, conf in local_ranges:
        if a > cursor:
            out.extend(fallback_paragraph_units(candidate[cursor:min(a, commit_stop + 1)], source="gap_fallback"))
        b = min(b, commit_stop)
        u = make_atomic_unit(len(out), unit_type, candidate[a:b + 1], title=title, confidence=conf, source=source)
        if u:
            out.append(u)
        cursor = b + 1
    if cursor <= commit_stop:
        out.extend(fallback_paragraph_units(candidate[cursor:commit_stop + 1], source="gap_fallback"))
    # reindex local
    for i, u in enumerate(out):
        u.idx = i
        u.uid = f"u{i:06d}"
    assign_sections(out)
    return out



def _clone_line_ref_with_text(ln: LineRef, text: str, suffix: str) -> LineRef:
    """Create a derived line ref for splitting a single very long OCR line.

    The original lid is preserved as a prefix so audit trails still point back
    to the source line, while downstream chunk assembly can use the derived
    text span as a normal LineRef.
    """
    return LineRef(
        lid=f"{ln.lid}_{suffix}",
        global_index=ln.global_index,
        file_index=ln.file_index,
        page_label=ln.page_label,
        block_id=ln.block_id,
        block_category=ln.block_category,
        line_index=ln.line_index,
        text=text,
    )


def split_long_line_ref(ln: LineRef, max_chars: int, target_chars: int) -> List[LineRef]:
    """Soft-split an OCR line that is itself too large to be one atomic unit."""
    text = clean_text(ln.text)
    if len(text) <= max_chars:
        return [ln]
    # Prefer Persian/Arabic/Latin sentence punctuation, then hard split.
    pieces = re.split(r"(?<=[\.؟\?!؛;])\s+", text)
    out: List[LineRef] = []
    cur = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if len(cur) + 1 + len(piece) <= target_chars:
            cur = cur + " " + piece
        else:
            out.append(_clone_line_ref_with_text(ln, cur, f"s{len(out):03d}"))
            cur = piece
    if cur:
        out.append(_clone_line_ref_with_text(ln, cur, f"s{len(out):03d}"))

    # If punctuation splitting was not enough, hard split the remaining pieces.
    final: List[LineRef] = []
    for x in out or [_clone_line_ref_with_text(ln, text, "s000")]:
        if len(x.text) <= max_chars:
            final.append(x)
            continue
        raw = x.text
        pos = 0
        while pos < len(raw):
            chunk = raw[pos:pos + target_chars].strip()
            if chunk:
                final.append(_clone_line_ref_with_text(ln, chunk, f"h{len(final):03d}"))
            pos += target_chars
    return final or [ln]


def split_large_atomic_units(
    units: List[AtomicUnit],
    args: argparse.Namespace,
    profile: Optional[BookProfile] = None,
    semantic_shape: Optional[str] = None,
) -> Tuple[List[AtomicUnit], Dict[str, Any]]:
    """Deterministically split oversized atomic units before chunk planning.

    Planner boundaries are only between AtomicUnit objects. If an atomizer emits
    a 4k-10k char unit, no planner can create clean chunk boundaries inside it.
    This pass turns such units into smaller contiguous units while preserving
    order, page refs, section path, unit type, and source metadata.
    """
    max_chars = int(getattr(args, "max_atomic_chars", 1200) or 1200)
    target_chars = int(getattr(args, "target_atomic_chars", min(max_chars, 850)) or min(max_chars, 850))
    max_lines = int(getattr(args, "max_atomic_lines", 18) or 18)
    enabled = bool(getattr(args, "split_oversized_atoms", True))
    if not enabled or max_chars <= 0:
        return units, {"enabled": False, "input_units": len(units), "output_units": len(units), "split_units": 0}

    out: List[AtomicUnit] = []
    split_units = 0
    derived_line_splits = 0
    max_seen = 0
    shape = semantic_shape or (profile.detected_shape if profile else "")

    def flush_group(src: AtomicUnit, group: List[LineRef], part_index: int) -> None:
        if not group:
            return
        title = src.title if part_index == 0 else None
        u = make_atomic_unit(
            len(out),
            src.unit_type,
            group,
            section_path=list(src.section_path),
            title=title,
            confidence=min(src.confidence, 0.92),
            source=f"{src.source}+split_large",
        )
        if u:
            out.append(u)

    for u in units:
        max_seen = max(max_seen, len(u.text))
        if len(u.text) <= max_chars and len(u.lines) <= max_lines:
            out.append(u)
            continue

        split_units += 1
        expanded_lines: List[LineRef] = []
        for ln in u.lines:
            parts = split_long_line_ref(ln, max_chars=max_chars, target_chars=target_chars)
            if len(parts) > 1:
                derived_line_splits += len(parts) - 1
            expanded_lines.extend(parts)

        cur: List[LineRef] = []
        part_index = 0
        for ln in expanded_lines:
            if not cur:
                cur = [ln]
                continue
            cur_text_len = len(line_range_text(cur))
            candidate_len = len(line_range_text(cur + [ln]))
            analytical_shape = shape in {"numbered_analytical_prose", "argumentative_philosophy"}
            # In analytical/philosophical prose, a numbered line is often a premise,
            # case, condition, or member of one enumeration rather than an independent
            # item.  The full-LLM atomizer already supplied the semantic unit, so this
            # safety split must not undo that decision merely because a line is numbered.
            structural_start = bool(
                QA_Q_RE.match(ln.text)
                or is_section_like_line(ln.text, ln.block_category)
                or (NUMBERED_LINE_RE.match(ln.text) and not analytical_shape)
            )
            semantic_risk = (
                semantic_boundary_penalty_text(line_range_text(cur), ln.text)
                if analytical_shape and cur else 0
            )
            semantic_overflow_cap = min(
                int(getattr(args, "generator_hard_max_chars", SAFE_GENERATOR_HARD_MAX_CHARS)),
                max(max_chars, int(max_chars * 1.30)),
            )
            preserve_dependency = bool(
                analytical_shape
                and semantic_risk >= 4
                and candidate_len <= semantic_overflow_cap
                and len(cur) < max_lines + 8
            )
            # Hard size still wins, but a modest overflow is allowed when cutting here
            # would separate a conclusion/reason/objection-answer/enumeration chain.
            should_flush = (
                (candidate_len > max_chars and not preserve_dependency)
                or (len(cur) >= max_lines and not preserve_dependency)
                or (
                    structural_start
                    and cur_text_len >= max(240, target_chars // 2)
                    and shape in {"item_book", "independent_items", "qa_book", "mixed"}
                )
            )
            if should_flush:
                flush_group(u, cur, part_index)
                part_index += 1
                cur = [ln]
            else:
                cur.append(ln)
        flush_group(u, cur, part_index)

    for i, u in enumerate(out):
        u.idx = i
        u.uid = f"u{i:06d}"
    assign_sections(out)
    stats = {
        "enabled": True,
        "input_units": len(units),
        "output_units": len(out),
        "split_units": split_units,
        "added_units": max(0, len(out) - len(units)),
        "derived_line_splits": derived_line_splits,
        "max_unit_chars_before": max_seen,
        "max_unit_chars_after": max((len(u.text) for u in out), default=0),
        "max_atomic_chars": max_chars,
        "target_atomic_chars": target_chars,
        "max_atomic_lines": max_lines,
    }
    return out, stats


def _script_counts(text: str) -> Tuple[int, int]:
    persian = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return persian, latin


def split_probable_noise_units(
    units: List[AtomicUnit],
    args: argparse.Namespace,
) -> Tuple[List[AtomicUnit], Dict[str, Any]]:
    """Isolate, but never delete, probable OCR/model hallucination runs.

    Long repeated Latin passages embedded after Persian content were observed
    in real OCR output.  Keeping them in the same chunk contaminates otherwise
    valuable material.  This pass only creates boundaries and a risk type;
    downstream_trust still writes every readable unit to the accepted stream.
    """
    enabled = bool(getattr(args, "isolate_ocr_noise", True))
    if not enabled or not units:
        return units, {
            "enabled": enabled,
            "input_units": len(units),
            "output_units": len(units),
            "noise_units": 0,
            "split_units": 0,
        }

    fingerprints: Counter[str] = Counter()
    for u in units:
        for ln in u.lines:
            txt = clean_text(ln.text)
            if len(txt) >= 80:
                fingerprints[re.sub(r"\s+", " ", txt.lower())[:600]] += 1

    def noisy_line(ln: LineRef) -> bool:
        txt = clean_text(ln.text)
        if len(txt) < 80:
            return False
        fa, en = _script_counts(txt)
        fp = re.sub(r"\s+", " ", txt.lower())[:600]
        repeated = fingerprints[fp] >= 2
        latin_dominant = en >= 70 and en >= max(1, fa * 4)
        symbol_garbage = len(re.findall(r"[^\w\s\u0600-\u06FF]", txt)) > len(txt) * 0.35
        return bool((latin_dominant and (repeated or len(txt) >= 450)) or symbol_garbage)

    out: List[AtomicUnit] = []
    split_units = 0
    noise_units = 0

    def emit(src: AtomicUnit, line_group: List[LineRef], noise: bool, part: int) -> None:
        nonlocal noise_units
        if not line_group:
            return
        unit_type = "ocr_noise" if noise else src.unit_type
        made = make_atomic_unit(
            len(out),
            unit_type,
            line_group,
            section_path=list(src.section_path),
            title=src.title if part == 0 and not noise else None,
            confidence=min(src.confidence, 0.40 if noise else 0.90),
            source=f"{src.source}+noise_isolation" if noise else src.source,
        )
        if made:
            out.append(made)
            noise_units += int(noise)

    for u in units:
        labels = [noisy_line(ln) for ln in u.lines]
        if not any(labels):
            out.append(u)
            continue
        if all(labels):
            u.unit_type = "ocr_noise"
            u.confidence = min(u.confidence, 0.40)
            u.source = f"{u.source}+noise_isolation"
            out.append(u)
            noise_units += 1
            continue
        split_units += 1
        start = 0
        part = 0
        for i in range(1, len(u.lines) + 1):
            if i == len(u.lines) or labels[i] != labels[start]:
                emit(u, u.lines[start:i], labels[start], part)
                part += 1
                start = i

    for i, u in enumerate(out):
        u.idx = i
        u.uid = f"u{i:06d}"
    assign_sections(out)
    return out, {
        "enabled": True,
        "input_units": len(units),
        "output_units": len(out),
        "noise_units": noise_units,
        "split_units": split_units,
    }


def llm_atomize(
    lines: List[LineRef],
    profile: BookProfile,
    cfg: LLMConfig,
    args: argparse.Namespace,
    debug_path: Optional[Path],
    semantic_shape: Optional[str] = None,
    book_plan: Optional[Dict[str, Any]] = None,
) -> List[AtomicUnit]:
    """LLM atomizer with adaptive shrink/split and production-safe fallback.

    Unlike the older strict implementation, a request-size failure in one batch
    no longer kills the entire book in production/balanced modes. It first
    shrinks line text and output budget, then recursively splits the window.
    Strict no-fallback behavior is enabled by --preset sft_dpo_quality or --strict-llm.
    """
    batches = make_line_batches(lines, args.atomizer_max_input_tokens, args.atomizer_max_lines)
    all_units: List[AtomicUnit] = []
    carry: List[LineRef] = []
    started = time.time()
    strict = bool(getattr(args, "strict_llm", False)) or getattr(args, "preset", "") == "sft_dpo_quality"

    def _units_from_plan_for_candidate(candidate: List[LineRef], plan: Dict[str, Any], is_final_window: bool, source: str) -> Tuple[List[AtomicUnit], List[LineRef], Dict[str, Any]]:
        ranges, carry_start = validate_atomizer_plan(plan, candidate)
        id_to_pos = {ln.lid: i for i, ln in enumerate(candidate)}
        if carry_start:
            if is_final_window:
                carry_start = None
            else:
                cpos = id_to_pos[carry_start]
                commit_stop = max(-1, cpos - 1)
                new_carry = candidate[cpos:]
                units = build_units_from_ranges_with_gap_fill(ranges, candidate, commit_stop, source=source) if commit_stop >= 0 else []
                return units, new_carry, {"accepted_ranges": len(ranges), "carry_start": carry_start}
        commit_stop = len(candidate) - 1
        units = build_units_from_ranges_with_gap_fill(ranges, candidate, commit_stop, source=source) if commit_stop >= 0 else []
        return units, [], {"accepted_ranges": len(ranges), "carry_start": None}

    def _atomize_candidate_adaptive(candidate: List[LineRef], is_final_window: bool, label: str, depth: int = 0) -> Tuple[List[AtomicUnit], List[LineRef], Dict[str, Any]]:
        if not candidate:
            return [], [], {"mode": "empty"}
        base_text = int(args.atomizer_text_limit)
        base_out = int(cfg.max_output_tokens)
        variants: List[Tuple[int, int]] = []
        for tl, ot in [
            (base_text, base_out),
            (min(base_text, 180), min(base_out, 768)),
            (min(base_text, 120), min(base_out, 640)),
            (min(base_text, 80), min(base_out, 512)),
            (min(base_text, 60), min(base_out, 384)),
        ]:
            item = (max(40, tl), max(256, ot))
            if item not in variants:
                variants.append(item)

        last_error: Optional[Exception] = None
        for attempt_i, (text_limit, out_tokens) in enumerate(variants, 1):
            payload = {
                "language": "fa",
                "profile_guess": asdict(profile),
                "semantic_shape": semantic_shape or profile.detected_shape,
                "unit_strategy": (book_plan or {}).get("unit_strategy") or {},
                "is_final_window": is_final_window,
                "lines": [compact_line_for_llm(ln, text_limit) for ln in candidate],
            }
            prompt_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            log(f"[ATOMIZER_LLM_CALL] {label} attempt={attempt_i}/{len(variants)} depth={depth} lines={candidate[0].lid}-{candidate[-1].lid} n={len(candidate)} text_limit={text_limit} out_tokens={out_tokens} chars={len(prompt_json)} est_tok={estimate_tokens(prompt_json)}")
            t0 = time.time()
            try:
                plan = call_llm_json(ATOMIZER_PROMPT, payload, cfg, max_output_tokens=out_tokens)
                units, new_carry, meta2 = _units_from_plan_for_candidate(candidate, plan, is_final_window, source="llm")
                if not units and not new_carry:
                    raise ValueError("atomizer returned no valid units")
                dt = time.time() - t0
                meta = {"mode": "llm", "attempt": attempt_i, "text_limit": text_limit, "out_tokens": out_tokens, "depth": depth, "plan": plan, "llm_sec": round(dt, 2), **meta2}
                log(f"[ATOMIZER_LLM_DONE] {label} depth={depth} llm={dt:.1f}s units={len(units)} carry={len(new_carry)}")
                return units, new_carry, meta
            except Exception as e:
                last_error = e
                log(f"[ATOMIZER_LLM_RETRY] {label} attempt={attempt_i}/{len(variants)} depth={depth} failed={e}")
                continue

        max_depth = int(getattr(args, "atomizer_adaptive_split_depth", 4) or 4)
        if len(candidate) > 1 and depth < max_depth:
            mid = max(1, len(candidate) // 2)
            left = candidate[:mid]
            right = candidate[mid:]
            log(f"[ATOMIZER_LLM_SPLIT] {label} depth={depth} n={len(candidate)} split={len(left)}+{len(right)} reason={last_error}")
            left_units, left_carry, left_meta = _atomize_candidate_adaptive(left, True, label + ".L", depth + 1)
            if left_carry:
                extra, extra_carry, _ = _atomize_candidate_adaptive(left_carry, True, label + ".Lcarry", depth + 1)
                left_units.extend(extra)
                if extra_carry and strict:
                    raise RuntimeError(f"atomizer carry remained after left split for {label}")
                elif extra_carry:
                    left_units.extend(rule_atomize(extra_carry, profile, semantic_shape))
            right_units, right_carry, right_meta = _atomize_candidate_adaptive(right, True, label + ".R", depth + 1)
            if right_carry:
                extra, extra_carry, _ = _atomize_candidate_adaptive(right_carry, True, label + ".Rcarry", depth + 1)
                right_units.extend(extra)
                if extra_carry and strict:
                    raise RuntimeError(f"atomizer carry remained after right split for {label}")
                elif extra_carry:
                    right_units.extend(rule_atomize(extra_carry, profile, semantic_shape))
            return left_units + right_units, [], {"mode": "llm_split", "depth": depth, "left": left_meta.get("mode"), "right": right_meta.get("mode")}

        if strict:
            raise RuntimeError(f"STRICT_ATOMIZER_LLM_FAILED_NO_FALLBACK {label}: {last_error}")
        log(f"[ATOMIZER_LLM_FALLBACK_RULE] {label} reason={last_error}")
        return rule_atomize(candidate, profile, semantic_shape), [], {"mode": "rule_fallback", "error": str(last_error)}

    llm_concurrency = max(1, int(getattr(args, "llm_concurrency", 1) or 1))
    if llm_concurrency > 1 and len(batches) > 1:
        # Parallel full-LLM atomization. Windows overlap for context, but only
        # boundaries whose END lies inside that batch's non-overlapping core are
        # committed. We then reconstruct one exact ordered partition of source
        # lines, so overlap can never duplicate or drop text.
        lid_to_pos = {ln.lid: i for i, ln in enumerate(lines)}
        overlap = max(0, int(getattr(args, "atomizer_overlap_lines", 8) or 0))
        jobs = []
        for bi, batch in enumerate(batches):
            core_start = lid_to_pos[batch[0].lid]
            core_end = lid_to_pos[batch[-1].lid]
            cand_start = max(0, core_start - overlap)
            cand_end = min(len(lines) - 1, core_end + overlap)
            candidate = lines[cand_start:cand_end + 1]
            jobs.append((bi, core_start, core_end, candidate))

        log(f"[ATOMIZER_PARALLEL] windows={len(jobs)} concurrency={llm_concurrency} overlap_lines={overlap}")
        cut_meta: Dict[int, Tuple[str, Optional[str], float, str]] = {}
        debug_rows: List[Dict[str, Any]] = []
        errors: List[Tuple[int, Exception]] = []

        def _run_atom_job(job):
            bi, core_start, core_end, candidate = job
            label = f"pwin={bi+1}/{len(jobs)}"
            units, rem, meta = _atomize_candidate_adaptive(candidate, True, label, 0)
            if rem:
                raise RuntimeError(f"parallel atomizer returned carry in final window: {label}")
            local = []
            for u in units:
                if not u.lines:
                    continue
                end_lid = u.lines[-1].lid
                if end_lid not in lid_to_pos:
                    continue
                end_pos = lid_to_pos[end_lid]
                if core_start <= end_pos <= core_end:
                    local.append((end_pos, u.unit_type, u.title, u.confidence, u.source))
            return bi, core_start, core_end, candidate, local, meta

        with ThreadPoolExecutor(max_workers=min(llm_concurrency, len(jobs))) as ex:
            futs = {ex.submit(_run_atom_job, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                bi = futs[fut]
                try:
                    bi2, core_start, core_end, candidate, local, meta = fut.result()
                    for end_pos, unit_type, title, conf, source in local:
                        cut_meta[end_pos] = (unit_type, title, conf, source)
                    debug_rows.append({
                        "batch": bi2 + 1,
                        "parallel": True,
                        "core_line_range": [lines[core_start].lid, lines[core_end].lid],
                        "window_line_range": [candidate[0].lid, candidate[-1].lid],
                        "meta": {k: v for k, v in meta.items() if k != "plan"},
                        "plan": meta.get("plan"),
                    })
                except Exception as e:
                    errors.append((bi, e))

        if errors:
            errors.sort(key=lambda x: x[0])
            raise RuntimeError("PARALLEL_ATOMIZER_FAILED " + "; ".join(f"batch={i+1}:{e}" for i, e in errors[:5]))

        # Always close the final source span. Other cuts come only from LLM.
        cut_meta.setdefault(len(lines) - 1, ("paragraph", None, 0.75, "llm_parallel_final"))
        cursor = 0
        all_units = []
        for cut in sorted(cut_meta):
            if cut < cursor:
                continue
            unit_type, title, conf, source = cut_meta[cut]
            u = make_atomic_unit(len(all_units), unit_type, lines[cursor:cut + 1], title=title, confidence=conf, source=source + "+parallel")
            if u:
                all_units.append(u)
            cursor = cut + 1
        if cursor < len(lines):
            u = make_atomic_unit(len(all_units), "paragraph", lines[cursor:], confidence=0.70, source="llm_parallel_tail")
            if u:
                all_units.append(u)
        if debug_path:
            for row in sorted(debug_rows, key=lambda r: r["batch"]):
                append_jsonl(debug_path, row)
        log(f"[ATOMIZER_PARALLEL_DONE] windows={len(jobs)} units={len(all_units)} elapsed={fmt_duration(time.time()-started)}")
    else:
        for bi, batch in enumerate(batches):
            is_final = bi == len(batches) - 1
            candidate = carry + batch
            if not candidate:
                continue
            label = f"batch={bi+1}/{len(batches)}"
            try:
                units, carry, meta = _atomize_candidate_adaptive(candidate, is_final, label, 0)
                all_units.extend(units)
                if len(carry) > args.atomizer_max_carry_lines:
                    forced = carry[:-args.atomizer_carry_keep_lines]
                    keep = carry[-args.atomizer_carry_keep_lines:]
                    if forced:
                        fu, fc, _ = _atomize_candidate_adaptive(forced, True, label + ".carryflush", 0)
                        all_units.extend(fu)
                        if fc:
                            all_units.extend(rule_atomize(fc, profile, semantic_shape))
                    carry = keep
                if debug_path:
                    append_jsonl(debug_path, {
                        "batch": bi + 1,
                        "line_range": [candidate[0].lid, candidate[-1].lid],
                        "meta": {k: v for k, v in meta.items() if k != "plan"},
                        "plan": meta.get("plan"),
                        "units_total": len(all_units),
                        "carry_lines": len(carry),
                    })
                log(f"[ATOMIZER_BATCH_DONE] {label} units_total={len(all_units)} carry={len(carry)} elapsed={fmt_duration(time.time()-started)}")
            except Exception as e:
                log(f"[ATOMIZER_LLM_ERROR] {label} {e}")
                if strict:
                    raise
                log(f"[ATOMIZER_LLM_ERROR] {label} fallback=rule")
                if is_final or len(candidate) <= args.atomizer_carry_keep_lines:
                    all_units.extend(rule_atomize(candidate, profile, semantic_shape))
                    carry = []
                else:
                    prefix = candidate[:-args.atomizer_carry_keep_lines]
                    carry = candidate[-args.atomizer_carry_keep_lines:]
                    all_units.extend(rule_atomize(prefix, profile, semantic_shape))

        if carry:
            if strict:
                units, carry2, _ = _atomize_candidate_adaptive(carry, True, "final_carry", 0)
                all_units.extend(units)
                if carry2:
                    raise RuntimeError("atomizer final carry remained in strict mode")
            else:
                all_units.extend(rule_atomize(carry, profile, semantic_shape))

    for i, u in enumerate(all_units):
        u.idx = i
        u.uid = f"u{i:06d}"
    assign_sections(all_units)
    return all_units

def should_use_llm_atomizer(
    profile: BookProfile,
    args: argparse.Namespace,
    book_plan: Optional[Dict[str, Any]] = None,
) -> bool:
    if args.atomizer == "never" or args.atomizer == "rule":
        return False
    if args.atomizer == "llm":
        return True

    # Auto means adaptive, not rule-only and not LLM-for-every-line.
    # For reliable regular structures, Python creates candidate units cheaply;
    # LLM stays the semantic core later in BookPlan/Planner/Review.
    # Use LLM atomizer only when the structural profile is uncertain or layout-heavy.
    if getattr(args, "preset", "") in {"production", "fast"}:
        return False
    if getattr(args, "preset", "") == "sft_dpo_quality":
        # strict audit mode can still force LLM atomizer when user asks --atomizer auto.
        return True

    threshold = float(getattr(args, "auto_llm_atomizer_confidence", 0.82) or 0.82)
    semantic_shape = semantic_book_shape(profile, book_plan)
    try:
        plan_confidence = float((book_plan or {}).get("confidence", profile.confidence))
    except Exception:
        plan_confidence = profile.confidence
    if semantic_shape in {"mixed", "table_heavy"}:
        return True
    # Dialogue and numbered analytical prose are easy to damage with a naive
    # paragraph atomizer when BookPlan itself is uncertain.
    if semantic_shape in {"dialogue", "numbered_analytical_prose"} and plan_confidence < 0.88:
        return True
    stable_shapes = {"item_book", "qa_book", "section_prose", "dense_prose", "poetry_or_aphorism"}
    if profile.confidence >= threshold and profile.detected_shape in stable_shapes:
        return False
    if profile.detected_shape in {"table_heavy"}:
        return True
    return profile.confidence < threshold

# -----------------------------------------------------------------------------
# Chunk planning
# -----------------------------------------------------------------------------


def unit_item_numbers(u: AtomicUnit) -> List[int]:
    return extract_item_numbers_from_text(u.text)


def group_text(units: Sequence[AtomicUnit]) -> str:
    parts: List[str] = []
    for u in units:
        if u.unit_type == "section_title":
            parts.append(clean_text(u.text))
        else:
            parts.append(clean_text(u.text))
    return clean_text("\n\n".join(p for p in parts if p))


def semantic_boundary_penalty_text(left_text: str, right_text: str) -> int:
    """Heuristic penalty for cutting a semantic dependency chain.

    This complements punctuation-based boundary checks.  It is intentionally
    conservative: a high score means the right side looks dependent on the left
    (reason, conclusion, objection-answer, enumeration continuation, etc.).
    """
    lt = clean_text(left_text or "")
    rt = clean_text(right_text or "")
    if not lt or not rt:
        return 0

    score = 0
    right_first = clean_text(rt.splitlines()[0]) if rt.splitlines() else rt
    left_lines = [clean_text(x) for x in lt.splitlines() if clean_text(x)]
    left_tail = "\n".join(left_lines[-6:]) if left_lines else lt[-900:]

    # Continuations, reasons, conclusions, examples, and qualifications normally
    # depend on the immediately preceding proposition.
    if DEPENDENT_CONTINUATION_RE.match(right_first):
        score += 4
    if CONCLUSION_START_RE.match(right_first) or REASON_START_RE.match(right_first):
        score += 3

    # Classical objection -> answer structures must remain visible together.
    if ANSWER_START_RE.match(right_first) and (any(OBJECTION_START_RE.match(x) for x in left_lines[-6:]) or "؟" in left_tail):
        score += 7
    if OBJECTION_START_RE.match(right_first):
        # An objection begins a dependent pair; cutting immediately before it is
        # less severe than separating it from its answer, but still suspicious.
        score += 2

    # Enumeration headers and sibling items are one semantic structure unless a
    # genuine new section starts.
    if ENUM_ITEM_RE.match(right_first):
        if ENUM_HEADER_RE.search(left_tail):
            score += 7
        elif any(ENUM_ITEM_RE.match(x) for x in left_lines[-4:]):
            score += 4

    # A colon often opens a definition, list, explanation, or proof step.
    if lt.rstrip().endswith((":", "：")):
        score += 4

    return score


def semantic_boundary_penalty(left: Sequence[AtomicUnit], right: Sequence[AtomicUnit]) -> int:
    if not left or not right:
        return 0
    return semantic_boundary_penalty_text(group_text(left), group_text(right))


def group_page_span(units: Sequence[AtomicUnit]) -> int:
    if not units:
        return 0
    files = [u.file_start for u in units] + [u.file_end for u in units]
    return max(files) - min(files) + 1


def same_major_section(a: List[str], b: List[str]) -> bool:
    if not a or not b:
        return True
    return a[0] == b[0]


def should_close_group(cur: List[AtomicUnit], nxt: AtomicUnit, policy: ChunkPolicy, profile: BookProfile) -> bool:
    if not cur:
        return False
    if nxt.unit_type == "ocr_noise" or cur[-1].unit_type == "ocr_noise":
        return True
    if nxt.unit_type == "section_title" and len(group_text(cur)) >= max(200, policy.min_chars // 2):
        return True
    if not policy.allow_cross_section and not same_major_section(cur[-1].section_path, nxt.section_path):
        if len(group_text(cur)) >= max(300, policy.min_chars // 2):
            return True
    cand = cur + [nxt]
    if len(group_text(cand)) > policy.max_chars:
        return True
    if group_page_span(cand) > policy.max_pages:
        return True
    if len(cur) >= policy.max_units:
        return True
    if profile.detected_shape == "item_book":
        item_count = sum(len(unit_item_numbers(u)) for u in cur)
        if item_count >= policy.target_units and len(group_text(cur)) >= policy.min_chars:
            return True
    if len(group_text(cur)) >= policy.target_chars:
        return True
    return False


def rule_pack_units(units: List[AtomicUnit], policy: ChunkPolicy, profile: BookProfile) -> List[List[AtomicUnit]]:
    content = [u for u in units if u.text.strip() and alpha_len(u.text) >= 3]
    groups: List[List[AtomicUnit]] = []
    cur: List[AtomicUnit] = []
    for u in content:
        if should_close_group(cur, u, policy, profile):
            groups.append(cur)
            cur = [u]
        else:
            cur.append(u)
    if cur:
        groups.append(cur)

    # Merge weak short groups when safe.
    merged: List[List[AtomicUnit]] = []
    for g in groups:
        if not merged:
            merged.append(g)
            continue
        prev = merged[-1]
        cand = prev + g
        safe = policy.allow_cross_section or same_major_section(prev[-1].section_path, g[0].section_path)
        noise_safe = not (
            prev[-1].unit_type == "ocr_noise"
            or g[0].unit_type == "ocr_noise"
        )
        if noise_safe and safe and len(group_text(prev)) < policy.min_chars and len(group_text(cand)) <= policy.max_chars and group_page_span(cand) <= policy.max_pages:
            merged[-1] = cand
        else:
            merged.append(g)
    return merged




def group_signature(g: List[AtomicUnit]) -> Tuple[str, str]:
    """Coarse signature used only for safe deterministic merging of too-short groups."""
    sec = section_for_group(g) if g else []
    major = sec[-1] if sec else ""
    kind = g[0].unit_type if g else ""
    return major, kind


def merge_short_groups_for_generator(groups: List[List[AtomicUnit]], policy: ChunkPolicy, args: argparse.Namespace) -> List[List[AtomicUnit]]:
    """Attach only obvious structural fragments; never undo semantic planning.

    Older versions merged every short planner group with an adjacent group that
    shared a coarse section label. Across heterogeneous books this mixed poems,
    quotations, story scenes, or independent entries. Length is a soft target;
    only headings/captions and tiny continuation fragments are merged here.
    """
    if not groups:
        return groups

    def is_structural_fragment(g: List[AtomicUnit]) -> bool:
        if not g:
            return False
        kinds = {u.unit_type for u in g}
        if kinds.issubset({"section_title", "caption"}):
            return True
        if len(g) == 1 and len(group_text(g)) <= 220:
            return g[0].unit_type in {"paragraph", "argument_block", "dialogue_turn"}
        return False

    def safe_join(left: List[AtomicUnit], right: List[AtomicUnit]) -> bool:
        if not left or not right:
            return False
        if left[-1].unit_type == "ocr_noise" or right[0].unit_type == "ocr_noise":
            return False
        cand = left + right
        safe_section = policy.allow_cross_section or same_major_section(left[-1].section_path, right[0].section_path)
        return bool(
            safe_section
            and len(group_text(cand)) <= min(policy.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS)
            and group_page_span(cand) <= policy.max_pages
            and len(cand) <= policy.max_units
        )

    out: List[List[AtomicUnit]] = []
    i = 0
    while i < len(groups):
        cur = list(groups[i])
        if is_structural_fragment(cur) and i + 1 < len(groups) and safe_join(cur, groups[i + 1]):
            out.append(cur + list(groups[i + 1]))
            i += 2
            continue
        if is_structural_fragment(cur) and out and safe_join(out[-1], cur):
            out[-1] = out[-1] + cur
        else:
            out.append(cur)
        i += 1
    return out


def optimize_short_groups_for_downstream(
    groups: List[List[AtomicUnit]],
    policy: ChunkPolicy,
    contract: GeneratorContract,
    args: argparse.Namespace,
) -> Tuple[List[List[AtomicUnit]], Dict[str, Any]]:
    """Reduce avoidable tiny chunks without forcing unrelated subjects together.

    The downstream generator can reject low-value material itself, so a short
    group is never dropped here. We merge it only with an adjacent group when
    the result stays within the downstream size/page/unit contract and the
    boundary has positive structural evidence (same section, continuation,
    heading-to-body, or an extremely small fragment). Remaining short groups
    are accepted later with a lower routing weight.
    """
    desired_min = max(200, int(getattr(args, "trust_merge_short_chars", 1000) or 1000))
    max_chars = min(
        int(policy.max_chars),
        int(contract.hard_max_chars),
        max(desired_min, int(getattr(args, "trust_merge_max_chars", 3000) or 3000)),
    )
    max_pages = min(
        max(1, int(getattr(args, "final_max_pages", 7) or 7)),
        max(1, int(getattr(args, "trust_merge_max_pages", 4) or 4)),
    )
    max_units = min(
        max(2, int(getattr(args, "final_max_units", 28) or 28)),
        max(2, int(getattr(args, "trust_merge_max_units", 20) or 20)),
    )
    enabled = bool(getattr(args, "trust_merge_short", True))

    def text_of(g: Sequence[AtomicUnit]) -> str:
        return group_text(g)

    def section_path(g: Sequence[AtomicUnit], from_right: bool) -> List[str]:
        if not g:
            return []
        ordered = reversed(g) if from_right else iter(g)
        for u in ordered:
            if u.section_path:
                return [clean_text(x) for x in u.section_path if clean_text(x)]
        return []

    def related_sections(left: Sequence[AtomicUnit], right: Sequence[AtomicUnit]) -> bool:
        lp = section_path(left, True)
        rp = section_path(right, False)
        if not lp or not rp:
            return True
        common = 0
        for a, b in zip(lp, rp):
            if a != b:
                break
            common += 1
        return common > 0 or lp[-1] == rp[-1]

    def feasible(left: Sequence[AtomicUnit], right: Sequence[AtomicUnit]) -> bool:
        if not left or not right:
            return False
        if left[-1].unit_type == "ocr_noise" or right[0].unit_type == "ocr_noise":
            return False
        if SCENE_BREAK_RE.match(clean_text(right[0].text)):
            return False
        cand = list(left) + list(right)
        return bool(
            cand
            and len(cand) <= max_units
            and len(text_of(cand)) <= max_chars
            and group_page_span(cand) <= max_pages
        )

    def join_score(left: Sequence[AtomicUnit], right: Sequence[AtomicUnit]) -> int:
        if not feasible(left, right):
            return -1000
        left_text = text_of(left)
        right_text = text_of(right)
        if not left_text or not right_text:
            return -1000

        related = related_sections(left, right)
        score = 5 if related else -6
        if not section_path(left, True) or not section_path(right, False):
            score += 1
        if left[-1].unit_type in {"section_title", "caption"}:
            score += 5
        if CONTINUATION_START_RE.match(right_text):
            score += 3
        if left_text[-1:] not in TERMINAL_CHARS:
            score += 2
        if min(len(left_text), len(right_text)) <= 300:
            score += 2
        if left[-1].unit_type == right[0].unit_type:
            score += 1
        # A new, unrelated titled section is a real semantic boundary.
        if right[0].unit_type == "section_title" and not related:
            score -= 5
        return score

    before_lengths = [len(text_of(g)) for g in groups]
    stats: Dict[str, Any] = {
        "enabled": enabled,
        "desired_min_chars": desired_min,
        "merge_max_chars": max_chars,
        "merge_max_pages": max_pages,
        "merge_max_units": max_units,
        "groups_before": len(groups),
        "groups_after": len(groups),
        "merges": 0,
        "below_desired_before": sum(n < desired_min for n in before_lengths),
        "below_desired_after": sum(n < desired_min for n in before_lengths),
    }
    if not enabled or not groups:
        return groups, stats

    current = [list(g) for g in groups]
    merges = 0
    # A few deterministic passes are enough for chains of adjacent fragments.
    for _ in range(4):
        out: List[List[AtomicUnit]] = []
        changed = False
        i = 0
        while i < len(current):
            cur = list(current[i])
            if len(text_of(cur)) >= desired_min:
                out.append(cur)
                i += 1
                continue

            left_score = join_score(out[-1], cur) if out else -1000
            right_score = join_score(cur, current[i + 1]) if i + 1 < len(current) else -1000
            best = max(left_score, right_score)
            if best < 2:
                out.append(cur)
                i += 1
            elif left_score >= right_score:
                out[-1] = out[-1] + cur
                i += 1
                merges += 1
                changed = True
            else:
                out.append(cur + list(current[i + 1]))
                i += 2
                merges += 1
                changed = True
        current = out
        if not changed:
            break

    after_lengths = [len(text_of(g)) for g in current]
    stats.update({
        "groups_after": len(current),
        "merges": merges,
        "below_desired_after": sum(n < desired_min for n in after_lengths),
        "min_chars_after": min(after_lengths) if after_lengths else 0,
        "median_chars_after": round(float(median(after_lengths)), 1) if after_lengths else 0.0,
        "max_chars_after": max(after_lengths) if after_lengths else 0,
    })
    return current, stats


def compact_unit_for_llm(u: AtomicUnit, text_limit: int) -> Dict[str, Any]:
    txt = clean_text(u.text)
    if len(txt) > text_limit:
        head = max(80, int(text_limit * 0.62))
        tail = max(40, text_limit - head - 5)
        txt = txt[:head].rstrip() + "\n…\n" + txt[-tail:].lstrip()
    return {
        "uid": u.uid,
        "type": u.unit_type,
        "pages": [u.page_start, u.page_end],
        "files": [u.file_start, u.file_end],
        "sec": u.section_path[-3:],
        "chars": len(u.text),
        "items": unit_item_numbers(u),
        "text": txt,
    }


def make_unit_batches(units: List[AtomicUnit], max_tokens: int, max_units: int, text_limit: int) -> List[List[AtomicUnit]]:
    batches: List[List[AtomicUnit]] = []
    cur: List[AtomicUnit] = []
    cur_tok = 0
    for u in units:
        tok = estimate_tokens(compact_unit_for_llm(u, text_limit)) + 8
        if cur and (cur_tok + tok > max_tokens or len(cur) >= max_units):
            batches.append(cur)
            cur = []
            cur_tok = 0
        cur.append(u)
        cur_tok += tok
    if cur:
        batches.append(cur)
    return batches


def validate_planner_plan(plan: Dict[str, Any], candidate: List[AtomicUnit]) -> Tuple[List[Tuple[int, int, Optional[str], Optional[str], Optional[str]]], Optional[str]]:
    """Validate planner output.

    v8 planner prefers implicit continuous boundaries:
        {"end_uids": ["u000004", "u000009"], "carry_start": null}

    This avoids the common LLM error where explicit start/end ranges leave gaps.
    For backward compatibility, it also accepts older chunk range schemas, but all
    returned ranges are normalized into complete, ordered ranges whenever possible.
    """
    id_to_pos = {u.uid: i for i, u in enumerate(candidate)}

    def _as_uid_list(x: Any) -> List[str]:
        if not isinstance(x, list):
            return []
        out: List[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(v.strip())
            elif isinstance(v, dict):
                # tolerate {"end":"u..."} or {"end_uid":"u..."}
                for k in ("end_uid", "end", "uid", "last_uid", "boundary"):
                    if isinstance(v.get(k), str):
                        out.append(v[k].strip())
                        break
        return out

    carry = plan.get("carry_start")
    if isinstance(carry, str):
        carry = carry.strip() or None
    if carry is not None and carry not in id_to_pos:
        carry = None

    commit_stop = id_to_pos[carry] - 1 if carry else len(candidate) - 1
    if commit_stop < 0:
        return [], carry

    # Preferred v8 schema and common aliases.
    end_uids: List[str] = []
    for key in ("end_uids", "chunk_end_uids", "chunk_ends", "boundaries", "ends"):
        end_uids = _as_uid_list(plan.get(key))
        if end_uids:
            break

    ranges: List[Tuple[int, int, Optional[str], Optional[str], Optional[str]]] = []

    if end_uids:
        cursor = 0
        last_end = -1
        for uid in end_uids:
            if uid not in id_to_pos:
                continue
            pos = id_to_pos[uid]
            if pos < cursor or pos > commit_stop:
                continue
            # Avoid zero-length or backwards ranges; starts are implicit.
            ranges.append((cursor, pos, None, None, "implicit_end_uid"))
            cursor = pos + 1
            last_end = pos
            if cursor > commit_stop:
                break
        return ranges, carry

    # Backward-compatible explicit chunk ranges. Accept several key spellings.
    raw_chunks = plan.get("chunks") or plan.get("ranges") or plan.get("chunk_ranges") or []
    if isinstance(raw_chunks, dict):
        raw_chunks = raw_chunks.get("chunks") or raw_chunks.get("ranges") or []
    if not isinstance(raw_chunks, list):
        raw_chunks = []

    explicit: List[Tuple[int, int, Optional[str], Optional[str], Optional[str]]] = []
    last_end = -1
    for ch in raw_chunks:
        if not isinstance(ch, dict):
            continue
        s = None
        e = None
        for sk in ("start", "start_uid", "first", "first_uid", "from"):
            if isinstance(ch.get(sk), str):
                s = ch.get(sk).strip()
                break
        for ek in ("end", "end_uid", "last", "last_uid", "to"):
            if isinstance(ch.get(ek), str):
                e = ch.get(ek).strip()
                break
        if e not in id_to_pos:
            continue
        b = id_to_pos[e]
        if b > commit_stop or b <= last_end:
            continue
        if s in id_to_pos:
            a = id_to_pos[s]
        else:
            # If start is missing/invalid, make it continuous from previous range.
            a = last_end + 1
        if a < 0 or a > b or a <= last_end:
            continue
        title = ch.get("title") if isinstance(ch.get("title"), str) else None
        quality = ch.get("quality") if isinstance(ch.get("quality"), str) else None
        reason = ch.get("reason") if isinstance(ch.get("reason"), str) else None
        explicit.append((a, b, title, quality, reason))
        last_end = b
    return explicit, carry

def split_oversize_group(g: List[AtomicUnit], policy: ChunkPolicy, profile: BookProfile) -> List[List[AtomicUnit]]:
    """Split only when hard limits require it, preferring low-dependency cuts.

    In full-LLM mode the planner may intentionally keep a long argument together.
    If a hard downstream limit forces a split, choose the least harmful boundary
    between AtomicUnits instead of falling back to target-length rule packing.
    """
    if len(group_text(g)) <= policy.max_chars and group_page_span(g) <= policy.max_pages and len(g) <= policy.max_units:
        return [g]
    if len(g) <= 1:
        return [g]

    out: List[List[AtomicUnit]] = []
    remaining = list(g)
    while remaining:
        if (
            len(group_text(remaining)) <= policy.max_chars
            and group_page_span(remaining) <= policy.max_pages
            and len(remaining) <= policy.max_units
        ):
            out.append(remaining)
            break

        feasible_cuts: List[int] = []
        for cut in range(1, len(remaining)):
            left = remaining[:cut]
            if (
                len(group_text(left)) <= policy.max_chars
                and group_page_span(left) <= policy.max_pages
                and len(left) <= policy.max_units
            ):
                feasible_cuts.append(cut)
            else:
                # Once size/unit count has exceeded the cap it will not recover.
                if len(group_text(left)) > policy.max_chars or len(left) > policy.max_units:
                    break
        if not feasible_cuts:
            out.append([remaining[0]])
            remaining = remaining[1:]
            continue

        def cut_cost(cut: int) -> float:
            left, right = remaining[:cut], remaining[cut:]
            dep = semantic_boundary_penalty(left, right)
            size = abs(len(group_text(left)) - policy.target_chars) / max(250.0, policy.target_chars)
            # A real section heading is a good place to cut; dependency is a very
            # bad place to cut and dominates the size preference.
            section_bonus = -3.0 if right and right[0].unit_type == "section_title" else 0.0
            short_penalty = 2.0 if len(group_text(left)) < max(250, policy.min_chars // 2) else 0.0
            return dep * 10.0 + size + short_penalty + section_bonus

        best_cut = min(feasible_cuts, key=cut_cost)
        out.append(remaining[:best_cut])
        remaining = remaining[best_cut:]
    return out


def normalize_group_coverage(
    groups: List[List[AtomicUnit]],
    units: List[AtomicUnit],
    policy: ChunkPolicy,
    profile: BookProfile,
) -> Tuple[List[List[AtomicUnit]], Dict[str, Any]]:
    """Rebuild planner groups as an exact ordered partition of atomic units.

    LLM carry/fallback plans may duplicate a range, omit a unit, or return
    overlapping ranges. Semantic cut positions are retained where possible, but
    membership is reconstructed from the canonical ``units`` sequence so the
    result is lossless and deterministic.
    """
    canonical = [u for u in units if u.text.strip()]
    if not canonical:
        return [], {
            "input_units": len(units),
            "canonical_units": 0,
            "raw_group_count": len(groups),
            "raw_unit_refs": 0,
            "raw_duplicate_refs": 0,
            "raw_missing_units": 0,
            "normalized_group_count": 0,
        }

    pos = {u.uid: i for i, u in enumerate(canonical)}
    raw_refs: List[str] = []
    cuts: List[int] = []
    for group in groups:
        positions = sorted({pos[u.uid] for u in group if u.uid in pos})
        raw_refs.extend(u.uid for u in group if u.uid in pos)
        if positions:
            cuts.append(positions[-1])

    raw_counter = Counter(raw_refs)
    raw_duplicate_refs = sum(max(0, count - 1) for count in raw_counter.values())
    raw_missing = [u.uid for u in canonical if u.uid not in raw_counter]

    cuts = sorted(set(cut for cut in cuts if 0 <= cut < len(canonical)))
    if not cuts or cuts[-1] != len(canonical) - 1:
        cuts.append(len(canonical) - 1)

    def lossless_split(segment: List[AtomicUnit]) -> List[List[AtomicUnit]]:
        # Preserve planner semantics whenever possible.  If a hard limit requires
        # a split, use the same dependency-aware cut selection as final oversize
        # handling instead of a first-exceed mechanical cut.
        return split_oversize_group(segment, policy, profile)

    rebuilt: List[List[AtomicUnit]] = []
    cursor = 0
    for cut in cuts:
        if cut < cursor:
            continue
        segment = canonical[cursor:cut + 1]
        if segment:
            rebuilt.extend(lossless_split(segment))
        cursor = cut + 1
    if cursor < len(canonical):
        rebuilt.extend(lossless_split(canonical[cursor:]))

    # ``rule_pack_units`` intentionally filters tiny non-alpha units. Restore
    # any such omissions by rebuilding one last time from the resulting cuts.
    final_cuts: List[int] = []
    for group in rebuilt:
        positions = [pos[u.uid] for u in group if u.uid in pos]
        if positions:
            final_cuts.append(max(positions))
    final_cuts = sorted(set(final_cuts))
    if not final_cuts or final_cuts[-1] != len(canonical) - 1:
        final_cuts.append(len(canonical) - 1)

    normalized: List[List[AtomicUnit]] = []
    cursor = 0
    for cut in final_cuts:
        if cut >= cursor:
            normalized.append(canonical[cursor:cut + 1])
            cursor = cut + 1
    if cursor < len(canonical):
        normalized.append(canonical[cursor:])

    flattened = [u.uid for group in normalized for u in group]
    expected = [u.uid for u in canonical]
    if flattened != expected:
        raise RuntimeError(
            "GROUP_COVERAGE_NORMALIZATION_FAILED: normalized groups are not an "
            "exact ordered partition of atomic units"
        )

    stats = {
        "input_units": len(units),
        "canonical_units": len(canonical),
        "raw_group_count": len(groups),
        "raw_unit_refs": len(raw_refs),
        "raw_duplicate_refs": raw_duplicate_refs,
        "raw_missing_units": len(raw_missing),
        "raw_missing_sample": raw_missing[:50],
        "normalized_group_count": len(normalized),
        "normalized_unit_refs": len(flattened),
        "normalized_duplicate_refs": 0,
        "normalized_missing_units": 0,
    }
    return normalized, stats


def _group_has_hard_boundary_type(group: Sequence[AtomicUnit]) -> bool:
    return any(u.unit_type == "ocr_noise" for u in group)


def boundary_penalty(left: Sequence[AtomicUnit], right: Sequence[AtomicUnit]) -> int:
    """Higher means a more suspicious syntactic or semantic cut."""
    if not left or not right:
        return 0
    if _group_has_hard_boundary_type(left) or _group_has_hard_boundary_type(right):
        return 0
    if right[0].unit_type == "section_title" or SCENE_BREAK_RE.match(clean_text(right[0].text)):
        return 0
    lt = clean_text(group_text(left))
    rt = clean_text(group_text(right))
    if not lt or not rt:
        return 0
    score = semantic_boundary_penalty_text(lt, rt)
    if not PROSE_TERMINAL_RE.search(lt):
        score += 3
    if CONTINUATION_START_RE.match(rt):
        score += 3
    if looks_like_speaker_label(lt.splitlines()[-1]):
        score += 4
    if rt[:1] in "،؛:»)]}":
        score += 2
    return score


def repair_group_boundaries(
    groups: List[List[AtomicUnit]],
    policy: ChunkPolicy,
    contract: GeneratorContract,
    args: argparse.Namespace,
) -> Tuple[List[List[AtomicUnit]], Dict[str, Any]]:
    """Repair bad prose cuts without changing text, order, or unit coverage."""
    enabled = bool(getattr(args, "boundary_repair", True)) and contract.semantic_shape not in {
        "independent_items", "qa_book", "poetry", "reference", "table_heavy",
    }
    before_bad = sum(boundary_penalty(a, b) >= 3 for a, b in zip(groups, groups[1:]))
    stats = {
        "enabled": enabled,
        "skipped_shape": None if enabled else contract.semantic_shape,
        "groups_before": len(groups),
        "groups_after": len(groups),
        "bad_boundaries_before": before_bad,
        "bad_boundaries_after": before_bad,
        "merges": 0,
        "rebalances": 0,
    }
    if not enabled or len(groups) < 2:
        return groups, stats

    max_chars = min(
        int(policy.max_chars),
        int(contract.hard_max_chars),
        int(getattr(args, "boundary_repair_max_chars", 3300) or 3300),
    )
    max_pages = max(1, min(policy.max_pages, int(getattr(args, "boundary_repair_max_pages", 5) or 5)))
    current = [list(g) for g in groups]
    merges = 0
    rebalances = 0

    def feasible(g: Sequence[AtomicUnit]) -> bool:
        return bool(
            g
            and len(g) <= policy.max_units
            and len(group_text(g)) <= max_chars
            and group_page_span(g) <= max_pages
        )

    for _ in range(3):
        changed = False
        out: List[List[AtomicUnit]] = []
        i = 0
        while i < len(current):
            if i + 1 >= len(current):
                out.append(current[i])
                break
            left, right = current[i], current[i + 1]
            old_penalty = boundary_penalty(left, right)
            if old_penalty < 3:
                out.append(left)
                i += 1
                continue

            combined = left + right
            if feasible(combined):
                out.append(combined)
                merges += 1
                changed = True
                i += 2
                continue

            old_cut = len(left)
            best_cut = old_cut
            best_score = -10_000.0
            old_score = -10_000.0
            for cut in range(1, len(combined)):
                a, b = combined[:cut], combined[cut:]
                if not feasible(a) or not feasible(b):
                    continue
                if b[0].unit_type == "ocr_noise" or a[-1].unit_type == "ocr_noise":
                    continue
                a_text, b_text = group_text(a), group_text(b)
                natural = 4 if PROSE_TERMINAL_RE.search(clean_text(a_text)) else 0
                continuation = 0 if CONTINUATION_START_RE.match(clean_text(b_text)) else 3
                label = -4 if looks_like_speaker_label(clean_text(a_text).splitlines()[-1]) else 0
                dependency = -2.0 * semantic_boundary_penalty_text(a_text, b_text)
                size = -abs(len(a_text) - contract.target_chars) / max(250.0, contract.target_chars)
                score = natural + continuation + label + dependency + size
                if cut == old_cut:
                    old_score = score
                if score > best_score:
                    best_score, best_cut = score, cut
            if best_cut != old_cut and best_score >= old_score + 1.5:
                out.extend([combined[:best_cut], combined[best_cut:]])
                rebalances += 1
                changed = True
                i += 2
            else:
                out.append(left)
                i += 1
        current = out
        if not changed:
            break

    stats.update({
        "groups_after": len(current),
        "bad_boundaries_after": sum(
            boundary_penalty(a, b) >= 3 for a, b in zip(current, current[1:])
        ),
        "merges": merges,
        "rebalances": rebalances,
    })
    return current, stats


def assert_exact_group_coverage(groups: List[List[AtomicUnit]], units: List[AtomicUnit]) -> None:
    expected = [u.uid for u in units if u.text.strip()]
    actual = [u.uid for group in groups for u in group]
    if actual != expected:
        expected_set = set(expected)
        counts = Counter(actual)
        duplicate_refs = sum(max(0, count - 1) for count in counts.values())
        missing = [uid for uid in expected if uid not in counts]
        foreign = [uid for uid in actual if uid not in expected_set]
        raise RuntimeError(
            "GROUP_COVERAGE_INVALID "
            f"expected={len(expected)} actual={len(actual)} "
            f"duplicates={duplicate_refs} missing={len(missing)} foreign={len(foreign)}"
        )


def build_groups_from_ranges_with_gap_fill(
    ranges: List[Tuple[int, int, Optional[str], Optional[str], Optional[str]]],
    candidate: List[AtomicUnit],
    commit_stop: int,
    policy: ChunkPolicy,
    profile: BookProfile,
) -> List[List[AtomicUnit]]:
    groups: List[List[AtomicUnit]] = []
    cursor = 0
    for a, b, _title, _quality, _reason in ranges:
        if a > commit_stop:
            break
        if a > cursor:
            groups.extend(rule_pack_units(candidate[cursor:min(a, commit_stop + 1)], policy, profile))
        b = min(b, commit_stop)
        for sg in split_oversize_group(candidate[a:b + 1], policy, profile):
            groups.append(sg)
        cursor = b + 1
    if cursor <= commit_stop:
        groups.extend(rule_pack_units(candidate[cursor:commit_stop + 1], policy, profile))
    return groups


def llm_plan_chunks(
    units: List[AtomicUnit],
    profile: BookProfile,
    policy: ChunkPolicy,
    cfg: LLMConfig,
    args: argparse.Namespace,
    debug_path: Optional[Path],
    book_plan: Optional[Dict[str, Any]] = None,
) -> List[List[AtomicUnit]]:
    """LLM planner with strict adaptive shrink/split and no semantic rule fallback.

    If a planner call fails because vLLM rejects the payload (HTTP 400), times out,
    or one endpoint refuses the request, we first rotate through the endpoint pool
    inside call_llm_json. If the whole pool fails, we reduce text/output budget and,
    if needed, split the unit batch and plan the smaller pieces with the LLM.

    With --preset sft_dpo_quality or --strict-llm, this function never turns a failed LLM planning call into
    final rule-made chunks. Without strict mode, rule packing is only a last-resort batch-level fallback.
    """
    batches = make_unit_batches(units, args.planner_max_input_tokens, args.planner_max_units, args.planner_text_limit)
    groups: List[List[AtomicUnit]] = []
    carry: List[AtomicUnit] = []
    started = time.time()
    strict = bool(getattr(args, "strict_llm", False)) or getattr(args, "preset", "") == "sft_dpo_quality"

    def _ranges_are_complete(ranges: List[Tuple[int, int, Optional[str], Optional[str], Optional[str]]], commit_stop: int) -> bool:
        cursor = 0
        for a, b, *_ in ranges:
            if a != cursor:
                return False
            cursor = b + 1
            if cursor > commit_stop:
                break
        return cursor == commit_stop + 1

    def _groups_from_llm_ranges(ranges, candidate, commit_stop, job_label: str):
        if strict and not _ranges_are_complete(ranges, commit_stop):
            raise ValueError(f"planner returned incomplete/non-contiguous ranges for {candidate[0].uid}-{candidate[commit_stop].uid}")
        out: List[List[AtomicUnit]] = []
        cursor = 0
        for a, b, _title, _quality, _reason in ranges:
            if a > commit_stop:
                break
            if a > cursor:
                if strict:
                    raise ValueError("planner left a gap in strict mode")
                out.extend(rule_pack_units(candidate[cursor:min(a, commit_stop + 1)], policy, profile))
            b = min(b, commit_stop)
            g = candidate[a:b + 1]

            # The LLM owns the semantic grouping decision, but the downstream
            # hard contract must still be respected. If an LLM group is too
            # large, split it only BETWEEN AtomicUnits using the existing
            # dependency-aware hard-limit splitter. This is not a semantic
            # rule fallback and never rewrites or splits AtomicUnit content.
            oversize = (
                len(group_text(g)) > policy.max_chars
                or group_page_span(g) > policy.max_pages
                or len(g) > policy.max_units
            )

            if oversize:
                split_groups = split_oversize_group(g, policy, profile)

                # A singleton AtomicUnit can theoretically still exceed the
                # current book policy. That case cannot be repaired here
                # without splitting the AtomicUnit itself, so strict mode
                # should still fail loudly.
                unresolved = [
                    sg for sg in split_groups
                    if (
                        len(group_text(sg)) > policy.max_chars
                        or group_page_span(sg) > policy.max_pages
                        or len(sg) > policy.max_units
                    )
                ]

                if unresolved:
                    bad = unresolved[0]
                    raise ValueError(
                        "planner hard-split still oversize "
                        f"units={len(bad)} "
                        f"chars={len(group_text(bad))} "
                        f"pages={group_page_span(bad)}"
                    )

                log(
                    f"[PLANNER_HARD_SPLIT] {job_label} "
                    f"units={len(g)} chars={len(group_text(g))} "
                    f"pages={group_page_span(g)} "
                    f"parts={len(split_groups)}"
                )
                out.extend(split_groups)
            else:
                out.append(g)

            cursor = b + 1
        if cursor <= commit_stop:
            if strict:
                raise ValueError("planner did not cover tail in strict mode")
            out.extend(rule_pack_units(candidate[cursor:commit_stop + 1], policy, profile))
        return out

    def _plan_candidate_adaptive(candidate: List[AtomicUnit], is_final_batch: bool, job_label: str, depth: int = 0) -> Tuple[List[List[AtomicUnit]], List[AtomicUnit], Dict[str, Any]]:
        if not candidate:
            return [], [], {"mode": "empty"}

        # Adaptive attempts: keep input context useful first, then reduce enough
        # to satisfy stricter vLLM context/request limits. This fixes the 400s at
        # the source without replacing the LLM with rules.
        variants: List[Tuple[int, int]] = []
        base_text = int(args.planner_text_limit)
        base_out = int(cfg.max_output_tokens)
        for tl, ot in [
            (base_text, base_out),
            (min(base_text, 220), min(base_out, 768)),
            (min(base_text, 160), min(base_out, 640)),
            (min(base_text, 100), min(base_out, 512)),
        ]:
            item = (max(60, tl), max(256, ot))
            if item not in variants:
                variants.append(item)

        last_error: Optional[Exception] = None
        for attempt_i, (text_limit, out_tokens) in enumerate(variants, 1):
            payload = {
                "language": "fa",
                "profile": asdict(profile),
                "book_plan": book_plan or {},
                "policy": asdict(policy),
                "generator_contract": {
                    "assessor_visible_chars": SFT_ASSESSOR_VISIBLE_CHARS,
                    "generator_visible_chars": SFT_GENERATOR_VISIBLE_CHARS,
                    "hard_max_chars": min(policy.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS),
                },
                "is_final_batch": is_final_batch,
                "strict_no_rule_fallback": strict,
                "units": [compact_unit_for_llm(u, text_limit) for u in candidate],
            }
            prompt_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            log(f"[PLANNER_LLM_CALL] {job_label} attempt={attempt_i}/{len(variants)} depth={depth} units={candidate[0].uid}-{candidate[-1].uid} n={len(candidate)} text_limit={text_limit} out_tokens={out_tokens} chars={len(prompt_json)} est_tok={estimate_tokens(prompt_json)}")
            t0 = time.time()
            try:
                plan = call_llm_json(PLANNER_PROMPT, payload, cfg, max_output_tokens=out_tokens)
                if is_final_batch and isinstance(plan, dict) and plan.get("carry_start"):
                    plan = dict(plan)
                    plan["model_carry_start_ignored"] = plan.get("carry_start")
                    plan["carry_start"] = None
                ranges, carry_start = validate_planner_plan(plan, candidate)
                if not ranges:
                    raise ValueError("planner returned no valid chunk ranges")
                id_to_pos = {u.uid: i for i, u in enumerate(candidate)}
                if carry_start:
                    if is_final_batch:
                        # Final means final. In non-strict mode any uncovered
                        # tail is deterministically filled below; in strict
                        # mode completeness validation triggers a retry.
                        commit_stop = len(candidate) - 1
                        new_carry = []
                    else:
                        cpos = id_to_pos[carry_start]
                        commit_stop = max(-1, cpos - 1)
                        new_carry = candidate[cpos:]
                else:
                    commit_stop = len(candidate) - 1
                    new_carry = []
                new_groups: List[List[AtomicUnit]] = []
                if commit_stop >= 0:
                    new_groups = _groups_from_llm_ranges(ranges, candidate, commit_stop, job_label)
                dt = time.time() - t0
                meta = {"mode": "llm", "attempt": attempt_i, "text_limit": text_limit, "out_tokens": out_tokens, "depth": depth, "plan": plan, "llm_sec": round(dt, 2)}
                log(f"[PLANNER_LLM_DONE] {job_label} depth={depth} llm={dt:.1f}s groups={len(new_groups)} carry={len(new_carry)}")
                return new_groups, new_carry, meta
            except (NameError, UnboundLocalError) as e:
                log(
                    f"[PLANNER_CODE_ERROR] {job_label} depth={depth} "
                    f"type={type(e).__name__} error={e}"
                )
                raise
            except Exception as e:
                last_error = e
                log(f"[PLANNER_LLM_RETRY] {job_label} attempt={attempt_i}/{len(variants)} depth={depth} failed={e}")
                continue

        # If every reduced request failed, split the candidate and plan both
        # halves with LLM. This is not rule fallback; it is request-size fallback.
        max_depth = int(getattr(args, "planner_adaptive_split_depth", 4))
        if len(candidate) > 1 and depth < max_depth:
            mid = max(1, len(candidate) // 2)
            left = candidate[:mid]
            right = candidate[mid:]
            log(f"[PLANNER_LLM_SPLIT] {job_label} depth={depth} n={len(candidate)} split={len(left)}+{len(right)} reason={last_error}")
            left_groups, left_carry, left_meta = _plan_candidate_adaptive(left, True, job_label + ".L", depth + 1)
            if left_carry:
                # Plan any carry as its own final piece with the LLM; do not rule-pack it.
                cg, cc, _ = _plan_candidate_adaptive(left_carry, True, job_label + ".Lcarry", depth + 1)
                left_groups.extend(cg)
                if cc:
                    raise RuntimeError(f"planner carry remained after adaptive split for {job_label}")
            right_groups, right_carry, right_meta = _plan_candidate_adaptive(right, True, job_label + ".R", depth + 1)
            if right_carry:
                cg, cc, _ = _plan_candidate_adaptive(right_carry, True, job_label + ".Rcarry", depth + 1)
                right_groups.extend(cg)
                if cc:
                    raise RuntimeError(f"planner carry remained after adaptive split for {job_label}")
            return left_groups + right_groups, [], {"mode": "llm_split", "depth": depth, "left": left_meta.get("mode"), "right": right_meta.get("mode")}

        raise RuntimeError(f"STRICT_PLANNER_LLM_FAILED_NO_FALLBACK {job_label}: {last_error}")

    llm_concurrency = max(1, int(getattr(args, "llm_concurrency", 1) or 1))
    if llm_concurrency > 1 and len(batches) > 1:
        # Parallel planner windows with overlapping context. Only chunk ENDs
        # whose end unit lies inside a window's non-overlapping core are
        # committed; final groups are reconstructed from the canonical global
        # unit sequence, guaranteeing no gaps/duplicates.
        uid_to_pos = {u.uid: i for i, u in enumerate(units)}
        overlap = max(0, int(getattr(args, "planner_overlap_units", 4) or 0))
        jobs = []
        for bi, batch in enumerate(batches):
            core_start = uid_to_pos[batch[0].uid]
            core_end = uid_to_pos[batch[-1].uid]
            cand_start = max(0, core_start - overlap)
            cand_end = min(len(units) - 1, core_end + overlap)
            candidate = units[cand_start:cand_end + 1]
            jobs.append((bi, core_start, core_end, candidate))

        log(f"[PLANNER_PARALLEL] windows={len(jobs)} concurrency={llm_concurrency} overlap_units={overlap}")
        cuts: set[int] = set()
        debug_rows: List[Dict[str, Any]] = []
        errors: List[Tuple[int, Exception]] = []

        def _run_plan_job(job):
            bi, core_start, core_end, candidate = job
            job_label = f"pwin={bi+1}/{len(jobs)}"
            local_groups, rem, meta = _plan_candidate_adaptive(candidate, True, job_label, 0)
            if rem:
                raise RuntimeError(f"parallel planner returned carry in final window: {job_label}")
            local_cuts = []
            for g in local_groups:
                if not g:
                    continue
                end_uid = g[-1].uid
                if end_uid not in uid_to_pos:
                    continue
                end_pos = uid_to_pos[end_uid]
                if core_start <= end_pos <= core_end:
                    local_cuts.append(end_pos)
            return bi, core_start, core_end, candidate, local_cuts, meta

        with ThreadPoolExecutor(max_workers=min(llm_concurrency, len(jobs))) as ex:
            futs = {ex.submit(_run_plan_job, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                bi = futs[fut]
                try:
                    bi2, core_start, core_end, candidate, local_cuts, meta = fut.result()
                    cuts.update(local_cuts)
                    debug_rows.append({
                        "batch": bi2 + 1,
                        "parallel": True,
                        "core_unit_range": [units[core_start].uid, units[core_end].uid],
                        "window_unit_range": [candidate[0].uid, candidate[-1].uid],
                        "meta": {k: v for k, v in meta.items() if k != "plan"},
                        "plan": meta.get("plan"),
                    })
                except Exception as e:
                    errors.append((bi, e))

        if errors:
            errors.sort(key=lambda x: x[0])
            raise RuntimeError("PARALLEL_PLANNER_FAILED " + "; ".join(f"batch={i+1}:{e}" for i, e in errors[:5]))

        cuts.add(len(units) - 1)
        groups = []
        cursor = 0
        for cut in sorted(cuts):
            if cut < cursor:
                continue
            g = units[cursor:cut + 1]
            if g:
                groups.append(g)
            cursor = cut + 1
        if cursor < len(units):
            groups.append(units[cursor:])

        # A parallel boundary set must still obey the downstream hard contract.
        # If a rare reconstructed span is oversize, re-plan ONLY that span with
        # the LLM (still no rule fallback) and replace it losslessly.
        repaired_groups: List[List[AtomicUnit]] = []
        for gi, g in enumerate(groups):
            oversize = (
                len(group_text(g)) > policy.max_chars
                or group_page_span(g) > policy.max_pages
                or len(g) > policy.max_units
            )
            if not oversize:
                repaired_groups.append(g)
                continue
            rg, rc, _ = _plan_candidate_adaptive(g, True, f"parallel_oversize={gi+1}", 0)
            if rc:
                raise RuntimeError("parallel oversize repair returned carry")
            repaired_groups.extend(rg)
        groups = repaired_groups
        assert_exact_group_coverage(groups, units)
        if debug_path:
            for row in sorted(debug_rows, key=lambda r: r["batch"]):
                append_jsonl(debug_path, row)
        log(f"[PLANNER_PARALLEL_DONE] windows={len(jobs)} groups={len(groups)} elapsed={fmt_duration(time.time()-started)}")
    else:
        for bi, batch in enumerate(batches):
            is_final = bi == len(batches) - 1
            candidate = carry + batch
            if not candidate:
                continue
            batch_label = f"batch={bi+1}/{len(batches)}"
            try:
                new_groups, new_carry, meta = _plan_candidate_adaptive(candidate, is_final, batch_label, 0)
                pending_groups = list(new_groups)
                if len(new_carry) > args.planner_max_carry_units:
                    forced = new_carry[:-args.planner_carry_keep_units]
                    keep = new_carry[-args.planner_carry_keep_units:]
                    fg, fc, _ = _plan_candidate_adaptive(forced, True, batch_label + ".carryflush", 0)
                    pending_groups.extend(fg)
                    if fc:
                        raise RuntimeError("planner carry flush returned carry")
                    new_carry = keep
                groups.extend(pending_groups)
                carry = new_carry
                if debug_path:
                    append_jsonl(debug_path, {
                        "batch": bi + 1,
                        "unit_range": [candidate[0].uid, candidate[-1].uid],
                        "meta": {k: v for k, v in meta.items() if k != "plan"},
                        "plan": meta.get("plan"),
                        "groups_total": len(groups),
                        "carry_units": len(carry),
                    })
                log(f"[PLANNER_BATCH_DONE] {batch_label} groups_total={len(groups)} carry={len(carry)} elapsed={fmt_duration(time.time()-started)}")
            except Exception as e:
                log(f"[PLANNER_LLM_ERROR] {batch_label} {e}")
                if strict:
                    raise
                log(f"[PLANNER_LLM_ERROR] {batch_label} fallback=rule")
                if is_final or len(candidate) <= args.planner_carry_keep_units:
                    groups.extend(rule_pack_units(candidate, policy, profile))
                    carry = []
                else:
                    prefix = candidate[:-args.planner_carry_keep_units]
                    carry = candidate[-args.planner_carry_keep_units:]
                    groups.extend(rule_pack_units(prefix, policy, profile))

        if carry:
            if strict:
                fg, fc, _ = _plan_candidate_adaptive(carry, True, "final_carry", 0)
                groups.extend(fg)
                if fc:
                    raise RuntimeError("planner final carry returned carry in strict mode")
            else:
                groups.extend(rule_pack_units(carry, policy, profile))
    return groups


def hybrid_plan_chunks(
    units: List[AtomicUnit],
    profile: BookProfile,
    policy: ChunkPolicy,
    cfg: LLMConfig,
    args: argparse.Namespace,
    debug_path: Optional[Path],
    book_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[AtomicUnit]], Dict[str, Any]]:
    """Rule-plan the book, then ask the LLM only about suspicious boundaries.

    This is deliberately bounded for a 20k-book run. Regular item/QA/poetry
    structures stay deterministic; long prose, narrative and philosophy get a
    small number of full-text local replanning windows.
    """
    base = rule_pack_units(units, policy, profile)
    shape = semantic_book_shape(profile, book_plan)
    max_windows = max(0, int(getattr(args, "hybrid_max_windows", 8) or 0))
    stats: Dict[str, Any] = {
        "mode": "hybrid",
        "shape": shape,
        "rule_groups": len(base),
        "candidate_windows": 0,
        "llm_calls": 0,
        "accepted_replans": 0,
        "groups_after": len(base),
    }
    if (
        max_windows == 0
        or shape in {"independent_items", "qa_book", "poetry", "reference", "table_heavy"}
        or len(base) < 2
    ):
        return base, stats

    current = [list(g) for g in base]
    out: List[List[AtomicUnit]] = []
    i = 0
    while i < len(current):
        if i + 1 >= len(current):
            out.append(current[i])
            break
        left, right = current[i], current[i + 1]
        old_penalty = boundary_penalty(left, right)
        if old_penalty < 3 or stats["llm_calls"] >= max_windows:
            out.append(left)
            i += 1
            continue

        candidate = left + right
        if len(candidate) > max(policy.max_units * 2, int(getattr(args, "planner_max_units", 28))):
            out.append(left)
            i += 1
            continue
        stats["candidate_windows"] += 1
        stats["llm_calls"] += 1
        label = f"hybrid={stats['llm_calls']}/{max_windows}"
        log(
            f"[HYBRID_LLM_WINDOW] {label} units={candidate[0].uid}-{candidate[-1].uid} "
            f"old_penalty={old_penalty} chars={len(group_text(candidate))}"
        )
        try:
            replanned = llm_plan_chunks(
                candidate, profile, policy, cfg, args, None, book_plan=book_plan
            )
            replanned, cov = normalize_group_coverage(replanned, candidate, policy, profile)
            assert_exact_group_coverage(replanned, candidate)
            new_penalty = sum(
                boundary_penalty(a, b) for a, b in zip(replanned, replanned[1:])
            )
            within_contract = all(
                len(group_text(g)) <= min(policy.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS)
                and group_page_span(g) <= policy.max_pages
                and len(g) <= policy.max_units
                for g in replanned
            )
            accept = bool(within_contract and new_penalty < old_penalty)
            if accept:
                out.extend(replanned)
                stats["accepted_replans"] += 1
            else:
                out.extend([left, right])
            if debug_path:
                append_jsonl(debug_path, {
                    "mode": "hybrid_window",
                    "unit_range": [candidate[0].uid, candidate[-1].uid],
                    "old_penalty": old_penalty,
                    "new_penalty": new_penalty,
                    "accepted": accept,
                    "raw_duplicates": cov.get("raw_duplicate_refs"),
                })
        except Exception as e:
            log(f"[HYBRID_LLM_FALLBACK] {label} reason={e}")
            out.extend([left, right])
        i += 2

    stats["groups_after"] = len(out)
    return out, stats


def should_use_llm_planner(profile: BookProfile, args: argparse.Namespace) -> bool:
    if args.planner == "never" or args.planner == "rule":
        return False
    if args.planner == "hybrid":
        return False
    if args.planner == "llm":
        return True
    # dataset_quality is LLM-first: auto means LLM planner.
    if getattr(args, "preset", "") in {"dataset_quality", "sft_dpo_quality"}:
        return True
    if getattr(args, "preset", "") in {"production", "fast"}:
        return False
    return True

# -----------------------------------------------------------------------------
# Chunk assembly and quality
# -----------------------------------------------------------------------------


def section_for_group(units: Sequence[AtomicUnit]) -> List[str]:
    sec: List[str] = []
    for u in units:
        for s in u.section_path:
            if s and s not in sec:
                sec.append(s)
    if sec:
        return sec[-4:]
    # Fallback: infer a year section from date-only lines if present.
    for u in units:
        for ln in u.text.splitlines():
            nd = normalize_digits(ln.strip())
            m = re.match(r"^(\d{2,4})\s*/\s*\d{1,2}\s*/\s*\d{1,2}$", nd)
            if m:
                y = int(m.group(1))
                if y < 100:
                    y += 1300
                return ["سال " + to_persian_digits(y)]
    return []



def clean_chunk_text_with_report(text: str) -> Tuple[str, List[str]]:
    """Post-assembly chunk cleaner. Returns cleaned text and removed noise lines."""
    out: List[str] = []
    removed: List[str] = []
    for raw in (text or "").splitlines():
        ln = clean_text(raw)
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        if is_general_noise_line(ln) or is_chunk_noise_line(ln):
            removed.append(ln)
            continue
        out.append(ln)
    return clean_text("\n".join(out)), removed


def clean_chunk_text(text: str) -> str:
    return clean_chunk_text_with_report(text)[0]


GENERIC_NON_CONTEXT_SECTIONS = {
    "فهرست", "مقدمه", "مقادم", "مقدم", "مقدّم", "پیشگفتار", "contents", "index", "toc",
}


def normalize_section_label(x: Any) -> str:
    return clean_text(str(x or "")).strip()


def is_generic_or_bad_section(label: str) -> bool:
    lab = normalize_section_label(label)
    bare = lab.strip().strip(":：").strip().lower()
    if not bare:
        return True
    if bare in GENERIC_NON_CONTEXT_SECTIONS:
        return True
    if is_chunk_noise_line(lab) or BOOK_FOOTER_DOTTED_RE.match(lab) or BOOK_FOOTER_DOTTED_RE_2.match(lab):
        return True
    return False


def sanitize_section_path(sec: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in sec or []:
        lab = normalize_section_label(x)
        if is_generic_or_bad_section(lab):
            continue
        if lab not in out:
            out.append(lab)
    return out[-4:]


def canonical_speaker_label(label: str) -> str:
    lab = normalize_section_label(label)
    if not lab:
        return ""
    if not lab.endswith((":", "：")):
        lab = lab.rstrip(".،؛") + ":"
    return lab


def extract_speaker_labels_from_text(text: str) -> List[str]:
    speakers: List[str] = []
    for raw in (text or "").splitlines():
        ln = clean_text(raw)
        if not ln or is_generic_or_bad_section(ln):
            continue
        if looks_like_speaker_label(ln):
            lab = canonical_speaker_label(ln)
            if lab and lab not in speakers:
                speakers.append(lab)
    return speakers


def likely_context_label(sec: Sequence[str]) -> Optional[str]:
    for x in reversed(sanitize_section_path(sec or [])):
        label = clean_text(str(x))
        if not label or is_generic_or_bad_section(label):
            continue
        if looks_like_speaker_label(label):
            return canonical_speaker_label(label)
    return None


def ensure_chunk_context_in_text(row: Dict[str, Any], inject_context: bool = True) -> Dict[str, Any]:
    """Clean final text and make attribution metadata safer for downstream generators."""
    raw_text = row.get("text") or ""
    text, removed = clean_chunk_text_with_report(raw_text)
    if removed:
        row["noise_removed"] = removed[:25]
        row["noise_removed_count"] = len(removed)
    else:
        row.pop("noise_removed", None)
        row.pop("noise_removed_count", None)

    sec = sanitize_section_path(row.get("sec") or [])
    row["sec"] = sec

    speakers = extract_speaker_labels_from_text(text)
    for lab in sec:
        if looks_like_speaker_label(lab):
            c = canonical_speaker_label(lab)
            if c and c not in speakers:
                speakers.append(c)
    row["speakers"] = speakers
    row["speaker_count"] = len(speakers)
    row.pop("context_injected", None)

    if inject_context and len(speakers) == 1:
        label = speakers[0]
        if text and not text.startswith(label):
            first = clean_text(text.splitlines()[0]) if text.splitlines() else ""
            if label.rstrip(":：") not in first:
                text = clean_text(label + "\n\n" + text)
                row["context_injected"] = label
    elif len(speakers) > 1 and str((row.get("generator_contract") or {}).get("semantic_shape") or "") != "dialogue":
        flags = list(row.get("quality_flags") or [])
        if "multiple_speakers" not in flags:
            flags.append("multiple_speakers")
        row["quality_flags"] = flags

    row["text"] = text
    row["char_count"] = len(text)
    return row


def apply_context_to_chunks(chunks: List[Dict[str, Any]], inject_context: bool = True) -> None:
    for c in chunks:
        ensure_chunk_context_in_text(c, inject_context=inject_context)


def extract_item_numbers_from_text(text: str) -> List[int]:
    """Find item numbers even when OCR puts date before number."""
    nums: List[int] = []
    for raw in (text or "").splitlines():
        ln = clean_text(raw)
        if not ln or is_general_noise_line(ln) or is_chunk_noise_line(ln):
            continue
        m = NUMBERED_LINE_RE.match(ln)
        if not m:
            norm = normalize_digits(ln)
            m = re.search(r"^\s*\d{2,4}\s*/\s*\d{1,2}\s*/\s*\d{1,2}\s+[\(\[]?(\d{1,5})[\)\]]?\s+\S+", norm)
        if not m:
            norm = normalize_digits(ln)
            m = re.match(r"^\s*[\(\[]?(\d{1,5})[\)\]]\s+\S+", norm)
        if m:
            try:
                n = int(normalize_digits(m.group(1)))
                if 1 <= n <= 100000 and n not in nums:
                    nums.append(n)
            except Exception:
                pass
    return nums


def chunk_row(
    doc_id: str,
    source_name: str,
    idx: int,
    units: List[AtomicUnit],
    min_alpha_chars: int = 25,
) -> Optional[Dict[str, Any]]:
    if not units:
        return None
    text = group_text(units)
    if not text or alpha_len(text) < max(0, min_alpha_chars):
        return None
    all_lines = [ln for u in units for ln in u.lines]
    if not all_lines:
        return None
    page_labels = sorted(set(ln.page_label for ln in all_lines))
    item_numbers: List[int] = []
    for u in units:
        for n in unit_item_numbers(u):
            if n not in item_numbers:
                item_numbers.append(n)
    return {
        "id": f"{doc_id}__{idx:06d}",
        "doc": doc_id,
        "src": source_name,
        "ps": min(ln.page_label for ln in all_lines),
        "pe": max(ln.page_label for ln in all_lines),
        "start": all_lines[0].lid,
        "end": all_lines[-1].lid,
        "sec": section_for_group(units),
        "text": text,
        "file_start": min(ln.file_index for ln in all_lines),
        "file_end": max(ln.file_index for ln in all_lines),
        "page_labels": page_labels,
        "unit_start": units[0].uid,
        "unit_end": units[-1].uid,
        "unit_ids": [u.uid for u in units],
        "unit_types": sorted(set(u.unit_type for u in units)),
        "item_numbers": item_numbers,
        "char_count": len(text),
        "unit_count": len(units),
    }


def is_training_worthy_chunk(row: Dict[str, Any]) -> bool:
    text = row.get("text", "")
    if not text or alpha_len(text) < 25:
        return False
    if HTML_TABLE_RE.search(text):
        return False
    return True


def propagate_missing_sections(chunks: List[Dict[str, Any]], max_file_gap: int = 3) -> None:
    last_sec: List[str] = []
    last_file: Optional[int] = None
    for c in chunks:
        if c.get("sec"):
            last_sec = list(c.get("sec") or [])
            last_file = int(c.get("file_end") or 0)
            continue
        if last_sec and last_file is not None:
            fs = int(c.get("file_start") or 0)
            if fs - last_file <= max_file_gap:
                c["sec"] = list(last_sec)


def item_numbers_in_lines(lines: List[LineRef]) -> List[int]:
    nums: List[int] = []
    for ln in lines:
        for n in extract_item_numbers_from_text(ln.text):
            if n not in nums:
                nums.append(n)
    return nums


def compact_number_ranges(values: Sequence[int]) -> List[str]:
    nums = sorted(set(int(x) for x in values))
    if not nums:
        return []
    out: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return out


def strict_item_sequence_report(lines: List[LineRef], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    source: List[int] = []
    for ln in lines:
        m = STRICT_ITEM_LINE_RE.match(ln.text)
        if not m:
            continue
        try:
            source.append(int(normalize_digits(m.group(1))))
        except Exception:
            pass
    unique = sorted(set(source))
    if len(source) >= 2:
        continuity = sum(
            b > a and (b - a) <= 3 for a, b in zip(source, source[1:])
        ) / max(1, len(source) - 1)
        uniqueness = len(unique) / len(source)
    else:
        continuity = uniqueness = 0.0
    missing: List[int] = []
    if unique and len(unique) >= 20 and uniqueness >= 0.80:
        missing = sorted(set(range(unique[0], unique[-1] + 1)) - set(unique))
    chunk_source_numbers: List[int] = []
    for c in chunks:
        text = str(c.get("text") or "")
        for raw in text.splitlines():
            m = STRICT_ITEM_LINE_RE.match(clean_text(raw))
            if m:
                try:
                    chunk_source_numbers.append(int(normalize_digits(m.group(1))))
                except Exception:
                    pass
    return {
        "detected_sequence": bool(len(source) >= 20 and continuity >= 0.72 and uniqueness >= 0.82),
        "source_markers": len(source),
        "source_unique": len(unique),
        "source_min": min(unique) if unique else None,
        "source_max": max(unique) if unique else None,
        "sequence_continuity": round(continuity, 4),
        "sequence_uniqueness": round(uniqueness, 4),
        "source_missing_count": len(missing),
        "source_missing_ranges": compact_number_ranges(missing)[:300],
        "chunk_markers": len(chunk_source_numbers),
        "chunk_marker_loss": max(0, len(source) - len(chunk_source_numbers)),
    }


def quality_report_for_chunks(
    chunks: List[Dict[str, Any]],
    lines: List[LineRef],
    policy: ChunkPolicy,
    args: argparse.Namespace,
    profile: Optional[BookProfile] = None,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    lengths = [len(c.get("text", "")) for c in chunks]

    for c in chunks:
        text = c.get("text", "")
        contract = c.get("generator_contract") or {}
        hard_max = int(contract.get("hard_max_chars") or min(policy.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS))
        final_min = int(contract.get("min_chars") or getattr(args, "final_min_chars", max(120, policy.min_chars * 0.40)))
        if len(text) > hard_max:
            issues.append({"id": c.get("id"), "issue": "too_long", "chars": len(text)})
        if len(text) < final_min:
            issues.append({"id": c.get("id"), "issue": "too_short_for_generator", "chars": len(text)})
        if len(text) > int(contract.get("assessor_visible_chars") or SFT_ASSESSOR_VISIBLE_CHARS):
            issues.append({
                "id": c.get("id"),
                "issue": "tail_not_seen_by_sft_assessor",
                "chars": len(text),
                "visible_chars": int(contract.get("assessor_visible_chars") or SFT_ASSESSOR_VISIBLE_CHARS),
            })
        file_span = int(c.get("file_end", 0)) - int(c.get("file_start", 0)) + 1
        if file_span > policy.max_pages + 1:
            issues.append({"id": c.get("id"), "issue": "too_many_files", "file_span": file_span})
        if c.get("noise_removed_count"):
            issues.append({"id": c.get("id"), "issue": "noise_removed", "count": c.get("noise_removed_count")})
        if int(c.get("speaker_count") or 0) > 1:
            issues.append({"id": c.get("id"), "issue": "multiple_speakers", "speakers": c.get("speakers")})
        txt_lines = [x.strip() for x in text.splitlines() if x.strip()]
        if txt_lines and DATE_ONLY_RE.match(txt_lines[0]):
            issues.append({"id": c.get("id"), "issue": "starts_with_date"})
        if txt_lines and looks_like_speaker_label(txt_lines[-1]):
            issues.append({"id": c.get("id"), "issue": "ends_with_label"})

    source_nums = item_numbers_in_lines(lines)
    chunk_nums: List[int] = []
    for c in chunks:
        chunk_nums.extend([int(x) for x in c.get("item_numbers") or [] if isinstance(x, int) or str(x).isdigit()])
    source_counter = Counter(source_nums)
    chunk_counter = Counter(chunk_nums)

    item_report: Dict[str, Any] = {
        "source_numbered_lines": len(source_nums),
        "chunk_numbered_items": len(chunk_nums),
        "unique_source_numbers": len(source_counter),
        "unique_chunk_numbers": len(chunk_counter),
        "duplicate_chunk_numbers": sorted([k for k, v in chunk_counter.items() if v > 1])[:200],
    }

    expected_missing: List[int] = []
    coverage_ratio: Optional[float] = None
    if args.expected_min_item is not None and args.expected_max_item is not None:
        expected = set(range(args.expected_min_item, args.expected_max_item + 1))
        found = set(chunk_counter.keys())
        expected_missing = sorted(expected - found)
        coverage_ratio = (len(expected) - len(expected_missing)) / max(1, len(expected))
        item_report.update({
            "expected_min_item": args.expected_min_item,
            "expected_max_item": args.expected_max_item,
            "expected_total": len(expected),
            "expected_missing_count": len(expected_missing),
            "expected_missing_sample": expected_missing[:300],
            "coverage_ratio": round(coverage_ratio, 4),
        })
        if coverage_ratio < args.min_item_coverage:
            issues.append({
                "issue": "item_coverage_below_threshold",
                "coverage_ratio": round(coverage_ratio, 4),
                "threshold": args.min_item_coverage,
                "missing_count": len(expected_missing),
            })

    passed = True
    fatal = []
    if args.fail_on_quality_issues and issues:
        passed = False
        fatal.append("quality_issues_present")
    if coverage_ratio is not None and coverage_ratio < args.min_item_coverage:
        passed = False
        fatal.append("item_coverage_below_threshold")

    sequence_report = strict_item_sequence_report(lines, chunks)
    if sequence_report["detected_sequence"] and sequence_report["source_missing_count"]:
        issues.append({
            "issue": "source_item_sequence_has_gaps",
            "missing_count": sequence_report["source_missing_count"],
            "missing_ranges": sequence_report["source_missing_ranges"][:50],
            "blocking": False,
        })
    if sequence_report["chunk_marker_loss"]:
        issues.append({
            "issue": "strict_item_marker_loss_after_chunking",
            "count": sequence_report["chunk_marker_loss"],
        })
        if bool(getattr(args, "fail_on_item_marker_loss", True)):
            passed = False
            fatal.append("strict_item_marker_loss_after_chunking")

    return {
        "passed": passed,
        "fatal": fatal,
        "chunks": len(chunks),
        "lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": round(sum(lengths) / max(1, len(lengths)), 1),
            "median": round(float(median(lengths)) if lengths else 0.0, 1),
        },
        "issues": issues,
        "issue_count": len(issues),
        "item_report": item_report,
        "strict_item_sequence": sequence_report,
        "profile": asdict(profile) if profile else None,
        "policy": asdict(policy),
    }




# -----------------------------------------------------------------------------
# LLM chunk review, repair queue, and repair pass
# -----------------------------------------------------------------------------

KEEP_DECISIONS = {"keep", "keep_sft_only", "keep_dpo_only"}
REPAIR_DECISIONS = {"repair"}
DROP_DECISIONS = {"drop"}
QUARANTINE_DECISIONS = {"quarantine"}
REPAIRABLE_ACTIONS = {"split", "reboundary", "merge_prev", "merge_next"}


def clamp_score(x: Any, default: float = 3.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    return max(1.0, min(5.0, v))


def compute_final_review_score(r: Dict[str, Any]) -> float:
    coherence = clamp_score(r.get("coherence_score"))
    self_contained = clamp_score(r.get("self_contained_score"))
    evidence = clamp_score(r.get("evidence_density_score"))
    sft = clamp_score(r.get("sft_value_score"))
    dpo = clamp_score(r.get("dpo_value_score"))
    boundary = clamp_score(r.get("boundary_quality_score"))
    noise = clamp_score(r.get("noise_score"), 1.0)
    # noise_score is inverted: 1 clean, 5 noisy.
    clean_score = 6.0 - noise
    return round((coherence * 1.25 + self_contained + evidence + sft + dpo + boundary + clean_score) / 7.25, 3)


def normalize_review(raw: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    r = dict(raw or {})
    r["chunk_id"] = str(r.get("chunk_id") or chunk_id)
    for key in [
        "coherence_score", "self_contained_score", "evidence_density_score",
        "sft_value_score", "dpo_value_score", "boundary_quality_score", "noise_score",
    ]:
        default = 1.0 if key == "noise_score" else 3.0
        r[key] = clamp_score(r.get(key), default)
    # Never trust an internally inconsistent model aggregate. Preserve it for
    # audit, but compute the gate score deterministically from component scores.
    r["model_final_score"] = clamp_score(r.get("final_score"), compute_final_review_score(r))
    r["final_score"] = compute_final_review_score(r)
    decision = str(r.get("decision") or "keep").strip().lower()
    if decision not in KEEP_DECISIONS | REPAIR_DECISIONS | DROP_DECISIONS | QUARANTINE_DECISIONS:
        decision = "repair" if r["final_score"] < 3.5 else "keep"
    r["decision"] = decision
    action = str(r.get("repair_action") or "none").strip().lower()
    if action not in {"none", "split", "merge_prev", "merge_next", "reboundary", "drop_noise"}:
        action = "none"
    r["repair_action"] = action
    error_type = str(r.get("error_type") or ("good" if decision in KEEP_DECISIONS else "uncertain")).strip().lower()
    r["error_type"] = error_type
    rep = str(r.get("repairability") or "none").strip().lower()
    if rep not in {"high", "medium", "low", "none"}:
        rep = "medium" if decision == "repair" else "none"
    r["repairability"] = rep
    r["reason"] = str(r.get("reason") or "")[:1200]
    return r


def local_review_for_chunk(c: Dict[str, Any], policy: ChunkPolicy, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    text = c.get("text", "") or ""
    chars = len(text)
    unit_count = int(c.get("unit_count") or len(c.get("unit_ids") or []))
    contract = c.get("generator_contract") or {}
    semantic_shape = str(contract.get("semantic_shape") or "")
    final_min = int(
        contract.get("min_chars")
        or (getattr(args, "final_min_chars", policy.min_chars) if args is not None else policy.min_chars)
    )
    soft_min = int(
        contract.get("ideal_min_chars")
        or (getattr(args, "final_soft_min_chars", max(final_min, policy.min_chars)) if args is not None else max(final_min, policy.min_chars))
    )
    hard_max = int(contract.get("hard_max_chars") or min(policy.max_chars, SAFE_GENERATOR_HARD_MAX_CHARS))
    min_final_units = int(
        contract.get("min_units")
        or (getattr(args, "final_min_units", 1) if args is not None else 1)
    )
    max_final_pages = int(getattr(args, "final_max_pages", policy.max_pages + 1) if args is not None else policy.max_pages + 1)

    flags = set(c.get("quality_flags") or [])
    if (
        args is not None
        and getattr(args, "preset", "") in {"dataset_quality", "sft_dpo_quality"}
        and str(c.get("planner_source") or "") in {"rule", "rule_fallback"}
    ):
        flags.add("non_llm_planner_requires_review")
    noise_removed_count = int(c.get("noise_removed_count") or 0)
    speaker_count = int(c.get("speaker_count") or len(c.get("speakers") or []))
    contaminated_sec = any(is_generic_or_bad_section(x) for x in c.get("sec") or [])

    noise = 1.0
    if HTML_TABLE_RE.search(text):
        noise = 4.0
    if noise_removed_count > 0:
        # Text was cleaned, so do not drop automatically, but force LLM review in flagged mode.
        noise = max(noise, 2.6)
        flags.add("noise_removed")
    if any(is_chunk_noise_line(x) for x in text.splitlines() if clean_text(x)):
        noise = max(noise, 4.2)
        flags.add("residual_noise")

    coherence = 3.9
    if chars < final_min:
        coherence = 2.6
    elif chars < soft_min and unit_count < min_final_units:
        coherence = 3.0
    if chars > hard_max:
        coherence = min(coherence, 2.7)
        flags.add("beyond_generator_contract")
    if unit_count > max(policy.max_units, int(getattr(args, "final_max_units", policy.max_units) if args is not None else policy.max_units)):
        coherence = min(coherence, 2.8)

    file_span = int(c.get("file_end", 0) or 0) - int(c.get("file_start", 0) or 0) + 1
    if file_span > max_final_pages:
        coherence = min(coherence, 2.8)
        flags.add("wide_page_span")

    self_contained = 3.8 if chars >= final_min else 2.8
    evidence = 3.8 if chars >= final_min else 2.8
    sft = 3.8 if chars >= final_min else 2.9
    dpo = 3.7 if chars >= max(final_min, 1800) else 2.8
    boundary = 3.7
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines and (DATE_ONLY_RE.match(lines[0]) or looks_like_colon_label(lines[-1])):
        boundary = 2.2
        flags.add("bad_edge")
    if not c.get("sec") and not c.get("speakers"):
        # Missing metadata is not the same as missing semantic context. Stories,
        # poems, and expository prose are often fully understandable from text.
        self_contained = min(self_contained, 3.5)
        if chars < final_min or (lines and DATE_ONLY_RE.match(lines[0])):
            flags.add("missing_context")
    if speaker_count > 1 and semantic_shape != "dialogue":
        self_contained = min(self_contained, 3.25)
        boundary = min(boundary, 3.15)
        flags.add("multiple_speakers")
    if contaminated_sec:
        self_contained = min(self_contained, 3.2)
        flags.add("contaminated_section")

    raw = {
        "chunk_id": c.get("id"),
        "coherence_score": coherence,
        "self_contained_score": self_contained,
        "evidence_density_score": evidence,
        "sft_value_score": sft,
        "dpo_value_score": dpo,
        "boundary_quality_score": boundary,
        "noise_score": noise,
        "repair_action": "none",
        "error_type": "good",
        "repairability": "none",
        "reason": "strict local triage review",
    }
    raw["final_score"] = compute_final_review_score(raw)

    if "residual_noise" in flags or noise >= 4:
        raw.update({"decision": "drop", "error_type": "noisy_ocr", "repairability": "none", "repair_action": "drop_noise"})
    elif chars < final_min:
        raw.update({"decision": "repair", "error_type": "too_short", "repairability": "medium", "repair_action": "merge_next"})
    elif flags:
        raw.update({"decision": "repair", "error_type": "+".join(sorted(flags))[:80], "repairability": "medium", "repair_action": "reboundary"})
    elif raw["final_score"] < 3.35:
        raw.update({"decision": "repair", "error_type": "low_information_or_bad_boundary", "repairability": "medium", "repair_action": "reboundary" if unit_count > 1 else "none"})
    else:
        raw["decision"] = "keep"
    rr = normalize_review(raw, str(c.get("id")))
    if flags:
        rr["local_flags"] = sorted(flags)
    return rr


def review_excerpt(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    part = max(200, limit // 3)
    mid = len(text) // 2
    return (
        text[:part].rstrip()
        + "\n…\n"
        + text[max(0, mid - part // 2): mid + part // 2].strip()
        + "\n…\n"
        + text[-part:].lstrip()
    )


def compact_chunk_for_review(c: Dict[str, Any], text_limit: int) -> Dict[str, Any]:
    text = review_excerpt(c.get("text", ""), text_limit)
    return {
        "chunk_id": c.get("id"),
        "pages": [c.get("ps"), c.get("pe")],
        "files": [c.get("file_start"), c.get("file_end")],
        "sec": c.get("sec") or [],
        "unit_start": c.get("unit_start"),
        "unit_end": c.get("unit_end"),
        "unit_count": c.get("unit_count"),
        "unit_types": c.get("unit_types") or [],
        "item_numbers": c.get("item_numbers") or [],
        "char_count": len(c.get("text", "") or ""),
        "generator_contract": c.get("generator_contract") or {},
        "text": text,
    }




def review_batch_cost(batch: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[int, int]:
    payload = [compact_chunk_for_review(c, args.review_text_limit) for c in batch]
    chars = sum(len(x.get("text", "")) for x in payload)
    toks = estimate_tokens(payload)
    return chars, toks


def make_review_batches(chunks: List[Dict[str, Any]], args: argparse.Namespace) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for c in chunks:
        trial = cur + [c]
        chars, toks = review_batch_cost(trial, args)
        if cur and (len(trial) > args.review_batch_size or chars > args.review_batch_max_chars or toks > args.review_batch_max_tokens):
            batches.append(cur)
            cur = [c]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


def mark_review_error_chunk(c: Dict[str, Any], policy: ChunkPolicy, args: argparse.Namespace, stage: str, error: Exception) -> Dict[str, Any]:
    mode = getattr(args, "review_fallback", "fail")
    if mode == "local":
        r = local_review_for_chunk(c, policy, args)
        r["reviewed_by_llm"] = False
        r["review_error"] = str(error)
        r["fallback"] = "local"
        r["stage"] = stage
        return r
    # Do not let local heuristics silently decide quality in strict mode.
    r = {
        "chunk_id": str(c.get("id")),
        "coherence_score": 1.0,
        "self_contained_score": 1.0,
        "evidence_density_score": 1.0,
        "sft_value_score": 1.0,
        "dpo_value_score": 1.0,
        "boundary_quality_score": 1.0,
        "noise_score": 5.0,
        "final_score": 1.0,
        "decision": "quarantine",
        "repair_action": "none",
        "error_type": "review_error",
        "repairability": "none",
        "reason": "LLM review failed; quarantined rather than judged by heuristic.",
        "reviewed_by_llm": False,
        "review_error": str(error),
        "fallback": "quarantine",
        "stage": stage,
    }
    return normalize_review(r, str(c.get("id")))

def review_chunks_with_llm(
    chunks: List[Dict[str, Any]],
    profile: BookProfile,
    policy: ChunkPolicy,
    cfg: LLMConfig,
    args: argparse.Namespace,
    review_path: Optional[Path],
    queue_path: Optional[Path],
    stage: str = "initial",
    book_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach LLM quality reviews to chunks and return a repair queue.

    Quality mode never silently uses heuristic review unless --review-fallback local is explicit.
    Review batches are token-budgeted, can run concurrently, and are recursively split on LLM failures.
    """
    if not chunks:
        return chunks, []
    repair_queue: List[Dict[str, Any]] = []
    batches = make_review_batches(chunks, args)
    total_batches = len(batches)
    results_by_id: Dict[str, Dict[str, Any]] = {}

    def review_one_batch(batch: List[Dict[str, Any]], batch_label: str, depth: int = 0) -> List[Dict[str, Any]]:
        use_llm = args.review_chunks in {"all", "llm", "flagged"}
        if args.review_chunks == "flagged":
            local = [local_review_for_chunk(c, policy, args) for c in batch]
            if all(r["final_score"] >= args.review_min_score and r["decision"] in KEEP_DECISIONS for r in local):
                for r in local:
                    r["reviewed_by_llm"] = False
                    r["stage"] = stage
                return local
        if not use_llm or args.review_chunks == "never":
            local = []
            for c in batch:
                r = local_review_for_chunk(c, policy, args)
                r["reviewed_by_llm"] = False
                r["stage"] = stage
                local.append(r)
            return local

        payload = {
            "language": "fa",
            "stage": stage,
            "profile": asdict(profile),
            "policy": asdict(policy),
            "book_plan": book_plan or {},
            "review_goal": "Primary target: grounded SFT. Judge whether each chunk is a coherent, self-contained evidence unit whose necessary semantic dependencies are visible. Do not create task-specific subsets.",
            "chunks": [compact_chunk_for_review(c, args.review_text_limit) for c in batch],
        }
        log(f"[REVIEW_LLM_CALL] stage={stage} batch={batch_label} chunks={len(batch)}")
        t0 = time.time()
        try:
            obj = call_llm_json(CHUNK_REVIEW_PROMPT, payload, cfg)
            raw_reviews = obj.get("reviews") or []
            by_id: Dict[str, Dict[str, Any]] = {}
            for rr in raw_reviews:
                if isinstance(rr, dict):
                    cid = str(rr.get("chunk_id") or "")
                    if cid:
                        by_id[cid] = rr
            out: List[Dict[str, Any]] = []
            for c in batch:
                cid = str(c.get("id"))
                if cid in by_id:
                    r = normalize_review(by_id[cid], cid)
                    r["reviewed_by_llm"] = True
                else:
                    raise RuntimeError(f"LLM review missing chunk_id={cid}")
                r["stage"] = stage
                out.append(r)
            log(f"[REVIEW_LLM_DONE] stage={stage} batch={batch_label} llm={time.time()-t0:.1f}s")
            return out
        except Exception as e:
            log(f"[REVIEW_LLM_ERROR] stage={stage} batch={batch_label} chunks={len(batch)} error={e}")
            # Token/time failures: split the batch; connection refused with strict fallback fails fast.
            if len(batch) > 1 and depth < args.review_split_depth:
                mid = len(batch) // 2
                left = review_one_batch(batch[:mid], batch_label + "a", depth + 1)
                right = review_one_batch(batch[mid:], batch_label + "b", depth + 1)
                return left + right
            if getattr(args, "review_fallback", "fail") == "fail":
                raise
            return [mark_review_error_chunk(c, policy, args, stage, e) for c in batch]

    # Review concurrently inside a book, but with bounded workers so the LLM pool is not flooded.
    if args.review_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max(1, args.review_workers)) as ex:
            futs = {
                ex.submit(review_one_batch, b, f"{i+1}/{total_batches}"): (i, b)
                for i, b in enumerate(batches)
            }
            for fut in as_completed(futs):
                _i, _b = futs[fut]
                for r in fut.result():
                    results_by_id[str(r.get("chunk_id"))] = r
                    if review_path:
                        append_jsonl(review_path, r)
    else:
        for i, b in enumerate(batches):
            for r in review_one_batch(b, f"{i+1}/{total_batches}"):
                results_by_id[str(r.get("chunk_id"))] = r
                if review_path:
                    append_jsonl(review_path, r)

    for c in chunks:
        cid = str(c.get("id"))
        r = results_by_id.get(cid)
        if not r:
            raise RuntimeError(f"No review result for chunk {cid}")
        c["quality"] = r

    for c in chunks:
        r = c.get("quality") or {}
        needs_repair = (
            r.get("decision") == "repair"
            and r.get("repair_action") in {"split", "reboundary", "merge_prev", "merge_next"}
            and r.get("repairability") in {"high", "medium"}
            and float(r.get("final_score") or 0) < args.review_keep_score
        )
        if needs_repair:
            q = {
                "doc": c.get("doc"),
                "src": c.get("src"),
                "chunk_id": c.get("id"),
                "attempt": int(c.get("repair_attempt", 0)) + 1,
                "unit_start": c.get("unit_start"),
                "unit_end": c.get("unit_end"),
                "unit_ids": c.get("unit_ids") or [],
                "repair_action": r.get("repair_action"),
                "error_type": r.get("error_type"),
                "repairability": r.get("repairability"),
                "scores": {k: r.get(k) for k in ["coherence_score", "self_contained_score", "sft_value_score", "dpo_value_score", "final_score"]},
                "reason": r.get("reason"),
            }
            repair_queue.append(q)
            if queue_path:
                append_jsonl(queue_path, q)
    return chunks, repair_queue


def validate_repair_ranges(plan: Dict[str, Any], candidate: List[AtomicUnit]) -> Tuple[str, List[Tuple[int, int, str]], str]:
    id_to_pos = {u.uid: i for i, u in enumerate(candidate)}
    decision = str(plan.get("repair_decision") or "drop").strip().lower()
    if decision not in {"split", "keep", "drop", "quarantine"}:
        decision = "drop"
    ranges: List[Tuple[int, int, str]] = []
    last_end = -1
    for ch in plan.get("chunks") or []:
        if not isinstance(ch, dict):
            continue
        s, e = ch.get("start"), ch.get("end")
        if s not in id_to_pos or e not in id_to_pos:
            continue
        a, b = id_to_pos[s], id_to_pos[e]
        if a > b or a <= last_end:
            continue
        reason = str(ch.get("reason") or "")[:500]
        ranges.append((a, b, reason))
        last_end = b
    return decision, ranges, str(plan.get("reason") or "")[:1000]


def repair_queued_chunks(
    repair_queue: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    uid_to_unit: Dict[str, AtomicUnit],
    doc_id: str,
    source_name: str,
    profile: BookProfile,
    policy: ChunkPolicy,
    cfg: LLMConfig,
    args: argparse.Namespace,
    repair_plans_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Repair flagged chunks with a lossless, neighbor-aware full-LLM window.

    split/reboundary operate on the original chunk.  merge_prev/merge_next expand
    the repair window to exactly one adjacent original chunk.  A successful plan
    must cover every AtomicUnit in the provided window exactly once, so repair
    can never silently delete source text or create overlapping replacements.
    """
    repaired: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    if args.repair_chunks == "never" or not repair_queue:
        return repaired, repair_queue

    ordered_ids = list(chunks_by_id.keys())
    index_by_id = {cid: i for i, cid in enumerate(ordered_ids)}
    consumed_chunk_ids: set[str] = set()

    def _window_for(q: Dict[str, Any]) -> Tuple[List[str], List[AtomicUnit], Optional[str]]:
        cid = str(q.get("chunk_id") or "")
        if cid not in chunks_by_id:
            return [], [], "missing_source_chunk"
        idx = index_by_id[cid]
        action = str(q.get("repair_action") or "")
        selected_ids = [cid]
        if action == "merge_prev":
            if idx <= 0:
                return [], [], "no_previous_chunk"
            selected_ids = [ordered_ids[idx - 1], cid]
        elif action == "merge_next":
            if idx + 1 >= len(ordered_ids):
                return [], [], "no_next_chunk"
            selected_ids = [cid, ordered_ids[idx + 1]]

        if any(x in consumed_chunk_ids for x in selected_ids):
            return [], [], "repair_window_already_consumed"

        unit_ids: List[str] = []
        seen_uids: set[str] = set()
        for sid in selected_ids:
            for uid in chunks_by_id[sid].get("unit_ids") or []:
                if uid in uid_to_unit and uid not in seen_uids:
                    seen_uids.add(uid)
                    unit_ids.append(uid)
        candidate = [uid_to_unit[uid] for uid in unit_ids]
        if not candidate:
            return selected_ids, [], "missing_units"
        return selected_ids, candidate, None

    def _ranges_cover_exactly(ranges: List[Tuple[int, int, str]], n: int) -> bool:
        if not ranges or n <= 0:
            return False
        cursor = 0
        for a, b, _reason in ranges:
            if a != cursor or b < a:
                return False
            cursor = b + 1
        return cursor == n

    for qi, q in enumerate(repair_queue):
        if int(q.get("attempt") or 1) > args.max_repair_attempts:
            unresolved.append({**q, "unresolved_reason": "max_repair_attempts"})
            continue
        action = str(q.get("repair_action") or "")
        if action not in REPAIRABLE_ACTIONS:
            unresolved.append({**q, "unresolved_reason": "unsafe_repair_action"})
            continue

        replace_ids, candidate, window_error = _window_for(q)
        if window_error:
            unresolved.append({**q, "unresolved_reason": window_error})
            continue

        payload = {
            "language": "fa",
            "profile": asdict(profile),
            "policy": asdict(policy),
            "review_error": q,
            "repair_window": {
                "replaces_chunk_ids": replace_ids,
                "repair_action": action,
                "must_cover_all_units_exactly_once": True,
            },
            "units": [compact_unit_for_llm(u, args.repair_text_limit) for u in candidate],
        }
        log(
            f"[REPAIR_LLM_CALL] {qi+1}/{len(repair_queue)} chunk={q.get('chunk_id')} "
            f"action={action} window={replace_ids} units={len(candidate)}"
        )
        t0 = time.time()
        try:
            plan = call_llm_json(CHUNK_REPAIR_PROMPT, payload, cfg)
            decision, ranges, reason = validate_repair_ranges(plan, candidate)
            if repair_plans_path:
                append_jsonl(
                    repair_plans_path,
                    {
                        "queue": q,
                        "repair_window": replace_ids,
                        "plan": plan,
                        "decision": decision,
                        "ranges": len(ranges),
                    },
                )
            if decision in {"drop", "quarantine"} or not ranges:
                unresolved.append({**q, "repair_decision": decision, "repair_reason": reason})
                continue
            if not _ranges_cover_exactly(ranges, len(candidate)):
                unresolved.append(
                    {
                        **q,
                        "unresolved_reason": "repair_plan_not_lossless",
                        "repair_decision": decision,
                        "repair_reason": reason,
                    }
                )
                continue

            made_rows: List[Dict[str, Any]] = []
            for a, b, rr in ranges:
                row = chunk_row(doc_id, source_name, len(repaired) + len(made_rows), candidate[a:b + 1])
                if not row or not is_training_worthy_chunk(row):
                    made_rows = []
                    break
                row["id"] = f"{doc_id}__repair_{len(repaired) + len(made_rows):06d}"
                row["repair"] = {
                    "from_chunk_id": q.get("chunk_id"),
                    "replaces_chunk_ids": list(replace_ids),
                    "attempt": q.get("attempt"),
                    "error_type": q.get("error_type"),
                    "repair_action": action,
                    "repair_reason": rr or reason,
                    "lossless_window": True,
                }
                made_rows.append(row)

            if not made_rows:
                unresolved.append({**q, "unresolved_reason": "repaired_chunk_not_training_worthy"})
                continue

            repaired.extend(made_rows)
            consumed_chunk_ids.update(replace_ids)
            log(
                f"[REPAIR_LLM_DONE] chunk={q.get('chunk_id')} replaces={replace_ids} "
                f"new_chunks={len(made_rows)} llm={time.time()-t0:.1f}s"
            )
        except Exception as e:
            log(f"[REPAIR_LLM_ERROR] chunk={q.get('chunk_id')} {e}")
            unresolved.append({**q, "repair_error": str(e)})
    return repaired, unresolved


TERMINAL_CHARS = set(".؟?!！؛;…»”\"')]}ـ")
CONTINUATION_START_RE = re.compile(
    r"^\s*(?:و|اما|ولی|لیکن|زیرا|چراکه|که|همچنین|بنابراین|ازاین‌رو|دراین‌حال)\s+"
)


def downstream_trust_review(c: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Attach non-blocking routing metadata and reject only fatal structure."""
    text = clean_text(c.get("text") or "")
    chars = len(text)
    alpha = alpha_len(text)
    contract = c.get("generator_contract") or {}
    shape = str(contract.get("semantic_shape") or "mixed")
    hard_max = int(
        contract.get("hard_max_chars")
        or getattr(args, "generator_hard_max_chars", SAFE_GENERATOR_HARD_MAX_CHARS)
    )
    min_chars = int(getattr(args, "trust_min_chars", 40) or 40)
    min_alpha = int(getattr(args, "trust_min_alpha_chars", 10) or 10)
    if shape in {"poetry", "qa_book", "independent_items", "dialogue"}:
        min_chars = min(min_chars, int(getattr(args, "trust_short_shape_min_chars", 20) or 20))

    fatal: List[str] = []
    if not text:
        fatal.append("empty_text")
    if chars < min_chars or alpha < min_alpha:
        fatal.append("unreadable_or_tiny")
    if chars > hard_max:
        fatal.append("hard_max_exceeded")
    if bool(getattr(args, "trust_reject_html_tables", True)) and HTML_TABLE_RE.search(text):
        fatal.append("residual_html_table")

    flags: List[str] = []
    weight = 1.0
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]

    if chars < 500:
        flags.append("very_short")
        weight -= 0.28
    elif chars < 1200:
        flags.append("short")
        weight -= 0.16

    if lines:
        first = lines[0]
        last = lines[-1]
        if CONTINUATION_START_RE.match(first):
            flags.append("continuation_start")
            weight -= 0.10
        if chars >= 250 and last[-1:] not in TERMINAL_CHARS and not DATE_ONLY_RE.match(last):
            flags.append("non_terminal_end")
            weight -= 0.16
        if looks_like_colon_label(last):
            flags.append("ends_with_label")
            weight -= 0.12

    speaker_count = int(c.get("speaker_count") or len(c.get("speakers") or []))
    if speaker_count > 1 and shape != "dialogue":
        flags.append("multiple_speakers")
        weight -= 0.10
    if len(c.get("sec") or []) > 3:
        flags.append("many_section_labels")
        weight -= 0.08
    if int(c.get("noise_removed_count") or 0) > 0:
        flags.append("noise_removed")
        weight -= 0.12
    if chars > int(contract.get("assessor_visible_chars") or SFT_ASSESSOR_VISIBLE_CHARS):
        flags.append("tail_not_seen_by_sft_assessor")
        weight -= 0.05

    unit_types = set(c.get("unit_types") or [])
    normalized_text = normalize_digits(text).lower()
    if "ocr_noise" in unit_types:
        flags.append("probable_ocr_noise")
        weight -= 0.45
    if unit_types and unit_types.issubset({"caption", "section_title"}):
        flags.append("metadata_like")
        weight -= 0.22
    elif re.search(r"(?:^|\n)\s*#{0,3}\s*(?:پی\s*نوشت|منابع|فهرست)\s*[:：]?", normalized_text):
        flags.append("back_matter_like")
        weight -= 0.18

    # Preserve order while removing duplicate flags.
    flags = list(dict.fromkeys(flags))
    weight = round(max(0.20, min(1.0, weight)), 3)
    accepted = not fatal
    if not accepted:
        weight = 0.0

    tier = "clean"
    if weight < 0.50:
        tier = "high_risk"
    elif weight < 0.75:
        tier = "medium_risk"
    elif weight < 0.95:
        tier = "low_risk"

    # In the old quality-gate path, risk flags could send a chunk to
    # quarantine. In downstream-trust mode those candidates are deliberately
    # promoted into the accepted stream; the second system owns semantic value
    # rejection. We retain an explicit marker so weighting/auditing is possible.
    promoted_from_quarantine = bool(accepted and flags)
    c["routing"] = {
        "accepted": accepted,
        "weight": weight,
        "tier": tier if accepted else "structural_reject",
        "flags": flags,
        "fatal_reasons": fatal,
        "promoted_from_quarantine": promoted_from_quarantine,
        "legacy_bucket": "quarantine_promoted" if promoted_from_quarantine else ("keep" if accepted else "drop"),
        "mode": "downstream_trust",
    }
    c["routing_weight"] = weight
    c["routing_flags"] = flags
    c["promoted_from_quarantine"] = promoted_from_quarantine

    compatibility_score = round(1.0 + 4.0 * weight, 3) if accepted else 1.0
    return {
        "chunk_id": str(c.get("id")),
        "coherence_score": compatibility_score,
        "self_contained_score": compatibility_score,
        "evidence_density_score": compatibility_score,
        "sft_value_score": compatibility_score,
        "dpo_value_score": compatibility_score,
        "boundary_quality_score": compatibility_score,
        "noise_score": 1.0 if accepted else 5.0,
        "final_score": compatibility_score,
        "decision": "keep" if accepted else "drop",
        "legacy_decision": "quarantine_promoted" if promoted_from_quarantine else ("keep" if accepted else "drop"),
        "repair_action": "none",
        "error_type": "good" if not flags and accepted else "+".join(flags or fatal)[:160],
        "repairability": "none",
        "reason": (
            "Promoted from the former quarantine path; downstream SFT/DPO service decides semantic value."
            if promoted_from_quarantine
            else "Passed structural routing; downstream SFT/DPO service decides semantic value."
            if accepted
            else "Rejected by fatal structural gate."
        ),
        "reviewed_by_llm": False,
        "stage": "downstream_trust",
    }


def route_chunks_downstream_trust(
    chunks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    keep: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for c in chunks:
        c["quality"] = downstream_trust_review(c, args)
        if bool((c.get("routing") or {}).get("accepted")):
            keep.append(c)
        else:
            rejected.append(c)
    # Quarantine is intentionally empty in downstream-trust mode.
    return keep, rejected, []


def split_chunks_by_decision(chunks: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Final quality gate for generator source chunks.

    The final output is a single high-quality chunks.jsonl for downstream SFT/DPO generators.
    No chunk marked repair is final. Very short chunks are not final either: they are
    quarantined/merge-needed unless explicitly allowed and exceptionally high-scoring.
    """
    keep: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    quarantine: List[Dict[str, Any]] = []
    allow_short = bool(getattr(args, "allow_short_final", False))
    short_exception_score = float(getattr(args, "short_exception_score", 4.65) or 4.65)

    for c in chunks:
        q = c.get("quality") or {}
        contract = c.get("generator_contract") or {}
        final_min = int(contract.get("min_chars") or getattr(args, "final_min_chars", 900) or 900)
        soft_min = int(contract.get("ideal_min_chars") or getattr(args, "final_soft_min_chars", 1200) or 1200)
        min_units = int(contract.get("min_units") or getattr(args, "final_min_units", 1) or 1)
        hard_max = int(contract.get("hard_max_chars") or getattr(args, "generator_hard_max_chars", SAFE_GENERATOR_HARD_MAX_CHARS))
        decision = str(q.get("decision") or "keep").lower()
        score = float(q.get("final_score") or 0)
        chars = len(c.get("text") or "")
        units = int(c.get("unit_count") or len(c.get("unit_ids") or []))

        if decision == "repair":
            c.setdefault("finalizer", {})["blocked_reason"] = "decision_repair_not_final"
            quarantine.append(c)
            continue

        if decision in DROP_DECISIONS or (score < args.review_drop_score and q.get("repairability") in {"low", "none"}):
            dropped.append(c)
            continue

        if decision in QUARANTINE_DECISIONS or score < args.review_min_score:
            quarantine.append(c)
            continue

        if chars > hard_max:
            c.setdefault("finalizer", {})["blocked_reason"] = "beyond_generator_hard_max"
            c.setdefault("finalizer", {})["generator_hard_max_chars"] = hard_max
            quarantine.append(c)
            continue

        # Generator-long gate: keep chunks long enough to be useful roots for SFT/DPO generation.
        # We do not keep short quote fragments just because they are clean.
        too_short = chars < final_min or (chars < soft_min and units < min_units)
        if too_short:
            if allow_short and score >= short_exception_score and q.get("coherence_score", 0) >= 4.5 and q.get("self_contained_score", 0) >= 4.5:
                c.setdefault("finalizer", {})["short_exception"] = True
                keep.append(c)
            else:
                c.setdefault("finalizer", {})["blocked_reason"] = "too_short_for_generator"
                c.setdefault("finalizer", {})["final_min_chars"] = final_min
                c.setdefault("finalizer", {})["final_soft_min_chars"] = soft_min
                quarantine.append(c)
            continue

        if decision in KEEP_DECISIONS or score >= args.review_min_score:
            keep.append(c)
        else:
            quarantine.append(c)
    return keep, dropped, quarantine



def write_unit_ledger(path: Optional[Path], units: List[AtomicUnit], final_chunks: List[Dict[str, Any]], dropped: List[Dict[str, Any]], quarantine: List[Dict[str, Any]], unresolved: List[Dict[str, Any]]) -> Dict[str, int]:
    status_by_uid: Dict[str, str] = {}
    chunk_by_uid: Dict[str, str] = {}
    for label, rows in [("final", final_chunks), ("dropped", dropped), ("quarantine", quarantine)]:
        for c in rows:
            for uid in c.get("unit_ids") or []:
                status_by_uid[str(uid)] = label
                chunk_by_uid[str(uid)] = str(c.get("id"))
    for q in unresolved:
        for uid in q.get("unit_ids") or []:
            status_by_uid.setdefault(str(uid), "repair_unresolved")
            chunk_by_uid.setdefault(str(uid), str(q.get("chunk_id")))
    counts = Counter()
    if path and path.exists():
        path.unlink()
    for u in units:
        st = status_by_uid.get(u.uid, "unaccounted")
        counts[st] += 1
        if path:
            append_jsonl(path, {
                "uid": u.uid,
                "status": st,
                "chunk_id": chunk_by_uid.get(u.uid),
                "unit_type": u.unit_type,
                "item_numbers": unit_item_numbers(u),
                "page_start": u.page_start,
                "page_end": u.page_end,
                "char_count": u.char_count,
            })
    return dict(counts)


# -----------------------------------------------------------------------------
# Debug writers
# -----------------------------------------------------------------------------


def write_debug_units(path: Path, units: List[AtomicUnit]) -> None:
    if path.exists():
        path.unlink()
    for u in units:
        append_jsonl(path, {
            "uid": u.uid,
            "idx": u.idx,
            "unit_type": u.unit_type,
            "pages": [u.page_start, u.page_end],
            "files": [u.file_start, u.file_end],
            "start": u.start_ref,
            "end": u.end_ref,
            "section_path": u.section_path,
            "confidence": u.confidence,
            "source": u.source,
            "chars": len(u.text),
            "item_numbers": unit_item_numbers(u),
            "text": u.text,
        })


def write_debug_groups(path: Path, groups: List[List[AtomicUnit]]) -> None:
    if path.exists():
        path.unlink()
    for i, g in enumerate(groups):
        append_jsonl(path, {
            "group_index": i,
            "unit_start": g[0].uid if g else None,
            "unit_end": g[-1].uid if g else None,
            "unit_ids": [u.uid for u in g],
            "unit_types": sorted(set(u.unit_type for u in g)),
            "chars": len(group_text(g)),
            "pages": [g[0].page_start, g[-1].page_end] if g else [],
            "files": [g[0].file_start, g[-1].file_end] if g else [],
            "item_numbers": [n for u in g for n in unit_item_numbers(u)],
            "text": group_text(g),
        })


# -----------------------------------------------------------------------------
# Folder processing
# -----------------------------------------------------------------------------


def process_folder(folder: Path, output_root: Path, llm_cfg: LLMConfig, args: argparse.Namespace) -> Dict[str, Any]:
    folder = folder.resolve()
    doc_id = stable_doc_id(folder)
    source_name = folder.name
    out_dir = output_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = out_dir / "chunks.jsonl"                  # final keep chunks
    chunks_initial_path = out_dir / "chunks_initial.jsonl"  # before review/repair
    # Optional task-specific subsets are disabled by default; generator service decides use.
    chunks_sft_path = out_dir / "chunks_sft.jsonl"
    chunks_dpo_path = out_dir / "chunks_dpo.jsonl"
    chunks_all_reviewed_path = out_dir / "chunks_all_reviewed.jsonl"
    unit_ledger_path = out_dir / "unit_ledger.jsonl"
    book_plan_path = out_dir / "book_plan.json"
    chunks_dropped_path = out_dir / "chunks_dropped.jsonl"
    chunks_rejected_structural_path = out_dir / "chunks_rejected_structural.jsonl"
    chunks_quarantine_path = out_dir / "chunks_quarantine.jsonl"
    review_path = out_dir / "chunk_reviews.jsonl"
    repair_queue_path = out_dir / "repair_queue.jsonl"
    repair_reviews_path = out_dir / "repair_reviews.jsonl"
    repair_unresolved_path = out_dir / "repair_unresolved.jsonl"
    manifest_path = out_dir / "manifest.json"
    profile_path = out_dir / "profile.json"
    quality_path = out_dir / "quality_report.json"
    lines_path = out_dir / "debug" / "clean_lines.jsonl"
    units_path = out_dir / "debug" / "atomic_units.jsonl"
    groups_path = out_dir / "debug" / "groups.jsonl"
    atomizer_plans_path = out_dir / "debug" / "atomizer_plans.jsonl"
    planner_plans_path = out_dir / "debug" / "planner_plans.jsonl"
    repair_plans_path = out_dir / "debug" / "repair_plans.jsonl"
    errors_path = out_dir / "errors.jsonl"
    save_audit = bool(args.debug or getattr(args, "output_mode", "minimal") == "audit")

    # Resume / skip-done for large corpus runs.  Do this before deleting any
    # previous output files.  A document is considered done when manifest and
    # chunks.jsonl exist and at least one final chunk was produced, unless
    # --resume-require-quality-pass is set.
    if (getattr(args, "resume", False) or getattr(args, "skip_done", False)) and not getattr(args, "force", False):
        if manifest_path.exists() and chunks_path.exists():
            try:
                prev_manifest = load_json(manifest_path)
                prev_chunks = int(prev_manifest.get("chunks") or 0)
                prev_passed = bool(prev_manifest.get("quality_passed"))
                require_pass = bool(getattr(args, "resume_require_quality_pass", False))
                if prev_chunks > 0 and (prev_passed or not require_pass):
                    prev_manifest["skipped_existing"] = True
                    prev_manifest["skip_reason"] = "resume_existing_manifest_and_chunks"
                    log(f"[SKIP] {source_name} existing chunks={prev_chunks} passed={prev_passed} out={out_dir}")
                    return prev_manifest
            except Exception as e:
                log(f"[RESUME_CHECK_ERROR] {source_name} {e}; reprocessing")

    all_output_files = [
        chunks_path, chunks_initial_path, chunks_sft_path, chunks_dpo_path,
        chunks_dropped_path, chunks_rejected_structural_path, chunks_quarantine_path, review_path, repair_queue_path,
        repair_reviews_path, repair_unresolved_path, chunks_all_reviewed_path, unit_ledger_path, errors_path,
    ]
    for fp in all_output_files:
        if fp.exists():
            fp.unlink()
    if args.debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        for fp in [lines_path, units_path, groups_path, atomizer_plans_path, planner_plans_path, repair_plans_path]:
            if fp.exists():
                fp.unlink()

    def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
        if path.exists():
            path.unlink()
        for row in rows:
            append_jsonl(path, row)

    started = time.time()
    thread_name = threading.current_thread().name
    pages, duplicate_pages_dropped, ingestion_stats = load_pages_sorted(folder)
    if not pages:
        error = (
            "No readable OCR pages found. JSON discovery is schema-based; "
            f"candidates={ingestion_stats.get('json_candidates', 0)} "
            f"parsed={ingestion_stats.get('json_parsed', 0)} "
            f"non_ocr={ingestion_stats.get('json_non_ocr', 0)} "
            f"invalid={ingestion_stats.get('json_invalid', 0)}"
        )
        append_jsonl(errors_path, {
            "stage": "ingestion",
            "error_type": "no_readable_pages",
            "error": error,
            "ingestion": ingestion_stats,
        })
        failure_manifest = {
            "doc_id": doc_id,
            "source_name": source_name,
            "source_path": str(folder),
            "status": "failed",
            "failed": True,
            "error_type": "no_readable_pages",
            "error": error,
            "pages_seen": 0,
            "duplicate_pages_dropped": duplicate_pages_dropped,
            "clean_lines": 0,
            "atomic_units": 0,
            "initial_chunks": 0,
            "chunks": 0,
            "sft_chunks": 0,
            "dpo_chunks": 0,
            "dropped_chunks": 0,
            "structural_reject_chunks": 0,
            "quarantine_chunks": 0,
            "quality_passed": False,
            "ingestion": ingestion_stats,
            "elapsed_sec": round(time.time() - started, 2),
        }
        write_json(manifest_path, failure_manifest)
        write_json(quality_path, {
            "passed": False,
            "fatal": ["no_readable_pages"],
            "chunks": 0,
            "issues": [],
            "issue_count": 0,
            "ingestion": ingestion_stats,
        })
        log(f"[FAILED_EMPTY_BOOK] [{thread_name}] {source_name} {error}")
        return failure_manifest

    repeated_noise = infer_repeated_noise_lines(pages)
    lines = build_clean_lines(pages, repeated_noise)
    profile = profile_book(pages, lines, repeated_noise)
    base_policy = choose_policy(profile, args)
    book_plan = {}
    if args.book_plan in {"auto", "llm"}:
        book_plan = llm_book_plan(profile, lines, llm_cfg, args, book_plan_path if save_audit else None)
    else:
        book_plan = {"book_shape": profile.detected_shape, "confidence": profile.confidence, "source": "disabled", "chunk_strategy": {}, "unit_strategy": {}}
        if save_audit:
            write_json(book_plan_path, {"book_plan": book_plan})
    policy = policy_from_book_plan(base_policy, book_plan, args)
    semantic_shape = semantic_book_shape(profile, book_plan)
    generator_contract = generator_contract_for_book(profile, policy, book_plan, args)

    if save_audit:
        write_json(profile_path, {
            "profile": asdict(profile),
            "base_policy": asdict(base_policy),
            "policy": asdict(policy),
            "generator_contract": asdict(generator_contract),
            "book_plan": book_plan,
            "repeated_noise_lines": sorted(repeated_noise)[:500],
            "ingestion": ingestion_stats,
        })
    log(f"[START] [{thread_name}] {source_name} doc_id={doc_id} pages={len(pages)} dup={duplicate_pages_dropped} lines={len(lines)} out={out_dir}")
    log(
        f"[PROFILE] [{thread_name}] {source_name} rule_shape={profile.detected_shape} "
        f"semantic_shape={semantic_shape} conf={profile.confidence} policy={policy.mode} "
        f"target={policy.target_chars} max={policy.max_chars} "
        f"final_min={generator_contract.min_chars} final_target={generator_contract.target_chars} "
        f"generator_hard_max={generator_contract.hard_max_chars}"
    )

    if args.debug:
        for ln in lines:
            append_jsonl(lines_path, asdict(ln))

    try:
        if should_use_llm_atomizer(profile, args, book_plan):
            units = llm_atomize(
                lines,
                profile,
                llm_cfg,
                args,
                atomizer_plans_path if args.debug else None,
                semantic_shape=semantic_shape,
                book_plan=book_plan,
            )
            atomizer_used = "llm"
        else:
            units = rule_atomize(lines, profile, semantic_shape)
            atomizer_used = "rule"
            log(f"[ATOMIZER_RULE] [{thread_name}] {source_name} units={len(units)}")
    except Exception as e:
        append_jsonl(errors_path, {"stage": "atomize", "error": str(e)})
        log(f"[ERROR] [{thread_name}] atomizer failed: {e}")
        if getattr(args, "preset", "") == "sft_dpo_quality":
            raise
        log(f"[ERROR] [{thread_name}] atomizer fallback=rule")
        units = rule_atomize(lines, profile, semantic_shape)
        atomizer_used = "rule_fallback"

    units_before_split = len(units)
    units, atom_split_stats = split_large_atomic_units(units, args, profile, semantic_shape)
    if atom_split_stats.get("split_units"):
        atomizer_used = f"{atomizer_used}+split_large"
        log(
            f"[ATOM_SPLIT] [{thread_name}] {source_name} units={units_before_split}->{len(units)} "
            f"split_units={atom_split_stats.get('split_units')} "
            f"max_chars={atom_split_stats.get('max_unit_chars_before')}->{atom_split_stats.get('max_unit_chars_after')}"
        )
    units, noise_isolation = split_probable_noise_units(units, args)
    if noise_isolation.get("noise_units"):
        atomizer_used = f"{atomizer_used}+noise_isolation"
        log(
            f"[NOISE_ISOLATION] [{thread_name}] {source_name} "
            f"units={noise_isolation.get('input_units')}->{noise_isolation.get('output_units')} "
            f"noise_units={noise_isolation.get('noise_units')} "
            f"split_units={noise_isolation.get('split_units')}"
        )

    if args.debug:
        write_debug_units(units_path, units)

    hybrid_stats: Dict[str, Any] = {"mode": "disabled", "llm_calls": 0, "accepted_replans": 0}
    try:
        if args.planner == "hybrid":
            groups, hybrid_stats = hybrid_plan_chunks(
                units,
                profile,
                policy,
                llm_cfg,
                args,
                planner_plans_path if args.debug else None,
                book_plan=book_plan,
            )
            planner_used = "hybrid"
            log(
                f"[PLANNER_HYBRID] [{thread_name}] {source_name} "
                f"groups={hybrid_stats.get('rule_groups')}->{hybrid_stats.get('groups_after')} "
                f"llm_calls={hybrid_stats.get('llm_calls')} "
                f"accepted={hybrid_stats.get('accepted_replans')}"
            )
        elif should_use_llm_planner(profile, args):
            groups = llm_plan_chunks(
                units,
                profile,
                policy,
                llm_cfg,
                args,
                planner_plans_path if args.debug else None,
                book_plan=book_plan,
            )
            planner_used = "llm"
        else:
            groups = rule_pack_units(units, policy, profile)
            planner_used = "rule"
            log(f"[PLANNER_RULE] [{thread_name}] {source_name} groups={len(groups)}")
    except Exception as e:
        append_jsonl(errors_path, {"stage": "plan", "error": str(e)})
        log(f"[ERROR] [{thread_name}] planner failed: {e}")
        if getattr(args, "preset", "") == "sft_dpo_quality":
            raise
        log(f"[ERROR] [{thread_name}] planner fallback=rule")
        groups = rule_pack_units(units, policy, profile)
        planner_used = "rule_fallback"

    # Normalize all planner output into an exact ordered partition before any
    # final assembly. This removes LLM carry overlaps/gaps while retaining its
    # semantic cut positions.
    packing_policy = policy
    if args.routing_mode == "downstream_trust" and generator_contract.hard_max_chars < policy.max_chars:
        packing_policy = ChunkPolicy(
            min_chars=min(policy.min_chars, generator_contract.hard_max_chars),
            target_chars=min(policy.target_chars, generator_contract.hard_max_chars),
            max_chars=generator_contract.hard_max_chars,
            max_pages=policy.max_pages,
            max_units=policy.max_units,
            min_units=policy.min_units,
            target_units=policy.target_units,
            mode=policy.mode + ":downstream_contract",
            allow_cross_section=policy.allow_cross_section,
        )
    groups, group_coverage = normalize_group_coverage(groups, units, packing_policy, profile)
    log(
        f"[GROUP_COVERAGE] [{thread_name}] {source_name} "
        f"raw_groups={group_coverage.get('raw_group_count')} "
        f"raw_dups={group_coverage.get('raw_duplicate_refs')} "
        f"raw_missing={group_coverage.get('raw_missing_units')} "
        f"normalized_groups={group_coverage.get('normalized_group_count')}"
    )

    groups, boundary_repair_stats = repair_group_boundaries(
        groups, packing_policy, generator_contract, args
    )
    log(
        f"[BOUNDARY_REPAIR] [{thread_name}] {source_name} "
        f"groups={boundary_repair_stats.get('groups_before')}->{boundary_repair_stats.get('groups_after')} "
        f"bad={boundary_repair_stats.get('bad_boundaries_before')}->{boundary_repair_stats.get('bad_boundaries_after')} "
        f"merges={boundary_repair_stats.get('merges')} "
        f"rebalances={boundary_repair_stats.get('rebalances')}"
    )

    # Optional structural short attachment. In Full-LLM no-review mode this is
    # disabled so the LLM planner output is not post-merged/revised.
    pre_merge_group_count = len(groups)
    if bool(getattr(args, "trust_merge_short", True)):
        groups = merge_short_groups_for_generator(groups, packing_policy, args)
        if len(groups) != pre_merge_group_count:
            log(
                f"[STRUCTURAL_ATTACH] [{thread_name}] {source_name} "
                f"groups={pre_merge_group_count}->{len(groups)}"
            )
    else:
        log(
            f"[STRUCTURAL_ATTACH_SKIP] [{thread_name}] {source_name} "
            f"groups={pre_merge_group_count} reason=post_planner_revision_disabled"
        )

    short_optimization: Dict[str, Any] = {
        "enabled": False,
        "groups_before": len(groups),
        "groups_after": len(groups),
        "merges": 0,
    }
    if args.routing_mode == "downstream_trust":
        groups, short_optimization = optimize_short_groups_for_downstream(
            groups,
            packing_policy,
            generator_contract,
            args,
        )
        log(
            f"[SHORT_OPTIMIZE] [{thread_name}] {source_name} "
            f"groups={short_optimization.get('groups_before')}->{short_optimization.get('groups_after')} "
            f"merges={short_optimization.get('merges')} "
            f"below_{short_optimization.get('desired_min_chars')}="
            f"{short_optimization.get('below_desired_before')}->{short_optimization.get('below_desired_after')}"
        )
    assert_exact_group_coverage(groups, units)

    if args.debug:
        write_debug_groups(groups_path, groups)

    # Assemble initial chunks deterministically from groups.
    initial_chunks: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for g in groups:
        row = chunk_row(
            doc_id,
            source_name,
            len(initial_chunks),
            g,
            min_alpha_chars=1 if args.routing_mode == "downstream_trust" else 25,
        )
        if not row:
            continue
        if args.routing_mode != "downstream_trust" and not is_training_worthy_chunk(row):
            continue
        th = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()
        if th in seen_hashes and args.routing_mode != "downstream_trust":
            continue
        seen_hashes.add(th)
        row["id"] = f"{doc_id}__{len(initial_chunks):06d}"
        row["stage"] = "initial"
        row["generator_contract"] = asdict(generator_contract)
        row["atomizer_source"] = atomizer_used
        row["planner_source"] = planner_used
        initial_chunks.append(row)

    if args.routing_mode == "downstream_trust":
        expected_uids = [u.uid for u in units if u.text.strip()]
        assembled_uids = [uid for row in initial_chunks for uid in row.get("unit_ids") or []]
        if assembled_uids != expected_uids:
            raise RuntimeError(
                "INITIAL_CHUNK_COVERAGE_INVALID "
                f"expected={len(expected_uids)} assembled={len(assembled_uids)}"
            )

    propagate_missing_sections(initial_chunks)
    apply_context_to_chunks(
        initial_chunks,
        inject_context=args.routing_mode != "downstream_trust",
    )
    if save_audit:
        write_rows(chunks_initial_path, initial_chunks)

    uid_to_unit = {u.uid: u for u in units}

    # Review initial chunks, queue bad/fixable ones, repair, then review repaired chunks.
    reviewed_initial: List[Dict[str, Any]] = list(initial_chunks)
    repair_queue: List[Dict[str, Any]] = []
    repaired_reviewed: List[Dict[str, Any]] = []
    unresolved_repairs: List[Dict[str, Any]] = []

    if args.routing_mode == "downstream_trust":
        # Semantic value is assessed by the downstream SFT/DPO service. Routing
        # metadata is attached after candidate assembly.
        pass
    elif args.review_chunks != "never":
        reviewed_initial, repair_queue = review_chunks_with_llm(
            reviewed_initial, profile, policy, llm_cfg, args,
            review_path=review_path if save_audit else None,
            queue_path=repair_queue_path if save_audit else None,
            stage="initial",
            book_plan=book_plan,
        )
        if args.repair_chunks != "never" and repair_queue:
            chunks_by_id = {str(c.get("id")): c for c in reviewed_initial}
            repaired_chunks, unresolved_repairs = repair_queued_chunks(
                repair_queue, chunks_by_id, uid_to_unit, doc_id, source_name,
                profile, policy, llm_cfg, args,
                repair_plans_path if args.debug else None,
            )
            if unresolved_repairs and save_audit:
                write_rows(repair_unresolved_path, unresolved_repairs)
            if repaired_chunks:
                for row in repaired_chunks:
                    row["generator_contract"] = asdict(generator_contract)
                    row["atomizer_source"] = atomizer_used
                    row["planner_source"] = planner_used
                propagate_missing_sections(repaired_chunks)
                apply_context_to_chunks(repaired_chunks, inject_context=True)
                repaired_reviewed, _repair_queue_2 = review_chunks_with_llm(
                    repaired_chunks, profile, policy, llm_cfg, args,
                    review_path=repair_reviews_path if save_audit else None,
                    queue_path=None,
                    stage="repair",
                    book_plan=book_plan,
                )
                # Do not run infinite repair loops; if repaired chunks still fail, they go to quarantine/drop.
    else:
        for c in reviewed_initial:
            c["quality"] = local_review_for_chunk(c, policy, args)
            c["quality"]["reviewed_by_llm"] = False
            c["quality"]["stage"] = "local_no_review"

    repair_ids = {str(q.get("chunk_id")) for q in repair_queue}
    replaced_chunk_ids: set[str] = set()
    for c in repaired_reviewed:
        rep = c.get("repair") or {}
        ids = rep.get("replaces_chunk_ids") or [rep.get("from_chunk_id")]
        replaced_chunk_ids.update(str(x) for x in ids if x)

    # Do not keep any original chunk that was successfully replaced by a lossless
    # repaired window.  This is essential for merge_prev/merge_next repairs: both
    # originals must disappear or the final dataset would contain overlapping text.
    candidates_for_final: List[Dict[str, Any]] = []
    for c in reviewed_initial:
        cid = str(c.get("id"))
        if cid in replaced_chunk_ids:
            continue
        candidates_for_final.append(c)
    candidates_for_final.extend(repaired_reviewed)

    # Store all reviewed candidates only in audit/debug mode.
    if save_audit:
        write_rows(chunks_all_reviewed_path, candidates_for_final)
    if args.routing_mode == "downstream_trust":
        keep_chunks, dropped_chunks, quarantine_chunks = route_chunks_downstream_trust(
            candidates_for_final,
            args,
        )
    else:
        keep_chunks, dropped_chunks, quarantine_chunks = split_chunks_by_decision(candidates_for_final, args)

    # Stable final IDs for final files, preserving original_id.
    final_chunks: List[Dict[str, Any]] = []
    final_seen: set[str] = set()
    for c in keep_chunks:
        th = hashlib.sha1((c.get("text") or "").encode("utf-8")).hexdigest()
        if th in final_seen and args.routing_mode != "downstream_trust":
            continue
        final_seen.add(th)
        c = dict(c)
        c["original_id"] = c.get("id")
        c["id"] = f"{doc_id}__final_{len(final_chunks):06d}"
        final_chunks.append(c)

    if args.write_task_subsets:
        final_by_text = {hashlib.sha1((c.get("text") or "").encode("utf-8")).hexdigest(): c for c in final_chunks}
        sft_final: List[Dict[str, Any]] = []
        dpo_final: List[Dict[str, Any]] = []
        if args.routing_mode == "downstream_trust":
            # The two downstream branches perform their own value assessment.
            sft_final = list(final_chunks)
            dpo_final = list(final_chunks)
        else:
            # Legacy compatibility: optional only. Generator should normally decide from chunks.jsonl quality metadata.
            for c in keep_chunks:
                q = c.get("quality") or {}
                th = hashlib.sha1((c.get("text") or "").encode("utf-8")).hexdigest()
                fc = final_by_text.get(th)
                if not fc:
                    continue
                if float(q.get("sft_value_score") or 0) >= args.sft_min_score and float(q.get("coherence_score") or 0) >= 3:
                    sft_final.append(fc)
                if float(q.get("dpo_value_score") or 0) >= args.dpo_min_score and float(q.get("evidence_density_score") or 0) >= 3:
                    dpo_final.append(fc)
        write_rows(chunks_sft_path, sft_final)
        write_rows(chunks_dpo_path, dpo_final)
    else:
        sft_final = []
        dpo_final = []

    write_rows(chunks_path, final_chunks)
    if args.routing_mode == "downstream_trust":
        write_rows(chunks_rejected_structural_path, dropped_chunks)
        if save_audit:
            write_rows(chunks_dropped_path, dropped_chunks)
    elif save_audit:
        write_rows(chunks_dropped_path, dropped_chunks)
    if save_audit:
        write_rows(chunks_quarantine_path, quarantine_chunks)
    ledger_counts = write_unit_ledger(unit_ledger_path if save_audit else None, units, final_chunks, dropped_chunks, quarantine_chunks, unresolved_repairs)
    if args.routing_mode == "downstream_trust":
        if int(ledger_counts.get("unaccounted", 0)) != 0:
            raise RuntimeError(f"UNIT_LEDGER_UNACCOUNTED {ledger_counts}")
        if int(ledger_counts.get("quarantine", 0)) != 0:
            raise RuntimeError(f"UNIT_LEDGER_UNEXPECTED_QUARANTINE {ledger_counts}")

    quality = quality_report_for_chunks(final_chunks, lines, policy, args, profile=profile)
    quality["atomic_split"] = atom_split_stats
    quality["noise_isolation"] = noise_isolation
    quality["hybrid_planner"] = hybrid_stats
    quality["boundary_repair"] = boundary_repair_stats
    quality["generator_contract"] = asdict(generator_contract)
    quality["review"] = {
        "enabled": args.review_chunks,
        "repair_enabled": args.repair_chunks,
        "initial_chunks": len(initial_chunks),
        "final_keep_chunks": len(final_chunks),
        "sft_chunks": len(sft_final),
        "dpo_chunks": len(dpo_final),
        "task_subset_files_written": bool(args.write_task_subsets),
        "dropped_chunks": len(dropped_chunks),
        "structural_reject_chunks": len(dropped_chunks) if args.routing_mode == "downstream_trust" else 0,
        "quarantine_chunks": len(quarantine_chunks),
        "repair_queue": len(repair_queue),
        "repaired_chunks": len(repaired_reviewed),
        "unresolved_repairs": len(unresolved_repairs),
        "review_path": str(review_path) if review_path.exists() else None,
        "repair_queue_path": str(repair_queue_path) if repair_queue_path.exists() else None,
        "unit_ledger_path": str(unit_ledger_path) if unit_ledger_path.exists() else None,
        "unit_ledger_counts": ledger_counts,
    }
    if args.routing_mode == "downstream_trust":
        weights = [float((c.get("routing") or {}).get("weight") or 0) for c in final_chunks]
        tiers = Counter(str((c.get("routing") or {}).get("tier") or "unknown") for c in final_chunks)
        flags = Counter(flag for c in final_chunks for flag in ((c.get("routing") or {}).get("flags") or []))
        promoted_count = sum(
            bool((c.get("routing") or {}).get("promoted_from_quarantine"))
            for c in final_chunks
        )
        quality["routing"] = {
            "mode": args.routing_mode,
            "accepted_chunks": len(final_chunks),
            "structural_reject_chunks": len(dropped_chunks),
            "quarantine_chunks": 0,
            "promoted_from_quarantine": promoted_count,
            "weight_min": round(min(weights), 3) if weights else 0.0,
            "weight_avg": round(sum(weights) / len(weights), 3) if weights else 0.0,
            "weight_max": round(max(weights), 3) if weights else 0.0,
            "tiers": dict(tiers),
            "flags": dict(flags),
        }
    # No output chunks is fatal in every routing mode.
    if not final_chunks:
        quality["passed"] = False
        quality.setdefault("fatal", []).append("no_accepted_chunks")
    quality["ingestion"] = ingestion_stats
    quality["group_coverage"] = group_coverage
    quality["short_optimization"] = short_optimization
    write_json(quality_path, quality)

    elapsed = time.time() - started
    document_failed = not final_chunks
    manifest = {
        "doc_id": doc_id,
        "source_name": source_name,
        "source_path": str(folder),
        "status": "failed" if document_failed else "completed",
        "failed": document_failed,
        "error_type": "no_accepted_chunks" if document_failed else None,
        "error": "No structurally readable chunks were produced." if document_failed else None,
        "pages_seen": len(pages),
        "duplicate_pages_dropped": duplicate_pages_dropped,
        "clean_lines": len(lines),
        "atomic_units": len(units),
        "initial_chunks": len(initial_chunks),
        "chunks": len(final_chunks),
        "sft_chunks": len(sft_final),
        "dpo_chunks": len(dpo_final),
        "task_subset_files_written": bool(args.write_task_subsets),
        "dropped_chunks": len(dropped_chunks),
        "structural_reject_chunks": len(dropped_chunks) if args.routing_mode == "downstream_trust" else 0,
        "quarantine_chunks": len(quarantine_chunks),
        "promoted_from_quarantine": int((quality.get("routing") or {}).get("promoted_from_quarantine") or 0),
        "repair_queue": len(repair_queue),
        "repaired_chunks": len(repaired_reviewed),
        "atomizer": atomizer_used,
        "planner": planner_used,
        "routing_mode": args.routing_mode,
        "review": args.review_chunks,
        "repair": args.repair_chunks,
        "quality_passed": quality.get("passed"),
        "quality_issue_count": quality.get("issue_count"),
        "chunks_path": str(chunks_path) if chunks_path.exists() else None,
        "chunks_initial_path": str(chunks_initial_path) if chunks_initial_path.exists() else None,
        "chunks_sft_path": str(chunks_sft_path) if chunks_sft_path.exists() else None,
        "chunks_dpo_path": str(chunks_dpo_path) if chunks_dpo_path.exists() else None,
        "chunks_all_reviewed_path": str(chunks_all_reviewed_path) if chunks_all_reviewed_path.exists() else None,
        "unit_ledger_path": str(unit_ledger_path) if unit_ledger_path.exists() else None,
        "book_plan_path": str(book_plan_path) if book_plan_path.exists() else None,
        "chunks_dropped_path": str(chunks_dropped_path) if chunks_dropped_path.exists() else None,
        "chunks_rejected_structural_path": (
            str(chunks_rejected_structural_path)
            if chunks_rejected_structural_path.exists()
            else None
        ),
        "chunks_quarantine_path": str(chunks_quarantine_path) if chunks_quarantine_path.exists() else None,
        "review_path": str(review_path) if review_path.exists() else None,
        "repair_queue_path": str(repair_queue_path) if repair_queue_path.exists() else None,
        "quality_report_path": str(quality_path),
        "profile_path": str(profile_path) if save_audit else None,
        "debug_dir": str(out_dir / "debug") if args.debug else None,
        "errors_path": str(errors_path) if errors_path.exists() else None,
        "elapsed_sec": round(elapsed, 2),
        "profile": asdict(profile),
        "semantic_shape": semantic_shape,
        "base_chunk_policy": asdict(base_policy),
        "chunk_policy": asdict(policy),
        "generator_contract": asdict(generator_contract),
        "book_plan": book_plan,
        "unit_ledger_counts": ledger_counts,
        "atomic_split": atom_split_stats,
        "noise_isolation": noise_isolation,
        "hybrid_planner": hybrid_stats,
        "boundary_repair": boundary_repair_stats,
        "ingestion": ingestion_stats,
        "group_coverage": group_coverage,
        "short_optimization": short_optimization,
        "routing": quality.get("routing"),
    }
    write_json(manifest_path, manifest)
    log(
        f"[DONE] [{thread_name}] {source_name} pages={len(pages)} lines={len(lines)} units={len(units)} "
        f"initial={len(initial_chunks)} final={len(final_chunks)} "
        f"structural_reject={len(dropped_chunks)} quarantine={len(quarantine_chunks)} "
        f"promoted={int((quality.get('routing') or {}).get('promoted_from_quarantine') or 0)} "
        f"atomizer={atomizer_used} planner={planner_used} routing={args.routing_mode} review={args.review_chunks} "
        f"issues={quality['issue_count']} passed={quality['passed']} elapsed={fmt_duration(elapsed)}"
    )
    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persian OCR semantic chunker v14 full-LLM no-review concurrent edition")
    p.add_argument("--input-folders", nargs="+", default=None, help="یک یا چند فولدر OCR extract شده. هر فولدر یک کتاب محسوب می‌شود.")
    p.add_argument("--input-root", default=None, help="ریشه‌ای که زیرش چند فولدر کتاب وجود دارد. اگر بدهی، همه زیرپوشه‌های مستقیم به عنوان کتاب پردازش می‌شوند.")
    p.add_argument(
        "--preset",
        choices=["downstream_trust", "dataset_quality", "production", "sft_dpo_quality", "balanced", "fast"],
        default="sft_dpo_quality",
        help="sft_dpo_quality: Full-LLM فقط برای BookPlan + Atomizer + Planner؛ بدون Review/Repair/quality-gate پس از chunking. presetهای دیگر برای سازگاری نگه داشته شده‌اند.",
    )
    p.add_argument("--output-root", required=True, help="ریشه خروجی. برای هر کتاب یک فولدر جدا ساخته می‌شود.")
    p.add_argument("--workers", type=int, default=1, help="تعداد کتاب‌های همزمان. داخل هر کتاب ترتیب حفظ می‌شود.")
    p.add_argument("--resume", action="store_true", help="اگر خروجی معتبر کتاب از قبل وجود دارد، دوباره پردازش نکن.")
    p.add_argument("--skip-done", action="store_true", help="معادل عملی --resume برای runهای بزرگ؛ خروجی‌های کامل قبلی را رد می‌کند.")
    p.add_argument("--force", action="store_true", help="حتی اگر خروجی قبلی وجود دارد، کتاب را دوباره پردازش کن.")
    p.add_argument("--resume-require-quality-pass", action="store_true", help="در resume فقط کتاب‌هایی را skip کن که quality_passed=true دارند.")

    p.add_argument("--llm-url", default=os.getenv("CHUNKER_LLM_URL") or os.getenv("VLLM_CHAT_URL") or "http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-urls", default=os.getenv("CHUNKER_LLM_URLS"), help="Comma-separated pool of OpenAI-compatible chat/completions URLs. If set, requests rotate across this pool.")
    p.add_argument("--llm-host", default=os.getenv("CHUNKER_LLM_HOST"), help="Host/IP for building URL pool from --llm-ports, e.g. 192.168.130.220")
    p.add_argument("--llm-ports", default=os.getenv("CHUNKER_LLM_PORTS"), help="Comma-separated ports for --llm-host, e.g. 8200,8300,8400,8500")
    p.add_argument("--llm-model", default=os.getenv("CHUNKER_LLM_MODEL") or os.getenv("VLLM_MODEL_NAME") or "auto", help="Model name, or auto to discover /v1/models per endpoint.")
    p.add_argument("--llm-api-key", default=os.getenv("CHUNKER_LLM_API_KEY") or os.getenv("VLLM_API_KEY"))
    p.add_argument("--llm-timeout", type=int, default=int(os.getenv("CHUNKER_LLM_TIMEOUT", "240")))
    p.add_argument("--llm-max-retries", type=int, default=1)
    p.add_argument("--llm-max-output-tokens", type=int, default=1024, help="سقف خروجی هر LLM call. برای vLLMهایی با context محدود بالا نگذار؛ مقدار زیاد باعث HTTP 400 می‌شود.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--response-format-json", action="store_true")
    p.add_argument("--llm-cache", dest="llm_cache", action="store_true", default=True, help="کش دیسکی برای پاسخ‌های LLM؛ برای resume/retry سرعت را بهتر می‌کند. پیش‌فرض روشن است.")
    p.add_argument("--no-llm-cache", dest="llm_cache", action="store_false", help="خاموش کردن cache دیسکی LLM.")
    p.add_argument("--llm-cache-dir", default=None, help="مسیر cache؛ پیش‌فرض output_root/_llm_cache است.")

    p.add_argument("--atomizer", choices=["auto", "llm", "rule", "never"], default="auto", help="auto/rule/llm. never معادل rule است.")
    p.add_argument("--planner", choices=["auto", "hybrid", "llm", "rule", "never"], default="auto", help="hybrid: rule lossless + چند LLM window فقط روی مرزهای مشکوک؛ llm: کل کتاب.")

    p.add_argument("--min-chars", type=int, default=1000)
    p.add_argument("--target-chars", type=int, default=2500)
    p.add_argument("--max-chars", type=int, default=3600)
    p.add_argument("--max-pages", type=int, default=5, help="در این نسخه بر اساس file_index/page window هم استفاده می‌شود.")
    p.add_argument("--max-units", type=int, default=20)
    p.add_argument("--min-units", type=int, default=3)
    p.add_argument("--target-units", type=int, default=8)
    p.add_argument("--allow-cross-section", action="store_true")
    p.add_argument("--fixed-policy", action="store_true", help="adaptive policy خاموش شود و مقادیر دستی CLI استفاده شود.")
    p.add_argument("--chunk-profile", choices=["generator_long", "standard"], default="generator_long", help="generator_long یعنی chunkهای نهایی بلند/خودبسنده برای سرویس generator؛ standard رفتار نرم‌تر دارد.")
    p.add_argument("--final-min-chars", type=int, default=1200, help="fallback حداقل نهایی وقتی adaptive-final-limits خاموش است.")
    p.add_argument("--final-soft-min-chars", type=int, default=1900, help="fallback حداقل نرم وقتی adaptive-final-limits خاموش است.")
    p.add_argument("--final-min-units", type=int, default=1, help="fallback حداقل unit وقتی adaptive-final-limits خاموش است.")
    p.add_argument("--final-max-pages", type=int, default=7, help="سقف page/file span برای merge کوتاه‌ها.")
    p.add_argument("--final-max-units", type=int, default=28, help="سقف unit برای merge کوتاه‌ها.")
    p.add_argument("--adaptive-final-limits", dest="adaptive_final_limits", action="store_true", default=True, help="حداقل نهایی را بر اساس داستان/شعر/QA/آیتم/فلسفه تطبیق بده. پیش‌فرض روشن است.")
    p.add_argument("--no-adaptive-final-limits", dest="adaptive_final_limits", action="store_false", help="از final-min/soft-min/min-units ثابت CLI استفاده کن.")
    p.add_argument("--generator-hard-max-chars", type=int, default=SAFE_GENERATOR_HARD_MAX_CHARS, help="سقف مطلق متن برای قرارداد سیستم SFT/DPO پایین‌دست؛ نباید از 4000 بیشتر شود.")
    p.add_argument("--allow-short-final", action="store_true", help="اجازه استثنایی برای chunk کوتاه ولی بسیار قوی. پیش‌فرض خاموش است.")
    p.add_argument("--short-exception-score", type=float, default=4.65)
    p.add_argument(
        "--routing-mode",
        choices=["auto", "downstream_trust", "quality_gate"],
        default="auto",
        help="downstream_trust همه chunkهای structurally readable را با weight عبور می‌دهد؛ quality_gate رفتار review/quarantine قدیمی است.",
    )
    p.add_argument("--trust-min-chars", type=int, default=40, help="حداقل سخت متن در downstream_trust؛ فقط fragmentهای تقریباً خالی رد می‌شوند.")
    p.add_argument("--trust-short-shape-min-chars", type=int, default=20, help="حداقل سخت برای شعر، QA، آیتم مستقل و دیالوگ.")
    p.add_argument("--trust-min-alpha-chars", type=int, default=10, help="حداقل حروف/اعداد خوانا برای رد نکردن chunk.")
    p.add_argument(
        "--trust-merge-short",
        dest="trust_merge_short",
        action="store_true",
        default=True,
        help="چانک کوتاه را فقط در صورت پیوستگی ساختاری امن با همسایه ادغام کن (پیش‌فرض روشن).",
    )
    p.add_argument(
        "--no-trust-merge-short",
        dest="trust_merge_short",
        action="store_false",
        help="بهینه‌سازی امن چانک‌های کوتاه را خاموش کن؛ چانک‌ها همچنان حذف یا قرنطینه نمی‌شوند.",
    )
    p.add_argument("--trust-merge-short-chars", type=int, default=1000, help="زیر این اندازه، ادغام امن با همسایه بررسی می‌شود.")
    p.add_argument("--trust-merge-max-chars", type=int, default=3000, help="سقف نتیجه ادغام کوتاه‌ها؛ برای دیده‌شدن کامل‌تر توسط ارزیاب SFT.")
    p.add_argument("--trust-merge-max-pages", type=int, default=4, help="حداکثر بازه فایل/صفحه در ادغام امن کوتاه‌ها.")
    p.add_argument("--trust-merge-max-units", type=int, default=20, help="حداکثر AtomicUnit در ادغام امن کوتاه‌ها.")
    p.add_argument(
        "--trust-reject-html-tables",
        dest="trust_reject_html_tables",
        action="store_true",
        default=True,
        help="HTML table خام را structural reject کن (پیش‌فرض روشن).",
    )
    p.add_argument(
        "--trust-allow-html-tables",
        dest="trust_reject_html_tables",
        action="store_false",
        help="HTML table خام را هم با risk flag عبور بده.",
    )

    p.add_argument("--atomizer-max-input-tokens", type=int, default=1800)
    p.add_argument("--atomizer-max-lines", type=int, default=48)
    p.add_argument("--atomizer-text-limit", type=int, default=240)
    p.add_argument("--atomizer-max-carry-lines", type=int, default=12)
    p.add_argument("--atomizer-carry-keep-lines", type=int, default=4)
    p.add_argument("--atomizer-adaptive-split-depth", type=int, default=4, help="اگر LLM atomizer به خاطر size/timeout fail شد، batch را تا این عمق نصف کند.")
    p.add_argument("--llm-concurrency", type=int, default=16, help="تعداد LLM call موازی داخل هر کتاب برای Atomizer/Planner. در این نسخه پیش‌فرض 16 است.")
    p.add_argument("--atomizer-overlap-lines", type=int, default=8, help="تعداد line زمینه از هر طرف برای Atomizer موازی؛ فقط core هر window commit می‌شود.")
    p.add_argument("--planner-overlap-units", type=int, default=4, help="تعداد AtomicUnit زمینه از هر طرف برای Planner موازی؛ فقط boundaryهای core commit می‌شوند.")
    p.add_argument("--auto-llm-atomizer-confidence", type=float, default=0.82, help="در atomizer auto فقط ساختارهای خیلی نامطمئن زیر این confidence به LLM می‌روند؛ production عملاً rule است.")
    p.add_argument("--strict-llm", action="store_true", help="در preset quality، خطای LLM atomizer را fail کن و rule fallback نزن.")
    p.add_argument("--split-oversized-atoms", dest="split_oversized_atoms", action="store_true", default=True, help="واحدهای اتمیک خیلی بزرگ را قبل از planner بشکن. پیش‌فرض روشن است.")
    p.add_argument("--no-split-oversized-atoms", dest="split_oversized_atoms", action="store_false", help="خاموش کردن split_large_atomic_units.")
    p.add_argument("--max-atomic-chars", type=int, default=2600, help="سقف نرم/ایمن AtomicUnit در full-LLM؛ زنجیره معنایی تا حد hard downstream می‌تواند کمی بزرگ‌تر بماند.")
    p.add_argument("--target-atomic-chars", type=int, default=1800, help="هدف نرم برای split واحد اتمیک بزرگ در full-LLM.")
    p.add_argument("--max-atomic-lines", type=int, default=36, help="سقف نرم تعداد line در AtomicUnit؛ dependencyهای معنایی می‌توانند کمی از آن عبور کنند.")
    p.add_argument("--isolate-ocr-noise", dest="isolate_ocr_noise", action="store_true", default=True, help="نویز OCR طولانی/تکراری را حذف نکن؛ در unit و chunk جدا با risk flag نگه دار.")
    p.add_argument("--no-isolate-ocr-noise", dest="isolate_ocr_noise", action="store_false")

    p.add_argument("--planner-max-input-tokens", type=int, default=3600)
    p.add_argument("--planner-max-units", type=int, default=28)
    p.add_argument("--planner-text-limit", type=int, default=300)
    p.add_argument("--planner-max-carry-units", type=int, default=8)
    p.add_argument("--planner-carry-keep-units", type=int, default=3)
    p.add_argument("--planner-adaptive-split-depth", type=int, default=4, help="در strict quality، اگر planner LLM به خاطر 400/timeout fail شد، batch را چند سطح نصف کند؛ rule fallback نمی‌زند.")
    p.add_argument("--hybrid-max-windows", type=int, default=8, help="حداکثر LLM call محلی برای مرزهای مشکوک هر کتاب در planner=hybrid.")
    p.add_argument("--boundary-repair", dest="boundary_repair", action="store_true", default=True, help="مرزهای وسط جمله/قبل continuation را بدون تغییر متن merge یا rebalance کن.")
    p.add_argument("--no-boundary-repair", dest="boundary_repair", action="store_false")
    p.add_argument("--boundary-repair-max-chars", type=int, default=3300)
    p.add_argument("--boundary-repair-max-pages", type=int, default=5)

    p.add_argument("--expected-min-item", type=int, default=None, help="برای item_bookهای شماره‌دار، مثلاً 1")
    p.add_argument("--expected-max-item", type=int, default=None, help="برای item_bookهای شماره‌دار، مثلاً 1000")
    p.add_argument("--min-item-coverage", type=float, default=0.98)
    p.add_argument("--fail-on-quality-issues", action="store_true")
    p.add_argument("--fail-on-item-marker-loss", dest="fail_on_item_marker_loss", action="store_true", default=True, help="اگر شماره آیتمی موجود در OCR طی chunking گم شد، کتاب fail شود.")
    p.add_argument("--allow-item-marker-loss", dest="fail_on_item_marker_loss", action="store_false")

    # LLM chunk review and repair queue. This is the quality-control layer for SFT/DPO.
    p.add_argument("--review-chunks", choices=["never", "flagged", "all", "llm"], default="flagged", help="all/llm: review every chunk with LLM; flagged: LLM only suspicious chunks; never: local only.")
    p.add_argument("--review-batch-size", type=int, default=2, help="در full-LLM هر batch کوچک است تا reviewer تقریباً کل chunk را ببیند.")
    p.add_argument("--review-text-limit", type=int, default=3600, help="حداکثر کاراکتر قابل مشاهده reviewer؛ نزدیک سقف واقعی chunk برای تشخیص completeness معنایی.")
    p.add_argument("--review-min-score", type=float, default=3.4, help="حداقل score برای نگه داشتن chunk.")
    p.add_argument("--review-keep-score", type=float, default=3.9, help="اگر chunk repair شده ولی بالاتر از این باشد می‌تواند keep شود.")
    p.add_argument("--review-drop-score", type=float, default=2.25, help="زیر این score معمولاً drop/quarantine می‌شود.")
    p.add_argument("--sft-min-score", type=float, default=3.5, help="حداقل sft_value برای chunks_sft.jsonl.")
    p.add_argument("--dpo-min-score", type=float, default=3.5, help="حداقل dpo_value برای chunks_dpo.jsonl.")
    p.add_argument("--repair-chunks", choices=["never", "auto"], default="never", help="chunkهای repairable را یک بار دیگر با علت خطا برای split/reboundary بفرست.")
    p.add_argument("--max-repair-attempts", type=int, default=1)
    p.add_argument("--repair-text-limit", type=int, default=1800)

    p.add_argument("--book-plan", choices=["auto", "llm", "rule", "never"], default="auto", help="یک call ابتدایی برای strategy/policy کتاب بر اساس rule profile + samples.")
    p.add_argument("--book-plan-sample-lines", type=int, default=30, help="نمونه‌های متوازن ابتدا/میانه/انتهای کتاب برای BookPlan.")
    p.add_argument("--book-plan-text-limit", type=int, default=100, help="حداکثر کاراکتر هر line نمونه در BookPlan.")

    p.add_argument("--review-batch-max-chars", type=int, default=8200, help="سقف کاراکتر batch review برای دو chunk تقریباً کامل.")
    p.add_argument("--review-batch-max-tokens", type=int, default=6500, help="سقف تقریبی توکن batch review در full-LLM.")
    p.add_argument("--review-workers", type=int, default=1, help="تعداد review batchهای موازی داخل هر کتاب. برای pool چند endpoint مقدار 2-4 خوب است.")
    p.add_argument("--review-split-depth", type=int, default=4, help="اگر batch review خطا/timeout خورد، تا این عمق نصف شود.")
    p.add_argument("--review-fallback", choices=["fail", "quarantine", "local"], default="fail", help="در full-LLM پیش‌فرض fail است؛ local کیفیت را با heuristic جایگزین می‌کند و توصیه نمی‌شود.")
    p.add_argument("--write-task-subsets", action="store_true", help="اختیاری/legacy: chunks_sft.jsonl و chunks_dpo.jsonl هم بنویسد. پیش‌فرض خاموش است.")
    p.add_argument("--output-mode", choices=["minimal", "audit"], default="minimal", help="minimal فقط chunks.jsonl/manifest/quality/errors را می‌نویسد؛ audit همه فایل‌های review/repair/ledger را هم ذخیره می‌کند.")

    p.add_argument("--debug", action="store_true", help="clean_lines، atomic_units، groups و LLM plans را ذخیره می‌کند.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Normalize the real downstream contract before any per-book work starts.
    args.generator_hard_max_chars = max(
        1800,
        min(int(args.generator_hard_max_chars), SFT_GENERATOR_VISIBLE_CHARS),
    )
    args.max_chars = max(500, min(int(args.max_chars), args.generator_hard_max_chars))
    args.target_chars = max(500, min(int(args.target_chars), args.max_chars))
    args.min_chars = max(200, min(int(args.min_chars), args.target_chars))
    args.max_atomic_chars = max(400, min(int(args.max_atomic_chars), args.generator_hard_max_chars))
    args.target_atomic_chars = max(300, min(int(args.target_atomic_chars), args.max_atomic_chars))
    args.trust_merge_short_chars = max(200, min(int(args.trust_merge_short_chars), args.generator_hard_max_chars))
    args.trust_merge_max_chars = max(
        args.trust_merge_short_chars,
        min(int(args.trust_merge_max_chars), args.generator_hard_max_chars),
    )
    args.trust_merge_max_pages = max(1, int(args.trust_merge_max_pages))
    args.trust_merge_max_units = max(2, int(args.trust_merge_max_units))
    args.hybrid_max_windows = max(0, min(32, int(args.hybrid_max_windows)))
    args.boundary_repair_max_chars = max(
        800, min(int(args.boundary_repair_max_chars), args.generator_hard_max_chars)
    )
    args.boundary_repair_max_pages = max(1, int(args.boundary_repair_max_pages))
    args.llm_concurrency = max(1, min(64, int(getattr(args, "llm_concurrency", 16) or 16)))
    args.atomizer_overlap_lines = max(0, min(32, int(getattr(args, "atomizer_overlap_lines", 8) or 0)))
    args.planner_overlap_units = max(0, min(16, int(getattr(args, "planner_overlap_units", 4) or 0)))

    # Chunk profile tuning. generator_long is the production default for creating
    # strong source chunks for downstream SFT/DPO generation, not short retrieval snippets.
    if getattr(args, "chunk_profile", "generator_long") == "standard":
        args.final_min_chars = min(args.final_min_chars, 500)
        args.final_soft_min_chars = min(args.final_soft_min_chars, 800)
        args.final_min_units = min(args.final_min_units, 1)

    # Preset tuning.
    # downstream_trust compatibility mode: one book-level strategy
    # call, deterministic lossless packing, and no duplicate semantic review.
    if args.preset == "downstream_trust":
        if args.book_plan == "auto":
            args.book_plan = "llm"
        if args.atomizer == "auto":
            args.atomizer = "rule"
        if args.planner == "auto":
            args.planner = "hybrid"
        args.review_chunks = "never"
        args.repair_chunks = "never"
        if args.routing_mode == "auto":
            args.routing_mode = "downstream_trust"
        args.resume = True if not args.force else args.resume

    # dataset_quality is the legacy balanced mode:
    # LLM is the semantic core at book/chunk quality level, not line-by-line.
    # BookPlan + Planner are LLM; Atomizer remains auto/adaptive; Review is targeted
    # via strict local gates; Repair is intentionally disabled in this pass.
    elif args.preset == "dataset_quality":
        if args.book_plan == "auto":
            args.book_plan = "llm"
        # Keep atomizer auto: regular item/QA/prose books use cheap structural units;
        # ambiguous/low-confidence books still call LLM atomizer.
        if args.atomizer == "auto":
            args.atomizer = "auto"
        if args.planner == "auto":
            args.planner = "llm"
        # Targeted LLM review: strict local gates decide which chunks need LLM judging.
        # Clean long generator-ready chunks can skip LLM review; suspicious chunks are batched.
        if args.review_chunks == "all":
            args.review_chunks = "flagged"
        if args.review_fallback == "fail":
            args.review_fallback = "quarantine"
        args.repair_chunks = "never"
        args.resume = True if not args.force else args.resume
    elif args.preset == "production":
        if args.planner == "auto":
            args.planner = "rule"
        if args.atomizer == "auto":
            args.atomizer = "rule"
        if args.book_plan == "auto":
            args.book_plan = "rule"
        if args.review_chunks == "all":
            args.review_chunks = "flagged"
        if args.review_fallback == "fail":
            args.review_fallback = "local"
        args.repair_chunks = "never"
        args.resume = True if not args.force else args.resume
    elif args.preset == "fast":
        if args.planner == "auto":
            args.planner = "rule"
        if args.atomizer == "auto":
            args.atomizer = "rule"
        if args.book_plan == "auto":
            args.book_plan = "never"
        args.review_chunks = "never"
        args.repair_chunks = "never"
    elif args.preset == "balanced":
        if args.planner == "auto":
            args.planner = "llm"
        if args.atomizer == "auto":
            args.atomizer = "llm"
        if args.book_plan == "auto":
            args.book_plan = "llm"
        if args.review_chunks == "all":
            args.review_chunks = "flagged"
        if args.review_fallback == "fail":
            args.review_fallback = "quarantine"
        args.repair_chunks = "auto"
    else:  # sft_dpo_quality / Full-LLM chunk-construction mode
        # Full-LLM means BookPlan + Atomizer + Planner are LLM-driven.
        # IMPORTANT: Project 1 must NOT semantically revise, review, drop, quarantine,
        # merge, or replace planner chunks for downstream SFT construction.
        # Project 2 owns semantic/value assessment.
        args.planner = "llm"
        args.atomizer = "llm"
        args.book_plan = "llm"
        args.review_chunks = "never"
        args.repair_chunks = "never"
        args.review_fallback = "fail"
        args.strict_llm = True

        # Preserve planner output instead of running the quality-gate path.
        if args.routing_mode == "auto":
            args.routing_mode = "downstream_trust"

        # Do not post-merge short planner chunks in this Full-LLM mode.
        args.trust_merge_short = False

        # Do not deterministically revise/rebalance boundaries after the LLM planner.
        # Semantic boundary quality must come from Atomizer/Planner themselves.
        args.boundary_repair = False

    if args.routing_mode == "auto":
        args.routing_mode = "quality_gate"

    input_folders: List[Path] = []
    if args.input_folders:
        input_folders.extend(Path(x).resolve() for x in args.input_folders)
    if args.input_root:
        root = Path(args.input_root).resolve()
        input_folders.extend(sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name))
    # de-duplicate while preserving order
    seen_paths = set()
    input_folders = [p for p in input_folders if not (str(p) in seen_paths or seen_paths.add(str(p)))]
    if not input_folders:
        print("No input folders. Use --input-folders or --input-root", file=sys.stderr)
        return 2
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in input_folders if not p.exists() or not p.is_dir()]
    if missing:
        print("Missing input folders:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 2

    endpoint_urls: List[str] = []
    if args.llm_urls:
        endpoint_urls = [u.strip() for u in str(args.llm_urls).split(",") if u.strip()]
    elif args.llm_host and args.llm_ports:
        host = str(args.llm_host).strip().rstrip("/")
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        ports = [p0.strip() for p0 in str(args.llm_ports).split(",") if p0.strip()]
        endpoint_urls = [f"{host}:{port}/v1/chat/completions" for port in ports]
    if not endpoint_urls:
        endpoint_urls = [args.llm_url]

    models_by_url: Dict[str, str] = {}
    if str(args.llm_model).strip().lower() == "auto":
        models_by_url = discover_models_for_endpoints(endpoint_urls, args.llm_api_key, timeout=min(15, max(3, args.llm_timeout)))
        if not models_by_url:
            print("No model could be discovered from LLM endpoints. Use --llm-model explicitly.", file=sys.stderr)
            return 2
        # A default is still needed; use first discovered. Per-endpoint values override it.
        args.llm_model = next(iter(models_by_url.values()))

    llm_cache_dir = Path(args.llm_cache_dir).resolve() if args.llm_cache_dir else (output_root / "_llm_cache")
    if args.llm_cache:
        llm_cache_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMConfig(
        url=endpoint_urls[0],
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_sec=args.llm_timeout,
        max_retries=args.llm_max_retries,
        temperature=args.temperature,
        max_output_tokens=args.llm_max_output_tokens,
        response_format_json=args.response_format_json,
        urls=endpoint_urls,
        models_by_url=models_by_url,
        cache_dir=llm_cache_dir if args.llm_cache else None,
        cache_enabled=bool(args.llm_cache),
    )

    log(f"[RUN] input_folders={len(input_folders)} workers={args.workers}")
    log(f"[RUN] output_root={output_root}")
    log(f"[RUN] llm_url_pool={','.join(endpoint_urls)}")
    if models_by_url:
        log(f"[RUN] llm_models_by_url={json.dumps(models_by_url, ensure_ascii=False)}")
    else:
        log(f"[RUN] llm_model={args.llm_model}")
    log(f"[RUN] version=v14_full_llm_noreview_c16_plannerclosurefix1 atomizer={args.atomizer} planner={args.planner} routing={args.routing_mode} review={args.review_chunks} repair={args.repair_chunks} preset={args.preset} chunk_profile={args.chunk_profile} llm_concurrency={args.llm_concurrency} atom_overlap={args.atomizer_overlap_lines} planner_overlap={args.planner_overlap_units}")
    log(f"[RUN] llm_cache={args.llm_cache} cache_dir={llm_cache_dir if args.llm_cache else None}")
    log(f"[RUN] runtime llm_max_output_tokens={args.llm_max_output_tokens} planner_max_input_tokens={args.planner_max_input_tokens} planner_max_units={args.planner_max_units} planner_text_limit={args.planner_text_limit} review_batch_size={args.review_batch_size} review_batch_max_tokens={args.review_batch_max_tokens}")

    manifests: List[Dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {ex.submit(process_folder, folder, output_root, llm_cfg, args): folder for folder in input_folders}
        for fut in as_completed(future_map):
            folder = future_map[fut]
            try:
                manifest = fut.result()
                manifests.append(manifest)
                if manifest.get("failed"):
                    log(
                        f"[FAILED] {folder.name}: "
                        f"{manifest.get('error_type')} {manifest.get('error')}"
                    )
                else:
                    log(
                        f"[OK] {folder.name}: chunks={manifest['chunks']} "
                        f"units={manifest['atomic_units']} atomizer={manifest['atomizer']} "
                        f"planner={manifest['planner']} routing={manifest.get('routing_mode')} "
                        f"passed={manifest['quality_passed']}"
                    )
            except Exception as e:
                # Production-scale diagnostics: persist the exact uncaught
                # traceback beside the failed book so one bad item can be
                # diagnosed and retried independently in a very large corpus.
                import traceback

                tb = traceback.format_exc()
                error_type = type(e).__name__
                resolved_folder = folder.resolve()
                failed_doc_id = stable_doc_id(resolved_folder)
                failed_out_dir = output_root / failed_doc_id
                failed_out_dir.mkdir(parents=True, exist_ok=True)
                failed_errors_path = failed_out_dir / "errors.jsonl"
                failed_traceback_path = failed_out_dir / "fatal_traceback.log"

                stage = "uncaught_exception"
                tb_lower = tb.lower()
                if "llm_book_plan" in tb_lower:
                    stage = "book_plan"
                elif "llm_atomize" in tb_lower or "_atomize_candidate_adaptive" in tb_lower:
                    stage = "atomizer"
                elif "llm_plan_chunks" in tb_lower or "_plan_candidate_adaptive" in tb_lower:
                    stage = "planner"
                elif "load_pages_sorted" in tb_lower or "pages_from_json" in tb_lower:
                    stage = "ingestion"
                elif "finalize" in tb_lower or "quality" in tb_lower:
                    stage = "finalization_or_quality"

                error_record = {
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage": stage,
                    "error_type": error_type,
                    "error": str(e),
                    "error_repr": repr(e),
                    "source_name": folder.name,
                    "source_path": str(resolved_folder),
                    "traceback": tb,
                }
                try:
                    append_jsonl(failed_errors_path, error_record)
                    failed_traceback_path.write_text(tb, encoding="utf-8")
                except Exception as diag_exc:
                    log(f"[DIAGNOSTIC_WRITE_ERROR] {folder}: {diag_exc}")

                # Mark the retry as failed even if an older successful manifest
                # happens to exist, so --skip-done cannot hide this failure.
                failure_manifest = {
                    "doc_id": failed_doc_id,
                    "source_name": folder.name,
                    "source_path": str(resolved_folder),
                    "status": "failed",
                    "failed": True,
                    "error_type": error_type,
                    "error_stage": stage,
                    "error": str(e),
                    "chunks": 0,
                    "quality_passed": False,
                    "errors_path": str(failed_errors_path),
                    "fatal_traceback_path": str(failed_traceback_path),
                }
                try:
                    write_json(failed_out_dir / "manifest.json", failure_manifest)
                except Exception as manifest_exc:
                    log(f"[FAILED_MANIFEST_WRITE_ERROR] {folder}: {manifest_exc}")

                log(
                    f"[FAILED] {folder} stage={stage} type={error_type}: {e}\n"
                    f"[TRACEBACK_BEGIN] {folder.name}\n{tb}[TRACEBACK_END] {folder.name}"
                )
                manifests.append(failure_manifest)

    summary_path = output_root / "run_manifest.json"
    failed_count = sum(1 for manifest in manifests if manifest.get("failed"))
    write_json(summary_path, {
        "input_count": len(input_folders),
        "workers": args.workers,
        "elapsed_sec": round(time.time() - started, 2),
        "completed_count": len(manifests) - failed_count,
        "failed_count": failed_count,
        "documents": manifests,
    })
    log(f"[RUN_DONE] manifest={summary_path} completed={len(manifests)-failed_count} failed={failed_count}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
