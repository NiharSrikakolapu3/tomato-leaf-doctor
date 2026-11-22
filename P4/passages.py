"""Fetch the fact sheets and cut them into numbered sentences for retrieval.

The sheets are copyrighted, so only their URLs and section rules live in the
repository. The text is cached locally (default P4/.cache/passages.json, git-ignored)
and each passage is fingerprinted, so a later run can tell when a page changed.
"""
import hashlib
import html
import json
import re
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from sources import SOURCES

DEFAULT_CACHE = Path(__file__).parent / ".cache" / "passages.json"
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "p", "li"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5"}
SKIPPED_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "figcaption", "noscript"}
# Abbreviations that end in a period but do not end a sentence.
NOT_SENTENCE_ENDS = ("e.g.", "i.e.", "Mr.", "Mrs.", "Dr.", "St.", "vs.", "approx.", "Fig.", "No.", "U.S.")


class BlockParser(HTMLParser):
    """Collect headings, paragraphs and list items in page order, skipping captions."""

    def __init__(self):
        super().__init__()
        self.blocks = []
        self.open_tags = []
        self.skip_depth = 0
        self.current = None

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class") or ""
        skipped = tag in SKIPPED_TAGS or "caption" in classes
        self.open_tags.append((tag, skipped))
        if skipped:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS and self.current is None and self.skip_depth == 0:
            self.current = [tag, ""]

    def handle_endtag(self, tag):
        while self.open_tags:
            open_tag, skipped = self.open_tags.pop()
            if skipped:
                self.skip_depth -= 1
            if open_tag == tag:
                break
        if self.current and self.current[0] == tag:
            text = re.sub(r"\s+", " ", html.unescape(self.current[1])).strip()
            if text:
                self.blocks.append((tag, text))
            self.current = None

    def handle_data(self, data):
        if self.current is not None and self.skip_depth == 0:
            self.current[1] += data


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf8", errors="ignore")


def page_blocks(raw_html):
    parser = BlockParser()
    parser.feed(raw_html)
    return parser.blocks


def starts_with(text, phrase):
    return text.lower().startswith(phrase.lower())


def section_blocks(blocks, start, stop):
    """Blocks after the heading that starts with `start`, up to the next `stop` heading."""
    begin = None
    for index, (tag, text) in enumerate(blocks):
        if tag in HEADING_TAGS and starts_with(text, start):
            begin = index + 1
            break
    if begin is None:
        raise ValueError(f"heading not found: {start!r}")

    kept = []
    for tag, text in blocks[begin:]:
        # A stop phrase can be a heading or a lead-in paragraph ("Host Crops: ...").
        if starts_with(text, stop):
            break
        if keep_block(tag, text):
            kept.append(text)
    return kept


def keep_block(tag, text):
    words = text.split()
    if tag in HEADING_TAGS:
        # Keep action subheadings ("Use reflective mulch"). Drop labels, questions, figure and table titles.
        is_title = text.startswith(("Figure", "TABLE", "Table"))
        return len(words) >= 3 and not is_title and not text.endswith("?")
    # Drop photo credits and other fragments without a sentence end.
    return len(words) >= 4 or text.endswith((".", ":"))


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘'\"(])", text)
    sentences = []
    for piece in pieces:
        if sentences and sentences[-1].endswith(NOT_SENTENCE_ENDS):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def passage_sentences(blocks, sections):
    sentences = []
    for start, stop in sections:
        for text in section_blocks(blocks, start, stop):
            sentences.extend(split_sentences(text))
    return sentences


def fingerprint(sentences):
    joined = "\n".join(sentences).encode("utf8")
    return hashlib.sha256(joined).hexdigest()[:16]


def build_passages():
    """Fetch every source page and return {class: passage record}."""
    passages = {}
    for label, source in SOURCES.items():
        blocks = page_blocks(fetch(source.url))
        symptoms = passage_sentences(blocks, source.symptoms)
        care = passage_sentences(blocks, source.care)
        key_sign = [i for i, text in enumerate(symptoms) if text.startswith(source.key_sign)]
        passages[label] = {
            "key_sign": key_sign[0] if source.key_sign and key_sign else None,
            "title": source.title,
            "publisher": source.publisher,
            "url": source.url,
            "fetched": date.today().isoformat(),
            "symptoms": symptoms,
            "care": care,
            "symptoms_sha": fingerprint(symptoms),
            "care_sha": fingerprint(care),
        }
    return passages


def load_passages(cache=DEFAULT_CACHE, refresh=False):
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    passages = build_passages()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(passages, indent=2, ensure_ascii=False) + "\n")
    return passages


if __name__ == "__main__":
    for label, record in load_passages(refresh=True).items():
        print(f"== {label}: {len(record['symptoms'])} symptom and {len(record['care'])} care sentences")
