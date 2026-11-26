"""Shared text helpers for the 10-K eval: HTML-to-text and span normalization."""
import re
from html.parser import HTMLParser

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
_CELL = {"td", "th"}


class _Text(HTMLParser):
    _HIDDEN = ("script", "style", "ix:header")  # ix:header holds hidden iXBRL facts

    def __init__(self):
        super().__init__()
        self.out, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self._HIDDEN:
            self._skip += 1
        elif tag in _BLOCK:
            self.out.append("\n")
        elif tag in _CELL:
            self.out.append("\t")

    def handle_endtag(self, tag):
        if tag in self._HIDDEN and self._skip:
            self._skip -= 1
        elif tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = "".join(p.out).replace(" ", " ")
    text = re.sub(r"[ \t]*\n[ \t\n]*", "\n", text)
    return text.strip()


def normalize(s: str) -> str:
    """Lowercase, unify quotes and collapse whitespace so spans match across chunkers."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


_TOKEN = re.compile(r"[a-z0-9$%][a-z0-9$%.,()\-]*")


def tokens(s: str) -> list[str]:
    return _TOKEN.findall(normalize(s))
