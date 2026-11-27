import re
from dataclasses import dataclass
from pathlib import Path

LABEL = re.compile(r"^(?:[-*]\s+)?(?:\*\*)?([\w \[\]()./'-]{1,64})(?:\*\*)?:(?:\*\*)?\s*(.*)$")
CITE = re.compile(r"\[T(\d+(?:\.\d+)?)\]")


@dataclass(frozen=True)
class Line:
    id: str
    turn: int
    speaker: str
    text: str

    def render(self):
        return f"[T{self.id}] {escape(self.speaker)}: {escape(self.text)}"


def escape(text):
    # Preserve literal transcript delimiters as visible, ordinary characters.
    return CITE.sub(lambda m: "［T" + m[1] + "］", text).replace("<|", "‹|").replace("|>", "|›")


def read_turns(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8-sig")
    else:
        raise ValueError("Expected a PDF, Markdown or TXT transcript")
    turns = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or (path.suffix.lower() == ".md" and re.match(r"^#{1,6}(?:\s|$)", raw)):
            continue
        match = LABEL.match(raw)
        if match:
            turns.append((match[1].strip("* "), match[2].strip()))
        elif turns and turns[-1][0] != "Transcript":
            speaker, previous = turns[-1]
            turns[-1] = (speaker, (previous + " " + raw).strip())
        else:
            turns.append(("Transcript", raw))
    turns = [(speaker, body) for speaker, body in turns if body]
    if not turns:
        raise ValueError("The transcript contains no extractable text")
    return turns


def split_turns(turns, tokens, limit):
    """Split exceptional turns without losing text or original turn identity."""
    lines = []
    for turn, (speaker, text) in enumerate(turns, 1):
        remaining = text
        parts = []
        while len(tokens(remaining)) > limit:
            low, high = 1, len(remaining)
            while low + 1 < high:
                mid = (low + high) // 2
                if len(tokens(remaining[:mid])) <= limit:
                    low = mid
                else:
                    high = mid
            space = remaining.rfind(" ", 0, low + 1)
            cut = space if space > low // 2 else low
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            parts.append(remaining)
        for part, body in enumerate(parts, 1):
            code = str(turn) if len(parts) == 1 else f"{turn}.{part}"
            lines.append(Line(code, turn, speaker, body))
    return lines


def make_chunks(lines, tokens, target):
    chunks, current, size = [], [], 0
    for i, line in enumerate(lines):
        length = len(tokens(line.render() + "\n"))
        if current and size + length > target:
            chunks.append(current)
            current, size = [], 0
        current.append(i)
        size += length
    if current:
        chunks.append(current)
    return chunks
