"""Abaqus input file parser."""

from typing import Any, Dict, List


class AbaqusInputParser:
    """Parser for Abaqus .inp files."""

    def __init__(self):
        self.cards: Dict[str, List[str]] = {}
        self.headers: Dict[str, List[str]] = {}  # card_name -> list of header lines

    def parse_file(self, filepath: str) -> Dict[str, List[str]]:
        """Parse Abaqus input file."""
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> Dict[str, List[str]]:
        """Parse Abaqus input content."""
        cards: Dict[str, List[str]] = {}
        headers: Dict[str, List[str]] = {}
        current_card = None

        for line in content.split("\n"):
            line = line.strip()

            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                card_name = line.split(",")[0][1:].upper()
                current_card = card_name
                if card_name not in cards:
                    cards[card_name] = []
                    headers[card_name] = []
                headers[card_name].append(line)
            elif current_card:
                cards[current_card].append(line)

        self.cards = cards
        self.headers = headers
        return cards
