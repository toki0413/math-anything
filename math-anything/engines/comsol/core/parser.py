"""COMSOL parameter file parser.

Parses a simplified COMSOL parameter summary file with *Card syntax.
Users can export parameters from COMSOL or create manually.
"""

from typing import Any, Dict, List


class ComsolJavaParser:
    """Parser for COMSOL model parameter files."""

    def __init__(self):
        self.cards: Dict[str, List[str]] = {}

    def parse_file(self, filepath: str) -> Dict[str, Any]:
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

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse COMSOL parameter content.

        Supports two formats:
        1. Simplified *Card format (like Abaqus)
        2. Java export file with key patterns
        """
        cards: Dict[str, List[str]] = {}
        current_card = None

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("%"):
                continue

            # Simplified *Card format
            if line.startswith("*"):
                card_name = line[1:].split()[0].upper()
                current_card = card_name
                if card_name not in cards:
                    cards[card_name] = []
                continue

            if current_card:
                cards[current_card].append(line)
                continue

            # Java export patterns
            if "model.physics()" in line or "model.physics(" in line:
                self._parse_java_physics(line, cards)
            elif "model.mesh()" in line or "model.mesh(" in line:
                self._parse_java_mesh(line, cards)
            elif "model.study()" in line or "model.study(" in line:
                self._parse_java_study(line, cards)
            elif ".feature(\"lemm1\").set(\"E\"" in line or ".feature(\"lemm1\").set(\"nu\"" in line:
                self._parse_java_material(line, cards)
            elif ".feature(\"fix1\").set(\"U0\"" in line or ".feature(\"bndl1\").set(\"F" in line:
                self._parse_java_boundary(line, cards)

        self.cards = cards
        return cards

    def _parse_java_physics(self, line: str, cards: Dict[str, List[str]]) -> None:
        if "\"solid\"" in line or "\"SolidMechanics\"" in line:
            cards.setdefault("PHYSICS", []).append("type solid_mechanics")
        elif "\"ht\"" in line or "\"HeatTransfer\"" in line:
            cards.setdefault("PHYSICS", []).append("type heat_transfer")
        elif "\"acdc\"" in line or "\"Electromagnetics\"" in line:
            cards.setdefault("PHYSICS", []).append("type electromagnetics")
        elif "\"fluid\"" in line or "\"LaminarFlow\"" in line:
            cards.setdefault("PHYSICS", []).append("type fluid_flow")

    def _parse_java_mesh(self, line: str, cards: Dict[str, List[str]]) -> None:
        if "\"hmax\"" in line:
            try:
                val = line.split("set(\"hmax\"")[1].split(")")[0].replace(",", "").strip().strip('"')
                cards.setdefault("MESH", []).append(f"max_element_size {val}")
            except IndexError:
                pass

    def _parse_java_study(self, line: str, cards: Dict[str, List[str]]) -> None:
        if "\"Stationary\"" in line or "\"stat\"" in line:
            cards.setdefault("STUDY", []).append("analysis_type stationary")
        elif "\"TimeDependent\"" in line or "\"time\"" in line:
            cards.setdefault("STUDY", []).append("analysis_type transient")
        elif "\"Eigenfrequency\"" in line or "\"eig\"" in line:
            cards.setdefault("STUDY", []).append("analysis_type eigenfrequency")

    def _parse_java_material(self, line: str, cards: Dict[str, List[str]]) -> None:
        if "\"E\"" in line:
            try:
                val = line.split("set(\"E\"")[1].split(")")[0].replace(",", "").strip().strip('"')
                cards.setdefault("MATERIAL", []).append(f"youngs_modulus {val}")
            except IndexError:
                pass
        elif "\"nu\"" in line:
            try:
                val = line.split("set(\"nu\"")[1].split(")")[0].replace(",", "").strip().strip('"')
                cards.setdefault("MATERIAL", []).append(f"poisson_ratio {val}")
            except IndexError:
                pass

    def _parse_java_boundary(self, line: str, cards: Dict[str, List[str]]) -> None:
        if "\"U0\"" in line or "\"fix\"" in line:
            cards.setdefault("BOUNDARY", []).append("fixed")
        elif "\"F\"" in line or "\"load\"" in line:
            cards.setdefault("BOUNDARY", []).append("load")
