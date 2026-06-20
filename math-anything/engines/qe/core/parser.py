"""Quantum ESPRESSO input file parser.

Parses pw.x style input files with Fortran namelists and card blocks.
"""

from typing import Any, Dict, List, Optional, Tuple


class QuantumEspressoInputParser:
    """Parser for Quantum ESPRESSO pw.x input files."""

    def __init__(self):
        self.namelists: Dict[str, Dict[str, Any]] = {}
        self.cards: Dict[str, List[str]] = {}

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse QE input file."""
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
        """Parse QE input content."""
        self.namelists = {}
        self.cards = {}

        lines = content.splitlines()
        i = 0
        current_namelist = None

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("!"):
                i += 1
                continue

            # Namelist start
            if line.startswith("&"):
                namelist_name = line[1:].strip().upper()
                current_namelist = namelist_name
                if current_namelist not in self.namelists:
                    self.namelists[current_namelist] = {}
                i += 1
                continue

            # Namelist end
            if line == "/" and current_namelist:
                current_namelist = None
                i += 1
                continue

            # Inside namelist
            if current_namelist:
                # Remove inline comments
                if "!" in line:
                    line = line.split("!")[0].strip()
                # Parse key = value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = self._parse_value(value.strip())
                    self.namelists[current_namelist][key] = value
                i += 1
                continue

            # Card block
            card_tokens = (
                "ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS",
                "CELL_PARAMETERS", "OCCUPATIONS", "CONSTRAINTS",
                "ATOMIC_FORCES", "ADDITIONAL_K_POINTS",
            )
            first_token = line.split()[0].upper() if line.split() else ""
            if first_token in card_tokens:
                card_name = first_token
                self.cards[card_name] = []
                i += 1
                # Read card data until next card, namelist, or empty block
                while i < len(lines):
                    data_line = lines[i].strip()
                    if not data_line or data_line.startswith("!"):
                        i += 1
                        continue
                    if data_line.startswith("&") or data_line == "/":
                        break
                    data_first = data_line.split()[0].upper() if data_line.split() else ""
                    if data_first in card_tokens:
                        break
                    self.cards[card_name].append(data_line)
                    i += 1
                continue

            i += 1

        return {"namelists": self.namelists, "cards": self.cards}

    def _parse_value(self, value: str) -> Any:
        """Parse a Fortran namelist value."""
        value = value.strip().rstrip(",")  # Remove trailing comma

        # String
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]

        # Logical
        upper = value.upper()
        if upper in (".TRUE.", "T", "TRUE"):
            return True
        if upper in (".FALSE.", "F", "FALSE"):
            return False

        # Array notation: celldm(1) = 10.2  (handled by key parsing)
        # But values can also be arrays: starting_magnetization(1) = 0.5

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float (handle Fortran d/D exponent)
        try:
            return float(value.replace("d", "e").replace("D", "E"))
        except ValueError:
            pass

        # Return as string if unparseable
        return value

    def get_namelist(self, name: str) -> Dict[str, Any]:
        """Get a namelist by name (case-insensitive)."""
        return self.namelists.get(name.upper(), {})

    def get_card(self, name: str) -> List[str]:
        """Get a card by name (case-insensitive)."""
        return self.cards.get(name.upper(), [])
