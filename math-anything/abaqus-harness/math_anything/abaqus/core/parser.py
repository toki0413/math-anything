"""Abaqus input file parser."""

from typing import Dict, Any


class AbaqusInputParser:
    """Parser for Abaqus .inp files."""
    
    def __init__(self):
        self.cards: Dict[str, list] = {}
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse Abaqus input file."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content)
    
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Abaqus input content."""
        cards = {}
        current_card = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('**'):
                continue
            
            if line.startswith('*'):
                card_name = line.split(',')[0][1:].upper()
                current_card = card_name
                if card_name not in cards:
                    cards[card_name] = []
            elif current_card:
                cards[current_card].append(line)
        
        self.cards = cards
        return cards