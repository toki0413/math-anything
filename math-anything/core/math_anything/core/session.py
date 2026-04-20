"""Session management for Math Anything extraction workflow.

Provides stateful session management with undo/redo capabilities.
"""

import copy
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..schemas import MathSchema


class ExtractionSession:
    """Manages extraction session state with undo/redo.
    
    This class maintains the state of an extraction workflow:
    - Current MathSchema being built
    - History of modifications (for undo/redo)
    - Associated file paths
    
    Example:
        ```python
        session = ExtractionSession()
        session.set_schema(schema, "model.json")
        
        # Make modifications
        session.snapshot("added boundary condition")
        schema.mathematical_model.boundary_conditions.append(bc)
        
        # Undo
        session.undo()
        ```
    """
    
    MAX_HISTORY = 50
    
    def __init__(self):
        self._schema: Optional[MathSchema] = None
        self._schema_path: Optional[str] = None
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._modified: bool = False
        self._created_at: datetime = datetime.now()
    
    @property
    def has_schema(self) -> bool:
        """Check if a schema is loaded."""
        return self._schema is not None
    
    @property
    def schema(self) -> Optional[MathSchema]:
        """Get current schema."""
        return self._schema
    
    @property
    def schema_path(self) -> Optional[str]:
        """Get current schema file path."""
        return self._schema_path
    
    @property
    def is_modified(self) -> bool:
        """Check if schema has been modified since last save."""
        return self._modified
    
    def set_schema(self, schema: MathSchema, path: Optional[str] = None):
        """Set the current schema.
        
        Args:
            schema: MathSchema to set as current.
            path: Optional file path for the schema.
        """
        self._schema = schema
        self._schema_path = path
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._modified = False
    
    def create_new(self, engine_name: str) -> MathSchema:
        """Create a new empty schema.
        
        Args:
            engine_name: Name of the extraction engine.
        
        Returns:
            New empty MathSchema.
        """
        from ..schemas import MetaInfo
        
        schema = MathSchema(
            meta=MetaInfo(
                extracted_by=f"math-anything-{engine_name}",
                extractor_version="0.1.0",
            )
        )
        self.set_schema(schema)
        return schema
    
    def load(self, path: str) -> MathSchema:
        """Load schema from JSON file.
        
        Args:
            path: Path to JSON file.
        
        Returns:
            Loaded MathSchema.
        
        Raises:
            FileNotFoundError: If file does not exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        schema = MathSchema.from_dict(data)
        self.set_schema(schema, path)
        return schema
    
    def save(self, path: Optional[str] = None) -> str:
        """Save schema to JSON file.
        
        Args:
            path: Path to save to. If None, uses current schema_path.
        
        Returns:
            Path where schema was saved.
        
        Raises:
            ValueError: If no path specified and no current path.
        """
        if self._schema is None:
            raise ValueError("No schema to save")
        
        save_path = path or self._schema_path
        if not save_path:
            raise ValueError("No save path specified")
        
        self._schema.save(save_path)
        self._schema_path = save_path
        self._modified = False
        return save_path
    
    def snapshot(self, description: str = ""):
        """Save current state to undo stack.
        
        Call this before making modifications to enable undo.
        
        Args:
            description: Description of the change about to be made.
        """
        if self._schema is None:
            return
        
        state = {
            "schema": copy.deepcopy(self._schema),
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }
        self._undo_stack.append(state)
        
        # Limit history size
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        
        # Clear redo stack on new modification
        self._redo_stack.clear()
        self._modified = True
    
    def undo(self) -> Optional[str]:
        """Undo last modification.
        
        Returns:
            Description of undone action, or None if nothing to undo.
        
        Raises:
            RuntimeError: If no schema loaded or nothing to undo.
        """
        if self._schema is None:
            raise RuntimeError("No schema loaded")
        
        if not self._undo_stack:
            raise RuntimeError("Nothing to undo")
        
        # Save current state to redo stack
        self._redo_stack.append({
            "schema": copy.deepcopy(self._schema),
            "description": "undo point",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Restore previous state
        state = self._undo_stack.pop()
        self._schema = state["schema"]
        self._modified = True
        
        return state.get("description", "")
    
    def redo(self) -> Optional[str]:
        """Redo last undone modification.
        
        Returns:
            Description of redone action, or None if nothing to redo.
        
        Raises:
            RuntimeError: If no schema loaded or nothing to redo.
        """
        if self._schema is None:
            raise RuntimeError("No schema loaded")
        
        if not self._redo_stack:
            raise RuntimeError("Nothing to redo")
        
        # Save current state to undo stack
        self._undo_stack.append({
            "schema": copy.deepcopy(self._schema),
            "description": "redo point",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Restore redo state
        state = self._redo_stack.pop()
        self._schema = state["schema"]
        self._modified = True
        
        return state.get("description", "")
    
    def get_status(self) -> Dict[str, Any]:
        """Get session status information.
        
        Returns:
            Dictionary with session status.
        """
        return {
            "has_schema": self.has_schema,
            "schema_path": self._schema_path,
            "modified": self._modified,
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "created_at": self._created_at.isoformat(),
        }
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get list of undo history entries.
        
        Returns:
            List of history entries with index, description, and timestamp.
        """
        result = []
        for i, state in enumerate(reversed(self._undo_stack)):
            result.append({
                "index": str(i),
                "description": state.get("description", ""),
                "timestamp": state.get("timestamp", ""),
            })
        return result
    
    def export_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Export a specific section of the schema.
        
        Args:
            section: Section name, e.g., 'mathematical_model.governing_equations'.
        
        Returns:
            Dictionary containing the section data, or None if not found.
        """
        if self._schema is None:
            return None
        
        data = self._schema.to_dict()
        parts = section.split(".")
        
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return None
        
        return data