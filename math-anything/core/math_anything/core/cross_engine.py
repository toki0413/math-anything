"""Cross-Engine Session - Multi-scale model extraction and coupling.

This module enables extraction from multiple computational engines simultaneously,
identifying coupling interfaces between micro-scale and macro-scale models.

Example:
    ```python
    from math_anything.core import CrossEngineSession

    session = CrossEngineSession()

    # Extract micro-scale (MD)
    session.extract("lammps", "micro", {
        "input": "in.md",
        "scale": "micro",
        "domain": "atomistic"
    })

    # Extract macro-scale (FEM)
    session.extract("abaqus", "macro", {
        "input": "model.inp",
        "scale": "macro",
        "domain": "continuum"
    })

    # Generate coupled schema
    coupled = session.generate_coupled_schema()
    coupled.save("coupled_model.json")
    ```
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schemas import MathSchema, MetaInfo
from .extractor import ExtractorEngine
from .harness import HarnessRegistry


class ModelScale(Enum):
    """Scales of computational modeling."""

    QUANTUM = "quantum"  # Electronic structure (Å, eV)
    ATOMISTIC = "atomistic"  # Molecular dynamics (nm, fs)
    MESOSCALE = "mesoscale"  # Coarse-grained (μm, ns)
    CONTINUUM = "continuum"  # FEM/CFD (mm, s)
    MACRO = "macro"  # Structural (m, hr)


class CouplingType(Enum):
    """Types of coupling between models."""

    HIERARCHICAL = auto()  # One-way: micro -> macro
    CONCURRENT = auto()  # Two-way simultaneous
    SEQUENTIAL = auto()  # Sequential execution
    EMBEDDED = auto()  # Micro embedded in macro


@dataclass
class CouplingInterface:
    """Interface between two models at different scales.

    Identifies how mathematical objects from one scale connect to another.
    """

    interface_id: str
    from_scale: str
    to_scale: str
    from_objects: List[str] = field(default_factory=list)
    to_objects: List[str] = field(default_factory=list)
    mapping_type: str = ""  # e.g., "homogenization", "interpolation"
    transfer_quantity: str = ""  # e.g., "stress", "temperature", "displacement"
    conservation_check: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "from_scale": self.from_scale,
            "to_scale": self.to_scale,
            "from_objects": self.from_objects,
            "to_objects": self.to_objects,
            "mapping_type": self.mapping_type,
            "transfer_quantity": self.transfer_quantity,
            "conservation_check": self.conservation_check,
        }


@dataclass
class ScaleModel:
    """A model at a specific scale."""

    model_id: str
    scale: ModelScale
    engine: str
    schema: MathSchema
    domain: str = ""  # e.g., "bulk", "interface", "defect"
    region: Optional[Dict[str, Any]] = None
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "scale": self.scale.value,
            "engine": self.engine,
            "domain": self.domain,
            "region": self.region,
            "extracted_at": self.extracted_at,
            "schema_summary": {
                "n_equations": len(self.schema.mathematical_model.governing_equations),
                "n_boundary_conditions": len(
                    self.schema.mathematical_model.boundary_conditions
                ),
            },
        }


@dataclass
class CoupledSchema:
    """Schema containing multiple coupled models.

    This extends a standard MathSchema with coupling information
    for multi-scale simulations.
    """

    schema_version: str = "1.0.0+coupled"
    meta: Optional[MetaInfo] = None
    models: Dict[str, ScaleModel] = field(default_factory=dict)
    coupling_interfaces: List[CouplingInterface] = field(default_factory=list)
    coupling_type: CouplingType = CouplingType.HIERARCHICAL

    def __post_init__(self):
        if self.meta is None:
            self.meta = MetaInfo(
                extracted_by="math-anything-cross-engine",
                extractor_version="0.1.0",
            )

    def add_model(self, model: ScaleModel):
        """Add a scale model."""
        self.models[model.model_id] = model

    def add_interface(self, interface: CouplingInterface):
        """Add a coupling interface."""
        self.coupling_interfaces.append(interface)

    def get_model(self, model_id: str) -> Optional[ScaleModel]:
        """Get model by ID."""
        return self.models.get(model_id)

    def get_models_by_scale(self, scale: ModelScale) -> List[ScaleModel]:
        """Get all models at a given scale."""
        return [m for m in self.models.values() if m.scale == scale]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
            "meta": self.meta.to_dict() if self.meta else {},
            "coupling_type": self.coupling_type.name,
            "models": {mid: model.to_dict() for mid, model in self.models.items()},
            "coupling_interfaces": [ci.to_dict() for ci in self.coupling_interfaces],
            "n_models": len(self.models),
            "n_interfaces": len(self.coupling_interfaces),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


class CrossEngineSession:
    """Session for coordinating multi-engine, multi-scale extraction.

    This class manages the extraction of mathematical models from multiple
    computational engines and identifies coupling interfaces between them.

    Example:
        ```python
        session = CrossEngineSession()

        # Extract atomistic model
        session.extract("lammps", "md_bulk", files={"input": "in.md"},
                       scale=ModelScale.ATOMISTIC, domain="bulk")

        # Extract continuum model
        session.extract("abaqus", "fem_structure", files={"input": "model.inp"},
                       scale=ModelScale.CONTINUUM, domain="structure")

        # Auto-detect coupling
        session.auto_detect_coupling()

        # Generate coupled schema
        coupled = session.generate_coupled_schema()
        ```
    """

    def __init__(self):
        self.engine = ExtractorEngine()
        self.models: Dict[str, ScaleModel] = {}
        self.coupling_interfaces: List[CouplingInterface] = []

    def extract(
        self,
        engine_name: str,
        model_id: str,
        files: Dict[str, str],
        scale: ModelScale,
        domain: str = "",
        region: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ScaleModel:
        """Extract a model from an engine.

        Args:
            engine_name: Name of the engine (e.g., 'lammps', 'abaqus')
            model_id: Unique identifier for this model
            files: Dictionary of file paths for extraction
            scale: Scale of the model
            domain: Domain description (e.g., 'bulk', 'interface')
            region: Spatial region definition
            options: Additional extraction options

        Returns:
            ScaleModel instance
        """
        # Extract schema
        schema = self.engine.extract(engine_name, files, options)

        # Create scale model
        model = ScaleModel(
            model_id=model_id,
            scale=scale,
            engine=engine_name,
            schema=schema,
            domain=domain,
            region=region,
        )

        self.models[model_id] = model
        return model

    def auto_detect_coupling(self) -> List[CouplingInterface]:
        """Automatically detect coupling interfaces between models.

        Analyzes mathematical objects from different scales and identifies
        potential coupling points.

        Returns:
            List of detected coupling interfaces
        """
        self.coupling_interfaces = []

        # Get all scale pairs
        model_list = list(self.models.values())

        for i, model_a in enumerate(model_list):
            for model_b in model_list[i + 1 :]:
                # Detect coupling between these two models
                interfaces = self._detect_coupling_between(model_a, model_b)
                self.coupling_interfaces.extend(interfaces)

        return self.coupling_interfaces

    def _detect_coupling_between(
        self, model_a: ScaleModel, model_b: ScaleModel
    ) -> List[CouplingInterface]:
        """Detect coupling between two specific models."""
        interfaces = []

        # Determine direction (micro -> macro)
        scale_order = [
            ModelScale.QUANTUM,
            ModelScale.ATOMISTIC,
            ModelScale.MESOSCALE,
            ModelScale.CONTINUUM,
            ModelScale.MACRO,
        ]

        idx_a = scale_order.index(model_a.scale) if model_a.scale in scale_order else -1
        idx_b = scale_order.index(model_b.scale) if model_b.scale in scale_order else -1

        if idx_a == -1 or idx_b == -1 or idx_a == idx_b:
            return interfaces

        micro_model = model_a if idx_a < idx_b else model_b
        macro_model = model_b if idx_a < idx_b else model_a

        # Check for stress/strain coupling (common in mechanics)
        micro_stress = self._has_quantity(micro_model, "stress")
        macro_stress = self._has_quantity(macro_model, "stress")

        if micro_stress and macro_stress:
            interfaces.append(
                CouplingInterface(
                    interface_id=f"{micro_model.model_id}_{macro_model.model_id}_stress",
                    from_scale=micro_model.model_id,
                    to_scale=macro_model.model_id,
                    from_objects=self._get_quantity_objects(micro_model, "stress"),
                    to_objects=self._get_quantity_objects(macro_model, "stress"),
                    mapping_type="homogenization",
                    transfer_quantity="stress_tensor",
                    conservation_check=True,
                )
            )

        # Check for temperature coupling
        micro_temp = self._has_quantity(micro_model, "temperature")
        macro_temp = self._has_quantity(macro_model, "temperature")

        if micro_temp and macro_temp:
            interfaces.append(
                CouplingInterface(
                    interface_id=f"{micro_model.model_id}_{macro_model.model_id}_thermal",
                    from_scale=micro_model.model_id,
                    to_scale=macro_model.model_id,
                    from_objects=self._get_quantity_objects(micro_model, "temperature"),
                    to_objects=self._get_quantity_objects(macro_model, "temperature"),
                    mapping_type="statistical_average",
                    transfer_quantity="temperature",
                    conservation_check=False,
                )
            )

        # Check for displacement coupling
        micro_disp = self._has_quantity(micro_model, "displacement")
        macro_disp = self._has_quantity(macro_model, "displacement")

        if micro_disp and macro_disp:
            interfaces.append(
                CouplingInterface(
                    interface_id=f"{micro_model.model_id}_{macro_model.model_id}_disp",
                    from_scale=micro_model.model_id,
                    to_scale=macro_model.model_id,
                    from_objects=self._get_quantity_objects(
                        micro_model, "displacement"
                    ),
                    to_objects=self._get_quantity_objects(macro_model, "displacement"),
                    mapping_type="coarse_graining",
                    transfer_quantity="displacement_field",
                    conservation_check=True,
                )
            )

        return interfaces

    def _has_quantity(self, model: ScaleModel, quantity: str) -> bool:
        """Check if model contains a specific quantity."""
        schema = model.schema

        # Check equations
        for eq in schema.mathematical_model.governing_equations:
            if quantity.lower() in eq.mathematical_form.lower():
                return True
            if quantity.lower() in " ".join(eq.variables).lower():
                return True

        # Check boundary conditions
        for bc in schema.mathematical_model.boundary_conditions:
            mo = bc.mathematical_object
            if quantity.lower() in mo.field.lower():
                return True
            if mo.tensor_form and quantity.lower() in mo.tensor_form.lower():
                return True

        return False

    def _get_quantity_objects(self, model: ScaleModel, quantity: str) -> List[str]:
        """Get object IDs related to a quantity."""
        objects = []
        schema = model.schema

        for eq in schema.mathematical_model.governing_equations:
            if quantity.lower() in eq.mathematical_form.lower():
                objects.append(f"equation:{eq.id}")

        for bc in schema.mathematical_model.boundary_conditions:
            if quantity.lower() in bc.mathematical_object.field.lower():
                objects.append(f"bc:{bc.id}")

        return objects

    def add_manual_interface(self, interface: CouplingInterface):
        """Manually add a coupling interface.

        Args:
            interface: CouplingInterface to add
        """
        self.coupling_interfaces.append(interface)

    def generate_coupled_schema(
        self, coupling_type: CouplingType = CouplingType.HIERARCHICAL
    ) -> CoupledSchema:
        """Generate the final coupled schema.

        Args:
            coupling_type: Type of coupling between models

        Returns:
            CoupledSchema containing all models and interfaces
        """
        coupled = CoupledSchema(
            coupling_type=coupling_type,
        )

        # Add all models
        for model in self.models.values():
            coupled.add_model(model)

        # Add all interfaces
        for interface in self.coupling_interfaces:
            coupled.add_interface(interface)

        return coupled

    def get_scale_hierarchy(self) -> List[Tuple[str, ModelScale]]:
        """Get models ordered by scale (micro to macro).

        Returns:
            List of (model_id, scale) tuples
        """
        scale_order = [
            ModelScale.QUANTUM,
            ModelScale.ATOMISTIC,
            ModelScale.MESOSCALE,
            ModelScale.CONTINUUM,
            ModelScale.MACRO,
        ]

        ordered = []
        for scale in scale_order:
            for model in self.models.values():
                if model.scale == scale:
                    ordered.append((model.model_id, scale))

        return ordered

    def check_consistency(self) -> Dict[str, Any]:
        """Check consistency of the coupled system.

        Returns:
            Consistency report with warnings and errors
        """
        report = {
            "consistent": True,
            "warnings": [],
            "errors": [],
            "interfaces_checked": len(self.coupling_interfaces),
        }

        # Check that all interface endpoints exist
        for interface in self.coupling_interfaces:
            if interface.from_scale not in self.models:
                report["errors"].append(
                    f"Interface source not found: {interface.from_scale}"
                )
                report["consistent"] = False

            if interface.to_scale not in self.models:
                report["errors"].append(
                    f"Interface target not found: {interface.to_scale}"
                )
                report["consistent"] = False

        # Check for orphaned models (no interfaces)
        for model_id in self.models:
            connected = any(
                iface.from_scale == model_id or iface.to_scale == model_id
                for iface in self.coupling_interfaces
            )
            if not connected:
                report["warnings"].append(
                    f"Model {model_id} has no coupling interfaces"
                )

        return report

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "n_models": len(self.models),
            "n_interfaces": len(self.coupling_interfaces),
            "scales_present": list(set(m.scale.value for m in self.models.values())),
            "engines_used": list(set(m.engine for m in self.models.values())),
            "model_ids": list(self.models.keys()),
        }
