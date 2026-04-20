"""Voxel Mathematical Structure Extractor.

Extracts mathematical semantics from voxel data used in computational simulations.

Key capabilities:
- Voxel grid as discretization domain with scale mapping
- Boundary condition numerical implementations (e.g., LBM bounce-back)
- Voxel field to continuous mathematics interpolation rules
- Geometric to mathematical semantic transformation

This enables LLMs to understand voxel-based simulations as mathematical structures
rather than just geometric data.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class VoxelGridType(Enum):
    """Types of voxel grids."""
    UNIFORM = "uniform"           # Regular cubic voxels
    NON_UNIFORM = "non_uniform"   # Variable spacing
    OCTREE = "octree"             # Hierarchical adaptive
    STAGGERED = "staggered"       # Staggered for velocities


class BoundaryScheme(Enum):
    """Numerical boundary condition schemes."""
    BOUNCE_BACK = "bounce_back"           # LBM standard
    SPECULAR_REFLECTION = "specular"      # Slip walls
    PERIODIC = "periodic"                 # Periodic BC
    OUTFLOW = "outflow"                   # Zero gradient
    INLET = "inlet"                       # Dirichlet for velocity
    CURVED_BOUNDARY = "curved"            # Interpolated bounce-back
    IMMERSED_BOUNDARY = "immersed"        # IBM approach


class InterpolationMethod(Enum):
    """Voxel to continuous interpolation methods."""
    NEAREST_NEIGHBOR = "nearest"
    TRILINEAR = "trilinear"
    TRICUBIC = "tricubic"
    SPECTRAL = "spectral"
    SHEPARD = "shepard"           # Inverse distance weighting
    RADIAL_BASIS = "rbf"


@dataclass
class VoxelField:
    """Represents a field defined on voxel grid."""
    name: str
    data: np.ndarray
    grid_type: VoxelGridType
    physical_origin: Tuple[float, float, float]
    voxel_size: Union[float, Tuple[float, float, float]]
    quantity_type: str              # scalar, vector, tensor
    interpolation_method: InterpolationMethod = InterpolationMethod.TRILINEAR
    boundary_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoxelBoundary:
    """Boundary condition in voxel context."""
    name: str
    location: str                   # xmin, xmax, ymin, ymax, zmin, zmax, or internal
    scheme: BoundaryScheme
    voxel_indices: np.ndarray       # Indices of boundary voxels
    mathematical_form: str          # Mathematical expression
    numerical_implementation: str   # How it's implemented in voxels
    ghost_layers: int = 1


@dataclass
class ScaleMapping:
    """Mapping from voxel indices to physical coordinates."""
    physical_origin: Tuple[float, float, float]
    voxel_size: Tuple[float, float, float]
    physical_dimensions: Tuple[float, float, float]
    index_to_physical_matrix: np.ndarray
    physical_to_index_matrix: np.ndarray
    
    def index_to_physical(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Convert voxel indices to physical coordinates."""
        x = self.physical_origin[0] + i * self.voxel_size[0]
        y = self.physical_origin[1] + j * self.voxel_size[1]
        z = self.physical_origin[2] + k * self.voxel_size[2]
        return (x, y, z)
    
    def physical_to_index(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[int, int, int]:
        """Convert physical coordinates to voxel indices."""
        i = int((x - self.physical_origin[0]) / self.voxel_size[0])
        j = int((y - self.physical_origin[1]) / self.voxel_size[1])
        k = int((z - self.physical_origin[2]) / self.voxel_size[2])
        return (i, j, k)


class VoxelMathExtractor:
    """Extracts mathematical structures from voxel data.
    
    Converts geometric voxel representations into mathematical semantics
    that LLMs can understand and reason about.
    
    Example:
        ```python
        extractor = VoxelMathExtractor()
        
        # Load voxel data
        voxel_data = np.load('simulation_grid.npy')
        
        # Extract mathematical structure
        math_structure = extractor.extract(
            voxel_data=voxel_data,
            physical_origin=(0.0, 0.0, 0.0),
            voxel_size=0.1,
            simulation_type='lattice_boltzmann',
        )
        
        # Get discretization schema
        print(math_structure['discretization'])
        
        # Get boundary conditions
        print(math_structure['boundary_conditions'])
        ```
    """
    
    def __init__(self):
        self.voxel_fields: List[VoxelField] = []
        self.boundaries: List[VoxelBoundary] = []
        self.scale_mapping: Optional[ScaleMapping] = None
        
    def extract(
        self,
        voxel_data: np.ndarray,
        physical_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Union[float, Tuple[float, float, float]] = 1.0,
        simulation_type: str = 'generic',
        boundary_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract mathematical structure from voxel data.
        
        Args:
            voxel_data: 3D or 4D array (nx, ny, nz) or (nx, ny, nz, n_components)
            physical_origin: Physical coordinates of voxel (0,0,0)
            voxel_size: Physical size of each voxel
            simulation_type: Type of simulation ('lattice_boltzmann', 'fdtd', 'fvm', etc.)
            boundary_mask: Optional mask indicating boundary voxels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with extracted mathematical structures
        """
        result = {
            'grid_info': {},
            'discretization': {},
            'boundary_conditions': [],
            'interpolation_rules': {},
            'mathematical_semantics': {},
        }
        
        # Extract grid information
        result['grid_info'] = self._extract_grid_info(
            voxel_data, physical_origin, voxel_size
        )
        
        # Extract scale mapping
        result['scale_mapping'] = self._extract_scale_mapping(
            voxel_data.shape, physical_origin, voxel_size
        )
        
        # Extract discretization semantics
        result['discretization'] = self._extract_discretization_semantics(
            voxel_data, simulation_type
        )
        
        # Extract boundary conditions
        if boundary_mask is not None or simulation_type == 'lattice_boltzmann':
            result['boundary_conditions'] = self._extract_boundary_conditions(
                voxel_data, boundary_mask, simulation_type
            )
        
        # Extract interpolation rules
        result['interpolation_rules'] = self._extract_interpolation_rules(
            voxel_data, simulation_type
        )
        
        # Extract mathematical semantics
        result['mathematical_semantics'] = self._extract_mathematical_semantics(
            voxel_data, simulation_type
        )
        
        return result
    
    def _extract_grid_info(
        self,
        voxel_data: np.ndarray,
        physical_origin: Tuple[float, float, float],
        voxel_size: Union[float, Tuple[float, float, float]],
    ) -> Dict[str, Any]:
        """Extract basic grid information."""
        shape = voxel_data.shape
        
        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        
        return {
            'dimensions': shape[:3],
            'num_voxels': shape[0] * shape[1] * shape[2],
            'num_components': shape[3] if len(shape) > 3 else 1,
            'physical_origin': physical_origin,
            'voxel_size': voxel_size,
            'physical_dimensions': (
                shape[0] * voxel_size[0],
                shape[1] * voxel_size[1],
                shape[2] * voxel_size[2],
            ),
            'grid_type': VoxelGridType.UNIFORM.value,
            'total_volume': shape[0] * shape[1] * shape[2] * voxel_size[0] * voxel_size[1] * voxel_size[2],
        }
    
    def _extract_scale_mapping(
        self,
        shape: Tuple[int, ...],
        physical_origin: Tuple[float, float, float],
        voxel_size: Union[float, Tuple[float, float, float]],
    ) -> Dict[str, Any]:
        """Extract scale mapping between index and physical space."""
        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        
        nx, ny, nz = shape[:3]
        
        # Create transformation matrices
        # Index to physical: x = origin + i * dx
        index_to_phys = np.array([
            [voxel_size[0], 0, 0, physical_origin[0]],
            [0, voxel_size[1], 0, physical_origin[1]],
            [0, 0, voxel_size[2], physical_origin[2]],
            [0, 0, 0, 1],
        ])
        
        # Physical to index: i = (x - origin) / dx
        phys_to_index = np.array([
            [1/voxel_size[0], 0, 0, -physical_origin[0]/voxel_size[0]],
            [0, 1/voxel_size[1], 0, -physical_origin[1]/voxel_size[1]],
            [0, 0, 1/voxel_size[2], -physical_origin[2]/voxel_size[2]],
            [0, 0, 0, 1],
        ])
        
        return {
            'transformation_type': 'linear_affine',
            'physical_origin': physical_origin,
            'voxel_size': voxel_size,
            'index_to_physical_matrix': index_to_phys.tolist(),
            'physical_to_index_matrix': phys_to_index.tolist(),
            'mathematical_expression': {
                'index_to_physical': 'r_phys = r_origin + i * Δr',
                'physical_to_index': 'i = (r_phys - r_origin) / Δr',
            },
            'discretization_semantics': {
                'continuum_limit': 'Δr → 0 as N → ∞',
                'resolution_refinement': 'Halving Δr doubles resolution in each dimension',
            },
        }
    
    def _extract_discretization_semantics(
        self,
        voxel_data: np.ndarray,
        simulation_type: str,
    ) -> Dict[str, Any]:
        """Extract discretization semantics."""
        shape = voxel_data.shape
        
        discretization = {
            'domain_type': 'voxel_grid',
            'spatial_discretization': {
                'type': 'finite_volume_like',
                'order': 1,  # First order in space
                'stencil_width': 1,  # Nearest neighbor coupling
            },
            'mathematical_interpretation': {
                'integral_form': '∫_Ω f(r) dV ≈ Σᵢ fᵢ ΔVᵢ',
                'differential_form': '∇f ≈ (f_{i+1} - f_{i-1}) / (2Δx)',
            },
        }
        
        if simulation_type == 'lattice_boltzmann':
            discretization.update({
                'method': 'Lattice Boltzmann Method (LBM)',
                'discretization_approach': 'discrete_velocity_boltzmann',
                'velocity_space': 'discrete_lattice',
                'time_marching': 'explicit_streaming_and_collision',
                'stencil': self._detect_lbm_stencil(voxel_data),
            })
        elif simulation_type == 'fdtd':
            discretization.update({
                'method': 'Finite-Difference Time-Domain',
                'discretization_approach': 'finite_difference',
                'staggered_grid': True,
                'yee_grid': True,
            })
        elif simulation_type == 'fvm':
            discretization.update({
                'method': 'Finite Volume Method',
                'discretization_approach': 'conservation_form',
                'cell_centered': True,
            })
        
        return discretization
    
    def _detect_lbm_stencil(self, voxel_data: np.ndarray) -> Dict[str, Any]:
        """Detect LBM lattice stencil from data."""
        # D3Q19 is most common for 3D
        # D3Q27 for more isotropy
        # D3Q15 for efficiency
        
        return {
            'likely_stencil': 'D3Q19',
            'dimensions': 3,
            'velocities': 19,
            'discrete_velocities': {
                'rest': 1,
                'face_neighbors': 6,
                'edge_neighbors': 12,
            },
            'weights': {
                'rest': '4/9',
                'face': '1/9',
                'edge': '1/36',
            },
        }
    
    def _extract_boundary_conditions(
        self,
        voxel_data: np.ndarray,
        boundary_mask: Optional[np.ndarray],
        simulation_type: str,
    ) -> List[Dict[str, Any]]:
        """Extract boundary condition numerical implementations."""
        boundaries = []
        shape = voxel_data.shape
        
        # Domain boundaries
        face_boundaries = [
            ('xmin', (0, slice(None), slice(None))),
            ('xmax', (shape[0]-1, slice(None), slice(None))),
            ('ymin', (slice(None), 0, slice(None))),
            ('ymax', (slice(None), shape[1]-1, slice(None))),
            ('zmin', (slice(None), slice(None), 0)),
            ('zmax', (slice(None), slice(None), shape[2]-1)),
        ]
        
        for name, idx in face_boundaries:
            boundary = self._create_boundary_description(
                name, idx, simulation_type
            )
            boundaries.append(boundary)
        
        # Internal boundaries (if mask provided)
        if boundary_mask is not None:
            internal = self._extract_internal_boundaries(
                boundary_mask, simulation_type
            )
            boundaries.extend(internal)
        
        return boundaries
    
    def _create_boundary_description(
        self,
        location: str,
        slice_idx: Tuple,
        simulation_type: str,
    ) -> Dict[str, Any]:
        """Create boundary description with numerical implementation."""
        
        if simulation_type == 'lattice_boltzmann':
            return {
                'location': location,
                'type': 'wall',
                'numerical_scheme': BoundaryScheme.BOUNCE_BACK.value,
                'mathematical_interpretation': 'u = 0 (no-slip)',
                'numerical_implementation': {
                    'algorithm': 'bounce_back',
                    'description': 'Incoming populations are reversed at boundary',
                    'equation': "f_{\bar{i}}(x_b, t+Δt) = f_i^*(x_b, t)",
                    'staircase_artifact': 'Geometry approximated as staircase',
                    'ghost_layers': 0,
                },
                'accuracy': '1st_order_in_space',
                'alternatives': [
                    {
                        'name': 'curved_boundary',
                        'description': 'Bouzidi et al. interpolated bounce-back',
                        'accuracy': '2nd_order',
                    },
                    {
                        'name': 'immersed_boundary',
                        'description': 'Explicit forcing for curved surfaces',
                        'accuracy': '1st_order',
                    },
                ],
            }
        elif simulation_type == 'fdtd':
            return {
                'location': location,
                'type': 'pml_or_metal',
                'numerical_scheme': BoundaryScheme.OUTFLOW.value,
                'mathematical_interpretation': 'Absorbing or perfect conductor',
                'numerical_implementation': {
                    'algorithm': 'pml_or_pec',
                    'description': 'Perfectly matched layer or perfect electric conductor',
                },
            }
        else:
            return {
                'location': location,
                'type': 'generic',
                'numerical_scheme': BoundaryScheme.PERIODIC.value,
                'mathematical_interpretation': 'Periodic or Dirichlet/Neumann',
            }
    
    def _extract_internal_boundaries(
        self,
        boundary_mask: np.ndarray,
        simulation_type: str,
    ) -> List[Dict[str, Any]]:
        """Extract internal boundary surfaces."""
        # Find boundary voxels
        boundary_voxels = np.argwhere(boundary_mask)
        
        return [{
            'location': 'internal',
            'type': 'obstacle_or_interface',
            'num_boundary_voxels': len(boundary_voxels),
            'numerical_scheme': BoundaryScheme.CURVED_BOUNDARY.value if simulation_type == 'lattice_boltzmann' else 'immersed_boundary',
            'mathematical_interpretation': 'Complex geometry embedded in grid',
        }]
    
    def _extract_interpolation_rules(
        self,
        voxel_data: np.ndarray,
        simulation_type: str,
    ) -> Dict[str, Any]:
        """Extract interpolation rules for continuous reconstruction."""
        
        rules = {
            'available_methods': [
                {
                    'name': 'nearest_neighbor',
                    'order': 0,
                    'continuous': False,
                    'formula': 'f(r) = f_{round(r/Δr)}',
                    'use_case': 'Quick visualization, classification',
                },
                {
                    'name': 'trilinear',
                    'order': 1,
                    'continuous': True,
                    'formula': 'f(r) = Σᵢ wᵢ(r) fᵢ, wᵢ = trilinear_weights',
                    'use_case': 'General purpose, C0 continuity',
                },
                {
                    'name': 'tricubic',
                    'order': 3,
                    'continuous': True,
                    'formula': 'f(r) = Σᵢⱼₖ cᵢⱼₖ φᵢ(x)φⱼ(y)φₖ(z)',
                    'use_case': 'Smooth reconstruction, derivative estimation',
                },
            ],
            'default_for_simulation': 'trilinear',
        }
        
        if simulation_type == 'lattice_boltzmann':
            rules['lbm_specific'] = {
                'macroscopic_reconstruction': {
                    'density': 'ρ = Σᵢ fᵢ',
                    'velocity': 'u = (1/ρ) Σᵢ cᵢ fᵢ',
                    'stress': 'σ = -(1-τ/Δt) Σᵢ cᵢcᵢ (fᵢ - fᵢ^eq)',
                },
                'interpolation_for_particles': {
                    'method': 'bilinear_or_trilinear',
                    'purpose': 'Velocity interpolation for Lagrangian particles',
                },
            }
        
        return rules
    
    def _extract_mathematical_semantics(
        self,
        voxel_data: np.ndarray,
        simulation_type: str,
    ) -> Dict[str, Any]:
        """Extract high-level mathematical semantics."""
        
        semantics = {
            'representation_type': 'discrete_field_on_cartesian_grid',
            'continuous_limit': {
                'description': 'As Δx → 0, voxel field converges to continuous field',
                'mathematical_statement': 'f_{voxel}(r) → f_{continuous}(r) as N → ∞',
            },
            'differential_operators': {
                'gradient': {
                    'discrete': '(f_{i+1} - f_{i-1}) / (2Δx)',
                    'accuracy': '2nd_order_central_difference',
                },
                'laplacian': {
                    'discrete': '(f_{i+1} - 2f_i + f_{i-1}) / Δx²',
                    'accuracy': '2nd_order',
                },
                'divergence': {
                    'discrete': 'Σ_{α=x,y,z} (v_{α,i+1} - v_{α,i-1}) / (2Δx_α)',
                    'accuracy': '2nd_order',
                },
            },
            'conservation_properties': {
                'mass': 'Conserved if Σ f_i conserved in LBM collision',
                'momentum': 'Conserved if symmetry preserved',
            },
        }
        
        if simulation_type == 'lattice_boltzmann':
            semantics.update({
                'methodology': 'mesoscopic_kinetic_approach',
                'governing_equation': 'discrete_velocity_boltzmann_equation',
                'collision_operator': 'bgk_or_trt_or_mrt',
                'chapman_enskog_analysis': {
                    'description': 'Recovers Navier-Stokes in hydrodynamic limit',
                    'viscosity': 'ν = (τ - 0.5) c_s² Δt',
                    'sound_speed': 'c_s = 1/√3 (lattice_units)',
                },
            })
        
        return semantics
    
    def generate_llm_prompt(self, extraction_result: Dict[str, Any]) -> str:
        """Generate LLM-friendly prompt from extraction result.
        
        Args:
            extraction_result: Result from extract()
            
        Returns:
            Formatted prompt describing mathematical structure
        """
        grid = extraction_result['grid_info']
        disc = extraction_result['discretization']
        bc_list = extraction_result['boundary_conditions']
        interp = extraction_result['interpolation_rules']
        math = extraction_result['mathematical_semantics']
        
        prompt = f"""# Voxel-Based Simulation Mathematical Structure

## Domain Discretization
- Grid dimensions: {grid['dimensions']} voxels
- Physical domain: {grid['physical_dimensions']}
- Voxel size: {grid['voxel_size']}
- Total voxels: {grid['num_voxels']:,}

## Scale Mapping (Index ↔ Physical)
The voxel grid implements a linear affine transformation:
- Physical coordinate: r_phys = r_origin + i × Δr
- Index coordinate: i = (r_phys - r_origin) / Δr

## Numerical Method
{disc.get('method', 'Voxel-based discretization')}

## Boundary Conditions
"""
        
        for bc in bc_list[:4]:  # Limit to first 4 for brevity
            prompt += f"""
### {bc['location']} Boundary
- Type: {bc['type']}
- Mathematical: {bc['mathematical_interpretation']}
- Numerical: {bc['numerical_scheme']}
- Implementation: {bc['numerical_implementation'].get('description', 'Standard')}
"""
        
        prompt += f"""
## Interpolation for Continuous Reconstruction
Default method: {interp['default_for_simulation']}
Available: {', '.join([m['name'] for m in interp['available_methods'][:3]])}

## Mathematical Semantics
{math.get('representation_type', 'Discrete field representation')}

In the continuum limit (Δx → 0), this voxel field converges to a continuous field.
Differential operators are approximated using finite differences on the voxel stencil.
"""
        
        return prompt
