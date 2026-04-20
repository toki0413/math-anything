"""Ansys extractor for mathematical structure extraction.

Extracts mathematical objects from Ansys FEA results.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class AnsysExtractor:
    """Extractor for Ansys FEA mathematical structures.
    
    Extracts:
    - Stress and strain tensors
    - Displacement fields
    - Natural frequencies and mode shapes
    - Reaction forces
    - Energy quantities
    """
    
    def __init__(self):
        self._current_data: Dict[str, Any] = {}
        
    def extract_stress_field(
        self,
        stress_data: np.ndarray,
        element_centroids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract stress field statistics and invariants.
        
        Args:
            stress_data: Array of stress tensors (n_elements, 6)
                        [Sxx, Syy, Szz, Sxy, Syz, Sxz]
            element_centroids: Optional element centroid coordinates
            
        Returns:
            Dictionary with stress statistics
        """
        if stress_data.shape[0] == 0:
            return {}
        
        # Compute von Mises stress
        sxx, syy, szz = stress_data[:, 0], stress_data[:, 1], stress_data[:, 2]
        sxy, syz, sxz = stress_data[:, 3], stress_data[:, 4], stress_data[:, 5]
        
        von_mises = np.sqrt(
            0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) +
            3 * (sxy**2 + syz**2 + sxz**2)
        )
        
        # Principal stresses
        principal_stresses = self._compute_principal_stresses(stress_data)
        
        # Stress invariants
        I1 = sxx + syy + szz
        I2 = sxx*syy + syy*szz + szz*sxx - sxy**2 - syz**2 - sxz**2
        I3 = (sxx*syy*szz + 2*sxy*syz*sxz - sxx*syz**2 - syy*sxz**2 - szz*sxy**2)
        
        return {
            'von_mises_max': float(np.max(von_mises)),
            'von_mises_min': float(np.min(von_mises)),
            'von_mises_mean': float(np.mean(von_mises)),
            'von_mises_std': float(np.std(von_mises)),
            'principal_max': float(np.max(principal_stresses[:, 0])),
            'principal_min': float(np.min(principal_stresses[:, 2])),
            'invariant_I1_mean': float(np.mean(I1)),
            'invariant_I2_mean': float(np.mean(I2)),
            'invariant_I3_mean': float(np.mean(I3)),
        }
    
    def _compute_principal_stresses(self, stress_data: np.ndarray) -> np.ndarray:
        """Compute principal stresses for each element."""
        n = stress_data.shape[0]
        principal = np.zeros((n, 3))
        
        for i in range(n):
            # Build stress tensor
            sigma = np.array([
                [stress_data[i, 0], stress_data[i, 3], stress_data[i, 5]],
                [stress_data[i, 3], stress_data[i, 1], stress_data[i, 4]],
                [stress_data[i, 5], stress_data[i, 4], stress_data[i, 2]],
            ])
            
            # Eigenvalues (principal stresses)
            eigenvalues = np.linalg.eigvalsh(sigma)
            principal[i] = np.sort(eigenvalues)[::-1]  # Descending order
        
        return principal
    
    def extract_strain_energy(
        self,
        stress_data: np.ndarray,
        strain_data: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract strain energy from stress and strain fields.
        
        Args:
            stress_data: Stress tensors (n_elements, 6)
            strain_data: Strain tensors (n_elements, 6)
            volumes: Element volumes
            
        Returns:
            Dictionary with energy quantities
        """
        # Strain energy density: U = ½ σ : ε
        energy_density = 0.5 * np.sum(stress_data * strain_data, axis=1)
        
        # Total strain energy
        strain_energy = np.sum(energy_density * volumes)
        
        return {
            'total_strain_energy': float(strain_energy),
            'max_energy_density': float(np.max(energy_density)),
            'mean_energy_density': float(np.mean(energy_density)),
        }
    
    def extract_modal_results(
        self,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray,
        mass_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract modal analysis results.
        
        Args:
            frequencies: Natural frequencies (Hz)
            mode_shapes: Mode shape vectors (n_modes, n_dofs)
            mass_matrix: Optional mass matrix for orthogonality check
            
        Returns:
            Dictionary with modal properties
        """
        n_modes = len(frequencies)
        
        results = {
            'num_modes': n_modes,
            'frequencies_hz': frequencies.tolist(),
            'frequencies_rad': (2 * np.pi * frequencies).tolist(),
            'periods': (1.0 / frequencies).tolist(),
        }
        
        # Compute modal mass and participation factors
        if mass_matrix is not None:
            modal_masses = []
            for i in range(n_modes):
                phi = mode_shapes[i]
                m_modal = np.dot(phi, np.dot(mass_matrix, phi))
                modal_masses.append(float(m_modal))
            
            results['modal_masses'] = modal_masses
            results['effective_modal_mass'] = sum(modal_masses)
        
        return results
    
    def extract_contact_info(
        self,
        contact_status: np.ndarray,
        contact_pressure: np.ndarray,
        contact_friction: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract contact mechanics information.
        
        Args:
            contact_status: Contact status array (0=open, 1=sliding, 2=sticking)
            contact_pressure: Contact pressure values
            contact_friction: Friction stress values
            
        Returns:
            Dictionary with contact statistics
        """
        n_contact = len(contact_status)
        
        open_count = np.sum(contact_status == 0)
        sliding_count = np.sum(contact_status == 1)
        sticking_count = np.sum(contact_status == 2)
        
        active = contact_status > 0
        
        return {
            'num_contact_elements': n_contact,
            'open_elements': int(open_count),
            'sliding_elements': int(sliding_count),
            'sticking_elements': int(sticking_count),
            'contact_area_fraction': float(np.sum(active) / n_contact) if n_contact > 0 else 0,
            'max_contact_pressure': float(np.max(contact_pressure[active])) if np.any(active) else 0.0,
            'mean_contact_pressure': float(np.mean(contact_pressure[active])) if np.any(active) else 0.0,
            'max_friction_stress': float(np.max(contact_friction[active])) if np.any(active) else 0.0,
        }
    
    def compute_reaction_forces(
        self,
        constraint_nodes: np.ndarray,
        constraint_dofs: np.ndarray,
        internal_forces: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute reaction forces at constraints.
        
        Args:
            constraint_nodes: Node numbers with constraints
            constraint_dofs: Constrained DOFs
            internal_forces: Internal force vector
            
        Returns:
            Dictionary with reaction force components
        """
        reactions = internal_forces[constraint_dofs]
        
        return {
            'total_reaction_magnitude': float(np.linalg.norm(reactions)),
            'max_reaction': float(np.max(np.abs(reactions))),
            'sum_reactions': float(np.sum(reactions)),
            'reaction_components': reactions.tolist(),
        }
    
    def extract_mesh_quality(
        self,
        element_types: np.ndarray,
        node_coords: np.ndarray,
        element_connectivity: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract mesh quality metrics.
        
        Args:
            element_types: Element type for each element
            node_coords: Node coordinates (n_nodes, 3)
            element_connectivity: Element connectivity
            
        Returns:
            Dictionary with mesh quality statistics
        """
        n_elements = len(element_types)
        
        # Compute aspect ratios (simplified)
        aspect_ratios = []
        for i in range(n_elements):
            nodes = element_connectivity[i]
            coords = node_coords[nodes - 1]  # ANSYS uses 1-based indexing
            
            # Simple edge length calculation
            edges = []
            for j in range(len(coords)):
                for k in range(j+1, len(coords)):
                    edge_len = np.linalg.norm(coords[j] - coords[k])
                    edges.append(edge_len)
            
            if edges:
                aspect_ratios.append(max(edges) / min(edges))
        
        aspect_ratios = np.array(aspect_ratios)
        
        return {
            'num_elements': n_elements,
            'num_nodes': len(node_coords),
            'mean_aspect_ratio': float(np.mean(aspect_ratios)) if len(aspect_ratios) > 0 else 0,
            'max_aspect_ratio': float(np.max(aspect_ratios)) if len(aspect_ratios) > 0 else 0,
            'bad_elements': int(np.sum(aspect_ratios > 10)) if len(aspect_ratios) > 0 else 0,
        }
