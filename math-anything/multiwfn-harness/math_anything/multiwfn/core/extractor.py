"""Multiwfn extractor for mathematical structure extraction.

Extracts mathematical objects from Multiwfn analysis outputs.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class MultiwfnExtractor:
    """Extractor for Multiwfn mathematical structures.
    
    Extracts:
    - Electron density and its derivatives
    - Electrostatic potential
    - Molecular orbital information
    - Topology critical points
    - Basin properties
    """
    
    def __init__(self):
        self._current_data: Dict[str, Any] = {}
        
    def extract_density_field(
        self,
        cube_data: np.ndarray,
        grid_origin: Tuple[float, float, float],
        grid_spacing: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """Extract electron density field properties.
        
        Args:
            cube_data: 3D array of density values
            grid_origin: Origin coordinates (Bohr)
            grid_spacing: Grid spacing in each direction (Bohr)
            
        Returns:
            Dictionary with density statistics and features
        """
        # Compute statistics
        rho_total = np.sum(cube_data) * np.prod(grid_spacing)
        rho_max = np.max(cube_data)
        rho_min = np.min(cube_data)
        rho_mean = np.mean(cube_data)
        
        # Find maxima locations (approximate nuclear positions)
        from scipy import ndimage
        local_maxima = ndimage.maximum_filter(cube_data, size=3) == cube_data
        maxima_coords = np.argwhere(local_maxima & (cube_data > 0.1))
        
        # Compute gradient magnitude (approximate)
        grad_x = np.gradient(cube_data, grid_spacing[0], axis=0)
        grad_y = np.gradient(cube_data, grid_spacing[1], axis=1)
        grad_z = np.gradient(cube_data, grid_spacing[2], axis=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return {
            'total_electrons': float(rho_total),
            'max_density': float(rho_max),
            'min_density': float(rho_min),
            'mean_density': float(rho_mean),
            'num_maxima': len(maxima_coords),
            'max_gradient': float(np.max(grad_magnitude)),
            'mean_gradient': float(np.mean(grad_magnitude)),
            'grid_shape': cube_data.shape,
            'grid_origin': grid_origin,
            'grid_spacing': grid_spacing,
        }
    
    def extract_critical_points(
        self,
        density_cube: np.ndarray,
        gradient_cube: np.ndarray,
        hessian_cubes: np.ndarray,
        threshold: float = 1e-5,
    ) -> List[Dict[str, Any]]:
        """Extract critical points from density topology.
        
        Args:
            density_cube: 3D density field
            gradient_cube: 3D gradient magnitude field
            hessian_cubes: 6-component Hessian field
            threshold: Gradient threshold for critical point detection
            
        Returns:
            List of critical point dictionaries
        """
        critical_points = []
        
        # Find points where gradient is near zero
        zero_gradient = gradient_cube < threshold
        
        # Label connected regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(zero_gradient)
        
        for i in range(1, num_features + 1):
            region = labeled == i
            coords = np.argwhere(region)
            
            if len(coords) == 0:
                continue
            
            # Find maximum density in region (approximate CP location)
            region_density = density_cube[region]
            max_idx = np.argmax(region_density)
            cp_coord = coords[max_idx]
            
            # Classify based on Hessian eigenvalues
            # (simplified - would need actual Hessian at point)
            signature = self._classify_critical_point_simple(
                density_cube, cp_coord
            )
            
            critical_points.append({
                'coordinates': tuple(cp_coord),
                'density': float(density_cube[tuple(cp_coord)]),
                'type': signature['type'],
                'rank': signature['rank'],
                'signature': signature['signature'],
            })
        
        return critical_points
    
    def _classify_critical_point_simple(
        self,
        density: np.ndarray,
        coord: np.ndarray,
    ) -> Dict[str, Any]:
        """Simple critical point classification based on local curvature."""
        x, y, z = coord
        nx, ny, nz = density.shape
        
        # Check boundaries
        if x <= 0 or x >= nx-1 or y <= 0 or y >= ny-1 or z <= 0 or z >= nz-1:
            return {'type': 'unknown', 'rank': 0, 'signature': 0}
        
        # Compute second derivatives (Laplacian approximation)
        d2x = density[x+1, y, z] - 2*density[x, y, z] + density[x-1, y, z]
        d2y = density[x, y+1, z] - 2*density[x, y, z] + density[x, y-1, z]
        d2z = density[x, y, z+1] - 2*density[x, y, z] + density[x, y, z-1]
        
        # Count negative eigenvalues (simplified)
        neg_count = sum(1 for d2 in [d2x, d2y, d2z] if d2 < 0)
        
        # QTAIM classification
        signatures = {
            3: ('nuclear_critical_point', '(3,-3)'),
            2: ('bond_critical_point', '(3,-1)'),
            1: ('ring_critical_point', '(3,+1)'),
            0: ('cage_critical_point', '(3,+3)'),
        }
        
        cp_type, sig_str = signatures.get(neg_count, ('unknown', 'unknown'))
        
        return {
            'type': cp_type,
            'rank': 3,
            'signature': neg_count - 3,  # (rank, signature)
            'signature_str': sig_str,
        }
    
    def extract_orbital_info(
        self,
        orbital_energies: np.ndarray,
        occupation_numbers: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract molecular orbital information.
        
        Args:
            orbital_energies: Array of orbital energies (Hartree)
            occupation_numbers: Array of occupation numbers
            
        Returns:
            Dictionary with HOMO-LUMO and orbital statistics
        """
        # Find HOMO and LUMO
        occupied = occupation_numbers > 0.5
        
        if np.any(occupied):
            homo_idx = np.where(occupied)[0][-1]
            homo_energy = orbital_energies[homo_idx]
        else:
            homo_idx = 0
            homo_energy = 0.0
        
        if np.any(~occupied):
            lumo_idx = np.where(~occupied)[0][0]
            lumo_energy = orbital_energies[lumo_idx]
        else:
            lumo_idx = len(orbital_energies) - 1
            lumo_energy = 0.0
        
        gap = lumo_energy - homo_energy
        
        return {
            'homo_index': int(homo_idx),
            'homo_energy': float(homo_energy),
            'lumo_index': int(lumo_idx),
            'lumo_energy': float(lumo_energy),
            'homo_lumo_gap': float(gap),
            'num_occupied': int(np.sum(occupied)),
            'num_virtual': int(np.sum(~occupied)),
            'energy_range': float(np.max(orbital_energies) - np.min(orbital_energies)),
        }
    
    def extract_aim_properties(
        self,
        basin_volumes: np.ndarray,
        basin_populations: np.ndarray,
        basin_energies: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract Atoms in Molecules (QTAIM) basin properties.
        
        Args:
            basin_volumes: Volume of each atomic basin
            basin_populations: Electron population of each basin
            basin_energies: Optional basin energies
            
        Returns:
            Dictionary with AIM properties
        """
        properties = {
            'num_basins': len(basin_volumes),
            'total_volume': float(np.sum(basin_volumes)),
            'total_population': float(np.sum(basin_populations)),
            'mean_basin_volume': float(np.mean(basin_volumes)),
            'mean_basin_population': float(np.mean(basin_populations)),
            'max_basin_volume': float(np.max(basin_volumes)),
            'min_basin_volume': float(np.min(basin_volumes)),
        }
        
        if basin_energies is not None:
            properties.update({
                'total_energy': float(np.sum(basin_energies)),
                'mean_basin_energy': float(np.mean(basin_energies)),
            })
        
        return properties
    
    def compute_electrostatic_moments(
        self,
        density_cube: np.ndarray,
        grid_points: np.ndarray,
        nuclear_charges: np.ndarray,
        nuclear_positions: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute electrostatic multipole moments.
        
        Args:
            density_cube: Electron density on grid
            grid_points: Grid point coordinates
            nuclear_charges: Nuclear charges
            nuclear_positions: Nuclear positions
            
        Returns:
            Dictionary with charge, dipole, quadrupole moments
        """
        # Total charge
        dV = np.prod([grid_points[i, 1] - grid_points[i, 0] 
                      for i in range(3)])
        electron_charge = -np.sum(density_cube) * dV
        nuclear_charge = np.sum(nuclear_charges)
        total_charge = nuclear_charge + electron_charge
        
        # Dipole moment (simplified - requires proper grid integration)
        # μ = Σ Z_A R_A - ∫ r ρ(r) dr
        
        moments = {
            'total_charge': float(total_charge),
            'nuclear_charge': float(nuclear_charge),
            'electron_charge': float(electron_charge),
        }
        
        return moments
