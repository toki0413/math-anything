"""Tier recommendation system for Math-Anything.

Analyzes input files and recommends appropriate analysis tier based on:
- System size
- Time scale
- Constraint complexity
- Data availability
"""

import os
from typing import List, Optional

from .tiered_schema import (
    AnalysisTier,
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TierRecommendation,
)


class TierRecommender:
    """Recommends analysis tier based on file properties."""

    THRESHOLDS = {
        "small_system": 100,
        "medium_system": 1000,
        "large_system": 10000,
        "short_time": 1e6,
        "medium_time": 1e8,
        "long_time": 1e9,
    }

    def recommend(self, file_analysis: FileAnalysis) -> TierRecommendation:
        """Recommend analysis tier based on file analysis."""

        complexity = self._calculate_complexity(file_analysis)
        suitable_tiers = self._get_suitable_tiers(file_analysis, complexity)
        recommended_tier = self._get_recommended_tier(file_analysis, complexity)
        reasons = self._generate_reasons(file_analysis, recommended_tier)
        resources = self._estimate_resources(recommended_tier, file_analysis)
        upgrade_triggers = self._identify_upgrade_triggers(
            file_analysis, recommended_tier
        )

        return TierRecommendation(
            recommended_tier=recommended_tier,
            suitable_tiers=suitable_tiers,
            reasons=reasons,
            complexity_score=complexity,
            estimated_time=self._format_time(resources.cpu_time_seconds),
            required_resources=resources,
            upgrade_triggers=upgrade_triggers,
        )

    def _calculate_complexity(self, analysis: FileAnalysis) -> ComplexityScore:
        """Calculate complexity score (0-100)."""
        score = ComplexityScore()

        if analysis.num_atoms > 0:
            score.system_size = min(analysis.num_atoms / 200, 50)

        if analysis.simulation_time > 0:
            score.time_scale = min(analysis.simulation_time / 2e7, 30)

        if analysis.has_constraints:
            score.constraints = 20

        if analysis.has_training_data:
            score.data_availability = min(analysis.training_data_size / 500, 10)

        score.total = min(
            score.system_size
            + score.time_scale
            + score.constraints
            + score.data_availability,
            100,
        )

        return score

    def _get_suitable_tiers(
        self, analysis: FileAnalysis, complexity: ComplexityScore
    ) -> List[AnalysisTier]:
        """Get list of suitable tiers."""
        tiers = []

        if complexity.total < 20:
            tiers = [AnalysisTier.BASIC, AnalysisTier.ENHANCED]
        elif complexity.total < 40:
            tiers = [
                AnalysisTier.BASIC,
                AnalysisTier.ENHANCED,
                AnalysisTier.PROFESSIONAL,
            ]
        elif complexity.total < 60:
            tiers = [
                AnalysisTier.ENHANCED,
                AnalysisTier.PROFESSIONAL,
                AnalysisTier.ADVANCED,
            ]
        elif complexity.total < 80:
            if analysis.has_training_data:
                tiers = [
                    AnalysisTier.PROFESSIONAL,
                    AnalysisTier.ADVANCED,
                    AnalysisTier.COMPLETE,
                ]
            else:
                tiers = [AnalysisTier.PROFESSIONAL, AnalysisTier.ADVANCED]
        else:
            if analysis.has_training_data:
                tiers = [AnalysisTier.ADVANCED, AnalysisTier.COMPLETE]
            else:
                tiers = [AnalysisTier.ADVANCED]

        return tiers

    def _get_recommended_tier(
        self, analysis: FileAnalysis, complexity: ComplexityScore
    ) -> AnalysisTier:
        """Get recommended tier."""

        if complexity.total < 20:
            return AnalysisTier.BASIC
        elif complexity.total < 40:
            return AnalysisTier.ENHANCED
        elif complexity.total < 60:
            return AnalysisTier.PROFESSIONAL
        elif complexity.total < 80:
            return AnalysisTier.ADVANCED
        else:
            if analysis.has_training_data:
                return AnalysisTier.COMPLETE
            return AnalysisTier.ADVANCED

    def _generate_reasons(
        self, analysis: FileAnalysis, tier: AnalysisTier
    ) -> List[str]:
        """Generate recommendation reasons."""
        reasons = []

        if analysis.num_atoms > self.THRESHOLDS["large_system"]:
            reasons.append(
                f"Large system ({analysis.num_atoms} atoms), advanced methods recommended"
            )
        elif analysis.num_atoms > self.THRESHOLDS["medium_system"]:
            reasons.append(
                f"Medium system ({analysis.num_atoms} atoms), topology analysis useful"
            )

        if analysis.simulation_time > self.THRESHOLDS["long_time"]:
            reasons.append(
                "Long simulation time, symplectic integrator needed for stability"
            )
        elif analysis.simulation_time > self.THRESHOLDS["medium_time"]:
            reasons.append("Extended simulation, stability considerations important")

        if analysis.has_constraints:
            reasons.append(
                "Constraints detected, manifold methods can reduce dimensionality"
            )

        if (
            analysis.has_training_data
            and analysis.num_atoms > self.THRESHOLDS["medium_system"]
        ):
            reasons.append(
                "Training data available for large system, latent space acceleration possible"
            )

        if not reasons:
            reasons.append(f"Standard analysis suitable for this complexity level")

        return reasons

    def _estimate_resources(
        self, tier: AnalysisTier, analysis: FileAnalysis
    ) -> ResourceRequirements:
        """Estimate resource requirements."""

        base_time = {1: 1.0, 2: 5.0, 3: 30.0, 4: 120.0, 5: 600.0}
        base_memory = {1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}

        time = base_time.get(tier.value, 5.0)
        memory = base_memory.get(tier.value, 1.0)

        if analysis.num_atoms > self.THRESHOLDS["medium_system"]:
            time *= 2
            memory *= 1.5

        gpu_required = (
            tier.value >= 4 and analysis.num_atoms > self.THRESHOLDS["medium_system"]
        )
        gpu_memory = 4.0 if gpu_required else 0.0

        packages = []
        if tier.value >= 3:
            packages.append("gudhi")
        if tier.value >= 4:
            packages.extend(["geomstats", "sympy"])
        if tier.value >= 5:
            packages.extend(["e3nn", "torch"])

        return ResourceRequirements(
            cpu_time_seconds=time,
            memory_gb=memory,
            gpu_required=gpu_required,
            gpu_memory_gb=gpu_memory,
            additional_packages=packages,
        )

    def _identify_upgrade_triggers(
        self, analysis: FileAnalysis, current_tier: AnalysisTier
    ) -> List[str]:
        """Identify conditions that would trigger tier upgrade."""
        triggers = []

        if (
            current_tier.value < 4
            and analysis.simulation_time > self.THRESHOLDS["medium_time"]
        ):
            triggers.append(
                "Long simulation time detected - consider upgrading for symplectic integrator"
            )

        if current_tier.value < 3 and analysis.has_constraints:
            triggers.append(
                "Constraints detected - topology analysis may provide insights"
            )

        if (
            current_tier.value < 5
            and analysis.has_training_data
            and analysis.num_atoms > self.THRESHOLDS["large_system"]
        ):
            triggers.append(
                "Training data available for large system - latent space acceleration possible"
            )

        return triggers

    def _format_time(self, seconds: float) -> str:
        """Format time estimate."""
        if seconds < 1:
            return "< 1 second"
        elif seconds < 60:
            return f"~{int(seconds)} seconds"
        elif seconds < 3600:
            return f"~{int(seconds / 60)} minutes"
        else:
            return f"~{int(seconds / 3600)} hours"


def analyze_file_properties(file_path: str) -> FileAnalysis:
    """Analyze file properties for tier recommendation."""

    analysis = FileAnalysis(file_path=file_path)

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".lmp", ".in"]:
        analysis.file_type = "lammps"
        analysis.engine = "lammps"
    elif ext == ".inp":
        analysis.file_type = "abaqus"
        analysis.engine = "abaqus"
    elif ext in [".mdp", ".gro", ".top"]:
        analysis.file_type = "gromacs"
        analysis.engine = "gromacs"
    elif ext in [".apdl", ".cdb"]:
        analysis.file_type = "ansys"
        analysis.engine = "ansys"
    elif ext == ".mph":
        analysis.file_type = "comsol"
        analysis.engine = "comsol"
    elif ext in [".fchk", ".wfn"]:
        analysis.file_type = "multiwfn"
        analysis.engine = "multiwfn"
    else:
        analysis.file_type = "unknown"

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read().lower()

            if analysis.engine == "lammps":
                for line in content.split("\n"):
                    if "atoms" in line and " #" not in line:
                        try:
                            analysis.num_atoms = int(line.split()[0])
                            break
                        except (ValueError, IndexError):
                            pass

                    if "run" in line and line.strip().startswith("run"):
                        try:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if p == "run" and i + 1 < len(parts):
                                    analysis.simulation_time = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass

                    if "shake" in line or "rigid" in line:
                        analysis.has_constraints = True
                        if "shake" not in analysis.constraint_types:
                            analysis.constraint_types.append("shake")
                    if "fix" in line and "rigid" in line:
                        analysis.has_constraints = True
                        if "rigid" not in analysis.constraint_types:
                            analysis.constraint_types.append("rigid")

        except Exception:
            pass

    return analysis
