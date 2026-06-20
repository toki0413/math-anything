"""插件发现系统。

使用 Python entry_points 标准机制替代 sys.path hack。
每个引擎通过 pyproject.toml 注册，无需手动添加路径。

使用:
    registry = PluginRegistry()
    registry.discover()
    vasp = registry.get("vasp")
    schema = vasp.extract({"ENCUT": 520})

引擎协议:
    每个引擎插件必须实现 EnginePlugin 协议，
    提供 engine_name, extract(params) → EnhancedMathSchema。
"""

from __future__ import annotations

from functools import partial
from importlib.metadata import entry_points
from typing import Any, Protocol

from .exceptions import PluginNotFoundError


class EnginePlugin(Protocol):
    """引擎插件协议.

    每个引擎实现此协议即可被自动发现。
    """

    @property
    def engine_name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def extract(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def supported_file_types(self) -> list[str]: ...


# ── 内置引擎的简单适配 ──


class _EngineWrapper:
    """包装一个导入路径为插件."""

    def __init__(self, name: str, version: str, module_path: str, extractor_class: str):
        self._name = name
        self._version = version
        self._module_path = module_path
        self._extractor_class = extractor_class
        self._extractor: Any = None

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def _load(self) -> None:
        if self._extractor is not None:
            return
        import importlib

        module = importlib.import_module(self._module_path)
        self._extractor = getattr(module, self._extractor_class)()

    def extract(self, params: dict[str, Any]) -> dict[str, Any]:
        self._load()
        return self._extractor.extract(params)

    def supported_file_types(self) -> list[str]:
        types: dict[str, list[str]] = {
            "vasp": ["INCAR", "POSCAR", "KPOINTS", "POTCAR", "OUTCAR"],
            "lammps": [".in", ".lammps", ".dump"],
            "abaqus": [".inp", ".cae"],
            "ansys": [".inp", ".dat"],
            "comsol": [".mph", ".m", ".java"],
            "gromacs": [".mdp", ".gro", ".top", ".itp"],
            "multiwfn": [".wfn", ".fchk", ".molden"],
            "qe": [".in", ".scf"],
            "solidworks": [".sldprt", ".step"],
            "voxel": [".npy", ".vox"],
            "gaussian": [".gjf", ".com", ".inp"],
        }
        return types.get(self._name, [])


# ── 内置引擎注册表 ──

BUILTIN_ENGINES: dict[str, tuple[str, str]] = {
    "vasp": ("engines.vasp.core.extractor_v2", "VaspExtractor"),
    "lammps": ("engines.lammps.core.extractor", "LammpsExtractor"),
    "abaqus": ("engines.abaqus.core.extractor", "AbaqusExtractor"),
    "ansys": ("engines.ansys.core.input_extractor", "AnsysInputExtractor"),
    "comsol": ("engines.comsol.core.extractor", "ComsolExtractor"),
    "gromacs": ("engines.gromacs.core.extractor", "GromacsExtractor"),
    "multiwfn": ("engines.multiwfn.core.extractor", "MultiwfnExtractor"),
    "qe": ("engines.qe.core.extractor", "QuantumEspressoExtractor"),
    "solidworks": ("engines.solidworks.core.extractor", "SolidworksExtractor"),
    "voxel": ("engines.voxel.core.harness", "VoxelHarness"),
    "openfoam": ("engines.openfoam.core.extractor", "OpenFOAMExtractor"),
    "cp2k": ("engines.cp2k.core.extractor", "CP2KExtractor"),
    "fluent": ("engines.fluent.core.extractor", "FluentExtractor"),
    "su2": ("engines.su2.core.extractor", "SU2Extractor"),
    "gamess": ("engines.gamess.core.extractor", "GAMESSExtractor"),
    "gaussian": ("engines.gaussian.core.extractor", "GaussianExtractor"),
    "nwchem": ("engines.nwchem.core.extractor", "NWChemExtractor"),
    "liggghts": ("engines.liggghts.core.extractor", "LIGGGHTSExtractor"),
    "dakota": ("engines.dakota.core.extractor", "DakotaExtractor"),
}


class PluginRegistry:
    """引擎插件注册表.

    自动发现通过 entry_points 注册的引擎，
    并回退到内置引擎的默认加载。
    """

    GROUP = "math_anything.engines"

    def __init__(self):
        self._plugins: dict[str, EnginePlugin] = {}
        self._discovered = False

    def discover(self) -> dict[str, EnginePlugin]:
        """自动发现所有引擎插件."""
        if self._discovered:
            return self._plugins

        from .logging import get_logger

        logger = get_logger()

        # 首先尝试 entry_points
        try:
            eps = entry_points(group=self.GROUP)
            for ep in eps:
                try:
                    plugin = ep.load()()
                    self._plugins[plugin.engine_name] = plugin
                    logger.plugin_loaded(plugin.engine_name, plugin.version)
                except (ImportError, AttributeError, TypeError) as e:
                    logger.warning(f"Failed to load plugin {ep.name}: {e}")
        except (ImportError, AttributeError):
            pass

        # 回退：加载内置引擎
        if not self._plugins:
            for name, (module_path, class_name) in BUILTIN_ENGINES.items():
                try:
                    wrapper = _EngineWrapper(name, "1.0.0", module_path, class_name)
                    self._plugins[name] = wrapper
                    logger.plugin_loaded(name, "1.0.0")
                except (ImportError, AttributeError, TypeError) as e:
                    logger.warning(f"Failed to load engine {name}: {e}")

        self._discovered = True
        return self._plugins

    def get(self, name: str) -> EnginePlugin:
        """获取指定引擎插件.

        Raises:
            PluginNotFoundError: 引擎不存在
        """
        self.discover()
        plugin = self._plugins.get(name.lower())
        if plugin is None:
            available = ", ".join(sorted(self._plugins.keys()))
            raise PluginNotFoundError(
                detail=f"Engine '{name}' not found.",
                suggestion=f"Available engines: {available}",
            )
        return plugin

    def list_engines(self) -> list[str]:
        self.discover()
        return sorted(self._plugins.keys())

    def register(self, plugin: EnginePlugin) -> None:
        self._plugins[plugin.engine_name] = plugin


# ── 全局单例 ──

_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.discover()
    return _registry


def list_engines() -> list[str]:
    return get_registry().list_engines()


# ── entry_points 可发现的插件工厂 ──
# pyproject.toml 中每个引擎注册为 math_anything.plugin:<name>_plugin，
# PluginRegistry 通过 ep.load()() 调用这些 partial 得到 _EngineWrapper 实例。
for _engine_name, (_module_path, _class_name) in BUILTIN_ENGINES.items():
    globals()[f"{_engine_name}_plugin"] = partial(_EngineWrapper, _engine_name, "1.0.0", _module_path, _class_name)


def get_engine(name: str) -> EnginePlugin:
    return get_registry().get(name)
