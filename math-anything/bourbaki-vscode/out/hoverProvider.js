"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.BourbakiHoverProvider = void 0;
const vscode = __importStar(require("vscode"));
const PARAM_DOCS = {
    ENCUT: 'Plane-wave cutoff energy in eV. Controls the size of the basis set.',
    EDIFF: 'SCF loop convergence criterion. Default is often 1e-6 eV.',
    ISMEAR: 'Smearing method: -5 (tetrahedron), 0 (Gaussian), 1 (Methfessel-Paxton), 2 (MP2).',
    SIGMA: 'Gaussian smearing width in eV. Keep < 0.05 eV for metals, ~0.2 for semiconductors.',
    ISPIN: '1 = non-spin-polarized, 2 = spin-polarized calculation.',
    IBRION: 'Ionic relaxation algorithm: -1 (no update), 0 (MD), 1 (RMM-DIIS), 2 (CG).',
    NSW: 'Maximum number of ionic steps in relaxation or MD.',
    ecutwfc: 'Kinetic-energy cutoff for wavefunctions in Ry.',
    ecutrho: 'Kinetic-energy cutoff for charge density in Ry (typically 4× ecutwfc).',
    conv_thr: 'SCF convergence threshold in Ry.',
    degauss: 'Gaussian spreading for Brillouin-zone integration in Ry.',
    occupations: 'Occupation scheme: smearing, fixed, or from_input.',
    timestep: 'Molecular dynamics integration timestep (fs once units is metal).',
    units: 'LAMMPS unit system: metal, real, lj, si, etc.',
    pair_style: 'Interatomic potential form: lj/cut, eam/alloy, reax/c, etc.',
    application: 'OpenFOAM solver application name, e.g. simpleFoam, pisoFoam.',
    deltaT: 'OpenFOAM time step size (seconds).',
    endTime: 'OpenFOAM simulation end time.',
    integrator: 'GROMACS integrator: md, sd, steep, etc.',
    dt: 'GROMACS integration timestep (ps).',
    nsteps: 'GROMACS number of integration steps.',
};
function extractKey(line, engine) {
    line = line.trim();
    if (engine === 'abaqus') {
        const m = line.match(/^\*([A-Za-z_][A-Za-z0-9_]*)/);
        return m ? `*${m[1]}` : null;
    }
    if (line.includes('=')) {
        return line.split('=')[0].trim();
    }
    if (line.split(/\s+/).length >= 2) {
        return line.split(/\s+/)[0].trim();
    }
    return null;
}
function detectEngine(document) {
    const name = document.fileName.split(/[\\/]/).pop()?.toLowerCase() || '';
    if (name === 'incar' || name === 'poscar') {
        return 'vasp';
    }
    const ext = name.split('.').pop() || '';
    if (ext === 'in') {
        return 'qe';
    }
    if (ext === 'lmp') {
        return 'lammps';
    }
    if (ext === 'inp') {
        return 'abaqus';
    }
    if (['top', 'itp'].includes(ext)) {
        return 'gromacs';
    }
    const basename = name.replace(/\.[^.]+$/, '');
    if (['controldict', 'fvschemes', 'fvsolution', 'transportproperties'].includes(basename)) {
        return 'openfoam';
    }
    return '';
}
class BourbakiHoverProvider {
    provideHover(document, position) {
        const engine = detectEngine(document);
        if (!engine) {
            return;
        }
        const line = document.lineAt(position).text;
        const key = extractKey(line, engine);
        if (!key) {
            return;
        }
        const doc = PARAM_DOCS[key] || PARAM_DOCS[key.replace(/^\*/, '')];
        if (!doc) {
            return;
        }
        const md = new vscode.MarkdownString(`**${key}** — ${doc}\n\n*Engine: ${engine}*`);
        return new vscode.Hover(md, document.lineAt(position).range);
    }
}
exports.BourbakiHoverProvider = BourbakiHoverProvider;
//# sourceMappingURL=hoverProvider.js.map