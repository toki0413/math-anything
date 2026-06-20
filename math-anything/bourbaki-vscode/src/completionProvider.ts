import * as vscode from 'vscode';

interface ParamInfo {
    name: string;
    description: string;
    type: string;
    domain: string;
}

const ENGINE_PARAMS: Record<string, ParamInfo[]> = {
    vasp: [
        { name: 'SYSTEM', description: 'Job title / description', type: 'string', domain: 'dft' },

        { name: 'ENCUT', description: 'Plane-wave cutoff energy (eV)', type: 'number', domain: 'dft' },
        { name: 'EDIFF', description: 'SCF convergence threshold', type: 'number', domain: 'dft' },
        { name: 'ISMEAR', description: 'Smearing method', type: 'integer', domain: 'dft' },
        { name: 'SIGMA', description: 'Smearing width (eV)', type: 'number', domain: 'dft' },
        { name: 'ISPIN', description: 'Spin polarization (1=non-spin, 2=spin)', type: 'integer', domain: 'dft' },
        { name: 'IBRION', description: 'Ionic relaxation algorithm', type: 'integer', domain: 'dft' },
        { name: 'NSW', description: 'Maximum number of ionic steps', type: 'integer', domain: 'dft' },
    ],
    qe: [
        { name: 'ecutwfc', description: 'Wavefunction cutoff (Ry)', type: 'number', domain: 'dft' },
        { name: 'ecutrho', description: 'Charge density cutoff (Ry)', type: 'number', domain: 'dft' },
        { name: 'conv_thr', description: 'SCF convergence threshold', type: 'number', domain: 'dft' },
        { name: 'degauss', description: 'Gaussian smearing width (Ry)', type: 'number', domain: 'dft' },
        { name: 'occupations', description: 'Occupation scheme', type: 'string', domain: 'dft' },
    ],
    lammps: [
        { name: 'timestep', description: 'Integration timestep', type: 'number', domain: 'md' },
        { name: 'units', description: 'Unit system', type: 'string', domain: 'md' },
        { name: 'pair_style', description: 'Interatomic potential style', type: 'string', domain: 'md' },
        { name: 'fix', description: 'Constraint/ensemble fix', type: 'string', domain: 'md' },
    ],
    abaqus: [
        { name: '*STEP', description: 'Analysis step definition', type: 'keyword', domain: 'fem' },
        { name: '*STATIC', description: 'Static analysis procedure', type: 'keyword', domain: 'fem' },
        { name: '*DYNAMIC', description: 'Dynamic analysis procedure', type: 'keyword', domain: 'fem' },
        { name: '*NLGEOM', description: 'Geometric nonlinearity flag', type: 'keyword', domain: 'fem' },
    ],
    openfoam: [
        { name: 'application', description: 'Solver application name', type: 'string', domain: 'cfd' },
        { name: 'startFrom', description: 'Start time control', type: 'string', domain: 'cfd' },
        { name: 'startTime', description: 'Start time value', type: 'number', domain: 'cfd' },
        { name: 'stopAt', description: 'Stop condition', type: 'string', domain: 'cfd' },
        { name: 'endTime', description: 'End time value', type: 'number', domain: 'cfd' },
        { name: 'deltaT', description: 'Time step size', type: 'number', domain: 'cfd' },
        { name: 'writeControl', description: 'Output write control', type: 'string', domain: 'cfd' },
        { name: 'writeInterval', description: 'Output write interval', type: 'number', domain: 'cfd' },
    ],
    gromacs: [
        { name: 'integrator', description: 'MD integrator algorithm', type: 'string', domain: 'md' },
        { name: 'dt', description: 'Integration timestep (ps)', type: 'number', domain: 'md' },
        { name: 'nsteps', description: 'Number of integration steps', type: 'integer', domain: 'md' },
        { name: 'cutoff-scheme', description: 'Neighbor list cutoff scheme', type: 'string', domain: 'md' },
        { name: 'coulombtype', description: 'Electrostatics treatment', type: 'string', domain: 'md' },
    ],
};

function detectEngine(document: vscode.TextDocument): string {
    const name = document.fileName.split(/[\\/]/).pop()?.toLowerCase() || '';
    if (name === 'incar' || name === 'poscar') { return 'vasp'; }
    const ext = name.split('.').pop() || '';
    if (ext === 'in') { return 'qe'; }
    if (ext === 'lmp') { return 'lammps'; }
    if (ext === 'inp') { return 'abaqus'; }
    if (['top', 'itp'].includes(ext)) { return 'gromacs'; }
    const basename = name.replace(/\.[^.]+$/, '');
    if (['controldict', 'fvschemes', 'fvsolution', 'transportproperties'].includes(basename)) { return 'openfoam'; }
    return '';
}

export class BourbakiCompletionProvider implements vscode.CompletionItemProvider {
    provideCompletionItems(
        document: vscode.TextDocument,
        _position: vscode.Position,
    ): vscode.CompletionItem[] {
        const engine = detectEngine(document);
        const params = ENGINE_PARAMS[engine];
        if (!params) { return []; }
        return params.map(p => {
            const item = new vscode.CompletionItem(p.name, vscode.CompletionItemKind.Property);
            item.detail = `${p.type} — ${p.domain}`;
            item.documentation = new vscode.MarkdownString(p.description);
            if (engine === 'abaqus') {
                item.insertText = p.name;
            } else {
                item.insertText = new vscode.SnippetString(`${p.name} = $0`);
            }
            return item;
        });
    }
}
