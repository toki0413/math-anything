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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const node_1 = require("vscode-languageclient/node");
const analysisProvider_1 = require("./analysisProvider");
const domainProvider_1 = require("./domainProvider");
const analysisWebview_1 = require("./analysisWebview");
const completionProvider_1 = require("./completionProvider");
const hoverProvider_1 = require("./hoverProvider");
let client;
function activate(context) {
    const config = vscode.workspace.getConfiguration('bourbaki');
    const pythonPath = config.get('pythonPath', 'python');
    const serverModule = context.asAbsolutePath('lsp_server.py');
    const serverOptions = {
        command: pythonPath,
        args: [serverModule],
        transport: node_1.TransportKind.stdio,
    };
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'vasp-incar' },
            { scheme: 'file', language: 'vasp-poscar' },
            { scheme: 'file', language: 'qe-in' },
            { scheme: 'file', language: 'lammps-input' },
            { scheme: 'file', language: 'abaqus-input' },
            { scheme: 'file', language: 'openfoam' },
            { scheme: 'file', language: 'gromacs-top' },
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{incar,poscar,in,lmp,inp,top,itp}'),
        },
        outputChannelName: 'Bourbaki LSP',
        traceOutputChannel: config.get('lsp.trace') ? undefined : undefined,
    };
    if (config.get('lsp.enabled', true)) {
        client = new node_1.LanguageClient('bourbaki', 'Bourbaki Language Server', serverOptions, clientOptions);
        client.start();
    }
    const analysisProvider = new analysisProvider_1.AnalysisProvider(client);
    const domainProvider = new domainProvider_1.DomainProvider(client);
    vscode.window.registerTreeDataProvider('bourbaki.analysis', analysisProvider);
    vscode.window.registerTreeDataProvider('bourbaki.domains', domainProvider);
    const supportedLanguages = ['vasp-incar', 'vasp-poscar', 'qe-in', 'lammps-input', 'abaqus-input', 'openfoam', 'gromacs-top'];
    for (const lang of supportedLanguages) {
        context.subscriptions.push(vscode.languages.registerCompletionItemProvider(lang, new completionProvider_1.BourbakiCompletionProvider(), '='), vscode.languages.registerHoverProvider(lang, new hoverProvider_1.BourbakiHoverProvider()));
    }
    context.subscriptions.push(vscode.commands.registerCommand('bourbaki.analyzeFile', async (uri) => {
        const target = uri ?? vscode.window.activeTextEditor?.document.uri;
        if (!target) {
            vscode.window.showWarningMessage('No file selected for analysis.');
            return;
        }
        const result = await sendRequest('bourbaki/analyzeFile', {
            filePath: target.fsPath,
            content: (await vscode.workspace.openTextDocument(target)).getText(),
        });
        analysisProvider.setResult(result);
        analysisWebview_1.AnalysisWebview.createOrShow(context.extensionUri, result);
    }), vscode.commands.registerCommand('bourbaki.analyzeWorkspace', async () => {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders) {
            vscode.window.showWarningMessage('No workspace folder open.');
            return;
        }
        const result = await sendRequest('bourbaki/analyzeWorkspace', { rootPath: folders[0].uri.fsPath });
        analysisProvider.setResult(result);
        vscode.window.showInformationMessage(`Found ${result.engines?.length ?? 0} engine configurations.`);
    }), vscode.commands.registerCommand('bourbaki.verifyEquation', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const statement = editor.document.getText(editor.selection);
        if (!statement) {
            vscode.window.showWarningMessage('Select an equation to verify.');
            return;
        }
        const result = await sendRequest('bourbaki/verifyEquation', { statement });
        analysisWebview_1.AnalysisWebview.createOrShow(context.extensionUri, result);
    }), vscode.commands.registerCommand('bourbaki.solveNumerical', async () => {
        const solverType = await vscode.window.showQuickPick(['symplectic', 'variational', 'eigenvalue', 'scf', 'conservation'], { placeHolder: 'Select solver type' });
        if (!solverType) {
            return;
        }
        const result = await sendRequest('bourbaki/solveNumerical', { solverType });
        analysisWebview_1.AnalysisWebview.createOrShow(context.extensionUri, result);
    }), vscode.commands.registerCommand('bourbaki.openWebview', () => {
        analysisWebview_1.AnalysisWebview.createOrShow(context.extensionUri, { title: 'Bourbaki Dashboard' });
    }));
    vscode.commands.executeCommand('setContext', 'bourbaki:enabled', true);
}
function deactivate() {
    return client?.stop();
}
async function sendRequest(method, params) {
    if (!client) {
        vscode.window.showErrorMessage('Bourbaki language server is disabled.');
        return {};
    }
    return client.sendRequest(method, params);
}
//# sourceMappingURL=extension.js.map