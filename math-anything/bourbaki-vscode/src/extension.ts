import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';
import { AnalysisProvider } from './analysisProvider';
import { DomainProvider } from './domainProvider';
import { AnalysisWebview } from './analysisWebview';
import { BourbakiCompletionProvider } from './completionProvider';
import { BourbakiHoverProvider } from './hoverProvider';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('bourbaki');
    const pythonPath = config.get<string>('pythonPath', 'python');

    const serverModule = context.asAbsolutePath('lsp_server.py');
    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: [serverModule],
        transport: TransportKind.stdio,
    };

    const clientOptions: LanguageClientOptions = {
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

    if (config.get<boolean>('lsp.enabled', true)) {
        client = new LanguageClient('bourbaki', 'Bourbaki Language Server', serverOptions, clientOptions);
        client.start();
    }

    const analysisProvider = new AnalysisProvider(client);
    const domainProvider = new DomainProvider(client);

    vscode.window.registerTreeDataProvider('bourbaki.analysis', analysisProvider);
    vscode.window.registerTreeDataProvider('bourbaki.domains', domainProvider);

    const supportedLanguages = ['vasp-incar', 'vasp-poscar', 'qe-in', 'lammps-input', 'abaqus-input', 'openfoam', 'gromacs-top'];
    for (const lang of supportedLanguages) {
        context.subscriptions.push(
            vscode.languages.registerCompletionItemProvider(lang, new BourbakiCompletionProvider(), '='),
            vscode.languages.registerHoverProvider(lang, new BourbakiHoverProvider())
        );
    }

    context.subscriptions.push(
        vscode.commands.registerCommand('bourbaki.analyzeFile', async (uri?: vscode.Uri) => {
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
            AnalysisWebview.createOrShow(context.extensionUri, result);
        }),

        vscode.commands.registerCommand('bourbaki.analyzeWorkspace', async () => {
            const folders = vscode.workspace.workspaceFolders;
            if (!folders) {
                vscode.window.showWarningMessage('No workspace folder open.');
                return;
            }
            const result = await sendRequest('bourbaki/analyzeWorkspace', { rootPath: folders[0].uri.fsPath });
            analysisProvider.setResult(result);
            vscode.window.showInformationMessage(`Found ${result.engines?.length ?? 0} engine configurations.`);
        }),

        vscode.commands.registerCommand('bourbaki.verifyEquation', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return; }
            const statement = editor.document.getText(editor.selection);
            if (!statement) {
                vscode.window.showWarningMessage('Select an equation to verify.');
                return;
            }
            const result = await sendRequest('bourbaki/verifyEquation', { statement });
            AnalysisWebview.createOrShow(context.extensionUri, result);
        }),

        vscode.commands.registerCommand('bourbaki.solveNumerical', async () => {
            const solverType = await vscode.window.showQuickPick(
                ['symplectic', 'variational', 'eigenvalue', 'scf', 'conservation'],
                { placeHolder: 'Select solver type' }
            );
            if (!solverType) { return; }
            const result = await sendRequest('bourbaki/solveNumerical', { solverType });
            AnalysisWebview.createOrShow(context.extensionUri, result);
        }),

        vscode.commands.registerCommand('bourbaki.openWebview', () => {
            AnalysisWebview.createOrShow(context.extensionUri, { title: 'Bourbaki Dashboard' });
        })
    );

    vscode.commands.executeCommand('setContext', 'bourbaki:enabled', true);
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}

async function sendRequest(method: string, params: any): Promise<any> {
    if (!client) {
        vscode.window.showErrorMessage('Bourbaki language server is disabled.');
        return {};
    }
    return client.sendRequest(method, params);
}
