import * as vscode from 'vscode';

export class AnalysisWebview {
    public static currentPanel: AnalysisWebview | undefined;
    private readonly _panel: vscode.WebviewPanel;

    private constructor(panel: vscode.WebviewPanel, public result: any) {
        this._panel = panel;
        this._panel.webview.html = this._getHtml(this.result);
        this._panel.onDidDispose(() => { AnalysisWebview.currentPanel = undefined; });
    }

    public static createOrShow(extensionUri: vscode.Uri, result: any): void {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;
        if (AnalysisWebview.currentPanel) {
            AnalysisWebview.currentPanel._panel.reveal(column);
            AnalysisWebview.currentPanel._panel.webview.html = AnalysisWebview.currentPanel._getHtml(result);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'bourbakiAnalysis',
            'Bourbaki Analysis',
            column || vscode.ViewColumn.One,
            { enableScripts: true, retainContextWhenHidden: true }
        );
        AnalysisWebview.currentPanel = new AnalysisWebview(panel, result);
    }

    private _getHtml(result: any): string {
        const title = result?.title || 'Bourbaki Analysis';
        const body = result ? `<pre>${JSON.stringify(result, null, 2)}</pre>` : '<p>No analysis result available.</p>';
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); background: var(--vscode-editor-background); }
        pre { background: var(--vscode-textCodeBlock-background); padding: 16px; border-radius: 6px; overflow: auto; }
        h1 { font-size: 1.2em; border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 8px; }
    </style>
</head>
<body>
    <h1>${title}</h1>
    ${body}
</body>
</html>`;
    }
}
