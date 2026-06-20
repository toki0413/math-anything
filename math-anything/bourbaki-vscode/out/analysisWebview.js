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
exports.AnalysisWebview = void 0;
const vscode = __importStar(require("vscode"));
class AnalysisWebview {
    result;
    static currentPanel;
    _panel;
    constructor(panel, result) {
        this.result = result;
        this._panel = panel;
        this._panel.webview.html = this._getHtml(this.result);
        this._panel.onDidDispose(() => { AnalysisWebview.currentPanel = undefined; });
    }
    static createOrShow(extensionUri, result) {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;
        if (AnalysisWebview.currentPanel) {
            AnalysisWebview.currentPanel._panel.reveal(column);
            AnalysisWebview.currentPanel._panel.webview.html = AnalysisWebview.currentPanel._getHtml(result);
            return;
        }
        const panel = vscode.window.createWebviewPanel('bourbakiAnalysis', 'Bourbaki Analysis', column || vscode.ViewColumn.One, { enableScripts: true, retainContextWhenHidden: true });
        AnalysisWebview.currentPanel = new AnalysisWebview(panel, result);
    }
    _getHtml(result) {
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
exports.AnalysisWebview = AnalysisWebview;
//# sourceMappingURL=analysisWebview.js.map