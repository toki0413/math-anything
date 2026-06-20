import * as vscode from 'vscode';
import { LanguageClient } from 'vscode-languageclient/node';

export class AnalysisProvider implements vscode.TreeDataProvider<AnalysisItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AnalysisItem | undefined | void> = new vscode.EventEmitter<AnalysisItem | undefined | void>();
    readonly onDidChangeTreeData: vscode.Event<AnalysisItem | undefined | void> = this._onDidChangeTreeData.event;

    private result: any = null;

    constructor(private client: LanguageClient | undefined) {}

    setResult(result: any) {
        this.result = result;
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: AnalysisItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: AnalysisItem): Thenable<AnalysisItem[]> {
        if (element) {
            return Promise.resolve(
                (element.children || []).map((c: any) => new AnalysisItem(c.label, c.value, c.children))
            );
        }
        if (!this.result) {
            return Promise.resolve([new AnalysisItem('Run "Analyze File" or "Analyze Workspace"', '', [])]);
        }
        return Promise.resolve(this._buildItems(this.result));
    }

    private _buildItems(obj: any, labelPrefix = ''): AnalysisItem[] {
        const items: AnalysisItem[] = [];
        for (const [key, value] of Object.entries(obj)) {
            const label = labelPrefix ? `${labelPrefix}.${key}` : key;
            if (Array.isArray(value)) {
                items.push(new AnalysisItem(
                    label,
                    `${value.length} item(s)`,
                    value.map((v, i) => ({ label: `[${i}]`, value: typeof v === 'object' ? '' : String(v), children: typeof v === 'object' ? this._leafChildren(v) : [] }))
                ));
            } else if (typeof value === 'object' && value !== null) {
                items.push(new AnalysisItem(label, '', this._leafChildren(value)));
            } else {
                items.push(new AnalysisItem(label, String(value), []));
            }
        }
        return items;
    }

    private _leafChildren(obj: any): any[] {
        return Object.entries(obj).map(([k, v]) => ({
            label: k,
            value: typeof v === 'object' ? JSON.stringify(v) : String(v),
            children: [],
        }));
    }
}

class AnalysisItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly value: string,
        public readonly children: any[],
    ) {
        super(label, children.length > 0 ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None);
        this.description = value;
        this.tooltip = value ? `${label}: ${value}` : label;
    }
}
