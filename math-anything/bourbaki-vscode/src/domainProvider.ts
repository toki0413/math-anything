import * as vscode from 'vscode';
import { LanguageClient } from 'vscode-languageclient/node';

export class DomainProvider implements vscode.TreeDataProvider<DomainItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<DomainItem | undefined | void> = new vscode.EventEmitter<DomainItem | undefined | void>();
    readonly onDidChangeTreeData: vscode.Event<DomainItem | undefined | void> = this._onDidChangeTreeData.event;

    private domains: any[] = [];

    constructor(private client: LanguageClient | undefined) {
        this.refresh();
    }

    refresh(): void {
        if (!this.client) {
            return;
        }
        this.client.sendRequest('bourbaki/listDomains', {}).then((res: any) => {
            this.domains = res.domains || [];
            this._onDidChangeTreeData.fire();
        });
    }

    getTreeItem(element: DomainItem): vscode.TreeItem {
        return element;
    }

    getChildren(_element?: DomainItem): Thenable<DomainItem[]> {
        if (this.domains.length === 0) {
            return Promise.resolve([new DomainItem('Loading domains...', '')]);
        }
        return Promise.resolve(this.domains.map(d => new DomainItem(d.name || String(d), d.description || '')));
    }
}

class DomainItem extends vscode.TreeItem {
    constructor(public readonly label: string, public readonly description: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.tooltip = description || label;
    }
}
