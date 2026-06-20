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
exports.DomainProvider = void 0;
const vscode = __importStar(require("vscode"));
class DomainProvider {
    client;
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    domains = [];
    constructor(client) {
        this.client = client;
        this.refresh();
    }
    refresh() {
        if (!this.client) {
            return;
        }
        this.client.sendRequest('bourbaki/listDomains', {}).then((res) => {
            this.domains = res.domains || [];
            this._onDidChangeTreeData.fire();
        });
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(_element) {
        if (this.domains.length === 0) {
            return Promise.resolve([new DomainItem('Loading domains...', '')]);
        }
        return Promise.resolve(this.domains.map(d => new DomainItem(d.name || String(d), d.description || '')));
    }
}
exports.DomainProvider = DomainProvider;
class DomainItem extends vscode.TreeItem {
    label;
    description;
    constructor(label, description) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.description = description;
        this.tooltip = description || label;
    }
}
//# sourceMappingURL=domainProvider.js.map