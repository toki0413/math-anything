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
exports.AnalysisProvider = void 0;
const vscode = __importStar(require("vscode"));
class AnalysisProvider {
    client;
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    result = null;
    constructor(client) {
        this.client = client;
    }
    setResult(result) {
        this.result = result;
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve((element.children || []).map((c) => new AnalysisItem(c.label, c.value, c.children)));
        }
        if (!this.result) {
            return Promise.resolve([new AnalysisItem('Run "Analyze File" or "Analyze Workspace"', '', [])]);
        }
        return Promise.resolve(this._buildItems(this.result));
    }
    _buildItems(obj, labelPrefix = '') {
        const items = [];
        for (const [key, value] of Object.entries(obj)) {
            const label = labelPrefix ? `${labelPrefix}.${key}` : key;
            if (Array.isArray(value)) {
                items.push(new AnalysisItem(label, `${value.length} item(s)`, value.map((v, i) => ({ label: `[${i}]`, value: typeof v === 'object' ? '' : String(v), children: typeof v === 'object' ? this._leafChildren(v) : [] }))));
            }
            else if (typeof value === 'object' && value !== null) {
                items.push(new AnalysisItem(label, '', this._leafChildren(value)));
            }
            else {
                items.push(new AnalysisItem(label, String(value), []));
            }
        }
        return items;
    }
    _leafChildren(obj) {
        return Object.entries(obj).map(([k, v]) => ({
            label: k,
            value: typeof v === 'object' ? JSON.stringify(v) : String(v),
            children: [],
        }));
    }
}
exports.AnalysisProvider = AnalysisProvider;
class AnalysisItem extends vscode.TreeItem {
    label;
    value;
    children;
    constructor(label, value, children) {
        super(label, children.length > 0 ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.value = value;
        this.children = children;
        this.description = value;
        this.tooltip = value ? `${label}: ${value}` : label;
    }
}
//# sourceMappingURL=analysisProvider.js.map