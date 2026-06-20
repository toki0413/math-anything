import { useState, useEffect, useRef } from 'react';
import { GitBranch, Search, CheckCircle2, XCircle, Loader2, Network, Activity } from 'lucide-react';
import { wsUrl } from '../lib/api';
import clsx from 'clsx';

interface Branch {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'pruned';
  hypothesis: string;
  objective: string;
  score: number | null;
  results: any;
  decisions: Decision[];
}

interface Decision {
  id: string;
  type: string;
  params: Record<string, any>;
  reason: string;
}

interface ExplorationState {
  active: boolean;
  objective: string;
  strategy: 'pareto' | 'bayesian' | 'grid';
  maxBranches: number;
  branches: Branch[];
  paretoFront: string[];
  logs: string[];
}

export default function ExplorationPage() {
  const [state, setState] = useState<ExplorationState>({
    active: false,
    objective: '',
    strategy: 'pareto',
    maxBranches: 5,
    branches: [],
    paretoFront: [],
    logs: [],
  });

  const [objectiveInput, setObjectiveInput] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Setup WebSocket connection
    let ws: WebSocket;
    wsUrl('/ws/agent').then((url) => {
      ws = new WebSocket(url);
      ws.onopen = () => {
        setState(p => ({ ...p, logs: [...p.logs, `[${new Date().toLocaleTimeString()}] Connected to backend]`] }));
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const ts = `[${new Date().toLocaleTimeString()}]`;

        switch (msg.type) {
          case 'branch_spawned': {
            const b = msg.branch as Branch;
            setState(p => ({
              ...p,
              branches: [...p.branches, { ...b, decisions: b.decisions ?? [] }],
              logs: [...p.logs, `${ts} Spawned: ${b.name} — ${b.hypothesis}`],
            }));
            break;
          }
          case 'branch_completed': {
            setState(p => ({
              ...p,
              branches: p.branches.map(b =>
                b.id === msg.branch_id
                  ? { ...b, status: 'completed' as const, score: msg.score, results: msg.results }
                  : b
              ),
              logs: [...p.logs, `${ts} Completed: ${msg.branch_id} (score: ${msg.score?.toFixed(3) ?? 'N/A'})`],
            }));
            break;
          }
          case 'branch_failed': {
            setState(p => ({
              ...p,
              branches: p.branches.map(b =>
                b.id === msg.branch_id ? { ...b, status: 'failed' as const } : b
              ),
              logs: [...p.logs, `${ts} Failed: ${msg.branch_id} — ${msg.error}`],
            }));
            break;
          }
          case 'branch_pruned': {
            setState(p => ({
              ...p,
              branches: p.branches.map(b =>
                b.id === msg.branch_id ? { ...b, status: 'pruned' as const } : b
              ),
              logs: [...p.logs, `${ts} Pruned: ${msg.branch_id}`],
            }));
            break;
          }
          case 'pareto_update': {
            setState(p => ({
              ...p,
              paretoFront: msg.front,
              logs: [...p.logs, `${ts} Pareto front updated: ${msg.front.join(', ')}`],
            }));
            break;
          }
          case 'log': {
            setState(p => ({ ...p, logs: [...p.logs, `${ts} ${msg.content}`] }));
            break;
          }
          case 'done': {
            setState(p => ({ ...p, active: false, logs: [...p.logs, `${ts} Exploration done.`] }));
            break;
          }
          case 'error': {
            setState(p => ({ ...p, active: false, logs: [...p.logs, `${ts} Error: ${msg.content}`] }));
            break;
          }
        }
      };
      wsRef.current = ws;
    });

    return () => {
      if (ws) ws.close();
    };
  }, []);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.logs]);

  const startExploration = () => {
    if (!objectiveInput.trim()) return;

    setState(prev => ({
      ...prev,
      active: true,
      objective: objectiveInput,
      branches: [],
      logs: [`[${new Date().toLocaleTimeString()}] Exploration started: ${objectiveInput}`],
    }));

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'explore_start',
        content: objectiveInput,
        strategy: state.strategy,
        max_branches: state.maxBranches,
      }));
    }
  };

  const stopExploration = () => {
    setState(prev => ({
      ...prev,
      active: false,
      logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] Exploration stopped.`],
    }));
  };

  const getStatusIcon = (status: Branch['status']) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running': return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'pruned': return <XCircle className="w-4 h-4 text-gray-400" />;
      default: return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: Branch['status']) => {
    const variants: Record<string, string> = {
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800',
      running: 'bg-blue-100 text-blue-800',
      pruned: 'bg-gray-100 text-gray-600',
      pending: 'bg-yellow-100 text-yellow-800',
    };
    return (
      <span className={clsx('px-2 py-0.5 rounded text-xs font-medium', variants[status] || '')}>
        {status}
      </span>
    );
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center gap-4 px-6 py-3 border-b border-border bg-bg-surface">
        <Network className="w-5 h-5 text-accent" />
        <h1 className="text-lg font-semibold text-text">Exploration Mode</h1>
        {state.active && (
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 animate-pulse">
            Active
          </span>
        )}
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left: Control Panel */}
        <div className="w-72 border-r border-border bg-bg p-4 overflow-y-auto space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-text mb-3">Objective</h3>
            <input
              className="w-full rounded border border-border bg-bg-surface px-3 py-2 text-sm text-text placeholder:text-text-3 focus:outline-none focus:ring-1 focus:ring-accent"
              placeholder="e.g., Find optimal dopants for NMC 333 cathode..."
              value={objectiveInput}
              onChange={(e) => setObjectiveInput(e.target.value)}
              disabled={state.active}
            />
          </div>

          <div>
            <label className="text-xs font-medium text-text-2">Strategy</label>
            <select
              value={state.strategy}
              onChange={(e) => setState(p => ({ ...p, strategy: e.target.value as any }))}
              disabled={state.active}
              className="w-full mt-1 rounded border border-border bg-bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-1 focus:ring-accent"
            >
              <option value="pareto">Pareto Frontier</option>
              <option value="bayesian">Bayesian Optimization</option>
              <option value="grid">Grid Search</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-text-2">Max Branches</label>
            <input
              type="number"
              value={state.maxBranches}
              onChange={(e) => setState(p => ({ ...p, maxBranches: parseInt(e.target.value) || 5 }))}
              disabled={state.active}
              className="w-full mt-1 rounded border border-border bg-bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-1 focus:ring-accent"
            />
          </div>

          {!state.active ? (
            <button
              onClick={startExploration}
              disabled={!objectiveInput.trim()}
              className="w-full flex items-center justify-center gap-2 rounded bg-accent text-white px-4 py-2 text-sm font-medium hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Search className="w-4 h-4" />
              Start Exploration
            </button>
          ) : (
            <button
              onClick={stopExploration}
              className="w-full flex items-center justify-center gap-2 rounded bg-red-500 text-white px-4 py-2 text-sm font-medium hover:bg-red-600 transition-colors"
            >
              <XCircle className="w-4 h-4" />
              Stop
            </button>
          )}

          {/* Stats */}
          <div className="border-t border-border pt-4">
            <h3 className="text-sm font-semibold text-text mb-3">Knowledge Graph</h3>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-bg-surface rounded border border-border p-2 text-center">
                <div className="text-xl font-bold text-accent">{state.branches.length}</div>
                <div className="text-xs text-text-3">Branches</div>
              </div>
              <div className="bg-bg-surface rounded border border-border p-2 text-center">
                <div className="text-xl font-bold text-green-600">
                  {state.branches.filter(b => b.status === 'completed').length}
                </div>
                <div className="text-xs text-text-3">Completed</div>
              </div>
            </div>
          </div>
        </div>

        {/* Middle: Branch Tree */}
        <div className="flex-1 overflow-y-auto p-4 bg-bg">
          <h2 className="text-base font-semibold text-text mb-3">Branches</h2>

          {state.branches.length === 0 && !state.active && (
            <div className="text-center text-text-3 mt-20">
              <GitBranch className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Enter an objective and start exploring</p>
            </div>
          )}

          <div className="space-y-3">
            {state.branches.map(branch => (
              <div key={branch.id} className="rounded-lg border border-border bg-bg-surface p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(branch.status)}
                    <div>
                      <div className="font-medium text-text">{branch.name}</div>
                      <div className="text-sm text-text-2">{branch.hypothesis}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {branch.score !== null && (
                      <div className="text-sm font-mono text-text-2">
                        Score: {branch.score.toFixed(3)}
                      </div>
                    )}
                    {getStatusBadge(branch.status)}
                  </div>
                </div>

                {branch.decisions.length > 0 && (
                  <div className="mt-3 pl-7">
                    <div className="text-xs text-text-3 mb-1">Decisions:</div>
                    {branch.decisions.map(d => (
                      <div key={d.id} className="text-sm bg-bg rounded px-2 py-1 mb-1 text-text-2">
                        <span className="font-medium text-text">{d.type}</span>: {d.reason}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Right: Logs */}
        <div className="w-64 border-l border-border bg-gray-900 text-gray-100 overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b border-gray-700">
            <h3 className="text-xs font-semibold uppercase tracking-wider">Execution Log</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-1">
            {state.logs.map((log, i) => (
              <div key={i} className="text-gray-300 leading-relaxed">{log}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
