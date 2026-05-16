import { useState } from "react";
import { sandboxExecute } from "../lib/api";

const EXAMPLE_CODE = `# Example: Validate a mathematical constraint
import math

# Check if a value satisfies a constraint
encut = 520
max_enmax = 500
passed = encut > max_enmax
print(f"ENCUT={encut}, max(ENMAX)={max_enmax}")
print(f"Constraint satisfied: {passed}")
`;

export default function SandboxPage() {
  const [code, setCode] = useState(EXAMPLE_CODE);
  const [timeout, setTimeout_] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runSandbox() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await sandboxExecute({
        code,
        timeout_seconds: timeout,
        backend: "auto",
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-800">Sandbox</h1>
        <p className="text-sm text-gray-500 mt-1">
          Execute Python code in an isolated sandbox for constraint validation
          and custom analysis.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700">
              Python Code
            </label>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-500">Timeout (s)</label>
              <input
                type="number"
                value={timeout}
                onChange={(e) => setTimeout_(Number(e.target.value))}
                className="w-20 border border-gray-300 rounded px-2 py-1 text-sm"
                min={1}
                max={60}
              />
            </div>
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm h-96 font-mono bg-gray-50 focus:ring-2 focus:ring-blue-300 focus:border-blue-400"
            spellCheck={false}
          />
          <button
            onClick={runSandbox}
            disabled={loading || !code.trim()}
            className="w-full bg-blue-600 text-white rounded px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Executing..." : "Run in Sandbox"}
          </button>
        </div>

        <div className="space-y-3">
          <label className="text-sm font-medium text-gray-700">Output</label>
          {error && (
            <div className="bg-red-50 border border-red-200 rounded px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          {result && (
            <div className="space-y-3">
              <div
                className={`flex items-center gap-2 px-3 py-2 rounded text-sm font-medium ${
                  result.success
                    ? "bg-green-50 text-green-700 border border-green-200"
                    : "bg-red-50 text-red-700 border border-red-200"
                }`}
              >
                <span>{result.success ? "✓ Success" : "✗ Failed"}</span>
                {result.execution_time_ms != null && (
                  <span className="text-xs opacity-70">
                    ({Number(result.execution_time_ms).toFixed(0)} ms, {String(result.backend)})
                  </span>
                )}
              </div>

              {result.stdout && (
                <div>
                  <span className="text-xs text-gray-500">stdout</span>
                  <pre className="bg-gray-900 text-green-400 rounded px-3 py-2 text-xs overflow-auto max-h-64 font-mono mt-1">
                    {String(result.stdout)}
                  </pre>
                </div>
              )}

              {result.stderr && (
                <div>
                  <span className="text-xs text-gray-500">stderr</span>
                  <pre className="bg-gray-900 text-yellow-400 rounded px-3 py-2 text-xs overflow-auto max-h-32 font-mono mt-1">
                    {String(result.stderr)}
                  </pre>
                </div>
              )}

              {result.error && (
                <div>
                  <span className="text-xs text-gray-500">Error</span>
                  <pre className="bg-red-50 border border-red-200 rounded px-3 py-2 text-xs overflow-auto max-h-32 font-mono mt-1 text-red-700">
                    {String(result.error)}
                  </pre>
                </div>
              )}
            </div>
          )}
          {!result && !error && (
            <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2 text-sm text-gray-400 h-96 flex items-center justify-center">
              Run code to see output
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
