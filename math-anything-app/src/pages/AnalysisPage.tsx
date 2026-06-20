import { useState } from "react";
import {
  analyzeSymmetry,
  analyzeSpectral,
  analyzeDynamics,
  analyzeTDA,
  analyzeSINDy,
  vizDOS,
  vizPhase,
  vizPersistence,
  vizSINDy,
} from "../lib/api";

type AnalysisTab = "symmetry" | "spectral" | "dynamics" | "tda" | "sindy";

const TABS: { key: AnalysisTab; label: string; icon: string }[] = [
  { key: "symmetry", label: "Symmetry", icon: "🔶" },
  { key: "spectral", label: "Spectral", icon: "📊" },
  { key: "dynamics", label: "Dynamics", icon: "🌀" },
  { key: "sindy", label: "SINDy", icon: "📐" },
  { key: "tda", label: "TDA", icon: "🔵" },
];

export function AnalysisPage() {
  const [tab, setTab] = useState<AnalysisTab>("symmetry");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [vizHtml, setVizHtml] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [spaceGroupHint, setSpaceGroupHint] = useState("225");
  const [eigenvaluesText, setEigenvaluesText] = useState("");
  const [timeSeriesText, setTimeSeriesText] = useState("");
  const [pointCloudText, setPointCloudText] = useState("");
  const [sindyText, setSindyText] = useState("");
  const [sindyDt, setSindyDt] = useState(1.0);
  const [sindyThreshold, setSindyThreshold] = useState(0.1);

  async function runSymmetry() {
    setLoading(true);
    setError(null);
    setResult(null);
    setVizHtml(null);
    try {
      const r = await analyzeSymmetry({ space_group_hint: spaceGroupHint });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runSpectral() {
    setLoading(true);
    setError(null);
    setResult(null);
    setVizHtml(null);
    try {
      const eigs = eigenvaluesText
        .split(/[\s,]+/)
        .map(Number)
        .filter((n) => !isNaN(n));
      if (eigs.length === 0) throw new Error("Enter eigenvalues");
      const r = await analyzeSpectral({ eigenvalues: eigs });
      setResult(r);
      const v = await vizDOS({ eigenvalues: eigs });
      if (v.html) setVizHtml(v.html);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runDynamics() {
    setLoading(true);
    setError(null);
    setResult(null);
    setVizHtml(null);
    try {
      const ts = timeSeriesText
        .split(/[\s,]+/)
        .map(Number)
        .filter((n) => !isNaN(n));
      if (ts.length < 10) throw new Error("Need at least 10 data points");
      const r = await analyzeDynamics({ time_series: ts });
      setResult(r);
      const v = await vizPhase({ time_series: ts });
      if (v.html) setVizHtml(v.html);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runTDA() {
    setLoading(true);
    setError(null);
    setResult(null);
    setVizHtml(null);
    try {
      const lines = pointCloudText.trim().split("\n");
      const cloud = lines
        .map((line) =>
          line
            .split(/[\s,]+/)
            .map(Number)
            .filter((n) => !isNaN(n))
        )
        .filter((arr) => arr.length >= 2);
      if (cloud.length < 3) throw new Error("Need at least 3 points");
      const r = await analyzeTDA({ point_cloud: cloud });
      setResult(r);
      const v = await vizPersistence({ point_cloud: cloud });
      if (v.html) setVizHtml(v.html);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runSINDy() {
    setLoading(true);
    setError(null);
    setResult(null);
    setVizHtml(null);
    try {
      if (sindyDt <= 0) throw new Error("dt must be positive");
      if (sindyThreshold < 0) throw new Error("threshold must be non-negative");
      const lines = sindyText.trim().split("\n");
      const data = lines
        .map((line) =>
          line
            .split(/[\s,]+/)
            .map(Number)
            .filter((n) => !isNaN(n))
        )
        .filter((arr) => arr.length >= 1);
      if (data.length < 10) throw new Error("Need at least 10 time steps");
      const r = await analyzeSINDy({
        time_series: data,
        dt: sindyDt,
        threshold: sindyThreshold,
      });
      setResult(r);
      const v = await vizSINDy({
        time_series: data,
        dt: sindyDt,
        threshold: sindyThreshold,
      });
      if (v.html) setVizHtml(v.html);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function runAnalysis() {
    switch (tab) {
      case "symmetry":
        return runSymmetry();
      case "spectral":
        return runSpectral();
      case "dynamics":
        return runDynamics();
      case "sindy":
        return runSINDy();
      case "tda":
        return runTDA();
    }
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">
        Mathematical Analysis
      </h1>
      <p className="text-gray-500 text-sm">
        Group theory, spectral analysis, dynamical systems, and topological
        data analysis
      </p>

      <div className="flex gap-2 border-b border-gray-200 pb-2">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => {
              setTab(t.key);
              setResult(null);
              setVizHtml(null);
              setError(null);
            }}
            className={`px-4 py-2 rounded-t text-sm font-medium transition-colors ${
              tab === t.key
                ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
            <h2 className="font-semibold text-gray-800">Input</h2>

            {tab === "symmetry" && (
              <div>
                <label className="block text-sm text-gray-600 mb-1">
                  Space Group Number or Symbol
                </label>
                <input
                  type="text"
                  value={spaceGroupHint}
                  onChange={(e) => setSpaceGroupHint(e.target.value)}
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                  placeholder="e.g. 225 or Fm-3m"
                />
              </div>
            )}

            {tab === "spectral" && (
              <div>
                <label className="block text-sm text-gray-600 mb-1">
                  Eigenvalues (space or comma separated)
                </label>
                <textarea
                  value={eigenvaluesText}
                  onChange={(e) => setEigenvaluesText(e.target.value)}
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm h-32 font-mono"
                  placeholder="-3.2, -2.1, -0.5, 0.3, 1.8, 3.5"
                />
              </div>
            )}

            {tab === "dynamics" && (
              <div>
                <label className="block text-sm text-gray-600 mb-1">
                  Time Series (space or comma separated)
                </label>
                <textarea
                  value={timeSeriesText}
                  onChange={(e) => setTimeSeriesText(e.target.value)}
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm h-32 font-mono"
                  placeholder="0.1 0.3 0.5 0.2 -0.1 0.4 ..."
                />
              </div>
            )}

            {tab === "tda" && (
              <div>
                <label className="block text-sm text-gray-600 mb-1">
                  Point Cloud (one point per line)
                </label>
                <textarea
                  value={pointCloudText}
                  onChange={(e) => setPointCloudText(e.target.value)}
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm h-32 font-mono"
                  placeholder={"0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9"}
                />
              </div>
            )}

            {tab === "sindy" && (
              <div className="space-y-3">
                <div>
                  <label className="block text-sm text-gray-600 mb-1">
                    Time Series Data (one row per time step, columns = variables)
                  </label>
                  <textarea
                    value={sindyText}
                    onChange={(e) => setSindyText(e.target.value)}
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm h-32 font-mono"
                    placeholder={"0.1 0.2\n0.3 0.5\n0.5 0.8\n..."}
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">dt</label>
                    <input
                      type="number"
                      value={sindyDt}
                      onChange={(e) => setSindyDt(Number(e.target.value))}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      step={0.01}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Threshold</label>
                    <input
                      type="number"
                      value={sindyThreshold}
                      onChange={(e) => setSindyThreshold(Number(e.target.value))}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      step={0.01}
                    />
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={runAnalysis}
              disabled={loading}
              className="w-full bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h2 className="font-semibold text-gray-800 mb-2">Results</h2>
              <ResultDisplay result={result} tab={tab} />
            </div>
          )}
        </div>

        <div className="space-y-4">
          {vizHtml && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h2 className="font-semibold text-gray-800 mb-2">
                Visualization
              </h2>
              <div
                className="w-full overflow-hidden"
                dangerouslySetInnerHTML={{ __html: vizHtml }}
              />
            </div>
          )}

          {!vizHtml && result && (
            <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center text-gray-400">
              <p className="text-4xl mb-2">📈</p>
              <p className="text-sm">
                Run analysis to see interactive visualization
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultDisplay({
  result,
  tab,
}: {
  result: Record<string, unknown>;
  tab: AnalysisTab;
}) {
  if (tab === "symmetry") {
    const sg = result.space_group as Record<string, unknown> | undefined;
    return (
      <div className="space-y-2 text-sm">
        {sg && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-500">Space Group:</span>
              <span className="font-mono">
                {String(sg.space_group_symbol || sg.space_group_number || "N/A")}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Crystal System:</span>
              <span>{String(sg.crystal_system || "N/A")}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Point Group:</span>
              <span>{String(sg.point_group || "N/A")}</span>
            </div>
          </>
        )}
        {Array.isArray(result.irreducible_representations) && (
          <div>
            <span className="text-gray-500">Irreducible Representations:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {(result.irreducible_representations as Record<string, unknown>[]).map(
                (irrep, i) => (
                  <span
                    key={i}
                    className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-xs"
                  >
                    {String(irrep.label)}
                    {irrep.dimension && Number(irrep.dimension) > 1
                      ? ` (${irrep.dimension}D)`
                      : ""}
                  </span>
                )
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (tab === "spectral") {
    const dos = result.dos as Record<string, unknown> | undefined;
    return (
      <div className="space-y-2 text-sm">
        {dos && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-500">Band Gap:</span>
              <span className="font-mono">
                {dos.band_gap != null ? `${dos.band_gap} eV` : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Gap Type:</span>
              <span>{String(dos.gap_type || "N/A")}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Fermi Energy:</span>
              <span className="font-mono">
                {dos.fermi_energy != null ? `${dos.fermi_energy} eV` : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">DOS at E_F:</span>
              <span className="font-mono">
                {dos.dos_at_fermi != null ? String(dos.dos_at_fermi) : "N/A"}
              </span>
            </div>
          </>
        )}
      </div>
    );
  }

  if (tab === "dynamics") {
    const chaos = result.chaos as Record<string, unknown> | undefined;
    const lyap = result.lyapunov as Record<string, unknown> | undefined;
    const fractal = result.fractal as Record<string, unknown> | undefined;
    return (
      <div className="space-y-2 text-sm">
        {chaos && (
          <div
            className={`px-3 py-2 rounded font-medium ${
              chaos.is_chaotic
                ? "bg-red-100 text-red-800"
                : "bg-green-100 text-green-800"
            }`}
          >
            {String(chaos.classification).toUpperCase()}
          </div>
        )}
        {lyap && (
          <div className="flex justify-between">
            <span className="text-gray-500">Max Lyapunov:</span>
            <span className="font-mono">{String(lyap.max_lyapunov ?? "N/A")}</span>
          </div>
        )}
        {fractal && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-500">Corr. Dimension:</span>
              <span className="font-mono">
                {String(fractal.correlation_dimension ?? "N/A")}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Hurst Exponent:</span>
              <span className="font-mono">
                {String(fractal.hurst_exponent ?? "N/A")}
              </span>
            </div>
          </>
        )}
      </div>
    );
  }

  if (tab === "tda") {
    const betti = result.betti_numbers as Record<string, unknown> | undefined;
    return (
      <div className="space-y-2 text-sm">
        {betti && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-500">β₀ (connected):</span>
              <span className="font-mono">{String(betti.beta_0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">β₁ (loops):</span>
              <span className="font-mono">{String(betti.beta_1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">β₂ (voids):</span>
              <span className="font-mono">{String(betti.beta_2)}</span>
            </div>
          </>
        )}
        {result.persistence_entropy != null && (
          <div className="flex justify-between">
            <span className="text-gray-500">Persistence Entropy:</span>
            <span className="font-mono">
              {Number(result.persistence_entropy).toFixed(4)}
            </span>
          </div>
        )}
      </div>
    );
  }

  if (tab === "sindy") {
    const equations = result.equations as Record<string, unknown>[] | undefined;
    const isFallback = String(result.optimizer_type || "").includes("fallback");
    return (
      <div className="space-y-3 text-sm">
        {isFallback && (
          <div className="bg-yellow-50 border border-yellow-200 rounded px-3 py-2 text-xs text-yellow-700">
            Using fallback STLSQ (pysindy not installed). Install pysindy for better results.
          </div>
        )}
        {result.model_score != null && (
          <div className="flex justify-between">
            <span className="text-gray-500">Model Score:</span>
            <span className="font-mono">{Number(result.model_score).toFixed(4)}</span>
          </div>
        )}
        {result.complexity != null && (
          <div className="flex justify-between">
            <span className="text-gray-500">Complexity:</span>
            <span className="font-mono">{String(result.complexity)} active terms</span>
          </div>
        )}
        {result.library_type && (
          <div className="flex justify-between">
            <span className="text-gray-500">Library:</span>
            <span>{String(result.library_type)}</span>
          </div>
        )}
        {equations && equations.length > 0 && (
          <div className="space-y-2">
            <span className="text-gray-500">Discovered Equations:</span>
            {equations.map((eq, i) => (
              <div
                key={i}
                className="bg-indigo-50 border border-indigo-200 rounded px-3 py-2 font-mono text-xs"
              >
                d{String(eq.variable_name)}/dt = {String(eq.equation || "0")}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <pre className="text-xs overflow-auto max-h-64">
      {JSON.stringify(result, null, 2)}
    </pre>
  );
}
