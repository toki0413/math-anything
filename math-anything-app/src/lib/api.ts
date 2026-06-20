let _baseUrl: string | null = null;

async function getBaseUrl(): Promise<string> {
  if (_baseUrl) return _baseUrl;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const port = await invoke<number>("get_backend_port");
    _baseUrl = `http://127.0.0.1:${port}`;
    return _baseUrl;
  } catch {
    _baseUrl = "http://localhost:8000";
    return _baseUrl;
  }
}

export async function api<T = unknown>(
  path: string,
  opts?: RequestInit
): Promise<T> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function apiUpload<T = unknown>(
  path: string,
  formData: FormData
): Promise<T> {
  const base = await getBaseUrl();
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    await api("/health");
    return true;
  } catch {
    return false;
  }
}

export function wsUrl(path: string): Promise<string> {
  return getBaseUrl().then((base) =>
    base.replace(/^http/, "ws") + path
  );
}

export async function analyzeSymmetry(params: {
  lattice?: number[][];
  positions?: number[][];
  numbers?: number[];
  space_group_hint?: string;
}) {
  return api<Record<string, unknown>>("/analyze/symmetry", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function analyzeSpectral(params: {
  eigenvalues: number[];
  k_grid?: number[][];
  weights?: number[];
  occupied_bands?: number;
  time_reversal?: boolean;
}) {
  return api<Record<string, unknown>>("/analyze/spectral", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function analyzeDynamics(params: {
  time_series: number[];
  min_tsep?: number;
}) {
  return api<Record<string, unknown>>("/analyze/dynamics", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function analyzeTDA(params: {
  point_cloud?: number[][];
  data_type?: string;
  max_dim?: number;
}) {
  return api<Record<string, unknown>>("/analyze/tda", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function vizDOS(params: {
  eigenvalues: number[];
  occupied_bands?: number;
}) {
  return api<{ html?: string; fallback?: boolean }>("/viz/dos", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function vizPhase(params: {
  time_series: number[];
}) {
  return api<{ html?: string; fallback?: boolean }>("/viz/phase", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function vizPersistence(params: {
  point_cloud: number[][];
  max_dim?: number;
}) {
  return api<{ html?: string; fallback?: boolean }>("/viz/persistence", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function analyzeSINDy(params: {
  time_series: number[][];
  dt?: number;
  poly_order?: number;
  threshold?: number;
  variable_names?: string[];
}) {
  return api<Record<string, unknown>>("/analyze/sindy", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function vizSINDy(params: {
  time_series: number[][];
  dt?: number;
  poly_order?: number;
  threshold?: number;
  variable_names?: string[];
}) {
  return api<{ html?: string; fallback?: boolean }>("/viz/sindy", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function sandboxExecute(params: {
  code: string;
  timeout_seconds?: number;
  backend?: string;
}) {
  return api<Record<string, unknown>>("/sandbox/execute", {
    method: "POST",
    body: JSON.stringify(params),
  });
}
