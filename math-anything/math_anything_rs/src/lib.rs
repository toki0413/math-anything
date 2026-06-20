//! math_anything_rs — High-performance Rust core for math-anything
//!
//! Provides fast implementations of:
//!   - EML operator evaluation and expression tree manipulation
//!   - Category graph traversal (morphism chains)
//!   - Buckingham π group computation (dimensional analysis)
//!   - Expression simplification

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::DMatrix;
use std::collections::{VecDeque, HashSet};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────
// EML operator
// ─────────────────────────────────────────────────────────

/// The EML (Exp-Minus-Log) operator: eml(x, y) = exp(x) - ln(y)
/// 
/// This is the universal binary operator from Odrzywolek's paper
/// (arXiv:2603.21852). All elementary functions can be expressed
/// as compositions of EML with the constants 0, 1, e, π, i.
#[pyfunction]
#[inline]
pub fn eml(x: f64, y: f64) -> f64 {
    if y <= 0.0 {
        return f64::INFINITY;
    }
    let exp_x = if x > 700.0 { f64::INFINITY } else if x < -700.0 { 0.0 } else { x.exp() };
    exp_x - y.ln()
}

/// Safe EML with clipping
#[inline]
pub fn eml_safe(x: f64, y: f64) -> f64 {
    let result = eml(x, y);
    if result.is_infinite() || result.is_nan() {
        if result.is_infinite() && result.is_sign_positive() { 1e10 }
        else if result.is_infinite() && result.is_sign_negative() { -1e10 }
        else { 0.0 }
    } else {
        result
    }
}

/// exp(x) = EML(x, 1) — derived from the universal operator
#[inline]
pub fn eml_exp(x: f64) -> f64 {
    eml(x, 1.0)
}

/// ln(x) = 1 - EML(0, x) — derived from the universal operator
#[inline]
pub fn eml_ln(x: f64) -> f64 {
    1.0 - eml(0.0, x)
}

/// sin(x) — expressible via EML with complex constants (Euler formula)
/// sin(x) = (e^{ix} - e^{-ix}) / (2i)
///        = (EML(ix, 1) - EML(-ix, 1)) / (2i)
#[inline]
pub fn eml_sin(x: f64) -> f64 {
    x.sin() // numerical fallback; the EML representation uses complex numbers
}

// ─────────────────────────────────────────────────────────
// EML closure computation
// ─────────────────────────────────────────────────────────

/// Compute the EML closure of a set of constants up to a given depth.
/// 
/// Starting from base constants {e, π, 0, 1}, repeatedly apply EML(x, y)
/// for all pairs (x, y) in the set, collecting results up to `max_depth` iterations.
/// 
/// This gives an approximation of the "elementary numbers" reachable from
/// the base set via finite EML compositions.
#[pyfunction]
pub fn eml_closure(base: Vec<f64>, max_depth: usize, max_size: usize) -> Vec<f64> {
    let mut closure: HashSet<u64> = HashSet::new(); // store as ordered float bits for dedup
    let mut current: Vec<f64> = base.iter().cloned().collect();
    
    for val in &base {
        closure.insert(val.to_bits());
    }
    
    for _depth in 0..max_depth {
        let mut next: Vec<f64> = Vec::new();
        
        for i in 0..current.len() {
            for j in 0..current.len() {
                let result = eml_safe(current[i], current[j]);
                let bits = result.to_bits();
                if !closure.contains(&bits) {
                    closure.insert(bits);
                    next.push(result);
                    if closure.len() >= max_size {
                        return closure.iter().map(|b| f64::from_bits(*b)).collect();
                    }
                }
            }
        }
        
        if next.is_empty() {
            break;
        }
        current = next;
    }
    
    closure.iter().map(|b| f64::from_bits(*b)).collect()
}

/// Find the simplest EML representation for a target value.
/// 
/// Uses breadth-first search over the EML closure tree.
/// Returns a representation string like "EML(EML(0,1), pi)" or None.
#[pyfunction]
pub fn find_eml_representation(target: f64, base: Vec<f64>, max_depth: usize) -> Option<String> {
    #[derive(Clone)]
    struct Node {
        value: f64,
        expr: String,
        depth: usize,
    }
    
    let mut queue: VecDeque<Node> = VecDeque::new();
    let mut generated: Vec<f64> = Vec::new();
    let eps: f64 = 1e-10;
    
    for (_i, &val) in base.iter().enumerate() {
        generated.push(val);
        queue.push_back(Node {
            value: val,
            expr: if val == std::f64::consts::E { "e".to_string() }
                 else if (val - std::f64::consts::PI).abs() < 1e-10 { "π".to_string() }
                 else if val.abs() < 1e-10 { "0".to_string() }
                 else if (val - 1.0).abs() < 1e-10 { "1".to_string() }
                 else { format!("{:.6}", val) },
            depth: 0,
        });
    }
    
    while let Some(node) = queue.pop_front() {
        if (node.value - target).abs() < eps * (1.0 + target.abs()) {
            return Some(node.expr);
        }
        
        if node.depth >= max_depth {
            continue;
        }
        
        let n_generated = generated.len();
        for i in 0..n_generated {
            for j in 0..n_generated {
                let result = eml_safe(generated[i], generated[j]);
                if result.is_finite() && !result.is_nan() {
                    generated.push(result);
                    queue.push_back(Node {
                        value: result,
                        expr: format!("EML({}, {})", 
                            if generated[i] == std::f64::consts::E { "e".to_string() }
                            else if (generated[i] - std::f64::consts::PI).abs() < 1e-10 { "π".to_string() }
                            else if generated[i].abs() < 1e-10 { "0".to_string() }
                            else if (generated[i] - 1.0).abs() < 1e-10 { "1".to_string() }
                            else { format!("{:.6}", generated[i]) },
                            if generated[j] == std::f64::consts::E { "e".to_string() }
                            else if (generated[j] - std::f64::consts::PI).abs() < 1e-10 { "π".to_string() }
                            else if generated[j].abs() < 1e-10 { "0".to_string() }
                            else if (generated[j] - 1.0).abs() < 1e-10 { "1".to_string() }
                            else { format!("{:.6}", generated[j]) }
                        ),
                        depth: node.depth + 1,
                    });
                }
            }
        }
    }
    
    None
}

// ─────────────────────────────────────────────────────────
// Buckingham π: matrix nullspace via nalgebra
// ─────────────────────────────────────────────────────────

/// Compute Buckingham π groups from a dimension matrix.
///
/// Input: dimension_matrix[ndims][nvars] — each column is a variable's dimension vector
/// Returns: list of π groups as exponent vectors (each row = one π group)
#[pyfunction]
pub fn buckingham_pi(matrix_rows: usize, matrix_cols: usize,
                     data: Vec<f64>) -> Vec<Vec<f64>> {
    buckingham_pi_single(matrix_rows, matrix_cols, &data)
}

/// Core Buckingham π computation (extracted for reuse by parallel version).
fn buckingham_pi_single(matrix_rows: usize, matrix_cols: usize,
                        data: &[f64]) -> Vec<Vec<f64>> {
    if matrix_rows == 0 || matrix_cols == 0 || data.len() != matrix_rows * matrix_cols {
        return Vec::new();
    }

    let dm = DMatrix::from_row_slice(matrix_rows, matrix_cols, data);

    // Compute nullspace via SVD
    let svd = dm.svd(true, true);
    let v_t = match svd.v_t {
        Some(ref m) => m,
        None => return Vec::new(),
    };
    let rank = svd.rank(1e-10);

    let nullspace_dim = matrix_cols.saturating_sub(rank);
    if nullspace_dim == 0 || v_t.nrows() < matrix_cols {
        return Vec::new();
    }

    let mut pi_groups = Vec::with_capacity(nullspace_dim);

    let start_row = matrix_cols.saturating_sub(nullspace_dim);
    for i in start_row..matrix_cols {
        if i >= v_t.nrows() {
            break;
        }
        let mut group = Vec::with_capacity(matrix_cols);
        for j in 0..matrix_cols.min(v_t.ncols()) {
            group.push(v_t[(i, j)]);
        }
        pi_groups.push(group);
    }

    pi_groups
}

/// Compute Buckingham Pi groups for multiple dimension matrices in parallel.
#[pyfunction]
pub fn parallel_buckingham_pi(matrices: Vec<(usize, usize, Vec<f64>)>) -> Vec<Vec<Vec<f64>>> {
    matrices.par_iter().map(|(rows, cols, data)| {
        buckingham_pi_single(*rows, *cols, data)
    }).collect()
}

// ─────────────────────────────────────────────────────────
// Graph traversal for category engine
// ─────────────────────────────────────────────────────────

/// Find shortest path between two nodes in an adjacency list.
/// Returns list of edge indices on the path, or empty if no path.
#[pyfunction]
pub fn shortest_path(
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    start: usize,
    end: usize,
) -> Vec<usize> {
    if start >= n_nodes || end >= n_nodes {
        return Vec::new();
    }
    
    // Build adjacency list
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_nodes];
    for (idx, &(u, v)) in edges.iter().enumerate() {
        if u < n_nodes && v < n_nodes {
            adj[u].push((v, idx));
        }
    }
    
    let mut queue = VecDeque::new();
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; n_nodes];
    let mut visited = vec![false; n_nodes];
    
    queue.push_back(start);
    visited[start] = true;
    
    while let Some(u) = queue.pop_front() {
        if u == end {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = end;
            while cur != start {
                if let Some((prev, edge_idx)) = parent[cur] {
                    path.push(edge_idx);
                    cur = prev;
                } else {
                    break;
                }
            }
            path.reverse();
            return path;
        }
        
        for &(v, edge_idx) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                parent[v] = Some((u, edge_idx));
                queue.push_back(v);
            }
        }
    }
    
    Vec::new()
}

// ─────────────────────────────────────────────────────────
// WHNF normalization for MLTT type checking
// ─────────────────────────────────────────────────────────

/// WHNF (Weak Head Normal Form) normalization for MLTT terms.
///
/// Takes a serialized term (as JSON string) and returns the WHNF result.
/// Supports: beta-reduction, projection reduction.
///
/// Input format: {"kind": "App", "func": {...}, "arg": {...}}
/// Output format: {"kind": "...", ...} (normalized)
#[pyfunction]
pub fn whnf_normalize(term_json: String) -> PyResult<String> {
    let term: serde_json::Value = serde_json::from_str(&term_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(
            format!("Invalid term JSON: {}", e)
        ))?;

    let result = whnf_step(&term);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Serialization error: {}", e)
        ))
}

fn whnf_step(term: &serde_json::Value) -> serde_json::Value {
    let kind = term.get("kind").and_then(|v| v.as_str()).unwrap_or("");

    match kind {
        "App" => {
            let func = term.get("func");
            let arg = term.get("arg");
            match (func, arg) {
                (Some(f), Some(a)) => {
                    let func_norm = whnf_step(f);
                    let func_kind = func_norm.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                    if func_kind == "Lam" {
                        // Beta reduction: (λx. body)(a) → body[x := a]
                        let var_name = func_norm.get("var_name").and_then(|v| v.as_str()).unwrap_or("x");
                        let body = func_norm.get("body").cloned().unwrap_or(serde_json::Value::Null);
                        substitute_json(&body, var_name, a)
                    } else {
                        let mut result = serde_json::Map::new();
                        result.insert("kind".to_string(), serde_json::Value::String("App".to_string()));
                        result.insert("func".to_string(), func_norm);
                        result.insert("arg".to_string(), a.clone());
                        serde_json::Value::Object(result)
                    }
                }
                _ => term.clone(),
            }
        }
        "Proj1" => {
            let pair = term.get("pair").and_then(|v| Some(whnf_step(v)));
            if let Some(p) = &pair {
                let p_kind = p.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                if p_kind == "Pair" {
                    return p.get("fst").cloned().unwrap_or(serde_json::Value::Null);
                }
            }
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Proj1".to_string()));
            if let Some(p) = pair { result.insert("pair".to_string(), p); }
            serde_json::Value::Object(result)
        }
        "Proj2" => {
            let pair = term.get("pair").and_then(|v| Some(whnf_step(v)));
            if let Some(p) = &pair {
                let p_kind = p.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                if p_kind == "Pair" {
                    return p.get("snd").cloned().unwrap_or(serde_json::Value::Null);
                }
            }
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Proj2".to_string()));
            if let Some(p) = pair { result.insert("pair".to_string(), p); }
            serde_json::Value::Object(result)
        }
        _ => term.clone(), // Already in WHNF
    }
}

fn substitute_json(term: &serde_json::Value, var_name: &str, replacement: &serde_json::Value) -> serde_json::Value {
    let kind = term.get("kind").and_then(|v| v.as_str()).unwrap_or("");

    match kind {
        "Var" => {
            let name = term.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if name == var_name { replacement.clone() } else { term.clone() }
        }
        "Lam" => {
            let lam_var = term.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let body = term.get("body").cloned().unwrap_or(serde_json::Value::Null);
            if lam_var == var_name {
                // Variable is shadowed
                term.clone()
            } else {
                let mut result = serde_json::Map::new();
                result.insert("kind".to_string(), serde_json::Value::String("Lam".to_string()));
                result.insert("var_name".to_string(), serde_json::Value::String(lam_var.to_string()));
                result.insert("body".to_string(), substitute_json(&body, var_name, replacement));
                serde_json::Value::Object(result)
            }
        }
        "App" => {
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("App".to_string()));
            result.insert("func".to_string(),
                substitute_json(term.get("func").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            result.insert("arg".to_string(),
                substitute_json(term.get("arg").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            serde_json::Value::Object(result)
        }
        "Pi" => {
            let pi_var = term.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Pi".to_string()));
            result.insert("var_name".to_string(), serde_json::Value::String(pi_var.to_string()));
            result.insert("domain".to_string(),
                substitute_json(term.get("domain").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            if pi_var == var_name {
                result.insert("codomain".to_string(), term.get("codomain").cloned().unwrap_or(serde_json::Value::Null));
            } else {
                result.insert("codomain".to_string(),
                    substitute_json(term.get("codomain").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            }
            serde_json::Value::Object(result)
        }
        "Sigma" => {
            let sig_var = term.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Sigma".to_string()));
            result.insert("var_name".to_string(), serde_json::Value::String(sig_var.to_string()));
            result.insert("fst_type".to_string(),
                substitute_json(term.get("fst_type").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            if sig_var == var_name {
                result.insert("snd_type".to_string(), term.get("snd_type").cloned().unwrap_or(serde_json::Value::Null));
            } else {
                result.insert("snd_type".to_string(),
                    substitute_json(term.get("snd_type").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            }
            serde_json::Value::Object(result)
        }
        "Pair" => {
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Pair".to_string()));
            result.insert("fst".to_string(),
                substitute_json(term.get("fst").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            result.insert("snd".to_string(),
                substitute_json(term.get("snd").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            serde_json::Value::Object(result)
        }
        "Identity" => {
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Identity".to_string()));
            result.insert("typ".to_string(),
                substitute_json(term.get("typ").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            result.insert("lhs".to_string(),
                substitute_json(term.get("lhs").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            result.insert("rhs".to_string(),
                substitute_json(term.get("rhs").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            serde_json::Value::Object(result)
        }
        "Refl" => {
            let mut result = serde_json::Map::new();
            result.insert("kind".to_string(), serde_json::Value::String("Refl".to_string()));
            result.insert("typ".to_string(),
                substitute_json(term.get("typ").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            result.insert("term".to_string(),
                substitute_json(term.get("term").unwrap_or(&serde_json::Value::Null), var_name, replacement));
            serde_json::Value::Object(result)
        }
        _ => term.clone(), // Universe, Var (already handled), etc.
    }
}

// ─────────────────────────────────────────────────────────
// Batch constraint propagation
// ─────────────────────────────────────────────────────────

/// Batch constraint propagation along morphism chains.
///
/// Given a list of morphisms (each with invariants_kept/lost/introduced),
/// and a list of invariant names, compute the propagation result for each
/// invariant through the entire chain.
///
/// Returns: list of (invariant_name, status, lost_at_step) tuples
///   status: "preserved" | "lost" | "emerged"
///   lost_at_step: step index where invariant was lost, or None
#[pyfunction]
pub fn propagate_constraints(
    invariant_names: Vec<String>,
    morphism_kept: Vec<Vec<String>>,
    morphism_lost: Vec<Vec<String>>,
    morphism_introduced: Vec<Vec<String>>,
) -> Vec<(String, String, Option<usize>)> {
    let mut results = Vec::new();

    for inv_name in &invariant_names {
        let mut status = "preserved".to_string();
        let mut lost_at: Option<usize> = None;
        let mut found_introduced = false;

        for (step, ((_kept, lost), introduced)) in morphism_kept.iter()
            .zip(morphism_lost.iter())
            .zip(morphism_introduced.iter())
            .enumerate()
        {
            if lost.contains(inv_name) {
                status = "lost".to_string();
                lost_at = Some(step);
                break;
            }
            if introduced.contains(inv_name) {
                found_introduced = true;
            }
            // If not in kept and not in lost, it's implicitly preserved
        }

        if found_introduced && status == "preserved" {
            status = "emerged".to_string();
        }

        results.push((inv_name.clone(), status, lost_at));
    }

    results
}

// ─────────────────────────────────────────────────────────
// Batch expression evaluator for PSRN symbolic regression
// ─────────────────────────────────────────────────────────

/// Batch evaluate mathematical expressions on data arrays.
///
/// This is the core acceleration for PSRN symbolic regression.
/// Supports: +, -, *, /, ^ (power), sin, cos, exp, log, sqrt, abs,
///           parentheses, variable names, numeric literals.
///
/// Args:
///   expressions: list of expression strings
///   var_names: list of variable names
///   data_rows: number of rows in data
///   data_flat: flattened data array (row-major, len = data_rows * var_names.len())
///
/// Returns: list of result arrays (one per expression), empty vec for parse errors
#[pyfunction]
pub fn batch_eval_expressions(
    expressions: Vec<String>,
    var_names: Vec<String>,
    data_rows: usize,
    data_flat: Vec<f64>,
) -> Vec<Vec<f64>> {
    let n_vars = var_names.len();
    if n_vars == 0 || data_rows == 0 || data_flat.len() != data_rows * n_vars {
        return expressions.iter().map(|_| Vec::new()).collect();
    }

    // Parse all expressions in parallel
    let parsed: Vec<Option<ExprNode>> = expressions.par_iter().map(|expr| {
        match parse_expr(expr) {
            Ok(ast) => Some(ast),
            Err(_) => None,
        }
    }).collect();

    // Evaluate each expression in parallel
    parsed.par_iter().map(|ast_opt| {
        match ast_opt {
            None => Vec::new(),
            Some(ast) => {
                let mut row_results = Vec::with_capacity(data_rows);
                for row in 0..data_rows {
                    let mut ctx: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
                    for (j, name) in var_names.iter().enumerate() {
                        ctx.insert(name.as_str(), data_flat[row * n_vars + j]);
                    }
                    row_results.push(eval_ast(ast, &ctx).unwrap_or(f64::NAN));
                }
                row_results
            }
        }
    }).collect()
}

// ── Simple expression AST ──

enum ExprNode {
    Num(f64),
    Var(String),
    Add(Box<ExprNode>, Box<ExprNode>),
    Sub(Box<ExprNode>, Box<ExprNode>),
    Mul(Box<ExprNode>, Box<ExprNode>),
    Div(Box<ExprNode>, Box<ExprNode>),
    Pow(Box<ExprNode>, Box<ExprNode>),
    Neg(Box<ExprNode>),
    Sin(Box<ExprNode>),
    Cos(Box<ExprNode>),
    Exp(Box<ExprNode>),
    Log(Box<ExprNode>),
    Sqrt(Box<ExprNode>),
    Abs(Box<ExprNode>),
}

fn eval_ast(node: &ExprNode, ctx: &std::collections::HashMap<&str, f64>) -> Result<f64, ()> {
    match node {
        ExprNode::Num(v) => Ok(*v),
        ExprNode::Var(name) => ctx.get(name.as_str()).copied().ok_or(()),
        ExprNode::Add(a, b) => Ok(eval_ast(a, ctx)? + eval_ast(b, ctx)?),
        ExprNode::Sub(a, b) => Ok(eval_ast(a, ctx)? - eval_ast(b, ctx)?),
        ExprNode::Mul(a, b) => Ok(eval_ast(a, ctx)? * eval_ast(b, ctx)?),
        ExprNode::Div(a, b) => {
            let d = eval_ast(b, ctx)?;
            if d.abs() < 1e-300 { Ok(f64::NAN) } else { Ok(eval_ast(a, ctx)? / d) }
        }
        ExprNode::Pow(a, b) => {
            let base = eval_ast(a, ctx)?;
            let exp = eval_ast(b, ctx)?;
            Ok(base.powf(exp))
        }
        ExprNode::Neg(a) => Ok(-eval_ast(a, ctx)?),
        ExprNode::Sin(a) => Ok(eval_ast(a, ctx)?.sin()),
        ExprNode::Cos(a) => Ok(eval_ast(a, ctx)?.cos()),
        ExprNode::Exp(a) => Ok(eval_ast(a, ctx)?.exp()),
        ExprNode::Log(a) => {
            let v = eval_ast(a, ctx)?;
            if v <= 0.0 { Ok(f64::NAN) } else { Ok(v.ln()) }
        }
        ExprNode::Sqrt(a) => {
            let v = eval_ast(a, ctx)?;
            if v < 0.0 { Ok(f64::NAN) } else { Ok(v.sqrt()) }
        }
        ExprNode::Abs(a) => Ok(eval_ast(a, ctx)?.abs()),
    }
}

// ── Recursive descent parser ──

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { chars: input.chars().peekable() }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() { self.chars.next(); } else { break; }
        }
    }

    fn parse_expr_inner(&mut self) -> Result<ExprNode, String> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<ExprNode, String> {
        let mut left = self.parse_multiplicative()?;
        loop {
            self.skip_whitespace();
            match self.chars.peek() {
                Some('+') => { self.chars.next(); let right = self.parse_multiplicative()?; left = ExprNode::Add(Box::new(left), Box::new(right)); }
                Some('-') => { self.chars.next(); let right = self.parse_multiplicative()?; left = ExprNode::Sub(Box::new(left), Box::new(right)); }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<ExprNode, String> {
        let mut left = self.parse_power()?;
        loop {
            self.skip_whitespace();
            match self.chars.peek() {
                Some('*') => { self.chars.next(); let right = self.parse_power()?; left = ExprNode::Mul(Box::new(left), Box::new(right)); }
                Some('/') => { self.chars.next(); let right = self.parse_power()?; left = ExprNode::Div(Box::new(left), Box::new(right)); }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<ExprNode, String> {
        let base = self.parse_unary()?;
        self.skip_whitespace();
        if self.chars.peek() == Some(&'^') {
            self.chars.next();
            let exp = self.parse_unary()?; // right-associative
            Ok(ExprNode::Pow(Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<ExprNode, String> {
        self.skip_whitespace();
        if self.chars.peek() == Some(&'-') {
            self.chars.next();
            let operand = self.parse_primary()?;
            Ok(ExprNode::Neg(Box::new(operand)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<ExprNode, String> {
        self.skip_whitespace();
        match self.chars.peek() {
            Some(&c) if c.is_ascii_digit() || c == '.' => self.parse_number(),
            Some(&c) if c.is_alphabetic() || c == '_' => self.parse_ident_or_func(),
            Some('(') => {
                self.chars.next();
                let inner = self.parse_expr_inner()?;
                self.skip_whitespace();
                if self.chars.peek() == Some(&')') { self.chars.next(); }
                Ok(inner)
            }
            _ => Err(format!("Unexpected character: {:?}", self.chars.peek())),
        }
    }

    fn parse_number(&mut self) -> Result<ExprNode, String> {
        let mut s = String::new();
        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+' {
                s.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        s.parse::<f64>().map(ExprNode::Num).map_err(|e| format!("Invalid number '{}': {}", s, e))
    }

    fn parse_ident_or_func(&mut self) -> Result<ExprNode, String> {
        let mut name = String::new();
        while let Some(&c) = self.chars.peek() {
            if c.is_alphanumeric() || c == '_' {
                name.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        self.skip_whitespace();
        // Check if it's a function call
        if self.chars.peek() == Some(&'(') {
            self.chars.next();
            let arg = self.parse_expr_inner()?;
            self.skip_whitespace();
            if self.chars.peek() == Some(&')') { self.chars.next(); }
            match name.as_str() {
                "sin" => Ok(ExprNode::Sin(Box::new(arg))),
                "cos" => Ok(ExprNode::Cos(Box::new(arg))),
                "exp" => Ok(ExprNode::Exp(Box::new(arg))),
                "log" | "ln" => Ok(ExprNode::Log(Box::new(arg))),
                "sqrt" => Ok(ExprNode::Sqrt(Box::new(arg))),
                "abs" => Ok(ExprNode::Abs(Box::new(arg))),
                _ => Err(format!("Unknown function: {}", name)),
            }
        } else {
            Ok(ExprNode::Var(name))
        }
    }
}

fn parse_expr(input: &str) -> Result<ExprNode, String> {
    let mut parser = Parser::new(input);
    let result = parser.parse_expr_inner()?;
    parser.skip_whitespace();
    if parser.chars.peek().is_some() {
        Err("Unexpected trailing characters".to_string())
    } else {
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────
// Definitional equality check for MLTT type theory
// ─────────────────────────────────────────────────────────

/// Check definitional equality of two MLTT terms.
///
/// Both terms are given as JSON strings. Returns true if they are definitionally equal.
/// Algorithm: normalize both to WHNF, then check structural equality with recursive def_eq.
#[pyfunction]
pub fn check_def_eq(term_a_json: String, term_b_json: String) -> PyResult<bool> {
    let a: serde_json::Value = serde_json::from_str(&term_a_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid term_a JSON: {}", e)))?;
    let b: serde_json::Value = serde_json::from_str(&term_b_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid term_b JSON: {}", e)))?;

    Ok(def_eq(&a, &b, 0))
}

fn def_eq(a: &serde_json::Value, b: &serde_json::Value, depth: usize) -> bool {
    if depth > 100 { return false; } // fuel limit

    let a_norm = whnf_step(a);
    let b_norm = whnf_step(b);

    structural_eq(&a_norm, &b_norm, depth)
}

fn structural_eq(a: &serde_json::Value, b: &serde_json::Value, depth: usize) -> bool {
    let a_kind = a.get("kind").and_then(|v| v.as_str()).unwrap_or("");
    let b_kind = b.get("kind").and_then(|v| v.as_str()).unwrap_or("");

    if a_kind != b_kind { return false; }

    match a_kind {
        "Universe" => {
            let a_lvl = a.get("level").and_then(|v| v.as_u64()).unwrap_or(0);
            let b_lvl = b.get("level").and_then(|v| v.as_u64()).unwrap_or(0);
            a_lvl == b_lvl
        }
        "Var" => {
            let a_name = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let b_name = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
            a_name == b_name
        }
        "Lam" => {
            let a_var = a.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let b_var = b.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            if a_var != b_var { return false; }
            let a_body = a.get("body").unwrap_or(&serde_json::Value::Null);
            let b_body = b.get("body").unwrap_or(&serde_json::Value::Null);
            def_eq(a_body, b_body, depth + 1)
        }
        "Pi" => {
            let a_var = a.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let b_var = b.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            if a_var != b_var { return false; }
            let a_dom = a.get("domain").unwrap_or(&serde_json::Value::Null);
            let b_dom = b.get("domain").unwrap_or(&serde_json::Value::Null);
            let a_cod = a.get("codomain").unwrap_or(&serde_json::Value::Null);
            let b_cod = b.get("codomain").unwrap_or(&serde_json::Value::Null);
            def_eq(a_dom, b_dom, depth + 1) && def_eq(a_cod, b_cod, depth + 1)
        }
        "App" => {
            let a_func = a.get("func").unwrap_or(&serde_json::Value::Null);
            let b_func = b.get("func").unwrap_or(&serde_json::Value::Null);
            let a_arg = a.get("arg").unwrap_or(&serde_json::Value::Null);
            let b_arg = b.get("arg").unwrap_or(&serde_json::Value::Null);
            def_eq(a_func, b_func, depth + 1) && def_eq(a_arg, b_arg, depth + 1)
        }
        "Sigma" => {
            let a_var = a.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            let b_var = b.get("var_name").and_then(|v| v.as_str()).unwrap_or("");
            if a_var != b_var { return false; }
            let a_fst = a.get("fst_type").unwrap_or(&serde_json::Value::Null);
            let b_fst = b.get("fst_type").unwrap_or(&serde_json::Value::Null);
            let a_snd = a.get("snd_type").unwrap_or(&serde_json::Value::Null);
            let b_snd = b.get("snd_type").unwrap_or(&serde_json::Value::Null);
            def_eq(a_fst, b_fst, depth + 1) && def_eq(a_snd, b_snd, depth + 1)
        }
        "Pair" => {
            let a_fst = a.get("fst").unwrap_or(&serde_json::Value::Null);
            let b_fst = b.get("fst").unwrap_or(&serde_json::Value::Null);
            let a_snd = a.get("snd").unwrap_or(&serde_json::Value::Null);
            let b_snd = b.get("snd").unwrap_or(&serde_json::Value::Null);
            def_eq(a_fst, b_fst, depth + 1) && def_eq(a_snd, b_snd, depth + 1)
        }
        "Identity" => {
            let a_typ = a.get("typ").unwrap_or(&serde_json::Value::Null);
            let b_typ = b.get("typ").unwrap_or(&serde_json::Value::Null);
            let a_lhs = a.get("lhs").unwrap_or(&serde_json::Value::Null);
            let b_lhs = b.get("lhs").unwrap_or(&serde_json::Value::Null);
            let a_rhs = a.get("rhs").unwrap_or(&serde_json::Value::Null);
            let b_rhs = b.get("rhs").unwrap_or(&serde_json::Value::Null);
            def_eq(a_typ, b_typ, depth + 1) && def_eq(a_lhs, b_lhs, depth + 1) && def_eq(a_rhs, b_rhs, depth + 1)
        }
        "Refl" => {
            let a_typ = a.get("typ").unwrap_or(&serde_json::Value::Null);
            let b_typ = b.get("typ").unwrap_or(&serde_json::Value::Null);
            let a_term = a.get("term").unwrap_or(&serde_json::Value::Null);
            let b_term = b.get("term").unwrap_or(&serde_json::Value::Null);
            def_eq(a_typ, b_typ, depth + 1) && def_eq(a_term, b_term, depth + 1)
        }
        _ => a == b, // fallback: JSON structural equality
    }
}

// ─────────────────────────────────────────────────────────
// Riemannian geometry: curvature computations
// ─────────────────────────────────────────────────────────

/// Compute Riemann curvature tensor from Christoffel symbols.
///
/// R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj}
///
/// Args:
///   christoffel: flattened Christoffel symbols [i][j][k] as (dim*dim*dim,) array
///   d_christoffel: flattened partial derivatives ∂_k Γ^i_{lj} as (dim*dim*dim*dim,) array
///                  indexed as [i][l][j][k] → flat index = i*dim^3 + l*dim^2 + j*dim + k
///   dim: manifold dimension
///
/// Returns: Riemann tensor R^i_{jkl} as (dim^4,) flattened array
///          indexed as [i][j][k][l] → flat index = i*dim^3 + j*dim^2 + k*dim + l
#[pyfunction]
pub fn compute_riemann_tensor(
    christoffel: Vec<f64>,
    d_christoffel: Vec<f64>,
    dim: usize,
) -> Vec<f64> {
    if dim == 0 || christoffel.len() != dim * dim * dim
       || d_christoffel.len() != dim * dim * dim * dim {
        return Vec::new();
    }

    let mut riemann = vec![0.0; dim * dim * dim * dim];
    let d3 = dim * dim * dim;
    let d2 = dim * dim;

    // R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Σ_m (Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj})
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                for l in 0..dim {
                    // ∂_k Γ^i_{lj}
                    let d1 = d_christoffel[i * d3 + l * d2 + j * dim + k];
                    // ∂_l Γ^i_{kj}
                    let d2_val = d_christoffel[i * d3 + k * d2 + j * dim + l];

                    let mut sum_m = 0.0;
                    for m in 0..dim {
                        // Γ^i_{km} * Γ^m_{lj}
                        let g1 = christoffel[i * d2 + k * dim + m];
                        let g2 = christoffel[m * d2 + l * dim + j];
                        // Γ^i_{lm} * Γ^m_{kj}
                        let g3 = christoffel[i * d2 + l * dim + m];
                        let g4 = christoffel[m * d2 + k * dim + j];
                        sum_m += g1 * g2 - g3 * g4;
                    }

                    riemann[i * d3 + j * d2 + k * dim + l] = d1 - d2_val + sum_m;
                }
            }
        }
    }

    riemann
}

/// Compute Ricci tensor by contracting the Riemann tensor.
///
/// R_{jk} = R^i_{jik} = Σ_i R^i_{jik}
///
/// Args:
///   riemann: flattened Riemann tensor (dim^4,)
///   dim: manifold dimension
///
/// Returns: Ricci tensor (dim*dim,) flattened
#[pyfunction]
pub fn compute_ricci_tensor(riemann: Vec<f64>, dim: usize) -> Vec<f64> {
    if dim == 0 || riemann.len() != dim * dim * dim * dim {
        return Vec::new();
    }

    let mut ricci = vec![0.0; dim * dim];
    let d3 = dim * dim * dim;
    let d2 = dim * dim;

    // R_{jk} = Σ_i R^i_{jik}
    for j in 0..dim {
        for k in 0..dim {
            let mut sum = 0.0;
            for i in 0..dim {
                // R^i_{jik} indexed as [i][j][i][k]
                sum += riemann[i * d3 + j * d2 + i * dim + k];
            }
            ricci[j * dim + k] = sum;
        }
    }

    ricci
}

/// Compute scalar curvature from Ricci tensor and inverse metric.
///
/// R = g^{jk} R_{jk}
///
/// Args:
///   ricci: flattened Ricci tensor (dim*dim,)
///   inv_metric: flattened inverse metric (dim*dim,)
///   dim: manifold dimension
///
/// Returns: scalar curvature (single f64)
#[pyfunction]
pub fn compute_scalar_curvature(
    ricci: Vec<f64>,
    inv_metric: Vec<f64>,
    dim: usize,
) -> f64 {
    if dim == 0 || ricci.len() != dim * dim || inv_metric.len() != dim * dim {
        return f64::NAN;
    }

    let mut scalar = 0.0;
    for j in 0..dim {
        for k in 0..dim {
            scalar += inv_metric[j * dim + k] * ricci[j * dim + k];
        }
    }
    scalar
}

// ─────────────────────────────────────────────────────────
// Python module initialization
// ─────────────────────────────────────────────────────────

#[pymodule]
fn math_anything_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eml, m)?)?;
    m.add_function(wrap_pyfunction!(eml_closure, m)?)?;
    m.add_function(wrap_pyfunction!(find_eml_representation, m)?)?;
    m.add_function(wrap_pyfunction!(buckingham_pi, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_buckingham_pi, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(whnf_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(batch_eval_expressions, m)?)?;
    m.add_function(wrap_pyfunction!(check_def_eq, m)?)?;
    m.add_function(wrap_pyfunction!(compute_riemann_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ricci_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(compute_scalar_curvature, m)?)?;
    
    // Constants
    let constants = PyDict::new(m.py());
    constants.set_item("E", std::f64::consts::E)?;
    constants.set_item("PI", std::f64::consts::PI)?;
    m.add("CONSTANTS", constants)?;
    
    Ok(())
}
