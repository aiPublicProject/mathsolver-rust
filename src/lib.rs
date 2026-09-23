//! mathsolver — BYOK AI math solver with independent verification.
//!
//! An answer is only `verified: true` when the model's verification
//! expression (pure arithmetic) is evaluated locally by this crate and
//! matches the answer numerically. No model output is ever executed as code.

use serde_json::{json, Value};
use std::fmt;

pub const SYSTEM_PROMPT: &str = "You are a precise math solver.
Reply with STRICT JSON only, no markdown fences, in this exact shape:
{\"program\": \"<string>\", \"steps\": [<string>, ...], \"check\": \"<string>\"}
Rules:
- \"program\" is a small JavaScript-like program that computes the final answer.
  One statement per line (or ; separated). Allowed statements:
      let NAME = EXPRESSION
      result = EXPRESSION
  EXPRESSIONs may use numbers, + - * / % ^ ( ), the functions
  abs sqrt sin cos tan ln log exp floor ceil round min max
  (log is base 10, ln is natural), the constants pi and e, and any
  variable defined by an earlier let. The value assigned to result
  is the answer. Never state the answer as a number in text.
- \"steps\" is an array of short plain-language explanation strings.
- \"check\" is a verification expression containing the placeholder {x}.
  After solving, {x} is replaced by the computed answer and the whole
  expression must evaluate to 0.
  For equations, substitute the answer back into the original equation
  (e.g. 2x+3=11 -> \"2*{x}+3-11\").
  For arithmetic, recompute via a different path and subtract the answer
  (e.g. 15% of 80 -> \"80*15/100-{x}\"). Provide \"check\" whenever possible.";

#[derive(Debug, Clone)]
pub struct SolverError {
    pub code: &'static str,
    pub message: String,
}

impl SolverError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for SolverError {}

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".into(),
            model: "gpt-4o-mini".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub answer: f64,
    pub steps: Vec<String>,
    pub expression: String,
    pub evaluated: Option<f64>,
    pub verified: bool,
    pub retries: u32,
}

/* ------------------------------------------------------------------ */
/* Expression evaluator (recursive descent, no deps)                    */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Id(String),
    Op(char),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, SolverError> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let mut num = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() || d == '.' {
                    num.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            // scientific notation
            if let Some(&e) = chars.peek() {
                if e == 'e' || e == 'E' {
                    let mut probe = chars.clone();
                    probe.next();
                    if let Some(&sign) = probe.peek() {
                        if sign == '+' || sign == '-' {
                            probe.next();
                        }
                        if probe.peek().map_or(false, |d| d.is_ascii_digit()) {
                            num.push(e);
                            chars.next();
                            if let Some(&s) = chars.peek() {
                                if s == '+' || s == '-' {
                                    num.push(s);
                                    chars.next();
                                }
                            }
                            while let Some(&d) = chars.peek() {
                                if d.is_ascii_digit() {
                                    num.push(d);
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            let value: f64 = num
                .parse()
                .map_err(|_| SolverError::new("EXPR_BAD_NUMBER", format!("bad number {}", num)))?;
            tokens.push(Tok::Num(value));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut id = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_alphanumeric() || d == '_' {
                    id.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Tok::Id(id));
            continue;
        }
        if "+-*/%^(),".contains(c) {
            tokens.push(Tok::Op(c));
            chars.next();
            continue;
        }
        return Err(SolverError::new("EXPR_BAD_CHAR", format!("unexpected character '{}'", c)));
    }
    Ok(tokens)
}

fn apply_fn(name: &str, args: &[f64]) -> Result<f64, SolverError> {
    let (a0, _a1) = (args.first().copied().unwrap_or(f64::NAN), args.get(1).copied().unwrap_or(f64::NAN));
    let v = match name {
        "abs" => a0.abs(),
        "sqrt" => a0.sqrt(),
        "sin" => a0.sin(),
        "cos" => a0.cos(),
        "tan" => a0.tan(),
        "ln" => a0.ln(),
        "log" => a0.log10(),
        "exp" => a0.exp(),
        "floor" => a0.floor(),
        "ceil" => a0.ceil(),
        "round" => a0.round(),
        "min" => args.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        _ => return Err(SolverError::new("EXPR_UNKNOWN_FUNC", format!("unknown function {}", name))),
    };
    Ok(v)
}

use std::collections::HashMap;

pub type Env = HashMap<String, f64>;

/// Evaluate a pure arithmetic expression string to a number (no variables).
pub fn eval_expression(src: &str) -> Result<f64, SolverError> {
    eval_expression_with(src, &Env::new())
}

/// Evaluate with variable bindings from let-statements.
pub fn eval_expression_with(src: &str, env: &Env) -> Result<f64, SolverError> {
    if src.trim().is_empty() {
        return Err(SolverError::new("EXPR_EMPTY", "empty expression"));
    }
    let tokens = tokenize(src)?;
    let mut pos = 0usize;

    fn peek(tokens: &[Tok], pos: usize) -> Option<&Tok> {
        tokens.get(pos)
    }

    fn eat(tokens: &[Tok], pos: &mut usize) -> Result<Tok, SolverError> {
        let tok = tokens.get(*pos).cloned();
        match tok {
            Some(t) => {
                *pos += 1;
                Ok(t)
            }
            None => Err(SolverError::new("EXPR_SYNTAX", "expected more tokens")),
        }
    }

    fn expr(tokens: &[Tok], pos: &mut usize, env: &Env) -> Result<f64, SolverError> {
        let mut v = term(tokens, pos, env)?;
        while let Some(Tok::Op(op)) = peek(tokens, *pos) {
            if *op != '+' && *op != '-' {
                break;
            }
            eat(tokens, pos)?;
            let r = term(tokens, pos, env)?;
            v = if *op == '+' { v + r } else { v - r };
        }
        Ok(v)
    }

    fn term(tokens: &[Tok], pos: &mut usize, env: &Env) -> Result<f64, SolverError> {
        let mut v = unary(tokens, pos, env)?;
        while let Some(Tok::Op(op)) = peek(tokens, *pos) {
            if *op != '*' && *op != '/' && *op != '%' {
                break;
            }
            eat(tokens, pos)?;
            let r = unary(tokens, pos, env)?;
            v = match op {
                '*' => v * r,
                '/' => v / r,
                _ => v % r,
            };
        }
        Ok(v)
    }

    fn unary(tokens: &[Tok], pos: &mut usize, env: &Env) -> Result<f64, SolverError> {
        if let Some(Tok::Op('-')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            return Ok(-unary(tokens, pos, env)?);
        }
        if let Some(Tok::Op('+')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            return unary(tokens, pos, env);
        }
        power(tokens, pos, env)
    }

    fn power(tokens: &[Tok], pos: &mut usize, env: &Env) -> Result<f64, SolverError> {
        let base = atom(tokens, pos, env)?;
        if let Some(Tok::Op('^')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            let exp = unary(tokens, pos, env)?; // right associative
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn atom(tokens: &[Tok], pos: &mut usize, env: &Env) -> Result<f64, SolverError> {
        match eat(tokens, pos)? {
            Tok::Num(v) => Ok(v),
            Tok::Id(id) => {
                if let Some(v) = env.get(id.as_str()) {
                    return Ok(*v);
                }
                let name = id.to_lowercase();
                if let Some(Tok::Op('(')) = peek(tokens, *pos) {
                    eat(tokens, pos)?;
                    let mut args = vec![expr(tokens, pos, env)?];
                    while let Some(Tok::Op(',')) = peek(tokens, *pos) {
                        eat(tokens, pos)?;
                        args.push(expr(tokens, pos, env)?);
                    }
                    if !matches!(eat(tokens, pos)?, Tok::Op(')')) {
                        return Err(SolverError::new("EXPR_SYNTAX", "expected )"));
                    }
                    apply_fn(&name, &args)
                } else if name == "pi" {
                    Ok(std::f64::consts::PI)
                } else if name == "e" {
                    Ok(std::f64::consts::E)
                } else {
                    Err(SolverError::new("EXPR_UNKNOWN_ID", format!("unknown identifier {}", name)))
                }
            }
            Tok::Op('(') => {
                let v = expr(tokens, pos, env)?;
                if !matches!(eat(tokens, pos)?, Tok::Op(')')) {
                    return Err(SolverError::new("EXPR_SYNTAX", "expected )"));
                }
                Ok(v)
            }
            Tok::Op(other) => Err(SolverError::new("EXPR_SYNTAX", format!("unexpected token {}", other))),
        }
    }

    let value = expr(&tokens, &mut pos, env)?;
    if pos != tokens.len() {
        return Err(SolverError::new("EXPR_TRAILING", "trailing tokens in expression"));
    }
    if !value.is_finite() {
        return Err(SolverError::new("EXPR_NON_FINITE", "expression evaluated to non-finite value"));
    }
    Ok(value)
}

/* ------------------------------------------------------------------ */
/* JSON helpers                                                         */
/* ------------------------------------------------------------------ */

struct Parsed {
    program: String,
    steps: Vec<String>,
    check: Option<String>,
}

fn parse_solver_json(text: &str) -> Result<Parsed, SolverError> {
    let start = text.find('{').ok_or_else(|| SolverError::new("INVALID_JSON", "no JSON object in reply"))?;
    let end = text.rfind('}').ok_or_else(|| SolverError::new("INVALID_JSON", "no JSON object in reply"))?;
    let value: Value = serde_json::from_str(&text[start..=end])
        .map_err(|_| SolverError::new("INVALID_JSON", "reply was not valid JSON"))?;
    let program = value["program"]
        .as_str()
        .ok_or_else(|| SolverError::new("INVALID_JSON", "missing program"))?
        .to_string();
    let check = value["check"].as_str().filter(|s| !s.trim().is_empty()).map(|s| s.to_string());
    let steps = match &value["steps"] {
        Value::Array(arr) => arr
            .iter()
            .map(|s| s.as_str().unwrap_or_default().to_string())
            .collect(),
        _ => Vec::new(),
    };
    Ok(Parsed { program, steps, check })
}

/// Execute a model-generated JS-dialect program (let / assignment / result).
pub fn run_program(src: &str) -> Result<f64, SolverError> {
    if src.trim().is_empty() {
        return Err(SolverError::new("PROGRAM_EMPTY", "empty program"));
    }
    let mut env = Env::new();
    let mut result_defined = false;
    let mut last_value: Option<f64> = None;
    for line in src.split(['\n', ';']) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("let ") {
            if let Some(eq) = rest.find('=') {
                let name = rest[..eq].trim();
                let val = eval_expression_with(rest[eq + 1..].trim(), &env)?;
                env.insert(name.to_string(), val);
                if name == "result" {
                    result_defined = true;
                }
                continue;
            }
        }
        if let Some(eq) = line.find('=') {
            let name = line[..eq].trim();
            let is_ident = !name.is_empty()
                && name.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_alphanumeric() || c == '_');
            if is_ident {
                let val = eval_expression_with(line[eq + 1..].trim(), &env)?;
                env.insert(name.to_string(), val);
                if name == "result" {
                    result_defined = true;
                }
                continue;
            }
        }
        last_value = Some(eval_expression_with(line, &env)?);
    }
    if result_defined {
        return Ok(env["result"]);
    }
    if let Some(v) = last_value {
        return Ok(v);
    }
    Err(SolverError::new("PROGRAM_NO_RESULT", "program produced no result"))
}

/// Substitute {x} with the computed answer; passes when value ~ 0.
pub fn run_check(check_src: &str, answer: f64) -> Result<(f64, bool), SolverError> {
    let substituted = check_src.replace("{x}", &format!("({})", answer));
    let value = eval_expression(&substituted)?;
    let passed = value.abs() <= 1e-6 * answer.abs().max(1.0);
    Ok((value, passed))
}

fn numerically_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/* ------------------------------------------------------------------ */
/* Client (instantiate once, solve many)                                */
/* ------------------------------------------------------------------ */

pub type TransportFn = Box<dyn Fn(&str, &str, &str) -> Result<String, SolverError> + Send + Sync>;

pub fn default_transport(url: &str, body: &str, api_key: &str) -> Result<String, SolverError> {
    let resp = ureq::post(url)
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(60))
        .send_string(body)
        .map_err(|e| SolverError::new("HTTP_ERROR", e.to_string()))?;
    let text = resp
        .into_string()
        .map_err(|_| SolverError::new("HTTP_ERROR", "failed reading API response"))?;
    let data: Value = serde_json::from_str(&text)
        .map_err(|_| SolverError::new("HTTP_ERROR", "invalid JSON from API"))?;
    data["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| SolverError::new("HTTP_ERROR", "API response missing message content"))
}

/// BYOK client for an OpenAI-compatible endpoint.
///
/// ```no_run
/// let solver = mathsolver::MathSolver::new("sk-...", "https://api.deepseek.com/v1")
///     .unwrap().model("deepseek-chat");
/// let r = solver.solve("2x + 3 = 11, solve for x").unwrap(); // r.verified == true
/// ```
pub struct MathSolver {
    api_key: String,
    base_url: String,
    model: String,
    transport: TransportFn,
}

impl std::fmt::Debug for MathSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MathSolver")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl MathSolver {
    /// Create a client with the built-in HTTP transport (ureq).
    pub fn new(api_key: &str, base_url: &str) -> Result<Self, SolverError> {
        Self::with_transport(api_key, base_url, default_transport)
    }

    /// Create a client with an injected transport `(url, body_json, api_key) -> model text`.
    pub fn with_transport<F>(api_key: &str, base_url: &str, transport: F) -> Result<Self, SolverError>
    where
        F: Fn(&str, &str, &str) -> Result<String, SolverError> + Send + Sync + 'static,
    {
        if api_key.is_empty() {
            return Err(SolverError::new("NO_API_KEY", "api_key is required (BYOK)"));
        }
        let base = base_url.trim_end_matches('/');
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(SolverError::new("BAD_BASE_URL", "base_url must be an http(s) URL, e.g. https://api.deepseek.com/v1"));
        }
        Ok(Self {
            api_key: api_key.to_string(),
            base_url: base.to_string(),
            model: "gpt-4o-mini".to_string(),
            transport: Box::new(transport),
        })
    }

    /// Builder-style model override.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Solve a math problem. The answer is the output of executing the model's
    /// program; `verified` is true only when the check expression passes
    /// (equations: answer substituted back must satisfy the original equation).
    pub fn solve(&self, problem: &str) -> Result<SolveResult, SolverError> {
        if problem.trim().is_empty() {
            return Err(SolverError::new("NO_PROBLEM", "problem must be non-empty"));
        }
        let url = format!("{}/chat/completions", self.base_url);

        let mut messages = vec![
            json!({"role": "system", "content": SYSTEM_PROMPT}),
            json!({"role": "user", "content": problem}),
        ];
        let call = |messages: &Vec<Value>| -> Result<String, SolverError> {
            let body = json!({"model": self.model, "messages": messages, "temperature": 0}).to_string();
            (self.transport)(&url, &body, &self.api_key)
        };

        let mut parsed = match parse_solver_json(&call(&messages)?) {
            Ok(p) => p,
            Err(e) if e.code == "INVALID_JSON" => {
                messages.push(json!({"role": "assistant", "content": "invalid JSON"}));
                messages.push(json!({"role": "user", "content": "Your reply was not valid JSON. Reply again with the exact strict JSON shape."}));
                parse_solver_json(&call(&messages)?)?
            }
            Err(e) => return Err(e),
        };

        // attempt: execute program + run check; reports ok/err without throwing.
        let attempt = |p: &Parsed| -> Result<(f64, Option<f64>, bool), SolverError> {
            let answer = run_program(&p.program)?;
            let mut check_value = None;
            let mut verified = false;
            if let Some(check) = &p.check {
                let (v, passed) = run_check(check, answer)?;
                check_value = Some(v);
                verified = passed;
            }
            Ok((answer, check_value, verified))
        };

        let mut program_error: Option<SolverError> = None;
        let (mut answer, mut check_value, mut verified) = match attempt(&parsed) {
            Ok(t) => t,
            Err(e) => { program_error = Some(e); (f64::NAN, None, false) }
        };
        let first_ok = program_error.is_none();

        let mut retries = 0u32;
        if !first_ok || !verified {
            retries = 1;
            let reason = if !first_ok {
                format!("program failed to execute ({})", program_error.as_ref().map(|e| e.message.clone()).unwrap_or_default())
            } else {
                format!("check evaluated to {:?} instead of 0", check_value)
            };
            messages.push(json!({"role": "assistant", "content": serde_json::to_string(&json!({
                "program": parsed.program, "steps": parsed.steps, "check": parsed.check
            })).unwrap_or_default()}));
            messages.push(json!({"role": "user", "content": format!(
                "Your submission failed verification: {}. Re-derive the problem carefully and reply again with the same strict JSON shape.", reason
            )}));
            match parse_solver_json(&call(&messages)?) {
                Ok(second) => match attempt(&second) {
                    Ok((a, cv, v)) => {
                        parsed = second;
                        answer = a; check_value = cv; verified = v;
                    }
                    Err(e) => return Err(e), // PROGRAM_* persisted after retry
                },
                Err(e) => return Err(e),
            }
        }

        Ok(SolveResult {
            answer,
            steps: parsed.steps,
            expression: parsed.program,
            evaluated: check_value,
            verified,
            retries,
        })
    }
}
