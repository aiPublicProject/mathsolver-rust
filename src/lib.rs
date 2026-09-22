//! mathsolver — BYOK AI math solver with independent verification.
//!
//! An answer is only `verified: true` when the model's verification
//! expression (pure arithmetic) is evaluated locally by this crate and
//! matches the answer numerically. No model output is ever executed as code.

use serde_json::{json, Value};
use std::fmt;

pub const SYSTEM_PROMPT: &str = "You are a precise math solver.\n\
Reply with STRICT JSON only, no markdown fences, in this exact shape:\n\
{\"answer\": <number>, \"steps\": [<string>, ...], \"verification\": {\"expression\": \"<string>\"}}\n\
Rules:\n\
- \"answer\" must be a single number (the final result).\n\
- \"steps\" must be an array of short plain-language explanation strings.\n\
- \"verification.expression\" must be a pure arithmetic expression that\n\
  evaluates to the answer. Allowed: numbers, + - * / % ^ ( ), and the\n\
  functions abs sqrt sin cos tan ln log exp floor ceil round min max\n\
  (log is base 10, ln is natural), and the constants pi and e.\n\
- The expression must recompute the answer independently.";

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

/// Evaluate a pure arithmetic expression string to a number.
pub fn eval_expression(src: &str) -> Result<f64, SolverError> {
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

    fn expr(tokens: &[Tok], pos: &mut usize) -> Result<f64, SolverError> {
        let mut v = term(tokens, pos)?;
        while let Some(Tok::Op(op)) = peek(tokens, *pos) {
            if *op != '+' && *op != '-' {
                break;
            }
            eat(tokens, pos)?;
            let r = term(tokens, pos)?;
            v = if *op == '+' { v + r } else { v - r };
        }
        Ok(v)
    }

    fn term(tokens: &[Tok], pos: &mut usize) -> Result<f64, SolverError> {
        let mut v = unary(tokens, pos)?;
        while let Some(Tok::Op(op)) = peek(tokens, *pos) {
            if *op != '*' && *op != '/' && *op != '%' {
                break;
            }
            eat(tokens, pos)?;
            let r = unary(tokens, pos)?;
            v = match op {
                '*' => v * r,
                '/' => v / r,
                _ => v % r,
            };
        }
        Ok(v)
    }

    fn unary(tokens: &[Tok], pos: &mut usize) -> Result<f64, SolverError> {
        if let Some(Tok::Op('-')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            return Ok(-unary(tokens, pos)?);
        }
        if let Some(Tok::Op('+')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            return unary(tokens, pos);
        }
        power(tokens, pos)
    }

    fn power(tokens: &[Tok], pos: &mut usize) -> Result<f64, SolverError> {
        let base = atom(tokens, pos)?;
        if let Some(Tok::Op('^')) = peek(tokens, *pos) {
            eat(tokens, pos)?;
            let exp = unary(tokens, pos)?; // right associative
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn atom(tokens: &[Tok], pos: &mut usize) -> Result<f64, SolverError> {
        match eat(tokens, pos)? {
            Tok::Num(v) => Ok(v),
            Tok::Id(id) => {
                let name = id.to_lowercase();
                if let Some(Tok::Op('(')) = peek(tokens, *pos) {
                    eat(tokens, pos)?;
                    let mut args = vec![expr(tokens, pos)?];
                    while let Some(Tok::Op(',')) = peek(tokens, *pos) {
                        eat(tokens, pos)?;
                        args.push(expr(tokens, pos)?);
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
                let v = expr(tokens, pos)?;
                if !matches!(eat(tokens, pos)?, Tok::Op(')')) {
                    return Err(SolverError::new("EXPR_SYNTAX", "expected )"));
                }
                Ok(v)
            }
            Tok::Op(other) => Err(SolverError::new("EXPR_SYNTAX", format!("unexpected token {}", other))),
        }
    }

    let value = expr(&tokens, &mut pos)?;
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
    answer: f64,
    steps: Vec<String>,
    expression: String,
}

fn parse_solver_json(text: &str) -> Result<Parsed, SolverError> {
    let start = text.find('{').ok_or_else(|| SolverError::new("INVALID_JSON", "no JSON object in reply"))?;
    let end = text.rfind('}').ok_or_else(|| SolverError::new("INVALID_JSON", "no JSON object in reply"))?;
    let value: Value = serde_json::from_str(&text[start..=end])
        .map_err(|_| SolverError::new("INVALID_JSON", "reply was not valid JSON"))?;
    let answer = match &value["answer"] {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| SolverError::new("INVALID_JSON", "answer is not numeric"))?,
        _ => return Err(SolverError::new("INVALID_JSON", "missing numeric answer")),
    };
    let expression = value["verification"]["expression"]
        .as_str()
        .ok_or_else(|| SolverError::new("INVALID_JSON", "missing verification.expression"))?
        .to_string();
    let steps = match &value["steps"] {
        Value::Array(arr) => arr
            .iter()
            .map(|s| s.as_str().unwrap_or_default().to_string())
            .collect(),
        _ => Vec::new(),
    };
    Ok(Parsed { answer, steps, expression })
}

fn numerically_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/* ------------------------------------------------------------------ */
/* Transport + solve                                                    */
/* ------------------------------------------------------------------ */

fn default_transport(url: &str, body: &str, api_key: &str) -> Result<String, SolverError> {
    let resp = ureq::post(url)
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_key))
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

/// Solve with an injected transport (url, body_json, api_key) -> model text. Used by tests.
pub fn solve_with_transport<F>(problem: &str, opts: &SolveOptions, mut transport: F) -> Result<SolveResult, SolverError>
where
    F: FnMut(&str, &str, &str) -> Result<String, SolverError>,
{
    if opts.api_key.is_empty() {
        return Err(SolverError::new("NO_API_KEY", "api_key is required (BYOK)"));
    }
    if problem.trim().is_empty() {
        return Err(SolverError::new("NO_PROBLEM", "problem must be non-empty"));
    }
    let url = format!("{}/chat/completions", opts.base_url.trim_end_matches('/'));

    let mut messages = vec![
        json!({"role": "system", "content": SYSTEM_PROMPT}),
        json!({"role": "user", "content": problem}),
    ];
    let mut call = |messages: &Vec<Value>| -> Result<String, SolverError> {
        let body = json!({"model": opts.model, "messages": messages, "temperature": 0}).to_string();
        transport(&url, &body, &opts.api_key)
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

    let evaluate = |p: &Parsed| -> (Option<f64>, bool) {
        match eval_expression(&p.expression) {
            Ok(ev) => (Some(ev), numerically_equal(ev, p.answer)),
            Err(_) => (None, false),
        }
    };

    let (mut evaluated, mut verified) = evaluate(&parsed);
    let mut retries = 0u32;
    if !verified {
        retries = 1;
        messages.push(json!({"role": "assistant", "content": serde_json::to_string(&json!({
            "answer": parsed.answer, "steps": parsed.steps, "verification": {"expression": parsed.expression}
        })).unwrap_or_default()}));
        messages.push(json!({"role": "user", "content": format!(
            "Your verification expression evaluated to {}, which does not match your answer {}. Re-derive carefully and reply again with the same strict JSON shape.",
            evaluated.map(|v| v.to_string()).unwrap_or_else(|| "an error".into()),
            parsed.answer
        )}));
        if let Ok(second) = parse_solver_json(&call(&messages)?) {
            let (ev2, ok2) = evaluate(&second);
            if ev2.is_some() {
                evaluated = ev2;
            }
            if ok2 {
                parsed = second;
                verified = true;
            }
        }
    }

    Ok(SolveResult {
        answer: parsed.answer,
        steps: parsed.steps,
        expression: parsed.expression,
        evaluated,
        verified,
        retries,
    })
}

/// Solve using the built-in HTTP transport (ureq).
pub fn solve(problem: &str, opts: &SolveOptions) -> Result<SolveResult, SolverError> {
    solve_with_transport(problem, opts, default_transport)
}
