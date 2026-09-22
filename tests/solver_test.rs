use mathsolver::{eval_expression, MathSolver, SolverError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const GOOD: &str = r#"{"answer": 4, "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "verification": {"expression": "(11-3)/2"}}"#;
const WRONG: &str = r#"{"answer": 4, "steps": ["..."], "verification": {"expression": "(11-3)/3"}}"#;

#[test]
fn evaluator_precedence() {
    assert_eq!(eval_expression("2*3+4").unwrap(), 10.0);
    assert_eq!(eval_expression("2+3*4").unwrap(), 14.0);
    assert_eq!(eval_expression("(2+3)*4").unwrap(), 20.0);
    assert_eq!(eval_expression("2^3^2").unwrap(), 512.0);
    assert_eq!(eval_expression("-3^2").unwrap(), -9.0);
    assert!((eval_expression("10%3").unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn evaluator_functions() {
    assert_eq!(eval_expression("sqrt(16)").unwrap(), 4.0);
    assert_eq!(eval_expression("min(3,5)").unwrap(), 3.0);
    assert!((eval_expression("pi").unwrap() - std::f64::consts::PI).abs() < 1e-12);
    assert!((eval_expression("log(1000)").unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn evaluator_rejects_bad_input() {
    assert!(eval_expression("std::mem::forget").is_err());
    assert!(eval_expression("1+2)").is_err());
    assert!(eval_expression("foo(1)").is_err());
    assert!(eval_expression("").is_err());
}

#[test]
fn new_validates_credentials() {
    let e = MathSolver::new("", "https://api.x").unwrap_err();
    assert_eq!(e.code, "NO_API_KEY");
    let e = MathSolver::new("sk", "not-a-url").unwrap_err();
    assert_eq!(e.code, "BAD_BASE_URL");
}

#[test]
fn solve_verified_first_try() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);
    let solver = MathSolver::with_transport("sk-test", "https://api.deepseek.com/v1", move |url, body, key| {
        calls_in.fetch_add(1, Ordering::SeqCst);
        assert!(url.starts_with("https://api.deepseek.com/v1/chat/completions"));
        assert!(body.contains("math solver"));
        assert_eq!(key, "sk-test");
        Ok(GOOD.to_string())
    })
    .unwrap()
    .model("deepseek-chat");
    let r = solver.solve("2x + 3 = 11, solve for x").unwrap();
    assert!(r.verified);
    assert_eq!(r.answer, 4.0);
    assert_eq!(r.evaluated, Some(4.0));
    assert_eq!(r.retries, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn solve_retry_recovers() {
    let n = Arc::new(AtomicUsize::new(0));
    let n_in = Arc::clone(&n);
    let solver = MathSolver::with_transport("sk", "https://api.x", move |_, _, _| {
        let i = n_in.fetch_add(1, Ordering::SeqCst);
        Ok(if i == 0 { WRONG.to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    let r = solver.solve("2x+3=11").unwrap();
    assert!(r.verified);
    assert_eq!(r.retries, 1);
}

#[test]
fn solve_invalid_json_then_ok() {
    let n = Arc::new(AtomicUsize::new(0));
    let n_in = Arc::clone(&n);
    let solver = MathSolver::with_transport("sk", "https://api.x", move |_, _, _| {
        let i = n_in.fetch_add(1, Ordering::SeqCst);
        Ok(if i == 0 { "no json here".to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    assert!(solver.solve("1+1").unwrap().verified);
}

#[test]
fn solve_invalid_json_twice_raises() {
    let solver = MathSolver::with_transport("sk", "https://api.x", |_, _, _| Ok("still nothing".to_string())).unwrap();
    let err = solver.solve("1+1").unwrap_err();
    assert_eq!(err.code, "INVALID_JSON");
}

#[test]
fn solve_no_api_key_at_construction() {
    let err = MathSolver::with_transport("", "https://x", |_, _, _| Ok(String::new())).unwrap_err();
    assert_eq!(err.code, "NO_API_KEY");
}

#[test]
fn solve_http_error_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);
    let solver = MathSolver::with_transport("sk", "https://api.x", move |_, _, _| {
        calls_in.fetch_add(1, Ordering::SeqCst);
        Err(SolverError::new("HTTP_ERROR", "401"))
    })
    .unwrap();
    let err = solver.solve("1+1").unwrap_err();
    assert_eq!(err.code, "HTTP_ERROR");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn solve_retry_still_wrong_unverified() {
    let solver = MathSolver::with_transport("sk", "https://api.x", |_, _, _| Ok(WRONG.to_string())).unwrap();
    let r = solver.solve("2x+3=11").unwrap();
    assert!(!r.verified);
    assert_eq!(r.retries, 1);
}
