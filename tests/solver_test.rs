use mathsolver::{eval_expression, run_program, run_check, MathSolver, SolverError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const GOOD: &str = r#"{"program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x", "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "check": "2*{x} + 3 - 11"}"#;
const NO_CHECK: &str = r#"{"program": "result = 0.15 * 80", "steps": ["15% of 80"]}"#;
const WRONG_CHECK: &str = r#"{"program": "let d = 11 - 3;\nresult = d / 2", "steps": ["..."], "check": "2*{x} + 3 - 12"}"#;
const BROKEN_PROGRAM: &str = r#"{"program": "result = undefinedvar + 1", "steps": []}"#;

#[test]
fn evaluator_precedence() {
    assert_eq!(eval_expression("2*3+4").unwrap(), 10.0);
    assert_eq!(eval_expression("2^3^2").unwrap(), 512.0);
    assert_eq!(eval_expression("-3^2").unwrap(), -9.0);
    assert_eq!(eval_expression("sqrt(16)").unwrap(), 4.0);
}

#[test]
fn run_program_let_and_result() {
    let p = "let d = 11 - 3;\nlet x = d / 2;\nresult = x";
    assert_eq!(run_program(p).unwrap(), 4.0);
    assert_eq!(run_program("let a = 3; let b = 4; a * b").unwrap(), 12.0);
    assert!(run_program("result = undefinedvar + 1").is_err());
    assert!(run_program("let a = 1; let b = 2").is_err()); // no result
}

#[test]
fn run_check_substitution() {
    let (v, ok) = run_check("2*{x} + 3 - 11", 4.0).unwrap();
    assert_eq!(v, 0.0);
    assert!(ok);
    let (v, ok) = run_check("2*{x} + 3 - 12", 4.0).unwrap();
    assert_eq!(v, -1.0);
    assert!(!ok);
}

#[test]
fn new_validates_credentials() {
    assert_eq!(MathSolver::new("", "https://api.x").unwrap_err().code, "NO_API_KEY");
    assert_eq!(MathSolver::new("sk", "not-a-url").unwrap_err().code, "BAD_BASE_URL");
}

#[test]
fn solve_answer_from_execution_first_try() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);
    let solver = MathSolver::with_transport("sk-test", "https://api.deepseek.com/v1", move |url, body, key| {
        calls_in.fetch_add(1, Ordering::SeqCst);
        assert!(url.starts_with("https://api.deepseek.com/v1/chat/completions"));
        assert!(body.contains("program"));
        assert_eq!(key, "sk-test");
        Ok(GOOD.to_string())
    })
    .unwrap()
    .model("deepseek-chat");
    let r = solver.solve("2x + 3 = 11, solve for x").unwrap();
    assert!(r.verified);
    assert_eq!(r.answer, 4.0);
    assert_eq!(r.evaluated, Some(0.0)); // check value
    assert_eq!(r.retries, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn solve_no_check_unverified() {
    let solver = MathSolver::with_transport("sk", "https://x", |_, _, _| Ok(NO_CHECK.to_string())).unwrap();
    let r = solver.solve("15% of 80").unwrap();
    assert_eq!(r.answer, 12.0);
    assert!(!r.verified);
}

#[test]
fn solve_check_fail_retry_recovers() {
    let n = Arc::new(AtomicUsize::new(0));
    let n_in = Arc::clone(&n);
    let solver = MathSolver::with_transport("sk", "https://x", move |_, _, _| {
        let i = n_in.fetch_add(1, Ordering::SeqCst);
        Ok(if i == 0 { WRONG_CHECK.to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    let r = solver.solve("2x+3=11").unwrap();
    assert!(r.verified);
    assert_eq!(r.retries, 1);
}

#[test]
fn solve_program_error_retry_recovers() {
    let n = Arc::new(AtomicUsize::new(0));
    let n_in = Arc::clone(&n);
    let solver = MathSolver::with_transport("sk", "https://x", move |_, _, _| {
        let i = n_in.fetch_add(1, Ordering::SeqCst);
        Ok(if i == 0 { BROKEN_PROGRAM.to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    let r = solver.solve("2x+3=11").unwrap();
    assert!(r.verified);
    assert_eq!(r.answer, 4.0);
}

#[test]
fn solve_program_error_persists_raises() {
    let solver = MathSolver::with_transport("sk", "https://x", |_, _, _| Ok(BROKEN_PROGRAM.to_string())).unwrap();
    let err = solver.solve("2x+3=11").unwrap_err();
    assert!(err.code.starts_with("PROGRAM_") || err.code.starts_with("EXPR_"), "got {}", err.code);
}

#[test]
fn solve_invalid_json_then_ok() {
    let n = Arc::new(AtomicUsize::new(0));
    let n_in = Arc::clone(&n);
    let solver = MathSolver::with_transport("sk", "https://x", move |_, _, _| {
        let i = n_in.fetch_add(1, Ordering::SeqCst);
        Ok(if i == 0 { "no json here".to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    assert!(solver.solve("1+1").unwrap().verified);
}

#[test]
fn solve_http_error_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);
    let solver = MathSolver::with_transport("sk", "https://x", move |_, _, _| {
        calls_in.fetch_add(1, Ordering::SeqCst);
        Err(SolverError::new("HTTP_ERROR", "401"))
    })
    .unwrap();
    let err = solver.solve("1+1").unwrap_err();
    assert_eq!(err.code, "HTTP_ERROR");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn solve_check_still_failing_unverified() {
    let solver = MathSolver::with_transport("sk", "https://x", |_, _, _| Ok(WRONG_CHECK.to_string())).unwrap();
    let r = solver.solve("2x+3=11").unwrap();
    assert!(!r.verified);
    assert_eq!(r.answer, 4.0); // still the executed answer
    assert_eq!(r.retries, 1);
}

#[test]
#[ignore = "smoke: set SMOKE_API_KEY to run (cargo test -- --ignored)"]
fn smoke_real_api() {
    let key = std::env::var("SMOKE_API_KEY").expect("SMOKE_API_KEY");
    let base = std::env::var("SMOKE_BASE_URL").ok().filter(|b| !b.is_empty()).unwrap_or_else(|| "https://api.openai.com/v1".into());
    let model = std::env::var("SMOKE_MODEL").ok().filter(|m| !m.is_empty()).unwrap_or_else(|| "gpt-4o-mini".into());
    let solver = MathSolver::new(&key, &base).unwrap().model(model);
    let r = solver.solve("2x + 3 = 11, solve for x").unwrap();
    println!("smoke: answer={} verified={} retries={}", r.answer, r.verified, r.retries);
    assert!(r.verified);
    assert_eq!(r.answer, 4.0);
}
