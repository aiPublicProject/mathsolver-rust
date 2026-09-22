use mathsolver::{eval_expression, solve_with_transport, SolveOptions, SolverError};

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
fn solve_verified_first_try() {
    let mut calls = 0;
    let opts = SolveOptions { api_key: "sk-test".into(), ..Default::default() };
    let r = solve_with_transport("2x + 3 = 11, solve for x", &opts, |url, body, key| {
        calls += 1;
        assert!(url.ends_with("/chat/completions"));
        assert!(body.contains("math solver"));
        assert_eq!(key, "sk-test");
        Ok(GOOD.to_string())
    })
    .unwrap();
    assert!(r.verified);
    assert_eq!(r.answer, 4.0);
    assert_eq!(r.evaluated, Some(4.0));
    assert_eq!(r.retries, 0);
    assert_eq!(calls, 1);
}

#[test]
fn solve_retry_recovers() {
    let mut n = 0;
    let opts = SolveOptions { api_key: "sk".into(), ..Default::default() };
    let r = solve_with_transport("2x+3=11", &opts, |_, _, _| {
        n += 1;
        Ok(if n == 1 { WRONG.to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    assert!(r.verified);
    assert_eq!(r.retries, 1);
}

#[test]
fn solve_invalid_json_then_ok() {
    let mut n = 0;
    let opts = SolveOptions { api_key: "sk".into(), ..Default::default() };
    let r = solve_with_transport("1+1", &opts, |_, _, _| {
        n += 1;
        Ok(if n == 1 { "no json here".to_string() } else { GOOD.to_string() })
    })
    .unwrap();
    assert!(r.verified);
}

#[test]
fn solve_invalid_json_twice_raises() {
    let opts = SolveOptions { api_key: "sk".into(), ..Default::default() };
    let err = solve_with_transport("1+1", &opts, |_, _, _| Ok("still nothing".to_string())).unwrap_err();
    assert_eq!(err.code, "INVALID_JSON");
}

#[test]
fn solve_no_api_key() {
    let err = solve_with_transport("1+1", &SolveOptions::default(), |_, _, _| Ok(String::new())).unwrap_err();
    assert_eq!(err.code, "NO_API_KEY");
}

#[test]
fn solve_http_error_no_retry() {
    let mut calls = 0;
    let opts = SolveOptions { api_key: "sk".into(), ..Default::default() };
    let err = solve_with_transport("1+1", &opts, |_, _, _| {
        calls += 1;
        Err(SolverError::new("HTTP_ERROR", "401"))
    })
    .unwrap_err();
    assert_eq!(err.code, "HTTP_ERROR");
    assert_eq!(calls, 1);
}

#[test]
fn solve_retry_still_wrong_unverified() {
    let opts = SolveOptions { api_key: "sk".into(), ..Default::default() };
    let r = solve_with_transport("2x+3=11", &opts, |_, _, _| Ok(WRONG.to_string())).unwrap();
    assert!(!r.verified);
    assert_eq!(r.retries, 1);
}
