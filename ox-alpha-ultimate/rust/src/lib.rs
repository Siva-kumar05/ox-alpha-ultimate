//! Risk Engine - Safety-critical risk management
//! Built with Rust for memory safety and zero-cost abstractions

pub mod engine;
pub mod limits;
pub mod portfolio;
pub mod var;
pub mod correlation;
pub mod stress_test;
pub mod types;

pub use engine::RiskEngine;
pub use limits::{RiskLimits, AgentRiskState, PortfolioLimits};
pub use types::{Position, Order, Signal, SymbolId, AgentId, Price, Quantity, Timestamp};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Position limit exceeded: {symbol} qty={qty} > max={max}")]
    PositionLimitExceeded { symbol: String, qty: i64, max: i64 },

    #[error("Order value exceeds limit: {value} > {max}")]
    OrderValueExceeded { value: i64, max: i64 },

    #[error("Daily loss limit exceeded: {loss_pct:.2}% > {limit_pct:.2}%")]
    DailyLossExceeded { loss_pct: f64, limit_pct: f64 },

    #[error("Drawdown limit exceeded: {drawdown_pct:.2}% > {limit_pct:.2}%")]
    DrawdownExceeded { drawdown_pct: f64, limit_pct: f64 },

    #[error("Leverage limit exceeded: {leverage:.2}x > {max}x")]
    LeverageExceeded { leverage: f64, max: f64 },

    #[error("Correlation limit exceeded: {corr:.2} > {limit:.2}")]
    CorrelationExceeded { corr: f64, limit: f64 },

    #[error("Insufficient margin: required={required}, available={available}")]
    InsufficientMargin { required: f64, available: f64 },

    #[error("Agent blocked: {reason}")]
    AgentBlocked { reason: String },

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type RiskResult<T> = Result<T, RiskError>;