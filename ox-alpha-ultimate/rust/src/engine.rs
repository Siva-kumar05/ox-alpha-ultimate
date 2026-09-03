use super::{RiskError, RiskResult, types::*};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Normal, ContinuousCDF};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_total_leverage: f64,
    pub max_portfolio_var_pct: f64,
    pub max_correlation_exposure: f64,
    pub max_sector_concentration: f64,
    pub max_single_position_pct: f64,
    pub max_daily_loss_pct: f64,
    pub max_drawdown_pct: f64,
    pub max_total_positions: usize,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_total_leverage: 5.0,
            max_portfolio_var_pct: 0.05,
            max_correlation_exposure: 0.7,
            max_sector_concentration: 0.3,
            max_single_position_pct: 0.15,
            max_daily_loss_pct: 0.03,
            max_drawdown_pct: 0.15,
            max_total_positions: 50,
        }
    }
}

#[derive(Debug, Default)]
pub struct AgentRiskState {
    pub agent_id: AgentId,
    pub current_leverage: f64,
    pub current_positions: usize,
    pub daily_pnl: f64,
    pub daily_loss_pct: f64,
    pub drawdown_pct: f64,
    pub var_pct: f64,
    pub last_update: Instant,
    pub blocked: bool,
    pub block_reason: String,
    pub positions: HashMap<SymbolId, Position>,
}

impl AgentRiskState {
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct PortfolioRiskState {
    pub total_leverage: f64,
    pub total_positions: usize,
    pub total_exposure: f64,
    pub portfolio_var_pct: f64,
    pub current_equity: f64,
    pub peak_equity: f64,
    pub daily_pnl: f64,
    pub drawdown_pct: f64,
    pub correlation_matrix: Option<DMatrix<f64>>,
    pub symbol_list: Vec<SymbolId>,
    pub last_update: Instant,
}

pub struct RiskEngine {
    limits: RiskLimits,
    agent_states: Arc<DashMap<AgentId, AgentRiskState>>,
    portfolio_state: Arc<RwLock<PortfolioRiskState>>,
    position_cache: Arc<DashMap<String, Position>>, // agent:symbol -> Position
    equity_history: Arc<RwLock<Vec<(Instant, f64)>>>,
    returns_history: Arc<RwLock<Vec<f64>>>,
    correlation_cache: Arc<RwLock<Option<(DMatrix<f64>, Instant)>>>,
    correlation_window: usize,
    var_confidence: f64,
    var_window: usize,
    stats: RiskStats,
}

#[derive(Debug, Default)]
pub struct RiskStats {
    pub signals_approved: AtomicU64,
    pub signals_rejected: AtomicU64,
    pub risk_checks: AtomicU64,
    pub var_calculations: AtomicU64,
    pub correlation_updates: AtomicU64,
    pub blocks: AtomicU64,
    pub unblocks: AtomicU64,
}

impl RiskEngine {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            agent_states: Arc::new(DashMap::new()),
            portfolio_state: Arc::new(RwLock::new(PortfolioRiskState::default())),
            position_cache: Arc::new(DashMap::new()),
            equity_history: Arc::new(RwLock::new(Vec::new())),
            returns_history: Arc::new(RwLock::new(Vec::new())),
            correlation_cache: Arc::new(RwLock::new(None)),
            correlation_window: 252,
            var_confidence: 0.99,
            var_window: 252,
            stats: RiskStats::default(),
        }
    }

    pub fn register_agent(&self, agent_id: AgentId, risk_params: HashMap<String, String>) {
        let mut state = AgentRiskState::new(agent_id);
        
        // Override defaults with agent-specific params
        if let Some(leverage) = risk_params.get("max_leverage") {
            if let Ok(l) = leverage.parse() {
                // Would apply to agent-specific limits
            }
        }
        
        self.agent_states.insert(agent_id, state);
        tracing::info!("Registered agent for risk management: {}", agent_id);
    }

    pub fn unregister_agent(&self, agent_id: &AgentId) {
        self.agent_states.remove(agent_id);
    }

    pub async fn approve_signal(&self, signal: &Signal) -> RiskResult<()> {
        self.stats.risk_checks.fetch_add(1, Ordering::Relaxed);
        
        let agent_id = signal.agent_id.clone();
        
        // Get or create agent state
        let mut agent_state = self.agent_states
            .entry(agent_id.clone())
            .or_insert_with(|| AgentRiskState::new(agent_id.clone()));
        
        // Check if blocked
        if agent_state.blocked {
            self.stats.signals_rejected.fetch_add(1, Ordering::Relaxed);
            return Err(RiskError::AgentBlocked {
                reason: agent_state.block_reason.clone(),
            });
        }

        // Check agent-level limits
        self.check_agent_limits(&agent_state, signal)?;
        
        // Check portfolio-level limits
        self.check_portfolio_limits(signal).await?;
        
        // Check correlation limits
        self.check_correlation_limits(&agent_state, signal).await?;
        
        self.stats.signals_approved.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn check_agent_limits(&self, agent_state: &AgentRiskState, signal: &Signal) -> RiskResult<()> {
        // Position count
        if agent_state.current_positions >= 10 { // Would come from config
            return Err(RiskError::InvalidOrder("Max positions reached".to_string()));
        }

        // Leverage
        if agent_state.current_leverage > self.limits.max_total_leverage {
            return Err(RiskError::LeverageExceeded {
                leverage: agent_state.current_leverage,
                max: self.limits.max_total_leverage,
            });
        }

        // Daily loss
        if agent_state.daily_loss_pct > self.limits.max_daily_loss_pct {
            return Err(RiskError::DailyLossExceeded {
                loss_pct: agent_state.daily_loss_pct * 100.0,
                limit_pct: self.limits.max_daily_loss_pct * 100.0,
            });
        }

        // Drawdown
        if agent_state.drawdown_pct > self.limits.max_drawdown_pct {
            return Err(RiskError::DrawdownExceeded {
                drawdown_pct: agent_state.drawdown_pct * 100.0,
                limit_pct: self.limits.max_drawdown_pct * 100.0,
            });
        }

        Ok(())
    }

    async fn check_portfolio_limits(&self, signal: &Signal) -> RiskResult<()> {
        let portfolio = self.portfolio_state.read();
        
        // Total leverage
        let new_leverage = portfolio.total_leverage + signal.leverage * signal.quantity as f64 * signal.price as f64 / portfolio.current_equity.max(1.0);
        if new_leverage > self.limits.max_total_leverage {
            return Err(RiskError::LeverageExceeded {
                leverage: new_leverage,
                max: self.limits.max_total_leverage,
            });
        }

        // Portfolio VaR
        if portfolio.portfolio_var_pct > self.limits.max_portfolio_var_pct {
            return Err(RiskError::InvalidOrder(
                format!("Portfolio VaR {:.2}% exceeds limit {:.2}%", 
                    portfolio.portfolio_var_pct * 100.0, 
                    self.limits.max_portfolio_var_pct * 100.0)
            ));
        }

        // Position count
        if portfolio.total_positions >= self.limits.max_total_positions {
            return Err(RiskError::InvalidOrder("Max total positions reached".to_string()));
        }

        // Single position size
        let position_value = signal.price as f64 * signal.quantity as f64;
        if position_value / portfolio.current_equity.max(1.0) > self.limits.max_single_position_pct {
            return Err(RiskError::OrderValueExceeded {
                value: position_value as i64,
                max: (portfolio.current_equity * self.limits.max_single_position_pct) as i64,
            });
        }

        Ok(())
    }

    async fn check_correlation_limits(&self, agent_state: &AgentRiskState, signal: &Signal) -> RiskResult<()> {
        // Would check correlation with existing positions
        // Simplified for now
        Ok(())
    }

    pub async fn update_agent_state(&self, agent_id: &AgentId, positions: &HashMap<SymbolId, Position>) {
        let mut agent_state = self.agent_states
            .entry(agent_id.clone())
            .or_insert_with(|| AgentRiskState::new(agent_id.clone()));

        agent_state.current_positions = positions.len();
        
        let total_exposure: f64 = positions.values()
            .map(|p| p.quantity as f64 * p.current_price)
            .sum();
        
        let equity = self.get_portfolio_equity().await;
        agent_state.current_leverage = total_exposure / equity.max(1.0);
        
        // Update positions cache
        for (symbol, pos) in positions {
            let key = format!("{}:{}", agent_state.agent_id, symbol);
            self.position_cache.insert(key, pos.clone());
        }

        agent_state.last_update = Instant::now();
    }

    async fn get_portfolio_equity(&self) -> f64 {
        let portfolio = self.portfolio_state.read();
        portfolio.current_equity.max(1.0)
    }

    pub async fn update_portfolio_equity(&self, equity: f64) {
        let mut portfolio = self.portfolio_state.write();
        portfolio.current_equity = equity;
        
        if equity > portfolio.peak_equity {
            portfolio.peak_equity = equity;
        }

        let drawdown = if portfolio.peak_equity > 0.0 {
            (portfolio.peak_equity - equity) / portfolio.peak_equity
        } else { 0.0 };
        
        portfolio.drawdown_pct = drawdown;
        
        // Update equity history for VaR
        let mut history = self.equity_history.write();
        history.push((Instant::now(), equity));
        
        // Keep last 1000 points
        if history.len() > 1000 {
            history.drain(0..history.len() - 1000);
        }

        // Update returns history
        if history.len() >= 2 {
            let len = history.len();
            let ret = (history[len - 1].1 / history[len - 2].1 - 1.0);
            self.returns_history.write().push(ret);
            
            if self.returns_history.read().len() > 1000 {
                self.returns_history.write().drain(0..1);
            }
        }

        // Update agent drawdowns
        let drawdown = portfolio.drawdown_pct;
        for mut state in self.agent_states.iter_mut() {
            state.drawdown_pct = drawdown;
            
            // Auto-block on excessive drawdown
            if drawdown > self.limits.max_drawdown_pct && !state.blocked {
                state.blocked = true;
                state.block_reason = format!("Portfolio drawdown {:.1}% exceeds limit {:.1}%", 
                    drawdown * 100.0, self.limits.max_drawdown_pct * 100.0);
                self.stats.blocks.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub async fn calculate_var(&self) -> f64 {
        self.stats.var_calculations.fetch_add(1, Ordering::Relaxed);
        
        let returns = self.returns_history.read();
        if returns.len() < self.var_window.min(30) {
            return 0.0;
        }

        let window = returns.len().min(self.var_window);
        let slice = &returns[returns.len() - window..];
        
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (slice.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        // VaR using normal distribution (Cornish-Fisher for better accuracy would be better)
        let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(1.0 - self.var_confidence);
        let var = -(mean + z * std_dev);
        
        var.max(0.0)
    }

    pub async fn calculate_expected_shortfall(&self) -> f64 {
        let returns = self.returns_history.read();
        if returns.len() < self.var_window.min(30) {
            return 0.0;
        }

        let window = returns.len().min(self.var_window);
        let slice = &returns[returns.len() - window..];
        
        let var = self.calculate_var().await;
        let tail_losses: Vec<f64> = slice.iter()
            .filter(|&&r| r < -var)
            .copied()
            .collect();
        
        if tail_losses.is_empty() {
            return var;
        }
        
        tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
    }

    pub async fn update_correlation_matrix(&self) {
        let returns = self.returns_history.read();
        if returns.len() < self.correlation_window {
            return;
        }

        // Build returns matrix for all symbols
        // Simplified - would build proper matrix from all symbols
        let symbols = {
            let portfolio = self.portfolio_state.read();
            portfolio.symbol_list.clone()
        };

        if symbols.len() < 2 {
            return;
        }

        let n = symbols.len();
        let mut corr_matrix = DMatrix::identity(n, n);
        
        // Would compute actual correlations here
        // For now, use identity as placeholder
        
        self.correlation_cache.write().replace((corr_matrix, Instant::now()));
        self.stats.correlation_updates.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_correlation(&self, symbol_a: SymbolId, symbol_b: SymbolId) -> f64 {
        let cache = self.correlation_cache.read();
        if let Some((matrix, _)) = cache.as_ref() {
            // Would look up indices
            0.0
        } else {
            0.0
        }
    }

    pub fn block_agent(&self, agent_id: &AgentId, reason: String) {
        if let Some(mut state) = self.agent_states.get_mut(agent_id) {
            state.blocked = true;
            state.block_reason = reason;
            self.stats.blocks.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn unblock_agent(&self, agent_id: &AgentId) {
        if let Some(mut state) = self.agent_states.get_mut(agent_id) {
            state.blocked = false;
            state.block_reason.clear();
            self.stats.unblocks.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn is_agent_blocked(&self, agent_id: &AgentId) -> bool {
        self.agent_states.get(agent_id)
            .map(|s| s.blocked)
            .unwrap_or(false)
    }

    pub fn get_risk_report(&self) -> RiskReport {
        let portfolio = self.portfolio_state.read();
        let var = self.calculate_var_sync();
        
        RiskReport {
            portfolio: PortfolioRiskSummary {
                equity: portfolio.current_equity,
                peak_equity: portfolio.peak_equity,
                drawdown_pct: portfolio.drawdown_pct,
                total_leverage: portfolio.total_leverage,
                total_positions: portfolio.total_positions,
                daily_pnl: portfolio.daily_pnl,
                var_99: var,
            },
            limits: self.limits.clone(),
            agents: self.agent_states.iter()
                .map(|entry| (entry.key().clone(), AgentRiskSummary {
                    leverage: entry.current_leverage,
                    positions: entry.current_positions,
                    daily_pnl: entry.daily_pnl,
                    daily_loss_pct: entry.daily_loss_pct,
                    drawdown_pct: entry.drawdown_pct,
                    blocked: entry.blocked,
                    block_reason: entry.block_reason.clone(),
                })).collect(),
            stats: RiskStatsSummary {
                signals_approved: self.stats.signals_approved.load(Ordering::Relaxed),
                signals_rejected: self.stats.signals_rejected.load(Ordering::Relaxed),
                risk_checks: self.stats.risk_checks.load(Ordering::Relaxed),
                var_calculations: self.stats.var_calculations.load(Ordering::Relaxed),
                correlation_updates: self.stats.correlation_updates.load(Ordering::Relaxed),
                blocks: self.stats.blocks.load(Ordering::Relaxed),
                unblocks: self.stats.unblocks.load(Ordering::Relaxed),
            },
            timestamp: Instant::now(),
        }
    }

    fn calculate_var_sync(&self) -> f64 {
        let returns = self.returns_history.read();
        if returns.len() < 30 {
            return 0.0;
        }

        let window = returns.len().min(self.var_window);
        let slice = &returns[returns.len() - window..];
        
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (slice.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(1.0 - self.var_confidence);
        let var = -(mean + z * std_dev);
        
        var.max(0.0)
    }

    pub async fn record_fill(&self, agent_id: &AgentId, symbol: SymbolId, side: &str, quantity: Quantity, price: Price) {
        // Update position cache
        let key = format!("{}:{}", agent_id, symbol);
        let mut position = self.position_cache.entry(key).or_insert(Position {
            symbol: symbol.clone(),
            quantity: 0,
            avg_price: 0.0,
            side: side.to_string(),
            entry_time: Instant::now(),
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        });

        let qty = quantity as f64;
        if side == "buy" {
            let new_qty = position.quantity + qty;
            position.avg_price = (position.avg_price * position.quantity as f64 + price * qty) / new_qty;
            position.quantity += qty;
        } else {
            position.quantity -= qty;
            position.realized_pnl += (price - position.avg_price) * qty;
        }
    }

    pub fn get_agent_exposure(&self, agent_id: &AgentId) -> f64 {
        self.position_cache.iter()
            .filter(|entry| entry.key().starts_with(&format!("{}:", agent_id)))
            .map(|entry| entry.value().quantity as f64 * entry.value().avg_price)
            .sum()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RiskReport {
    pub portfolio: PortfolioRiskSummary,
    pub limits: RiskLimits,
    pub agents: HashMap<AgentId, AgentRiskSummary>,
    pub stats: RiskStatsSummary,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PortfolioRiskSummary {
    pub equity: f64,
    pub peak_equity: f64,
    pub drawdown_pct: f64,
    pub total_leverage: f64,
    pub total_positions: usize,
    pub daily_pnl: f64,
    pub var_99: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentRiskSummary {
    pub leverage: f64,
    pub positions: usize,
    pub daily_pnl: f64,
    pub daily_loss_pct: f64,
    pub drawdown_pct: f64,
    pub blocked: bool,
    pub block_reason: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RiskStatsSummary {
    pub signals_approved: u64,
    pub signals_rejected: u64,
    pub risk_checks: u64,
    pub var_calculations: u64,
    pub correlation_updates: u64,
    pub blocks: u64,
    pub unblocks: u64,
}