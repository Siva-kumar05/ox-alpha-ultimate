use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::time::Instant;

pub type SymbolId = String;
pub type AgentId = String;
pub type Price = i64;      // Fixed-point 1e8
pub type Quantity = i64;
pub type Timestamp = i64;   // Nanoseconds since epoch

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    Stop = 2,
    StopLimit = 3,
    IOC = 4,
    FOK = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    New = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,
    Rejected = 4,
    Expired = 5,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: SymbolId,
    pub agent_id: AgentId,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Price,
    pub quantity: Quantity,
    pub filled_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub status: OrderStatus,
    pub timestamp: Timestamp,
    pub leverage: f64,
    pub stop_price: Option<Price>,
    pub take_profit: Option<Price>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub signal_id: String,
    pub agent_id: AgentId,
    pub symbol: SymbolId,
    pub action: SignalAction,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub leverage: f64,
    pub stop_loss: Option<Price>,
    pub take_profit: Option<Price>,
    pub strength: f64,  // 0.0 to 1.0
    pub timestamp: Timestamp,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalAction {
    Buy = 0,
    Sell = 1,
    Close = 2,
    Modify = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: SymbolId,
    pub agent_id: AgentId,
    pub side: Side,
    pub quantity: Quantity,
    pub avg_entry_price: Price,
    pub current_price: Price,
    pub entry_time: Instant,
    pub stop_loss: Option<Price>,
    pub take_profit: Option<Price>,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub leverage: f64,
    pub metadata: serde_json::Value,
}

impl Position {
    pub fn new(symbol: SymbolId, agent_id: AgentId, side: Side, quantity: Quantity, price: Price) -> Self {
        Self {
            symbol,
            agent_id,
            side,
            quantity,
            avg_entry_price: price,
            current_price: price,
            entry_time: Instant::now(),
            stop_loss: None,
            take_profit: None,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            leverage: 1.0,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn update_price(&mut self, price: Price) {
        self.current_price = price;
        let pnl_per_unit = match self.side {
            Side::Buy => price as f64 - self.avg_entry_price as f64,
            Side::Sell => self.avg_entry_price as f64 - price as f64,
        };
        self.unrealized_pnl = pnl_per_unit * self.quantity as f64;
    }

    pub fn notional(&self) -> f64 {
        self.quantity as f64 * self.current_price as f64 / 1e8
    }

    pub fn pnl_pct(&self) -> f64 {
        if self.avg_entry_price == 0 {
            return 0.0;
        }
        match self.side {
            Side::Buy => (self.current_price as f64 - self.avg_entry_price as f64) / self.avg_entry_price as f64,
            Side::Sell => (self.avg_entry_price as f64 - self.current_price as f64) / self.avg_entry_price as f64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub symbol: SymbolId,
    pub exchange: String,
    pub price: Price,
    pub size: Quantity,
    pub side: Side,
    pub timestamp: Timestamp,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub orders: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: SymbolId,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: Timestamp,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub agent_id: AgentId,
    pub equity: f64,
    pub available_margin: f64,
    pub used_margin: f64,
    pub total_pnl: f64,
    pub daily_pnl: f64,
    pub positions: Vec<Position>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: String,
    pub order_id: String,
    pub symbol: SymbolId,
    pub agent_id: AgentId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub fee: f64,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: SymbolId,
    pub exchange: String,
    pub rate: f64,
    pub next_funding_time: Timestamp,
    pub predicted_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Liquidation {
    pub symbol: SymbolId,
    pub exchange: String,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
}