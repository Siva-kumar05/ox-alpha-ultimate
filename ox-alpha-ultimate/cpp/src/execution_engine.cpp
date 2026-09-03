/**
 * Execution Engine Implementation
 */

#include "execution_engine.hpp"
#include <random>
#include <chrono>

ExecutionEngine::ExecutionEngine() = default;

ExecutionEngine::~ExecutionEngine() {
    stop();
}

bool ExecutionEngine::start(const RiskLimits& limits) {
    risk_limits_ = limits;
    running_ = true;
    
    workers_.emplace_back([this] { market_data_worker(); });
    workers_.emplace_back([this] { order_worker(); });
    workers_.emplace_back([this] { fill_worker(); });
    workers_.emplace_back([this] { risk_worker(); });
    
    return true;
}

void ExecutionEngine::stop() {
    running_ = false;
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

bool ExecutionEngine::on_market_tick(const MarketTick& tick) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Update order book
    auto& book = order_books_[tick.symbol_id];
    if (tick.side == 0) {
        book.update_bid(0, tick.price, tick.size, 1);
    } else if (tick.side == 1) {
        book.update_ask(0, tick.price, tick.size, 1);
    }
    
    // Push to processing queue
    if (!market_data_queue_.try_push(tick)) {
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    latency_sum_ns_.fetch_add(latency, std::memory_order_relaxed);
    ticks_processed_.fetch_add(1, std::memory_order_relaxed);
    
    return true;
}

bool ExecutionEngine::submit_order(const Order& order) {
    if (!pre_trade_risk_check(order)) {
        rejected_orders_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    
    if (!order_queue_.try_push(order)) {
        return false;
    }
    return true;
}

void ExecutionEngine::set_order_sent_callback(std::function<void(const Order&)> cb) { 
    on_order_sent_ = cb; 
}
void ExecutionEngine::set_fill_callback(std::function<void(const Order&)> cb) { 
    on_fill_ = cb; 
}
void ExecutionEngine::set_reject_callback(std::function<void(const Order&)> cb) { 
    on_reject_ = cb; 
}
void ExecutionEngine::set_position_update_callback(std::function<void(uint32_t, int64_t, int64_t)> cb) { 
    on_position_update_ = cb; 
}

ExecutionEngine::Stats ExecutionEngine::get_stats() const {
    Stats s;
    s.ticks_processed = ticks_processed_.load();
    s.orders_sent = orders_sent_.load();
    s.fills_received = fills_received_.load();
    s.rejected_orders = rejected_orders_.load();
    s.risk_rejections = risk_rejections_.load();
    uint64_t ticks = ticks_processed_.load();
    s.avg_latency_us = ticks > 0 ? latency_sum_ns_.load() / ticks / 1000.0 : 0;
    return s;
}

bool ExecutionEngine::pre_trade_risk_check(const Order& order) {
    auto pos_it = positions_.find(order.symbol_id);
    int64_t current_pos = pos_it != positions_.end() ? pos_it->second.quantity : 0;
    int64_t new_pos = order.side == 0 ? current_pos + order.quantity : current_pos - order.quantity;
    
    if (std::abs(new_pos) > risk_limits_.max_position_size) return false;
    
    int64_t order_value = order.price * order.quantity;
    if (order_value > risk_limits_.max_order_value) return false;
    
    return true;
}

void ExecutionEngine::market_data_worker() {
    MarketTick tick;
    while (running_) {
        if (market_data_queue_.try_pop(tick)) {
            process_tick(tick);
        } else {
            std::this_thread::yield();
        }
    }
}

void ExecutionEngine::order_worker() {
    Order order;
    while (running_) {
        if (order_queue_.try_pop(order)) {
            send_order_to_exchange(order);
        } else {
            std::this_thread::yield();
        }
    }
}

void ExecutionEngine::fill_worker() {
    Order fill;
    while (running_) {
        if (fill_queue_.try_pop(fill)) {
            process_fill(fill);
        } else {
            std::this_thread::yield();
        }
    }
}

void ExecutionEngine::risk_worker() {
    while (running_) {
        check_portfolio_risk();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void ExecutionEngine::process_tick(const MarketTick& tick) {
    auto it = positions_.find(tick.symbol_id);
    if (it != positions_.end()) {
        it->second.mark_price = tick.price;
        it->second.unrealized_pnl = (tick.price - it->second.avg_entry_price) * it->second.quantity;
        it->second.last_update_ns = tick.timestamp_ns;
        
        if (on_position_update_) {
            on_position_update_(tick.symbol_id, it->second.quantity, it->second.unrealized_pnl);
        }
    }
}

void ExecutionEngine::send_order_to_exchange(const Order& order) {
    orders_sent_.fetch_add(1, std::memory_order_relaxed);
    if (on_order_sent_) on_order_sent_(order);
    
    // Simulate fill for testing
    Order fill = order;
    fill.status = 2; // Filled
    fill.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    if (!fill_queue_.try_push(fill)) {
        // Queue full
    }
}

void ExecutionEngine::process_fill(const Order& fill) {
    auto& pos = positions_[fill.symbol_id];
    if (fill.side == 0) { // Buy
        int64_t new_qty = pos.quantity + fill.quantity;
        pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity) / new_qty;
        pos.quantity = new_qty;
    } else { // Sell
        int64_t new_qty = pos.quantity - fill.quantity;
        pos.realized_pnl += (fill.price - pos.avg_entry_price) * fill.quantity;
        pos.quantity = new_qty;
        if (pos.quantity == 0) pos.avg_entry_price = 0;
    }
    
    pos.mark_price = fill.price;
    pos.unrealized_pnl = (pos.mark_price - pos.avg_entry_price) * pos.quantity;
    pos.last_update_ns = fill.timestamp_ns;
    
    fills_received_.fetch_add(1, std::memory_order_relaxed);
    if (on_fill_) on_fill_(fill);
    if (on_position_update_) on_position_update_(fill.symbol_id, pos.quantity, pos.unrealized_pnl);
}

void ExecutionEngine::check_portfolio_risk() {
    // Portfolio-level risk checks
}