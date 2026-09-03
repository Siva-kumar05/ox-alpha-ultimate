/**
 * Pro Max Execution Engine - C++17
 * Ultra-low latency order execution and market data processing
 * Target: < 10 microseconds tick-to-trade
 */

#ifndef EXECUTION_ENGINE_HPP
#define EXECUTION_ENGINE_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <queue>
#include <unordered_map>

// Lock-free ring buffer for ultra-low latency
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    alignas(64) std::array<T, Capacity> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    
public:
    bool try_push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) & (Capacity - 1);
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Full
        }
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }
    
    bool try_pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false; // Empty
        }
        item = buffer_[tail];
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }
    
    size_t size() const {
        return (head_.load() - tail_.load()) & (Capacity - 1);
    }
};

// Market data structures (packed for cache efficiency)
#pragma pack(push, 1)
struct MarketTick {
    uint64_t timestamp_ns;    // Nanosecond timestamp
    uint32_t symbol_id;       // Symbol identifier
    uint32_t exchange_id;     // Exchange identifier
    int64_t price;            // Price in fixed-point (1e8)
    uint64_t size;            // Volume
    uint8_t side;             // 0=bid, 1=ask, 2=trade
    uint8_t flags;            // Bit flags
    uint16_t sequence;        // Sequence number
};

struct Order {
    uint64_t order_id;
    uint64_t client_order_id;
    uint32_t symbol_id;
    uint32_t exchange_id;
    int64_t price;            // Fixed-point price
    uint64_t quantity;
    uint64_t remaining_qty;
    uint8_t side;             // 0=buy, 1=sell
    uint8_t type;             // 0=market, 1=limit, 2=stop, 3=ioc, 4=fok
    uint8_t tif;              // Time in force
    uint8_t status;           // 0=new, 1=partial, 2=filled, 3=cancelled, 4=rejected
    uint64_t timestamp_ns;
};

struct Position {
    uint32_t symbol_id;
    int64_t quantity;         // Positive = long, negative = short
    int64_t avg_entry_price;
    int64_t mark_price;
    int64_t unrealized_pnl;
    int64_t realized_pnl;
    uint64_t last_update_ns;
};
#pragma pack(pop)

// Lock-free order book (simplified L2)
template<size_t MaxLevels = 32>
class OrderBook {
    struct Level {
        int64_t price;
        uint64_t quantity;
        uint32_t order_count;
    };
    
    std::array<Level, MaxLevels> bids_;
    std::array<Level, MaxLevels> asks_;
    uint8_t bid_depth_ = 0;
    uint8_t ask_depth_ = 0;
    std::atomic<uint64_t> version_{0};
    
public:
    void update_bid(uint8_t level, int64_t price, uint64_t qty, uint32_t count) {
        if (level < MaxLevels) {
            bids_[level] = {price, qty, count};
            if (level >= bid_depth_) bid_depth_ = level + 1;
            version_.fetch_add(1, std::memory_order_release);
        }
    }
    
    void update_ask(uint8_t level, int64_t price, uint64_t qty, uint32_t count) {
        if (level < MaxLevels) {
            asks_[level] = {price, qty, count};
            if (level >= ask_depth_) ask_depth_ = level + 1;
            version_.fetch_add(1, std::memory_order_release);
        }
    }
    
    int64_t best_bid() const { return bid_depth_ > 0 ? bids_[0].price : 0; }
    int64_t best_ask() const { return ask_depth_ > 0 ? asks_[0].price : 0; }
    int64_t mid_price() const { 
        int64_t b = best_bid(), a = best_ask();
        return (b > 0 && a > 0) ? (b + a) / 2 : 0;
    }
    int64_t spread() const { return best_ask() - best_bid(); }
    uint64_t version() const { return version_.load(); }
};

// High-performance execution engine
class ExecutionEngine {
    // Lock-free queues for zero-copy data flow
    LockFreeRingBuffer<MarketTick, 65536> market_data_queue_;
    LockFreeRingBuffer<Order, 16384> order_queue_;
    LockFreeRingBuffer<Order, 16384> fill_queue_;
    
    // Order books per symbol
    std::unordered_map<uint32_t, OrderBook<32>> order_books_;
    
    // Position tracking
    std::unordered_map<uint32_t, Position> positions_;
    
    // Risk limits (hot path - cache friendly)
    struct RiskLimits {
        int64_t max_position_value;
        int64_t max_order_value;
        int64_t max_daily_loss;
        uint32_t max_orders_per_sec;
        int64_t max_position_size;
    } risk_limits_;
    
    // Statistics
    alignas(64) std::atomic<uint64_t> ticks_processed_{0};
    alignas(64) std::atomic<uint64_t> orders_sent_{0};
    alignas(64) std::atomic<uint64_t> fills_received_{0};
    alignas(64) std::atomic<uint64_t> rejected_orders_{0};
    alignas(64) std::atomic<uint64_t> risk_rejections_{0};
    alignas(64) std::atomic<uint64_t> latency_sum_ns_{0};
    
    // Callbacks
    std::function<void(const Order&)> on_order_sent_;
    std::function<void(const Order&)> on_fill_;
    std::function<void(const Order&)> on_reject_;
    std::function<void(uint32_t, int64_t, int64_t)> on_position_update_;
    
    // Worker threads
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    
public:
    ExecutionEngine() = default;
    ~ExecutionEngine() { stop(); }
    
    bool start(const RiskLimits& limits) {
        risk_limits_ = limits;
        running_ = true;
        
        // Start processing threads (pin to cores)
        workers_.emplace_back([this] { market_data_worker(); });
        workers_.emplace_back([this] { order_worker(); });
        workers_.emplace_back([this] { fill_worker(); });
        workers_.emplace_back([this] { risk_worker(); });
        
        return true;
    }
    
    void stop() {
        running_ = false;
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }
    
    // Hot path: submit market tick (called from network thread)
    bool on_market_tick(const MarketTick& tick) {
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
            // Queue full - drop tick (monitor this metric)
            return false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        latency_sum_ns_.fetch_add(latency, std::memory_order_relaxed);
        ticks_processed_.fetch_add(1, std::memory_order_relaxed);
        
        return true;
    }
    
    // Submit order (called from strategy)
    bool submit_order(const Order& order) {
        // Pre-trade risk checks (inline for speed)
        if (!pre_trade_risk_check(order)) {
            rejected_orders_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        if (!order_queue_.try_push(order)) {
            return false;
        }
        return true;
    }
    
    // Callbacks
    void set_order_sent_callback(std::function<void(const Order&)> cb) { on_order_sent_ = cb; }
    void set_fill_callback(std::function<void(const Order&)> cb) { on_fill_ = cb; }
    void set_reject_callback(std::function<void(const Order&)> cb) { on_reject_ = cb; }
    void set_position_update_callback(std::function<void(uint32_t, int64_t, int64_t)> cb) { on_position_update_ = cb; }
    
    // Statistics
    struct Stats {
        uint64_t ticks_processed;
        uint64_t orders_sent;
        uint64_t fills_received;
        uint64_t rejected_orders;
        uint64_t risk_rejections;
        double avg_latency_us;
    };
    
    Stats get_stats() const {
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
    
private:
    bool pre_trade_risk_check(const Order& order) {
        // Position limit
        auto pos_it = positions_.find(order.symbol_id);
        int64_t current_pos = pos_it != positions_.end() ? pos_it->second.quantity : 0;
        int64_t new_pos = order.side == 0 ? current_pos + order.quantity : current_pos - order.quantity;
        
        if (std::abs(new_pos) > risk_limits_.max_position_size) return false;
        
        // Order value limit
        int64_t order_value = order.price * order.quantity;
        if (order_value > risk_limits_.max_order_value) return false;
        
        // Daily loss check (simplified)
        // Would check actual P&L here
        
        return true;
    }
    
    void market_data_worker() {
        MarketTick tick;
        while (running_) {
            if (market_data_queue_.try_pop(tick)) {
                process_tick(tick);
            } else {
                std::this_thread::yield();
            }
        }
    }
    
    void order_worker() {
        Order order;
        while (running_) {
            if (order_queue_.try_pop(order)) {
                send_order_to_exchange(order);
            } else {
                std::this_thread::yield();
            }
        }
    }
    
    void fill_worker() {
        Order fill;
        while (running_) {
            if (fill_queue_.try_pop(fill)) {
                process_fill(fill);
            } else {
                std::this_thread::yield();
            }
        }
    }
    
    void risk_worker() {
        while (running_) {
            check_portfolio_risk();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void process_tick(const MarketTick& tick) {
        // Update position marks
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
    
    void send_order_to_exchange(const Order& order) {
        // Would send to exchange via network
        // For now, simulate
        orders_sent_.fetch_add(1, std::memory_order_relaxed);
        if (on_order_sent_) on_order_sent_(order);
    }
    
    void process_fill(const Order& fill) {
        // Update position
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
    
    void check_portfolio_risk() {
        // Check portfolio-level risk limits
        // Would implement portfolio VaR, correlation checks, etc.
    }
};

#endif // EXECUTION_ENGINE_HPP