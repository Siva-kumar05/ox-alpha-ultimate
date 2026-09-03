/**
 * Main entry point for execution engine
 */

#include "execution_engine.hpp"
#include "market_data_decoder.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>

std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "=== Ox Alpha Execution Engine ===" << std::endl;
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    
    // Risk limits
    ExecutionEngine::RiskLimits limits;
    limits.max_position_value = 10000000;      // 1 Cr
    limits.max_order_value = 500000;           // 5 Lakh
    limits.max_daily_loss = 200000;            // 2 Lakh
    limits.max_orders_per_sec = 1000;
    limits.max_position_size = 2000000;        // 20 Lakh
    
    ExecutionEngine engine;
    if (!engine.start(limits)) {
        std::cerr << "Failed to start execution engine" << std::endl;
        return 1;
    }
    
    // Set callbacks
    engine.set_order_sent_callback([](const Order& order) {
        std::cout << "[ORDER SENT] " << order.order_id 
                  << " " << (order.side == 0 ? "BUY" : "SELL")
                  << " " << order.quantity << "@" << order.price / 1e8 << std::endl;
    });
    
    engine.set_fill_callback([](const Order& fill) {
        std::cout << "[FILL] " << fill.order_id
                  << " " << fill.quantity << "@" << fill.price / 1e8 << std::endl;
    });
    
    engine.set_reject_callback([](const Order& order) {
        std::cout << "[REJECT] " << order.order_id << std::endl;
    });
    
    engine.set_position_update_callback([](uint32_t symbol_id, int64_t qty, int64_t pnl) {
        std::cout << "[POSITION] Symbol: " << symbol_id 
                  << " Qty: " << qty << " PnL: " << pnl / 1e8 << std::endl;
    });
    
    std::cout << "Engine started. Press Ctrl+C to stop." << std::endl;
    
    // Simulate market data
    std::thread market_thread([&]() {
        ExecutionEngine::MarketTick tick{};
        tick.exchange_id = 1;
        tick.symbol_id = 1; // RELIANCE
        tick.sequence = 0;
        
        int64_t base_price = 250000000000LL; // 2500.00
        
        while (g_running) {
            // Simulate price movement
            double change = (rand() % 200 - 100) / 10000.0; // +/- 1%
            base_price = static_cast<int64_t>(base_price * (1.0 + change));
            
            tick.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            tick.price = base_price;
            tick.size = 100 + rand() % 900;
            tick.side = rand() % 2;
            
            engine.on_market_tick(tick);
            
            // Also send opposite side
            tick.side = 1 - tick.side;
            engine.on_market_tick(tick);
            
            tick.sequence++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Order simulation thread
    std::thread order_thread([&]() {
        uint64_t order_id = 1000000;
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            Order order{};
            order.order_id = order_id++;
            order.client_order_id = order_id;
            order.symbol_id = 1;
            order.exchange_id = 1;
            order.side = rand() % 2;
            order.type = 1; // Limit
            order.price = 250000000000LL + (rand() % 1000000000 - 500000000);
            order.quantity = 10 + rand() % 90;
            order.tif = 0; // Day
            order.status = 0;
            order.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            engine.submit_order(order);
        }
    });
    
    // Stats reporting thread
    std::thread stats_thread([&]() {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            auto stats = engine.get_stats();
            std::cout << "\n=== STATS ===" << std::endl;
            std::cout << "Ticks processed: " << stats.ticks_processed << std::endl;
            std::cout << "Orders sent: " << stats.orders_sent << std::endl;
            std::cout << "Fills received: " << stats.fills_received << std::endl;
            std::cout << "Rejected orders: " << stats.rejected_orders << std::endl;
            std::cout << "Risk rejections: " << stats.risk_rejections << std::endl;
            std::cout << "Avg latency: " << stats.avg_latency_us << " us" << std::endl;
        }
    });
    
    // Wait for shutdown
    market_thread.join();
    order_thread.join();
    stats_thread.join();
    
    engine.stop();
    std::cout << "Engine stopped gracefully." << std::endl;
    
    return 0;
}