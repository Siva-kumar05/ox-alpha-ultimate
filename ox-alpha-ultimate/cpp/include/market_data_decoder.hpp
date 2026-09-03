/**
 * Market Data Decoder - C++17
 * Ultra-fast protocol parsing for exchange feeds
 * Supports: Dhan, Binance, Bybit, Coinbase, etc.
 */

#ifndef MARKET_DATA_DECODER_HPP
#define MARKET_DATA_DECODER_HPP

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <optional>

#include "execution_engine.hpp"

// Fast float parsing (avoids std::stod overhead)
inline double fast_atof(const char* str, size_t len) {
    double result = 0.0;
    double div = 1.0;
    bool neg = false;
    bool decimal = false;
    
    for (size_t i = 0; i < len; ++i) {
        char c = str[i];
        if (c == '-') { neg = true; }
        else if (c == '.') { decimal = true; }
        else if (c >= '0' && c <= '9') {
            if (decimal) {
                div *= 10.0;
                result += (c - '0') / div;
            } else {
                result = result * 10.0 + (c - '0');
            }
        }
    }
    return neg ? -result : result;
}

inline int64_t fast_atoi(const char* str, size_t len) {
    int64_t result = 0;
    bool neg = false;
    for (size_t i = 0; i < len; ++i) {
        char c = str[i];
        if (c == '-') { neg = true; }
        else if (c >= '0' && c <= '9') {
            result = result * 10 + (c - '0');
        }
    }
    return neg ? -result : result;
}

// Dhan 20-level depth decoder
class DhanDepthDecoder {
    static constexpr size_t PACKET_SIZE = 332;
    static constexpr size_t HEADER_SIZE = 12;
    static constexpr size_t LEVEL_SIZE = 16;
    static constexpr uint8_t BID_CODE = 41;
    static constexpr uint8_t ASK_CODE = 51;
    
    struct alignas(16) DepthLevel {
        int64_t price;      // Fixed-point 1e8
        uint64_t quantity;
        uint32_t order_count;
    };
    
    struct alignas(16) DepthPacket {
        uint32_t security_id;
        uint8_t side;       // 0=bid, 1=ask
        uint32_t sequence;
        DepthLevel levels[20];
    };
    
public:
    using Callback = std::function<void(uint32_t security_id, uint8_t side, const DepthPacket&)>;
    
    DhanDepthDecoder(Callback cb) : callback_(cb) {}
    
    // Zero-copy parsing - processes buffer in place
    size_t parse(const uint8_t* data, size_t len) {
        size_t parsed = 0;
        const uint8_t* ptr = data;
        const uint8_t* end = data + len;
        
        while (ptr + PACKET_SIZE <= end) {
            // Parse header (12 bytes)
            uint16_t msg_len = *reinterpret_cast<const uint16_t*>(ptr);
            ptr += 2;
            uint8_t response_code = *ptr++;
            ptr++; // segment
            uint32_t security_id = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += 4;
            uint32_t sequence = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += 4;
            
            if (msg_len != 332 || (response_code != BID_CODE && response_code != ASK_CODE)) {
                ptr = data + parsed + PACKET_SIZE; // Skip to next packet
                parsed += PACKET_SIZE;
                continue;
            }
            
            DepthPacket packet;
            packet.security_id = security_id;
            packet.side = (response_code == BID_CODE) ? 0 : 1;
            packet.sequence = sequence;
            
            // Parse 20 levels (16 bytes each = 320 bytes)
            for (int i = 0; i < 20; ++i) {
                double price = *reinterpret_cast<const double*>(ptr);
                ptr += 8;
                uint64_t qty = *reinterpret_cast<const uint64_t*>(ptr);
                ptr += 8;
                uint32_t count = *reinterpret_cast<const uint32_t*>(ptr);
                ptr += 4;
                
                if (price > 0 && qty > 0) {
                    packet.levels[i] = {
                        static_cast<int64_t>(price * 100000000),
                        qty,
                        count
                    };
                }
            }
            
            callback_(security_id, packet.side, packet);
            parsed += PACKET_SIZE;
        }
        
        return parsed;
    }
    
private:
    std::function<void(uint32_t, uint8_t, const DepthPacket&)> callback_;
};

// Binance WebSocket decoder (JSON + binary)
class BinanceDecoder {
public:
    struct StreamData {
        enum Type { TRADE, DEPTH, TICKER, KLINE, BOOK_TICKER } type;
        std::string symbol;
        int64_t event_time;
        // Trade
        int64_t price;
        int64_t qty;
        bool is_buyer_maker;
        // Depth
        std::vector<std::pair<int64_t, int64_t>> bids;
        std::vector<std::pair<int64_t, int64_t>> asks;
        // Ticker
        int64_t high, low, open, close, volume;
    };
    
    using Callback = std::function<void(const StreamData&)>;
    
    BinanceDecoder(Callback cb) : callback_(cb) {}
    
    void parse(const char* data, size_t len) {
        // Fast JSON parsing for Binance streams
        // Would use simdjson or custom parser
        // Simplified for header
    }
    
private:
    std::function<void(const StreamData&)> callback_;
};

// Bybit decoder
class BybitDecoder {
public:
    struct PublicTrade {
        std::string symbol;
        int64_t price;
        int64_t qty;
        char side; // 'B'uy or 'S'ell
        int64_t timestamp;
    };
    
    struct OrderbookDelta {
        std::string symbol;
        std::vector<std::pair<int64_t, int64_t>> bids;
        std::vector<std::pair<int64_t, int64_t>> asks;
        uint64_t timestamp;
    };
    
    using Callback = std::function<void(const PublicTrade&)>;
    using DepthCallback = std::function<void(const OrderbookDelta&)>;
    
    BybitDecoder(Callback trade_cb, DepthCallback depth_cb)
        : trade_cb_(trade_cb), depth_cb_(depth_cb) {}
    
    void parse(const char* data, size_t len) {
        // Parse Bybit WebSocket messages
    }
    
private:
    std::function<void(const PublicTrade&)> trade_cb_;
    std::function<void(const OrderbookDelta&)> depth_cb_;
};

// Coinbase Pro decoder
class CoinbaseDecoder {
public:
    struct Ticker {
        std::string product_id;
        int64_t price;
        int64_t open_24h;
        int64_t low_24h;
        int64_t high_24h;
        int64_t volume_24h;
        int64_t timestamp;
    };
    
    struct Level2Update {
        std::string product_id;
        std::vector<std::tuple<char, int64_t, int64_t>> changes; // side, price, size
    };
    
    using Callback = std::function<void(const Ticker&)>;
    using Level2Callback = std::function<void(const Level2Update&)>;
    
    CoinbaseDecoder(Callback cb, Level2Callback l2cb) 
        : cb_(cb), l2cb_(l2cb) {}
    
    void parse(const char* data, size_t len) {
        // Parse Coinbase Pro JSON messages
    }
    
private:
    std::function<void(const Ticker&)> cb_;
    std::function<void(const Level2Update&)> l2cb_;
};

// Generic fast CSV parser for historical data
class FastCSVParser {
public:
    struct Row {
        std::array<const char*, 16> fields;
        uint8_t field_count = 0;
    };
    
    using Callback = std::function<bool(const Row&)>; // Return false to stop
    
    static void parse(const char* data, size_t len, Callback cb) {
        const char* ptr = data;
        const char* end = data + len;
        Row row;
        
        while (ptr < end) {
            row.field_count = 0;
            
            while (ptr < end && *ptr != '\n' && row.field_count < 16) {
                row.fields[row.field_count++] = ptr;
                
                // Find next comma or newline
                while (ptr < end && *ptr != ',' && *ptr != '\n') ++ptr;
                
                if (ptr < end && *ptr == ',') {
                    ++ptr;
                }
            }
            
            if (row.field_count > 0) {
                if (!cb(row)) break;
            }
            
            if (ptr < end && *ptr == '\n') ++ptr;
        }
    }
};

// Binary protocol encoder for order entry
class OrderEncoder {
public:
    // Dhan order format
    static size_t encode_dhan_order(uint8_t* buffer, const Order& order) {
        size_t offset = 0;
        
        // Header
        *reinterpret_cast<uint16_t*>(buffer + offset) = 0; // Length placeholder
        offset += 2;
        buffer[offset++] = 1; // Request code: New Order
        buffer[offset++] = 1; // Segment: NSE_EQ
        *reinterpret_cast<uint32_t*>(buffer + offset) = order.symbol_id;
        offset += 4;
        *reinterpret_cast<uint32_t*>(buffer + offset) = order.client_order_id;
        offset += 4;
        
        // Order fields
        buffer[offset++] = order.side; // 0=Buy, 1=Sell
        buffer[offset++] = order.type; // 1=Market, 2=Limit
        *reinterpret_cast<int64_t*>(buffer + offset) = order.price;
        offset += 8;
        *reinterpret_cast<uint64_t*>(buffer + offset) = order.quantity;
        offset += 8;
        
        // Update length
        *reinterpret_cast<uint16_t*>(buffer) = static_cast<uint16_t>(offset);
        
        return offset;
    }
    
    // Binance order format (JSON)
    static std::string encode_binance_order(const Order& order) {
        // Would use rapidjson or fmt
        return "";
    }
};

#endif // MARKET_DATA_DECODER_HPP