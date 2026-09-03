/**
 * Market Data Decoder Implementation
 */

#include "market_data_decoder.hpp"
#include <cmath>

size_t DhanDepthDecoder::parse(const uint8_t* data, size_t len) {
    size_t parsed = 0;
    const uint8_t* ptr = data;
    const uint8_t* end = data + len;
    
    while (ptr + PACKET_SIZE <= end) {
        uint16_t msg_len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += 2;
        uint8_t response_code = *ptr++;
        ptr++; // segment
        uint32_t security_id = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        uint32_t sequence = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        
        if (msg_len != PACKET_SIZE || (response_code != BID_CODE && response_code != ASK_CODE)) {
            ptr = data + parsed + PACKET_SIZE;
            parsed += PACKET_SIZE;
            continue;
        }
        
        DepthPacket packet;
        packet.security_id = security_id;
        packet.side = (response_code == BID_CODE) ? 0 : 1;
        packet.sequence = sequence;
        
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

void BinanceDecoder::parse(const char* data, size_t len) {
    // Simplified - would use simdjson for production
}

void BybitDecoder::parse(const char* data, size_t len) {
    // Parse Bybit WebSocket messages
}

void CoinbaseDecoder::parse(const char* data, size_t len) {
    // Parse Coinbase Pro JSON messages
}

void FastCSVParser::parse(const char* data, size_t len, Callback cb) {
    const char* ptr = data;
    const char* end = data + len;
    Row row;
    
    while (ptr < end) {
        row.field_count = 0;
        
        while (ptr < end && *ptr != '\n' && row.field_count < 16) {
            row.fields[row.field_count++] = ptr;
            
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

size_t OrderEncoder::encode_dhan_order(uint8_t* buffer, const Order& order) {
    size_t offset = 0;
    
    *reinterpret_cast<uint16_t*>(buffer + offset) = 0;
    offset += 2;
    buffer[offset++] = 1;
    buffer[offset++] = 1;
    *reinterpret_cast<uint32_t*>(buffer + offset) = order.symbol_id;
    offset += 4;
    *reinterpret_cast<uint32_t*>(buffer + offset) = order.client_order_id;
    offset += 4;
    
    buffer[offset++] = order.side;
    buffer[offset++] = order.type;
    *reinterpret_cast<int64_t*>(buffer + offset) = order.price;
    offset += 8;
    *reinterpret_cast<uint64_t*>(buffer + offset) = order.quantity;
    offset += 8;
    
    *reinterpret_cast<uint16_t*>(buffer) = static_cast<uint16_t>(offset);
    
    return offset;
}

std::string OrderEncoder::encode_binance_order(const Order& order) {
    return "";
}