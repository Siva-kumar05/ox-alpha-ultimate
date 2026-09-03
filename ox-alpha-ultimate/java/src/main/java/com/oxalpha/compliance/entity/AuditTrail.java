package com.oxalpha.compliance.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.envers.Audited;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "audit_trail", indexes = {
    @Index(name = "idx_audit_agent_timestamp", columnList = "agent_id, timestamp"),
    @Index(name = "idx_audit_event_type", columnList = "event_type"),
    @Index(name = "idx_audit_symbol", columnList = "symbol"),
    @Index(name = "idx_audit_correlation", columnList = "correlation_id")
})
@Audited
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AuditTrail {
    
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    @Column(name = "correlation_id", nullable = false)
    private UUID correlationId;
    
    @Column(name = "agent_id", nullable = false, length = 100)
    private String agentId;
    
    @Column(name = "event_type", nullable = false, length = 50)
    @Enumerated(EnumType.STRING)
    private EventType eventType;
    
    @Column(name = "symbol", length = 50)
    private String symbol;
    
    @Column(name = "side", length = 10)
    @Enumerated(EnumType.STRING)
    private Side side;
    
    @Column(name = "quantity", precision = 20, scale = 8)
    private BigDecimal quantity;
    
    @Column(name = "price", precision = 20, scale = 8)
    private BigDecimal price;
    
    @Column(name = "pnl", precision = 20, scale = 8)
    private BigDecimal pnl;
    
    @Column(name = "commission", precision = 20, scale = 8)
    private BigDecimal commission;
    
    @Column(name = "leverage", precision = 10, scale = 4)
    private BigDecimal leverage;
    
    @Column(name = "metadata", columnDefinition = "jsonb")
    @Convert(converter = JsonConverter.class)
    private String metadata;
    
    @Column(name = "ip_address", length = 45)
    private String ipAddress;
    
    @Column(name = "user_agent", length = 500)
    private String userAgent;
    
    @CreationTimestamp
    @Column(name = "timestamp", nullable = false, updatable = false)
    private Instant timestamp;
    
    @Column(name = "processed", nullable = false)
    @Builder.Default
    private Boolean processed = false;
    
    @Column(name = "processing_error", length = 1000)
    private String processingError;
    
    public enum EventType {
        SIGNAL_GENERATED,
        SIGNAL_VALIDATED,
        SIGNAL_REJECTED,
        ORDER_PLACED,
        ORDER_FILLED,
        ORDER_PARTIAL_FILL,
        ORDER_CANCELLED,
        ORDER_REJECTED,
        POSITION_OPENED,
        POSITION_CLOSED,
        POSITION_MODIFIED,
        STOP_LOSS_HIT,
        TAKE_PROFIT_HIT,
        TIME_STOP_HIT,
        RISK_LIMIT_BREACH,
        MARGIN_CALL,
        LIQUIDATION,
        CAPITAL_ALLOCATED,
        CAPITAL_DEALLOCATED,
        RISK_LIMIT_UPDATED,
        AGENT_STARTED,
        AGENT_STOPPED,
        AGENT_PAUSED,
        AGENT_RESUMED,
        SYSTEM_ALERT,
        MANUAL_OVERRIDE
    }
    
    public enum Side {
        BUY, SELL, NONE
    }
}