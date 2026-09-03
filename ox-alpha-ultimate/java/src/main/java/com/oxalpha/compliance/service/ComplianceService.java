package com.oxalpha.compliance.service;

import com.oxalpha.compliance.entity.AuditTrail;
import com.oxalpha.compliance.repository.AuditTrailRepository;
import com.oxalpha.compliance.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class ComplianceService {
    
    private final AuditTrailRepository auditRepository;
    private final ReportGenerator reportGenerator;
    private final AlertService alertService;
    
    // Record all trading events for audit trail
    @KafkaListener(topics = "trading.events", groupId = "compliance-service")
    public void handleTradingEvent(TradingEvent event) {
        try {
            AuditTrail audit = AuditTrail.builder()
                .correlationId(event.getCorrelationId())
                .agentId(event.getAgentId())
                .eventType(AuditTrail.EventType.valueOf(event.getEventType()))
                .symbol(event.getSymbol())
                .side(event.getSide() != null ? AuditTrail.Side.valueOf(event.getSide()) : AuditTrail.Side.NONE)
                .quantity(event.getQuantity())
                .price(event.getPrice())
                .pnl(event.getPnl())
                .commission(event.getCommission())
                .leverage(event.getLeverage())
                .metadata(event.getMetadata())
                .ipAddress(event.getIpAddress())
                .userAgent(event.getUserAgent())
                .timestamp(Instant.ofEpochMilli(event.getTimestamp()))
                .build();
            
            auditRepository.save(audit);
            
        } catch (Exception e) {
            log.error("Failed to record audit trail: {}", e.getMessage(), e);
        }
    }
    
    // Generate daily compliance report
    @Scheduled(cron = "0 0 6 * * ?") // 6 AM daily
    @Transactional
    public void generateDailyReport() {
        LocalDate reportDate = LocalDate.now().minusDays(1);
        log.info("Generating daily compliance report for {}", reportDate);
        
        DailyComplianceReport report = DailyComplianceReport.builder()
            .reportDate(reportDate)
            .generatedAt(Instant.now())
            .build();
        
        // Fetch trades for the day
        Instant start = reportDate.atStartOfDay().toInstant(ZoneOffset.UTC);
        Instant end = start.plusDays(1);
        
        List<AuditTrail> trades = auditRepository.findByEventTypeInAndTimestampBetween(
            Arrays.asList(
                AuditTrail.EventType.ORDER_PLACED,
                AuditTrail.EventType.ORDER_FILLED,
                AuditTrail.EventType.ORDER_PARTIAL_FILL,
                AuditTrail.EventType.ORDER_CANCELLED,
                AuditTrail.EventType.ORDER_REJECTED
            ),
            start, end
        );
        
        // Calculate metrics
        report.setTotalTrades(trades.size());
        report.setTotalVolume(trades.stream()
            .map(t -> t.getQuantity().multiply(t.getPrice()))
            .reduce(BigDecimal.ZERO, BigDecimal::add));
        report.setTotalPnl(trades.stream()
            .filter(t -> t.getPnl() != null)
            .map(AuditTrail::getPnl)
            .reduce(BigDecimal.ZERO, BigDecimal::add));
        report.setTotalCommission(trades.stream()
            .filter(t -> t.getCommission() != null)
            .map(AuditTrail::getCommission)
            .reduce(BigDecimal.ZERO, BigDecimal::add));
        
        // Group by agent
        Map<String, AgentSummary> agentSummaries = trades.stream()
            .collect(Collectors.groupingBy(
                AuditTrail::getAgentId,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    list -> AgentSummary.builder()
                        .agentId(list.get(0).getAgentId())
                        .tradeCount(list.size())
                        .totalPnl(list.stream()
                            .filter(t -> t.getPnl() != null)
                            .map(AuditTrail::getPnl)
                            .reduce(BigDecimal.ZERO, BigDecimal::add))
                        .totalVolume(list.stream()
                            .map(t -> t.getQuantity().multiply(t.getPrice()))
                            .reduce(BigDecimal.ZERO, BigDecimal::add))
                        .build()
                )
            ));
        
        report.setAgentSummaries(new ArrayList<>(agentSummaries.values()));
        
        // Risk events
        List<AuditTrail> riskEvents = auditRepository.findByEventTypeInAndTimestampBetween(
            Arrays.asList(
                AuditTrail.EventType.RISK_LIMIT_BREACH,
                AuditTrail.EventType.MARGIN_CALL,
                AuditTrail.EventType.LIQUIDATION
            ),
            start, end
        );
        
        report.setRiskEvents(riskEvents.stream()
            .map(this::toRiskEventDto)
            .collect(Collectors.toList()));
        
        // Generate report files
        String pdfPath = reportGenerator.generateDailyPdf(report);
        String excelPath = reportGenerator.generateDailyExcel(report);
        
        report.setPdfPath(pdfPath);
        report.setExcelPath(excelPath);
        
        // Save report
        // reportRepository.save(report);
        
        log.info("Daily compliance report generated: {} trades, {} agents, {} risk events",
            report.getTotalTrades(), report.getAgentSummaries().size(), riskEvents.size());
    }
    
    // Generate weekly summary
    @Scheduled(cron = "0 0 7 * * MON") // Monday 7 AM
    @Transactional
    public void generateWeeklyReport() {
        LocalDate endDate = LocalDate.now().minusDays(1);
        LocalDate startDate = endDate.minusDays(6);
        
        log.info("Generating weekly compliance report for {} to {}", startDate, endDate);
        
        WeeklyComplianceReport report = WeeklyComplianceReport.builder()
            .startDate(startDate)
            .endDate(endDate)
            .generatedAt(Instant.now())
            .build();
        
        // Aggregate daily reports
        List<DailyComplianceReport> dailyReports = // fetch from repository
            reportRepository.findByReportDateBetween(startDate, endDate);
        
        report.setDailyReports(dailyReports);
        
        // Weekly aggregates
        report.setTotalTrades(dailyReports.stream()
            .map(DailyComplianceReport::getTotalTrades)
            .reduce(0, Integer::sum));
        report.setTotalPnl(dailyReports.stream()
            .map(DailyComplianceReport::getTotalPnl)
            .reduce(BigDecimal.ZERO, BigDecimal::add));
        
        // Performance attribution
        report.setAttribution(calculateAttribution(dailyReports));
        
        // Generate files
        reportGenerator.generateWeeklyPdf(report);
        reportGenerator.generateWeeklyExcel(report);
    }
    
    // Generate monthly regulatory report
    @Scheduled(cron = "0 0 8 1 * ?") // 1st of month 8 AM
    @Transactional
    public void generateMonthlyReport() {
        YearMonth month = YearMonth.now().minusMonths(1);
        log.info("Generating monthly regulatory report for {}", month);
        
        // SEBI/Exchange specific reports
        // - Trade reporting
        // - Position reporting
        // - Margin reporting
        // - Investor grievance summary
    }
    
    // Real-time risk monitoring
    @KafkaListener(topics = "risk.alerts", groupId = "compliance-risk")
    public void handleRiskAlert(RiskAlert alert) {
        log.warn("Risk alert: {} for agent {}", alert.getAlertType(), alert.getAgentId());
        
        // Create audit entry
        AuditTrail audit = AuditTrail.builder()
            .correlationId(UUID.randomUUID())
            .agentId(alert.getAgentId())
            .eventType(AuditTrail.EventType.RISK_LIMIT_BREACH)
            .metadata(toJson(Map.of(
                "alert_type", alert.getAlertType(),
                "severity", alert.getSeverity(),
                "message", alert.getMessage(),
                "current_value", alert.getCurrentValue(),
                "limit_value", alert.getLimitValue()
            )))
            .timestamp(Instant.now())
            .build();
        
        auditRepository.save(audit);
        
        // Alert operations team
        if ("CRITICAL".equals(alert.getSeverity())) {
            alertService.sendCriticalAlert(alert);
        }
    }
    
    // Query methods
    public Page<AuditTrail> searchAuditTrail(AuditSearchRequest request, Pageable pageable) {
        return auditRepository.search(
            request.getAgentId(),
            request.getEventTypes(),
            request.getSymbol(),
            request.getStartTime(),
            request.getEndTime(),
            pageable
        );
    }
    
    public TradeReconstruction reconstructTrade(UUID correlationId) {
        List<AuditTrail> events = auditRepository.findByCorrelationIdOrderByTimestampAsc(correlationId);
        return TradeReconstruction.builder()
            .correlationId(correlationId)
            .events(events)
            .reconstructedAt(Instant.now())
            .build();
    }
    
    // Helper methods
    private RiskEventDto toRiskEventDto(AuditTrail audit) {
        return RiskEventDto.builder()
            .eventId(audit.getId())
            .agentId(audit.getAgentId())
            .eventType(audit.getEventType().name())
            .symbol(audit.getSymbol())
            .timestamp(audit.getTimestamp())
            .metadata(audit.getMetadata())
            .build();
    }
    
    private String toJson(Object obj) {
        try {
            return new ObjectMapper().writeValueAsString(obj);
        } catch (Exception e) {
            return "{}";
        }
    }
    
    private AttributionAnalysis calculateAttribution(List<DailyComplianceReport> reports) {
        // Calculate performance attribution
        return AttributionAnalysis.builder().build();
    }
}