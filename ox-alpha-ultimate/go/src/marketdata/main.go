package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/valyala/fasthttp"
	"github.com/gorilla/websocket"
	"github.com/segmentio/kafka-go"
	"gopkg.in/yaml.v3"
)

// ... (continuing from previous)

func (s *MarketDataService) broadcastTick(tick MarketTick) {
	data, _ := json.Marshal(tick)

	s.wsMutex.RLock()
	for client := range s.wsClients {
		if err := client.WriteMessage(websocket.TextMessage, data); err != nil {
			client.Close()
			delete(s.wsClients, client)
		}
	}
	s.wsMutex.RUnlock()
}

func (s *MarketDataService) reportStats() {
	defer s.wg.Done()
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.statsMutex.RLock()
			s.wsMutex.RLock()
			s.stats.WSClients = len(s.wsClients)
			log.Printf("Stats: Ticks=%d Published=%d WSClients=%d Errors=%v Uptime=%v",
				s.stats.TicksReceived, s.stats.TicksPublished, s.stats.WSClients,
				s.stats.ExchangeErrors, time.Since(s.stats.StartTime).Round(time.Second))
			s.wsMutex.RUnlock()
			s.statsMutex.RUnlock()
		}
	}
}

// HTTP Handlers
func (s *MarketDataService) handleWS(ctx *fasthttp.RequestCtx) {
	conn, err := s.wsUpgrader.Upgrade(ctx, nil, nil)
	if err != nil {
		log.Printf("WS upgrade error: %v", err)
		return
	}

	s.wsMutex.Lock()
	s.wsClients[conn] = true
	s.wsMutex.Unlock()

	// Keep connection alive
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			break
		}
	}

	s.wsMutex.Lock()
	delete(s.wsClients, conn)
	s.wsMutex.Unlock()
	conn.Close()
}

func (s *MarketDataService) handleHealth(ctx *fasthttp.RequestCtx) {
	s.statsMutex.RLock()
	s.wsMutex.RLock()
	resp := map[string]interface{}{
		"status":          "healthy",
		"ticks_received":  s.stats.TicksReceived,
		"ticks_published": s.stats.TicksPublished,
		"ws_clients":      len(s.wsClients),
		"exchanges":       len(s.exchanges),
		"uptime_sec":      time.Since(s.stats.StartTime).Seconds(),
	}
	s.wsMutex.RUnlock()
	s.statsMutex.RUnlock()

	ctx.SetContentType("application/json")
	json.NewEncoder(ctx).Encode(resp)
}

func (s *MarketDataService) handleMetrics(ctx *fasthttp.RequestCtx) {
	s.statsMutex.RLock()
	defer s.statsMutex.RUnlock()

	ctx.SetContentType("text/plain; version=0.0.4")
	fmt.Fprintf(ctx, "ticks_received_total %d\n", s.stats.TicksReceived)
	fmt.Fprintf(ctx, "ticks_published_total %d\n", s.stats.TicksPublished)
	fmt.Fprintf(ctx, "ws_clients %d\n", len(s.wsClients))
	
	for exch, errs := range s.stats.ExchangeErrors {
		fmt.Fprintf(ctx, "exchange_errors{exchange=\"%s\"} %d\n", exch, errs)
	}
}

func (s *MarketDataService) Run() error {
	// HTTP server
	server := &fasthttp.Server{
		Handler: func(ctx *fasthttp.RequestCtx) {
			switch string(ctx.Path()) {
			case "/ws":
				s.handleWS(ctx)
			case "/health":
				s.handleHealth(ctx)
			case "/metrics":
				s.handleMetrics(ctx)
			default:
				ctx.Error("Not Found", fasthttp.StatusNotFound)
			}
		},
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	go func() {
		addr := fmt.Sprintf(":%d", s.config.Server.HTTPPort)
		log.Printf("HTTP server listening on %s", addr)
		if err := server.ListenAndServe(addr); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	s.cancel()
	s.wg.Wait()
	s.kafkaWriter.Close()

	for _, conn := range s.exchanges {
		conn.Close()
	}

	return nil
}

// ============================================================================
// Exchange Connectors
// ============================================================================

// Binance Connector
type BinanceConnector struct {
	config    ExchangeConfig
	wsConn    *websocket.Conn
	mu        sync.Mutex
	ctx       context.Context
	cancel    context.CancelFunc
	msgHandler func([]byte)
}

func NewBinanceConnector(cfg ExchangeConfig) *BinanceConnector {
	ctx, cancel := context.WithCancel(context.Background())
	return &BinanceConnector{
		config: cfg,
		ctx:    ctx,
		cancel: cancel,
	}
}

func (b *BinanceConnector) Connect(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	url := b.config.WSURL + "/stream"
	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, _, err := dialer.DialContext(ctx, url, nil)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	b.wsConn = conn
	return nil
}

func (b *BinanceConnector) Subscribe(symbols []string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.wsConn == nil {
		return fmt.Errorf("not connected")
	}

	streams := make([]string, len(symbols))
	for i, s := range symbols {
		streams[i] = strings.ToLower(s) + "@trade"
		streams = append(streams, strings.ToLower(s)+"@depth@100ms")
		streams = append(streams, strings.ToLower(s)+"@bookTicker")
	}

	msg := map[string]interface{}{
		"method": "SUBSCRIBE",
		"params": streams,
		"id":     1,
	}
	return b.wsConn.WriteJSON(msg)
}

func (b *BinanceConnector) OnMessage(msg []byte) error {
	// Parse Binance stream messages
	// Would parse and emit to tick channel
	return nil
}

func (b *BinanceConnector) Close() error {
	b.cancel()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.wsConn != nil {
		return b.wsConn.Close()
	}
	return nil
}

func (b *BinanceConnector) Name() string { return "binance" }

// Dhan Connector (Indian market)
type DhanConnector struct {
	config    ExchangeConfig
	wsConn    *websocket.Conn
	mu        sync.Mutex
	ctx       context.Context
	cancel    context.CancelFunc
}

func NewDhanConnector(cfg ExchangeConfig) *DhanConnector {
	ctx, cancel := context.WithCancel(context.Background())
	return &DhanConnector{
		config: cfg,
		ctx:    ctx,
		cancel: cancel,
	}
}

func (d *DhanConnector) Connect(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	conn, _, err := websocket.DefaultDialer.DialContext(ctx, d.config.WSURL, nil)
	if err != nil {
		return err
	}
	d.wsConn = conn
	return d.authenticate()
}

func (d *DhanConnector) authenticate() error {
	msg := map[string]interface{}{
		"token":     d.config.APIKey,
		"clientId":  d.config.Passphrase,
		"authType":  2,
	}
	return d.wsConn.WriteJSON(msg)
}

func (d *DhanConnector) Subscribe(symbols []string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	instruments := make([]map[string]interface{}, len(symbols))
	for i, s := range symbols {
		instruments[i] = map[string]interface{}{
			"ExchangeSegment": "NSE_EQ",
			"SecurityId":      s, // Would map to security ID
		}
	}

	msg := map[string]interface{}{
		"RequestCode":      23,
		"InstrumentCount":  len(instruments),
		"InstrumentList":   instruments,
	}
	return d.wsConn.WriteJSON(msg)
}

func (d *DhanConnector) OnMessage(msg []byte) error {
	// Parse Dhan binary depth packets
	return nil
}

func (d *DhanConnector) Close() error {
	d.cancel()
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.wsConn != nil {
		return d.wsConn.Close()
	}
	return nil
}

func (d *DhanConnector) Name() string { return "dhan" }

// ============================================================================
// Main
// ============================================================================

func main() {
	configPath := "config.yaml"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}

	service, err := NewMarketDataService(configPath)
	if err != nil {
		log.Fatalf("Failed to create service: %v", err)
	}

	if err := service.Start(); err != nil {
		log.Fatalf("Failed to start service: %v", err)
	}

	if err := service.Run(); err != nil {
		log.Fatalf("Service error: %v", err)
	}
}