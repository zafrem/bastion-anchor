// Package hooks provides a lightweight, non-blocking hook/extension-point
// system for Anchor (Foundation Architecture doc 01 §2.3).
//
// Cross-cutting coordinators (e.g. Tracker) register Hooks; core operations
// fire them asynchronously so they never block the data path.
package hooks

import (
	"context"
	"sync"
)

// EventType identifies the operation that fired a hook.
type EventType string

const (
	EventEmbeddingSecured  EventType = "embedding_secured"
	EventBiasDetected      EventType = "bias_detected"
	EventAnomalyDetected   EventType = "anomaly_detected"
	EventQualityDegraded   EventType = "quality_degraded"
	EventResponseVerified  EventType = "response_verified"
	EventSemanticDrift     EventType = "semantic_drift_detected"
)

// Event carries the hook payload.
type Event struct {
	Type      EventType
	RequestID string
	TenantID  string
	TraceID   string
	SpanID    string
	Data      map[string]interface{}
	Ctx       context.Context
}

// Hook is implemented by any object that wants to observe Anchor events.
type Hook interface {
	Name() string
	OnEvent(Event)
}

// Manager registers hooks and dispatches events to them.
type Manager struct {
	mu    sync.RWMutex
	hooks map[EventType][]Hook
}

// New returns an empty Manager.
func New() *Manager {
	return &Manager{hooks: make(map[EventType][]Hook)}
}

// Register adds hook h to the listener set for eventType.
func (m *Manager) Register(eventType EventType, h Hook) {
	m.mu.Lock()
	m.hooks[eventType] = append(m.hooks[eventType], h)
	m.mu.Unlock()
}

// Fire dispatches ev to all registered hooks for ev.Type asynchronously.
// The caller is never blocked.
func (m *Manager) Fire(ev Event) {
	m.mu.RLock()
	hs := m.hooks[ev.Type]
	m.mu.RUnlock()
	for _, h := range hs {
		go h.OnEvent(ev)
	}
}
