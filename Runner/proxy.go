package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func validHex32(s string) bool {
	if len(s) != 32 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// Session proxy protocol: the server keeps tiered history; the client sends only the newest frame.
func randHex32() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

type StepResult struct{ SessionID, Action, ActionID, Protocol string }

// ProxySession is sequential: callers must not use it concurrently.
type ProxySession struct {
	HTTP      *http.Client
	Base      string
	Key       string
	SessionID string
	// Set ONLY by MarkExecuted after real execution; never from dry-run.
	lastAction *string
	lastID     *string
	pendingID  string
	pending    string
	hasPending bool
}

func NewProxySession(h *http.Client, endpoint, key string) (*ProxySession, error) {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("endpoint must be an http(s) /v1 base URL without credentials/query")
	}
	sid, err := randHex32()
	if err != nil {
		return nil, err
	}
	return &ProxySession{HTTP: h, Base: strings.TrimRight(u.String(), "/"), Key: key, SessionID: sid}, nil
}

func (p *ProxySession) client() *http.Client {
	if p.HTTP != nil {
		return p.HTTP
	}
	return &http.Client{Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

func (p *ProxySession) post(ctx context.Context, path string, body any) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", p.Base+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.Key != "" {
		req.Header.Set("Authorization", "Bearer "+p.Key)
	}
	resp, err := p.client().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 1<<20 {
		return nil, fmt.Errorf("proxy response too large")
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("proxy HTTP %d (check server logs)", resp.StatusCode)
	}
	return b, nil
}

// Reset asks the server to drop this session's history, then starts a fresh local session id.
func (p *ProxySession) Reset(ctx context.Context) error {
	_, err := p.post(ctx, "/session/reset", map[string]string{"session_id": p.SessionID})
	p.forget()
	return err
}

func (p *ProxySession) forget() {
	p.lastAction, p.lastID, p.hasPending = nil, nil, false
	if sid, err := randHex32(); err == nil {
		p.SessionID = sid
	}
}

// Step sends the newest frame plus the last REALLY executed action (nil if none).
func (p *ProxySession) Step(ctx context.Context, frame image.Image) (StepResult, error) {
	var res StepResult
	padded, err := PaddedFrame(frame)
	if err != nil {
		return res, err
	}
	var buf bytes.Buffer
	if err = (&png.Encoder{CompressionLevel: png.BestSpeed}).Encode(&buf, padded); err != nil {
		return res, err
	}
	rid, err := randHex32()
	if err != nil {
		return res, err
	}
	body := map[string]any{
		"session_id": p.SessionID, "request_id": rid,
		"frame":                "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes()),
		"last_executed_action": p.lastAction, "executed_action_id": p.lastID,
	}
	b, err := p.post(ctx, "/session/step", body)
	if err != nil {
		return res, err
	}
	var r struct {
		SessionID string `json:"session_id"`
		Action    string `json:"action"`
		ActionID  string `json:"action_id"`
		Protocol  string `json:"protocol"`
	}
	if err = json.Unmarshal(b, &r); err != nil {
		return res, fmt.Errorf("invalid proxy JSON: %w", err)
	}
	if r.SessionID != p.SessionID || !validHex32(r.ActionID) || r.Protocol != "lumine-session-v1" {
		return res, fmt.Errorf("proxy response rejected (session/action/action_id mismatch)")
	}
	if _, err := ParseAction(r.Action); err != nil {
		return res, fmt.Errorf("proxy action rejected: %w", err)
	}
	res = StepResult{r.SessionID, r.Action, r.ActionID, r.Protocol}
	p.lastAction, p.lastID = nil, nil // consumed only after a successful step
	p.pending, p.pendingID, p.hasPending = r.Action, r.ActionID, true
	return res, nil
}

// MarkExecuted must be called only after Play succeeded for this exact action id.
func (p *ProxySession) MarkExecuted(actionID string) {
	if !p.hasPending || p.pendingID != actionID {
		p.lastAction, p.lastID, p.hasPending = nil, nil, false
		return
	}
	a, id := p.pending, p.pendingID
	p.lastAction, p.lastID, p.hasPending = &a, &id, false
}
