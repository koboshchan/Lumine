package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProxyContinuity(t *testing.T) {
	var got []map[string]any
	resets := 0
	n := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			w.WriteHeader(401)
			return
		}
		var m map[string]any
		json.NewDecoder(r.Body).Decode(&m)
		if r.URL.Path == "/v1/session/reset" {
			resets++
			w.Write([]byte("{}"))
			return
		}
		got = append(got, m)
		n++
		id := fmt.Sprintf("%032x", n)
		json.NewEncoder(w).Encode(map[string]string{"session_id": m["session_id"].(string), "action": "<|action_start|>0 0 0;;;;;;<|action_end|>", "action_id": id, "protocol": "lumine-session-v1"})
	}))
	defer s.Close()
	p, err := NewProxySession(nil, s.URL+"/v1/", "k")
	if err != nil || len(p.SessionID) != 32 {
		t.Fatal(err)
	}
	f := image.NewRGBA(image.Rect(0, 0, 1280, 720))
	r1, err := p.Step(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	// dry-run: no MarkExecuted
	if _, err = p.Step(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if got[1]["last_executed_action"] != nil || got[1]["executed_action_id"] != nil {
		t.Fatal("dry-run action leaked")
	}
	p.MarkExecuted(r1.ActionID) // stale id must not be accepted
	if p.lastAction != nil {
		t.Fatal("mismatched id acknowledged")
	}
	r3, _ := p.Step(context.Background(), f)
	p.MarkExecuted(r3.ActionID)
	p.Step(context.Background(), f)
	if got[3]["last_executed_action"] != r3.Action || got[3]["executed_action_id"] != r3.ActionID {
		t.Fatal("executed action not sent", got[3])
	}
	old := p.SessionID
	if err = p.Reset(context.Background()); err != nil || resets != 1 || p.SessionID == old || p.lastAction != nil {
		t.Fatal("reset")
	}
	bad, _ := NewProxySession(nil, s.URL+"/v1", "wrong")
	if _, err = bad.Step(context.Background(), f); err == nil {
		t.Fatal("auth error not failing closed")
	}
}

func TestProxyRejectsUnsafeOrIncompatibleResponse(t *testing.T) {
	cases := []struct{ name, protocol, action, id, session string }{
		{"bad protocol", "other", "<|action_start|>0 0 0;;;;;;<|action_end|>", "0123456789abcdef0123456789abcdef", ""},
		{"invalid id", "lumine-session-v1", "<|action_start|>0 0 0;;;;;;<|action_end|>", "g123456789abcdef0123456789abcdef", ""},
		{"unsafe action", "lumine-session-v1", "<|action_start|>0 0 0;Alt Four;;;;;<|action_end|>", "0123456789abcdef0123456789abcdef", ""},
		{"wrong session", "lumine-session-v1", "<|action_start|>0 0 0;;;;;;<|action_end|>", "0123456789abcdef0123456789abcdef", "00000000000000000000000000000000"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var m map[string]any
				json.NewDecoder(r.Body).Decode(&m)
				sid := m["session_id"].(string)
				if tc.session != "" {
					sid = tc.session
				}
				json.NewEncoder(w).Encode(map[string]string{"session_id": sid, "action": tc.action, "action_id": tc.id, "protocol": tc.protocol})
			}))
			defer server.Close()
			p, _ := NewProxySession(nil, server.URL+"/v1", "secret")
			if _, err := p.Step(context.Background(), image.NewRGBA(image.Rect(0, 0, 1280, 720))); err == nil {
				t.Fatal("accepted incompatible response")
			}
			if p.hasPending {
				t.Fatal("rejected response committed")
			}
		})
	}
}

func TestProxyCancellationDoesNotAcknowledge(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("cancelled request reached upstream") }))
	defer server.Close()
	p, _ := NewProxySession(nil, server.URL+"/v1", "secret")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Step(ctx, image.NewRGBA(image.Rect(0, 0, 1280, 720))); err == nil {
		t.Fatal("accepted cancelled request")
	}
	if p.hasPending || p.lastAction != nil {
		t.Fatal("cancelled action entered state")
	}
}
