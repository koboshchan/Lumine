package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// Inspect Windows wiring on every host too: a cross-build alone cannot detect
// accidental reintroduction of a desktop-capture branch or selector.
func TestWindowsCaptureIsOBSOnly(t *testing.T) {
	mainSource, err := os.ReadFile("main_windows.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(mainSource)
	for _, forbidden := range []string{"CaptureOBS", "captureID", "capture(cfg.Window)", "Select GDI", "GDI selected"} {
		if strings.Contains(s, forbidden) {
			t.Fatalf("alternate capture wiring remains: %s", forbidden)
		}
	}
	for _, required := range []string{"obs, err := DialOBS(ctx, cfg.OBS)", "return obs.Capture(ctx)", "OBS WebSocket v5 required", "defer obs.Close()"} {
		if !strings.Contains(s, required) {
			t.Fatalf("missing mandatory OBS wiring: %s", required)
		}
	}
	// A failed connection must return before the capture/inference loop.
	start := strings.Index(s, "obs, err := DialOBS")
	if start < 0 {
		t.Fatal("missing OBS setup")
	}
	end := strings.Index(s[start:], "samples, captureDone :=")
	if end < 0 {
		t.Fatal("missing OBS setup before capture")
	}
	setup := s[start : start+end]
	if !strings.Contains(setup, "if err != nil {") || !strings.Contains(setup, "\n\t\treturn\n") {
		t.Fatal("OBS connection failure must stop the run")
	}
	native, err := os.ReadFile("win_windows.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"func capture(", "\"GetDC\"", "\"CreateCompatibleDC\"", "\"CreateDIBSection\"", "\"StretchBlt\"", "\"BitBlt\"", "\"PrintWindow\""} {
		if strings.Contains(string(native), forbidden) {
			t.Fatalf("desktop frame acquisition remains: %s", forbidden)
		}
	}
}

func TestOBSUnavailableFailsClosed(t *testing.T) {
	// A real local HTTP endpoint that is not OBS must never produce a session.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "OBS unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	obs, err := DialOBS(ctx, OBSConfig{Address: "ws" + strings.TrimPrefix(server.URL, "http")})
	if obs != nil {
		obs.Close()
		t.Fatal("non-OBS server produced a capture session")
	}
	if err == nil {
		t.Fatal("unavailable OBS did not fail")
	}
}
