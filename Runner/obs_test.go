package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"image"
	"image/jpeg"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func obsTestJPEG(w, h int) string {
	var b bytes.Buffer
	_ = jpeg.Encode(&b, image.NewRGBA(image.Rect(0, 0, w, h)), &jpeg.Options{Quality: 90})
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(b.Bytes())
}

type obsMockOptions struct {
	auth       bool
	password   string
	mode       string
	data       string
	delay      time.Duration
	identified atomic.Int32
	requests   atomic.Int32
	entered    chan struct{}
}

func obsMock(t *testing.T, opts *obsMockOptions) OBSConfig {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{Subprotocols: []string{"obswebsocket.json"}})
		if err != nil {
			return
		}
		defer c.CloseNow()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		hello := map[string]any{"rpcVersion": 1}
		if opts.auth {
			hello["authentication"] = map[string]string{"salt": "test-salt", "challenge": "test-challenge"}
		}
		if opts.mode == "hello-rpc" {
			hello["rpcVersion"] = 0
		}
		if opts.mode == "hello-timeout" {
			<-ctx.Done()
			return
		}
		_ = wsjson.Write(ctx, c, map[string]any{"op": 0, "d": hello})
		var identify struct {
			Op int            `json:"op"`
			D  map[string]any `json:"d"`
		}
		if wsjson.Read(ctx, c, &identify) != nil {
			return
		}
		opts.identified.Add(1)
		if identify.Op != 1 || identify.D["rpcVersion"] != float64(1) || identify.D["eventSubscriptions"] != float64(0) {
			t.Error("invalid identify")
			return
		}
		if opts.auth {
			a := sha256.Sum256([]byte(opts.password + "test-salt"))
			b := sha256.Sum256([]byte(base64.StdEncoding.EncodeToString(a[:]) + "test-challenge"))
			if identify.D["authentication"] != base64.StdEncoding.EncodeToString(b[:]) {
				_ = c.Close(4009, "SECRET_PASSWORD_ADDRESS")
				return
			}
		} else if _, ok := identify.D["authentication"]; ok {
			t.Error("auth sent without challenge")
			return
		}
		version := 1
		if opts.mode == "rpc" {
			version = 2
		}
		_ = wsjson.Write(ctx, c, map[string]any{"op": 2, "d": map[string]int{"negotiatedRpcVersion": version}})
		latestScene := ""
		for {
			var req struct {
				Op int `json:"op"`
				D  struct {
					Type string         `json:"requestType"`
					ID   string         `json:"requestId"`
					Data map[string]any `json:"requestData"`
				} `json:"d"`
			}
			if wsjson.Read(ctx, c, &req) != nil {
				return
			}
			opts.requests.Add(1)
			if req.Op != 6 {
				t.Error("wrong request opcode")
				return
			}
			if opts.entered != nil {
				select {
				case opts.entered <- struct{}{}:
				default:
				}
			}
			if opts.mode == "disconnect" {
				_ = c.Close(4000, "SECRET_PASSWORD_ADDRESS")
				return
			}
			if opts.mode == "oversize" {
				_ = c.Write(ctx, websocket.MessageText, []byte(strings.Repeat("x", obsMessageLimit+1)))
				return
			}
			if opts.delay > 0 {
				select {
				case <-time.After(opts.delay):
				case <-ctx.Done():
					return
				}
			}
			data := map[string]any{"imageData": opts.data}
			if req.D.Type == "GetCurrentProgramScene" {
				latestScene = fmt.Sprintf("Scene%d", opts.requests.Load())
				data = map[string]any{"sceneName": latestScene}
				if opts.mode == "legacy" {
					latestScene = "LegacyScene"
					data = map[string]any{"currentProgramSceneName": latestScene}
				}
			} else {
				if req.D.Type != "GetSourceScreenshot" || req.D.Data["imageFormat"] != "jpg" || req.D.Data["imageWidth"] != float64(1280) || req.D.Data["imageHeight"] != float64(720) || req.D.Data["imageCompressionQuality"] != float64(90) {
					t.Error("invalid screenshot request")
				}
				wantSource := "Game"
				if opts.mode == "program" || opts.mode == "legacy" {
					wantSource = latestScene
				}
				if req.D.Data["sourceName"] != wantSource {
					t.Errorf("wrong source got %v want %s", req.D.Data["sourceName"], wantSource)
				}
			}
			id := req.D.ID
			kind := req.D.Type
			op := 7
			success := true
			code := 100
			switch opts.mode {
			case "id":
				id = "wrong"
			case "type":
				kind = "wrong"
			case "opcode":
				op = 5
			case "status":
				success = false
				code = 600
			case "status-inconsistent":
				code = 600
			}
			response := map[string]any{"op": op, "d": map[string]any{"requestType": kind, "requestId": id, "requestStatus": map[string]any{"result": success, "code": code, "comment": "SECRET_PASSWORD_ADDRESS"}, "responseData": data}}
			if opts.mode == "binary" {
				b, _ := json.Marshal(response)
				_ = c.Write(ctx, websocket.MessageBinary, b)
			} else {
				_ = wsjson.Write(ctx, c, response)
			}
		}
	}))
	t.Cleanup(server.Close)
	return OBSConfig{Address: "ws" + strings.TrimPrefix(server.URL, "http"), Source: "Game"}
}
func TestOBSAuthentication(t *testing.T) {
	for _, tc := range []struct {
		name, password, serverPassword string
		auth, wantErr                  bool
	}{
		{"required", "correct", "correct", true, false},
		{"none-blank", "", "", false, false},
		{"none-password", "ignored", "", false, false},
		{"missing", "", "correct", true, true},
		{"wrong", "wrong", "correct", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := &obsMockOptions{auth: tc.auth, password: tc.serverPassword}
			cfg := obsMock(t, opts)
			cfg.Password = tc.password
			c, err := DialOBS(context.Background(), cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("dial error %v", err)
			}
			if c != nil {
				_ = c.Close()
			}
			if tc.name == "missing" && opts.identified.Load() != 0 {
				t.Fatal("Identify sent without required password")
			}
			if err != nil && (!strings.Contains(err.Error(), "password") || strings.Contains(err.Error(), "SECRET")) {
				t.Fatalf("bad auth error %v", err)
			}
		})
	}
}
func TestOBSCaptureModes(t *testing.T) {
	jpegData := obsTestJPEG(1280, 720)
	for _, mode := range []string{"direct", "program", "legacy"} {
		t.Run(mode, func(t *testing.T) {
			opts := &obsMockOptions{mode: mode, data: jpegData}
			cfg := obsMock(t, opts)
			if mode != "direct" {
				cfg.Source = ""
			}
			c, err := DialOBS(context.Background(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			for i := 0; i < 2; i++ {
				img, err := c.Capture(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				if img.Bounds() != image.Rect(0, 0, 1280, 720) || img.RGBAAt(0, 0).A != 255 {
					t.Fatal("invalid RGBA")
				}
			}
			want := int32(2)
			if mode != "direct" {
				want = 4
			}
			if opts.requests.Load() != want {
				t.Fatal("program scene not queried every capture")
			}
		})
	}
}
func TestOBSInvalidScreenshots(t *testing.T) {
	validRaw, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(obsTestJPEG(1280, 720), "data:image/jpeg;base64,"))
	truncated := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(validRaw[:len(validRaw)/2])
	bomb := append([]byte(nil), validRaw...)
	for i := 0; i+9 < len(bomb); i++ {
		if bomb[i] == 0xff && bomb[i+1] == 0xc0 {
			bomb[i+5] = 0xff
			bomb[i+6] = 0xff
			bomb[i+7] = 0xff
			bomb[i+8] = 0xff
			break
		}
	}
	huge := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bomb)
	for name, data := range map[string]string{
		"allocation-bomb": huge, "truncated": truncated,
		"geometry": obsTestJPEG(640, 480), "small": obsTestJPEG(640, 360),
		"bad-jpeg": "data:image/jpeg;base64,YmFk", "base64": "data:image/jpeg;base64,!!!",
		"png": "data:image/png;base64,YmFk", "empty": "",
	} {
		t.Run(name, func(t *testing.T) {
			opts := &obsMockOptions{data: data}
			cfg := obsMock(t, opts)
			c, err := DialOBS(context.Background(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if _, err = c.Capture(context.Background()); err == nil {
				t.Fatal("accepted bad screenshot")
			}
			if !c.closed.Load() {
				t.Fatal("failed stream not closed")
			}
		})
	}
}
func TestOBSProtocolFailures(t *testing.T) {
	for _, mode := range []string{"id", "type", "opcode", "status", "status-inconsistent", "disconnect", "oversize", "binary", "rpc", "hello-rpc"} {
		t.Run(mode, func(t *testing.T) {
			cfg := obsMock(t, &obsMockOptions{mode: mode, data: obsTestJPEG(1280, 720)})
			c, err := DialOBS(context.Background(), cfg)
			if err == nil {
				defer c.Close()
				_, err = c.Capture(context.Background())
			}
			if err == nil {
				t.Fatal("accepted invalid server response")
			}
			if strings.Contains(err.Error(), "SECRET") {
				t.Fatal("server detail leaked")
			}
		})
	}
}
func TestOBSURL(t *testing.T) {
	for _, address := range []string{"http://localhost", "ws://user:SECRET@localhost", "ws://localhost/?SECRET", "ws://localhost/#SECRET", "ws://localhost/?", "ws://localhost/#", "ws://"} {
		if _, err := obsURL(OBSConfig{Address: address}); err == nil {
			t.Errorf("accepted unsafe address %q", address)
		}
	}
	for _, port := range []string{"0", "65536", "secret"} {
		if _, err := obsURL(OBSConfig{Port: port}); err == nil {
			t.Fatal("bad port accepted")
		}
	}
	for _, cfg := range []OBSConfig{{}, {Address: "wss://example.com", Port: "4455"}, {Address: "ws://[::1]:4455"}} {
		if _, err := obsURL(cfg); err != nil {
			t.Fatal(err)
		}
	}
}
func TestOBSCancelTimeoutConcurrency(t *testing.T) {
	for _, mode := range []string{"cancel", "timeout", "close"} {
		t.Run(mode, func(t *testing.T) {
			opts := &obsMockOptions{delay: 5 * time.Second, entered: make(chan struct{}, 1)}
			c, err := DialOBS(context.Background(), obsMock(t, opts))
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			start := time.Now()
			go func() { _, err := c.Capture(ctx); done <- err }()
			<-opts.entered
			if _, err := c.Capture(context.Background()); err == nil || !strings.Contains(err.Error(), "progress") {
				t.Fatal("concurrent Capture accepted")
			}
			if mode == "cancel" {
				cancel()
			}
			if mode == "close" {
				_ = c.Close()
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("expected error")
				}
			case <-time.After(4 * time.Second):
				t.Fatal("capture deadline exceeded")
			}
			if mode == "timeout" && time.Since(start) < 2800*time.Millisecond {
				t.Fatal("timeout unexpectedly early")
			}
			if _, err := c.Capture(context.Background()); err == nil {
				t.Fatal("reused failed stream")
			}
		})
	}
}
func TestOBSHandshakeTimeout(t *testing.T) {
	cfg := obsMock(t, &obsMockOptions{mode: "hello-timeout"})
	start := time.Now()
	if _, err := DialOBS(context.Background(), cfg); err == nil {
		t.Fatal("missing Hello accepted")
	}
	if time.Since(start) > 4*time.Second {
		t.Fatal("handshake exceeded timeout")
	}
}
func TestOBSMockLatency(t *testing.T) {
	cfg := obsMock(t, &obsMockOptions{data: obsTestJPEG(1280, 720)})
	c, err := DialOBS(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	samples := make([]time.Duration, 30)
	var total time.Duration
	for i := range samples {
		start := time.Now()
		if _, err := c.Capture(context.Background()); err != nil {
			t.Fatal(err)
		}
		samples[i] = time.Since(start)
		total += samples[i]
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	t.Logf("MOCK ONLY, pre-encoded JPEG, includes request/decode/RGBA: mean=%s p95=%s n=%d; not native OBS latency", total/time.Duration(len(samples)), samples[28], len(samples))
}

// The official protocol's password/salt/challenge example, with its expected
// digest independently calculated using Python hashlib rather than this client.
func TestOBSOfficialAuthVector(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = wsjson.Write(ctx, c, map[string]any{"op": 0, "d": map[string]any{"rpcVersion": 1, "authentication": map[string]string{"salt": "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI=", "challenge": "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY="}}})
		var id struct {
			Op int `json:"op"`
			D  struct {
				Auth string `json:"authentication"`
			} `json:"d"`
		}
		if err := wsjson.Read(ctx, c, &id); err != nil {
			t.Error(err)
			return
		}
		if id.Op != 1 || id.D.Auth != "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4=" {
			t.Error("official authentication vector mismatch")
			return
		}
		_ = wsjson.Write(ctx, c, map[string]any{"op": 2, "d": map[string]int{"negotiatedRpcVersion": 1}})
		_, _, _ = c.Read(ctx)
	}))
	defer server.Close()
	c, err := DialOBS(context.Background(), OBSConfig{Address: "ws" + strings.TrimPrefix(server.URL, "http"), Password: "supersecretpassword"})
	if err != nil {
		t.Fatal(err)
	}
	_ = c.Close()
}
