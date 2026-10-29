package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

const obsTimeout = 3 * time.Second
const obsMessageLimit = 4 << 20

// OBSConfig is session-only. Passwords must never be persisted.
type OBSConfig struct{ Address, Port, Password, Source string }

type OBSCapture struct {
	conn     *websocket.Conn
	source   string
	busy     sync.Mutex
	closed   atomic.Bool
	sequence uint64 // protected by busy
}

type obsEnvelope struct {
	Op *int            `json:"op"`
	D  json.RawMessage `json:"d"`
}

func obsURL(cfg OBSConfig) (string, error) {
	address := strings.TrimSpace(cfg.Address)
	if address == "" {
		address = "ws://127.0.0.1"
	}
	if !strings.Contains(address, "://") {
		address = "ws://" + address
	}
	u, err := url.Parse(address)
	if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || strings.Contains(address, "#") {
		return "", errors.New("invalid OBS address; use ws or wss without credentials, query or fragment")
	}
	port := strings.TrimSpace(cfg.Port)
	if port == "" {
		port = u.Port()
		if port == "" {
			port = "4455"
		}
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return "", errors.New("invalid OBS port")
	}
	u.Host = net.JoinHostPort(u.Hostname(), port)
	return u.String(), nil
}

func obsTransportError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return fmt.Errorf("OBS operation canceled or timed out: %w", ctx.Err())
	}
	if websocket.CloseStatus(err) == 4009 {
		return errors.New("OBS authentication failed; check password")
	}
	return errors.New("OBS connection failed or closed; reconnect and check OBS websocket settings")
}

func (o *OBSCapture) read(ctx context.Context, op int, dest any) error {
	typ, data, err := o.conn.Read(ctx)
	if err != nil {
		return obsTransportError(ctx, err)
	}
	var env obsEnvelope
	if typ != websocket.MessageText || json.Unmarshal(data, &env) != nil || env.Op == nil || *env.Op != op || len(env.D) == 0 || string(env.D) == "null" {
		return errors.New("invalid OBS protocol message")
	}
	if json.Unmarshal(env.D, dest) != nil {
		return errors.New("invalid OBS protocol data")
	}
	return nil
}
func (o *OBSCapture) write(ctx context.Context, op int, data any) error {
	payload, err := json.Marshal(struct {
		Op int `json:"op"`
		D  any `json:"d"`
	}{op, data})
	if err != nil || len(payload) > obsMessageLimit {
		return errors.New("invalid OBS request")
	}
	if err = o.conn.Write(ctx, websocket.MessageText, payload); err != nil {
		return obsTransportError(ctx, err)
	}
	return nil
}

func DialOBS(ctx context.Context, cfg OBSConfig) (*OBSCapture, error) {
	address, err := obsURL(cfg)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, obsTimeout)
	defer cancel()
	// Disallow redirects so credentials derived for one server cannot reach another.
	hc := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	conn, _, err := websocket.Dial(ctx, address, &websocket.DialOptions{HTTPClient: hc, Subprotocols: []string{"obswebsocket.json"}})
	if err != nil {
		return nil, obsTransportError(ctx, err)
	}
	conn.SetReadLimit(obsMessageLimit)
	o := &OBSCapture{conn: conn, source: strings.TrimSpace(cfg.Source)}
	ready := false
	defer func() {
		if !ready {
			_ = o.Close()
		}
	}()
	var hello struct {
		RPCVersion     int             `json:"rpcVersion"`
		Authentication json.RawMessage `json:"authentication"`
	}
	if err = o.read(ctx, 0, &hello); err != nil {
		return nil, err
	}
	if hello.RPCVersion < 1 {
		return nil, errors.New("OBS requires RPC version 1")
	}
	identify := map[string]any{"rpcVersion": 1, "eventSubscriptions": 0}
	if len(hello.Authentication) > 0 {
		var auth struct {
			Salt      string `json:"salt"`
			Challenge string `json:"challenge"`
		}
		if json.Unmarshal(hello.Authentication, &auth) != nil || auth.Salt == "" || auth.Challenge == "" {
			return nil, errors.New("invalid OBS authentication challenge")
		}
		if cfg.Password == "" {
			return nil, errors.New("OBS authentication requires a password")
		}
		secret := sha256.Sum256([]byte(cfg.Password + auth.Salt))
		response := sha256.Sum256([]byte(base64.StdEncoding.EncodeToString(secret[:]) + auth.Challenge))
		identify["authentication"] = base64.StdEncoding.EncodeToString(response[:])
	}
	if err = o.write(ctx, 1, identify); err != nil {
		return nil, err
	}
	var identified struct {
		Version int `json:"negotiatedRpcVersion"`
	}
	if err = o.read(ctx, 2, &identified); err != nil {
		return nil, err
	}
	if identified.Version != 1 {
		return nil, errors.New("unsupported OBS negotiated RPC version")
	}
	ready = true
	return o, nil
}

// Close immediately releases the session and interrupts an in-flight capture.
func (o *OBSCapture) Close() error {
	if o != nil && o.closed.CompareAndSwap(false, true) {
		_ = o.conn.CloseNow()
	}
	return nil
}

func (o *OBSCapture) request(ctx context.Context, kind string, data any, dest any) error {
	o.sequence++
	id := strconv.FormatUint(o.sequence, 10)
	if err := o.write(ctx, 6, map[string]any{"requestType": kind, "requestId": id, "requestData": data}); err != nil {
		return err
	}
	var response struct {
		Type   string `json:"requestType"`
		ID     string `json:"requestId"`
		Status struct {
			Result bool `json:"result"`
			Code   int  `json:"code"`
		} `json:"requestStatus"`
		Data json.RawMessage `json:"responseData"`
	}
	if err := o.read(ctx, 7, &response); err != nil {
		return err
	}
	if response.Type != kind || response.ID != id {
		return errors.New("OBS response does not match request")
	}
	if !response.Status.Result || response.Status.Code != 100 {
		return fmt.Errorf("OBS request failed (status %d)", response.Status.Code)
	}
	if len(response.Data) == 0 || string(response.Data) == "null" || json.Unmarshal(response.Data, dest) != nil {
		return errors.New("invalid OBS response data")
	}
	return nil
}

func (o *OBSCapture) Capture(ctx context.Context) (frame *image.RGBA, err error) {
	if o == nil {
		return nil, errors.New("OBS capture is closed")
	}
	if !o.busy.TryLock() {
		return nil, errors.New("OBS capture already in progress")
	}
	defer o.busy.Unlock()
	if o.closed.Load() {
		return nil, errors.New("OBS capture is closed")
	}
	if ctx.Err() != nil {
		return nil, obsTransportError(ctx, ctx.Err())
	}
	ctx, cancel := context.WithTimeout(ctx, obsTimeout)
	defer cancel()
	// Any uncertain response invalidates the stream; never reuse a late response.
	defer func() {
		if err != nil {
			_ = o.Close()
		}
	}()
	source := o.source
	if source == "" {
		var scene struct {
			Name   string `json:"sceneName"`
			Legacy string `json:"currentProgramSceneName"`
		}
		if err = o.request(ctx, "GetCurrentProgramScene", map[string]any{}, &scene); err != nil {
			return nil, err
		}
		source = scene.Name
		if source == "" {
			source = scene.Legacy
		}
		if source == "" {
			return nil, errors.New("OBS program scene is empty")
		}
	}
	var screenshot struct {
		Data string `json:"imageData"`
	}
	if err = o.request(ctx, "GetSourceScreenshot", map[string]any{"sourceName": source, "imageFormat": "jpg", "imageWidth": 1280, "imageHeight": 720, "imageCompressionQuality": 90}, &screenshot); err != nil {
		return nil, err
	}
	const prefix = "data:image/jpeg;base64,"
	if !strings.HasPrefix(screenshot.Data, prefix) {
		return nil, errors.New("OBS screenshot is not a JPEG data URL")
	}
	encoded := strings.TrimPrefix(screenshot.Data, prefix)
	if len(encoded) > obsMessageLimit {
		return nil, errors.New("OBS screenshot exceeds size limit")
	}
	raw, decodeErr := base64.StdEncoding.Strict().DecodeString(encoded)
	if decodeErr != nil {
		return nil, errors.New("invalid OBS screenshot encoding")
	}
	config, decodeErr := jpeg.DecodeConfig(bytes.NewReader(raw))
	if decodeErr != nil {
		return nil, errors.New("invalid OBS JPEG screenshot")
	}
	if config.Width != 1280 || config.Height != 720 {
		return nil, errors.New("OBS screenshot must be exactly 1280x720; use a 16:9 source or program scene")
	}
	decoded, decodeErr := jpeg.Decode(bytes.NewReader(raw))
	if decodeErr != nil {
		return nil, errors.New("invalid OBS JPEG screenshot")
	}
	if ctx.Err() != nil {
		return nil, obsTransportError(ctx, ctx.Err())
	}
	frame = image.NewRGBA(image.Rect(0, 0, 1280, 720))
	draw.Draw(frame, frame.Bounds(), decoded, decoded.Bounds().Min, draw.Src)
	return frame, nil
}
