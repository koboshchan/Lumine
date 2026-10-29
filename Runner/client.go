package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// PaddedFrame mirrors VeOmni's image_frame_width/image_frame_height path.
func PaddedFrame(src image.Image) (*image.RGBA, error) {
	if src.Bounds().Dx() != 1280 || src.Bounds().Dy() != 720 {
		return nil, fmt.Errorf("capture must be 1280x720")
	}
	dst := image.NewRGBA(image.Rect(0, 0, 1280, 736))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(image.Black), image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(0, 0, 1280, 720), src, src.Bounds().Min, draw.Src)
	return dst, nil
}

type Config struct {
	Endpoint, Model, APIKey string
	Dry, Proxy              bool
	Window                  uintptr
	CaptureOBS              bool
	OBS                     OBSConfig
}
type Client struct{ HTTP *http.Client }

func (c Client) Infer(ctx context.Context, cfg Config, frame image.Image) (string, error) {
	u, err := url.Parse(cfg.Endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("endpoint must be an http(s) /v1 base URL without credentials/query")
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return "", fmt.Errorf("model name required")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/chat/completions"
	padded, err := PaddedFrame(frame)
	if err != nil {
		return "", err
	}
	var pngbuf bytes.Buffer
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	if err = enc.Encode(&pngbuf, padded); err != nil {
		return "", err
	}
	body := map[string]any{
		"model": cfg.Model, "temperature": 0, "max_tokens": 192, "stream": false, "skip_special_tokens": false,
		// Stage-1 training has ONE image-only user message, no system/instruction/history.
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngbuf.Bytes())}}}}},
	}
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", u.String(), bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	h := c.HTTP
	if h == nil {
		h = &http.Client{Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	}
	resp, err := h.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("endpoint HTTP %d (check server logs)", resp.StatusCode)
	}
	var r struct {
		Choices []struct{ Message struct{ Content string } }
	}
	if err = json.Unmarshal(b, &r); err != nil {
		return "", fmt.Errorf("invalid response JSON: %w", err)
	}
	if len(r.Choices) != 1 {
		return "", fmt.Errorf("expected one completion")
	}
	return r.Choices[0].Message.Content, nil
}
