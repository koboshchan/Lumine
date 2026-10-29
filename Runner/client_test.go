package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPadding(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 1280, 720))
	src.Set(1279, 719, color.RGBA{12, 34, 56, 255})
	dst, e := PaddedFrame(src)
	if e != nil {
		t.Fatal(e)
	}
	if dst.Bounds() != image.Rect(0, 0, 1280, 736) || dst.RGBAAt(1279, 719) != (color.RGBA{12, 34, 56, 255}) {
		t.Fatal("pixels moved")
	}
	for y := 720; y < 736; y++ {
		for x := 0; x < 1280; x++ {
			if dst.RGBAAt(x, y) != (color.RGBA{0, 0, 0, 255}) {
				t.Fatal("padding not opaque black")
			}
		}
	}
	if _, e = PaddedFrame(image.NewRGBA(image.Rect(0, 0, 1280, 736))); e == nil {
		t.Fatal("accepted double padding")
	}
}
func TestTrainingRequest(t *testing.T) {
	const action = "<|action_start|><use_absolute> 10 20 0 ; W ; ; ; ; ; <|action_end|>"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != "POST" || r.Header.Get("Authorization") != "Bearer test-secret" {
			t.Errorf("wrong request metadata")
		}
		var b struct {
			Model       string
			Temperature int
			MaxTokens   int  `json:"max_tokens"`
			Skip        bool `json:"skip_special_tokens"`
			Messages    []struct {
				Role    string
				Content []struct {
					Type  string
					Image struct{ URL string } `json:"image_url"`
				}
			}
		}
		var raw map[string]any
		dec := json.NewDecoder(r.Body)
		if e := dec.Decode(&raw); e != nil {
			t.Error(e)
			return
		}
		data, _ := json.Marshal(raw)
		json.Unmarshal(data, &b)
		if _, ok := raw["skip_special_tokens"]; !ok {
			t.Error("token preservation missing")
		}
		if b.Model != "lumine-mini-2b" || b.Skip || b.MaxTokens != 192 || len(b.Messages) != 1 || b.Messages[0].Role != "user" || len(b.Messages[0].Content) != 1 || b.Messages[0].Content[0].Type != "image_url" {
			t.Errorf("training format mismatch: %s", data)
		}
		uri := b.Messages[0].Content[0].Image.URL
		decoded, e := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, "data:image/png;base64,"))
		if e != nil {
			t.Error(e)
			return
		}
		im, e := png.Decode(strings.NewReader(string(decoded)))
		if e != nil || im.Bounds() != image.Rect(0, 0, 1280, 736) {
			t.Error("wrong PNG geometry", e)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]string{"content": action}}}})
	}))
	defer server.Close()
	result, e := (Client{}).Infer(context.Background(), Config{Endpoint: server.URL + "/v1/", Model: "lumine-mini-2b", APIKey: "test-secret"}, image.NewRGBA(image.Rect(0, 0, 1280, 720)))
	if e != nil || result != action {
		t.Fatal(result, e)
	}
}
func TestEndpointFailures(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 1280, 720))
	for _, ep := range []string{"file:///tmp/x", "not-a-url", "https://user:pass@host/v1", "https://host/v1?key=secret"} {
		if _, e := (Client{}).Infer(context.Background(), Config{Endpoint: ep, Model: "m"}, im); e == nil {
			t.Fatal(ep)
		}
	}
	for _, body := range []string{"bad JSON", `{"choices":[]}`, `{"choices":[{},{}]}`} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
		if _, e := (Client{}).Infer(context.Background(), Config{Endpoint: server.URL, Model: "m"}, im); e == nil {
			t.Error(body)
		}
		server.Close()
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := (Client{}).Infer(ctx, Config{Endpoint: "http://127.0.0.1:1/v1", Model: "m"}, im); e == nil {
		t.Fatal("ignored cancel")
	}
}
func TestNoCredentialRedirect(t *testing.T) {
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("followed redirect with endpoint credentials") }))
	defer destination.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, destination.URL, 302) }))
	defer redirect.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, e := (Client{}).Infer(ctx, Config{Endpoint: redirect.URL, Model: "m", APIKey: "secret"}, image.NewRGBA(image.Rect(0, 0, 1280, 720))); e == nil {
		t.Fatal("accepted redirect")
	}
}
