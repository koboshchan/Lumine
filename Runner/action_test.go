package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestDatasetSamples(t *testing.T) {
	b, e := os.ReadFile("testdata/actions.json")
	if e != nil {
		t.Fatal(e)
	}
	var rows []string
	if e = json.Unmarshal(b, &rows); e != nil {
		t.Fatal(e)
	}
	var abs, rel bool
	for _, r := range rows {
		a, e := ParseAction(r)
		if e != nil {
			t.Fatalf("%s: %v", r, e)
		}
		if a.Absolute {
			abs = true
		} else {
			rel = true
		}
	}
	if !abs || !rel {
		t.Fatal("need both cursor modes")
	}
}
func TestChunksAndBounds(t *testing.T) {
	a, e := ParseAction("<|action_start|><use_absolute> 1279 719 -5 ; W LB ; W ; ; one ; Shift Space ; <|action_end|>")
	if e != nil {
		t.Fatal(e)
	}
	if !a.Absolute || a.X != 1279 || a.Y != 719 || a.Wheel != -5 || !reflect.DeepEqual(a.Chunks[0], []string{"W", "LB"}) || len(a.Chunks[5]) != 0 {
		t.Fatal(a)
	}
	a, e = ParseAction("<|action_start|>-995 996 5 ; ; ; ; ; ; <|action_end|><|im_end|>")
	if e != nil || a.X != -995 || a.Y != 996 {
		t.Fatal(a, e)
	}
}
func TestMalformedRejected(t *testing.T) {
	for _, s := range []string{
		"",
		"<|action_start|>0 0 0 ; ; ; ; ; <|action_end|>",
		"<|action_start|>0 0 0 ; ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 6 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>1 0 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>1000 0 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 -1000 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|><use_absolute>1280 0 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|><use_absolute>0 -1 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; W W ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; W A S D Space ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; Win ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; Alt Four ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; Ctrl Esc ; ; ; ; ;<|action_end|>",
		"<|action_start|>NaN 0 0 ; ; ; ; ; ;<|action_end|>",
		"text <|action_start|>0 0 0 ; ; ; ; ; ;<|action_end|>",
		"<|action_start|>0 0 0 ; ; ; ; ; ;<|action_end|> trailing",
	} {
		if _, e := ParseAction(s); e == nil {
			t.Errorf("accepted %q", s)
		}
	}
}

type recordingSink struct {
	keys           [][]string
	times          []time.Time
	release, moves int
	fail           bool
}

func (s *recordingSink) Move(Action) error { s.moves++; return nil }
func (s *recordingSink) Keys(k []string) error {
	s.keys = append(s.keys, k)
	s.times = append(s.times, time.Now())
	if s.fail {
		return errors.New("injection refused")
	}
	return nil
}
func (s *recordingSink) Release() { s.release++ }
func TestTimingAndRelease(t *testing.T) {
	if ChunkDeadline(0) != 0 || ChunkDeadline(3) != 100*time.Millisecond || ChunkDeadline(6) != 200*time.Millisecond {
		t.Fatal("chunk drift")
	}
	a, _ := ParseAction("<|action_start|>0 0 0 ; W ; W LB ; ; A ; A ; <|action_end|>")
	s := &recordingSink{}
	start := time.Now()
	if e := Play(context.Background(), a, s); e != nil {
		t.Fatal(e)
	}
	if s.moves != 6 || s.release != 1 || !reflect.DeepEqual(s.keys, a.Chunks[:]) {
		t.Fatal(s)
	}
	if time.Since(start) < FrameDuration {
		t.Fatal("action ended early")
	}
	for i, at := range s.times {
		if at.Sub(start) < ChunkDeadline(i) {
			t.Fatalf("chunk %d early", i)
		}
	}
}
func TestCancelAndErrorRelease(t *testing.T) {
	a := Action{}
	s := &recordingSink{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := Play(ctx, a, s); !errors.Is(e, context.Canceled) || s.moves != 0 || s.release != 1 {
		t.Fatal(e, s)
	}
	s = &recordingSink{fail: true}
	if e := Play(context.Background(), a, s); e == nil || s.release != 1 {
		t.Fatal(e, s)
	}
	s = &recordingSink{}
	ctx, cancel = context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if e := Play(ctx, a, s); !errors.Is(e, context.DeadlineExceeded) || s.release != 1 || len(s.keys) != 1 {
		t.Fatal(e, s)
	}
}
