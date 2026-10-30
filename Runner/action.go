package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const FrameDuration = 200 * time.Millisecond

type Action struct {
	Absolute    bool
	X, Y, Wheel int
	Chunks      [6][]string
}

var keyCodes = func() map[string]uint16 {
	m := map[string]uint16{"Esc": 27, "Tab": 9, "Caps": 20, "Shift": 16, "Ctrl": 17, "Alt": 18, "Space": 32, "Back": 8, "Enter": 13, "Left": 37, "Up": 38, "Right": 39, "Down": 40, "LB": 0, "RB": 0, "MB": 0}
	for _, c := range "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
		m[string(c)] = uint16(c)
	}
	for i, n := range []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"} {
		m[n] = uint16(48 + i)
	}
	for i, n := range []string{"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"} {
		m[n] = uint16(112 + i)
	}
	return m
}()

func ParseAction(text string) (Action, error) {
	var a Action
	s := strings.TrimSpace(text)
	// Some compatible servers expose the terminating ChatML token.
	s = strings.TrimSpace(strings.TrimSuffix(s, "<|im_end|>"))
	if !strings.HasPrefix(s, "<|action_start|>") || !strings.HasSuffix(s, "<|action_end|>") {
		return a, fmt.Errorf("missing action delimiters")
	}
	s = strings.TrimSuffix(strings.TrimPrefix(s, "<|action_start|>"), "<|action_end|>")
	p := strings.Split(s, ";")
	if len(p) != 7 {
		return a, fmt.Errorf("expected movement and exactly six chunks, got %d", len(p)-1)
	}
	f := strings.Fields(p[0])
	if len(f) > 0 && f[0] == "<|use_abs|>" {
		a.Absolute = true
		f = f[1:]
	}
	if len(f) != 3 {
		return a, fmt.Errorf("movement needs three integers")
	}
	vals := []*int{&a.X, &a.Y, &a.Wheel}
	for i, v := range f {
		n, e := strconv.Atoi(v)
		if e != nil {
			return a, fmt.Errorf("invalid integer %q", v)
		}
		*vals[i] = n
	}
	if a.Wheel < -5 || a.Wheel > 5 {
		return a, fmt.Errorf("wheel outside training bounds")
	}
	if a.Absolute {
		if a.X < 0 || a.X > 1279 || a.Y < 0 || a.Y > 719 {
			return a, fmt.Errorf("absolute coordinate outside frame")
		}
	} else if a.X < -995 || a.X > 995 || a.Y < -996 || a.Y > 996 || a.X%5 != 0 || a.Y%4 != 0 {
		return a, fmt.Errorf("relative delta outside training quantization")
	}
	for i := 0; i < 6; i++ {
		ks := strings.Fields(p[i+1])
		if len(ks) > 4 {
			return a, fmt.Errorf("chunk %d exceeds four keys", i+1)
		}
		seen := map[string]bool{}
		for _, k := range ks {
			if _, ok := keyCodes[k]; !ok || seen[k] {
				return a, fmt.Errorf("unknown or repeated key %q", k)
			}
			seen[k] = true
		}
		// Never allow model output to invoke OS task switching or termination.
		if seen["Alt"] && (seen["Tab"] || seen["Four"] || seen["Esc"]) || seen["Ctrl"] && seen["Esc"] {
			return a, fmt.Errorf("OS shortcut rejected")
		}
		a.Chunks[i] = ks
	}
	return a, nil
}

// Integer division of the full duration avoids six rounded 33ms chunks drifting.
func ChunkDeadline(i int) time.Duration { return time.Duration(i) * FrameDuration / 6 }

type ActionSink interface {
	Move(Action) error
	Keys([]string) error
	Release()
}

func Play(ctx context.Context, a Action, sink ActionSink) error {
	defer sink.Release() // Never hold predicted input during network inference.
	if ctx.Err() != nil {
		return ctx.Err()
	}
	start := time.Now()
	for i, keys := range a.Chunks {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Delta is a total over the 200ms horizon, not six repeated deltas.
		part := a
		if a.Absolute {
			if i != 0 {
				part.Absolute = false
				part.X = 0
				part.Y = 0
			}
		} else {
			part.X = a.X*(i+1)/6 - a.X*i/6
			part.Y = a.Y*(i+1)/6 - a.Y*i/6
		}
		part.Wheel = a.Wheel*(i+1)/6 - a.Wheel*i/6
		if err := sink.Move(part); err != nil {
			return err
		}
		if err := sink.Keys(keys); err != nil {
			return err
		}
		t := time.NewTimer(time.Until(start.Add(ChunkDeadline(i + 1))))
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
	return nil
}
