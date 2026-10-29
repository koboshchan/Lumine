package main

import (
	"context"
	"errors"
	"image"
	"sync/atomic"
	"testing"
	"time"
)

func TestCapturePacedSingleLatestSlot(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var active, maximum atomic.Int32
	var sequence atomic.Uint32
	starts := make(chan time.Time, 16)
	capture := func(ctx context.Context) (*image.RGBA, error) {
		a := active.Add(1)
		if a > maximum.Load() {
			maximum.Store(a)
		}
		defer active.Add(-1)
		starts <- time.Now()
		f := image.NewRGBA(image.Rect(0, 0, 1, 1))
		f.Pix[0] = byte(sequence.Add(1))
		return f, nil
	}
	samples, done := captureLatest(ctx, capture, nil)
	first := <-starts
	second := <-starts
	third := <-starts
	if second.Sub(first) < FrameDuration-time.Millisecond || third.Sub(second) < FrameDuration-time.Millisecond {
		t.Fatal("capture starts exceed 5Hz")
	}
	// Observe() is called before publication; reading third's start only tells
	// us it started, so wait for its completion via an observer in the next test.
	cancel()
	<-done
	if maximum.Load() != 1 {
		t.Fatal("overlapping capture")
	}
	if len(samples) != 1 {
		t.Fatal("capture backlog or lost latest slot", len(samples))
	}
	latest := <-samples
	if latest.frame.Pix[0] < 2 || latest.err != nil {
		t.Fatal("not latest completed frame")
	}
}

func TestCaptureLatestReplacementAndCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	observed := make(chan byte, 8)
	var seq byte
	samples, done := captureLatest(ctx, func(ctx context.Context) (*image.RGBA, error) {
		seq++
		f := image.NewRGBA(image.Rect(0, 0, 1, 1))
		f.Pix[0] = seq
		return f, nil
	}, func(f *image.RGBA) { observed <- f.Pix[0] })
	for i := 1; i <= 3; i++ {
		select {
		case <-observed:
		case <-time.After(2 * time.Second):
			t.Fatal("capture stalled")
		}
	}
	cancel()
	<-done
	if len(samples) != 1 {
		t.Fatal("must have only one latest sample")
	}
	if f := <-samples; f.frame.Pix[0] < 2 {
		t.Fatal("oldest frame retained")
	}
}

func TestCaptureSlowCallNeverOverlapsOrQueues(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{}, 8)
	var calls atomic.Int32
	samples, done := captureLatest(ctx, func(ctx context.Context) (*image.RGBA, error) {
		calls.Add(1)
		entered <- struct{}{}
		<-ctx.Done()
		return nil, ctx.Err()
	}, nil)
	<-entered
	// Deliberately span multiple nominal frame intervals; no capture goroutine
	// is spawned by a timer and cancellation must terminate the active call.
	timer := time.NewTimer(2*FrameDuration + time.Millisecond)
	<-timer.C
	if calls.Load() != 1 || len(entered) != 0 || len(samples) != 0 {
		t.Fatal("overlapping captures or queued work")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("capture cancellation stalled")
	}
	if calls.Load() != 1 {
		t.Fatal("queued capture ran after cancellation")
	}
}

func TestCaptureErrorStopsAndIsDelivered(t *testing.T) {
	var calls atomic.Int32
	want := errors.New("capture failed")
	samples, done := captureLatest(context.Background(), func(context.Context) (*image.RGBA, error) {
		calls.Add(1)
		return nil, want
	}, nil)
	<-done
	s, ok := <-samples
	if !ok || !errors.Is(s.err, want) || calls.Load() != 1 {
		t.Fatal("capture error swallowed or retried")
	}
	if _, ok = <-samples; ok {
		t.Fatal("closed stream has backlog")
	}
}

func TestCaptureAlreadyCancelledNeverStarts(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	samples, done := captureLatest(ctx, func(context.Context) (*image.RGBA, error) {
		t.Error("capture started after cancellation")
		return nil, nil
	}, nil)
	<-done
	if _, ok := <-samples; ok {
		t.Fatal("sample after cancellation")
	}
}
