package main

import (
	"context"
	"image"
	"time"
)

type captureSample struct {
	frame   *image.RGBA
	err     error
	latency time.Duration
}

// captureLatest owns a single synchronous capture and a single replacement slot.
// Slow capture never creates overlapping calls, queued ticks, or catch-up bursts.
// observe must be quick; it only swaps the immutable preview pointer in the GUI.
func captureLatest(ctx context.Context, capture func(context.Context) (*image.RGBA, error), observe func(*image.RGBA)) (<-chan captureSample, <-chan struct{}) {
	samples := make(chan captureSample, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer close(samples)
		for ctx.Err() == nil {
			start := time.Now()
			frame, err := capture(ctx)
			elapsed := time.Since(start)
			if ctx.Err() != nil {
				return
			}
			if err == nil && observe != nil {
				observe(frame)
			}
			select {
			case <-samples:
			default:
			}
			select {
			case samples <- captureSample{frame: frame, err: err, latency: elapsed}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
			wait := time.Until(start.Add(FrameDuration))
			if wait <= 0 {
				continue
			} // previous operation already consumed its own slot
			timer := time.NewTimer(wait)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return
			}
		}
	}()
	return samples, done
}
