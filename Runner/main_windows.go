//go:build windows && amd64

package main

import (
	"context"
	"fmt"
	"image"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

var mainWindow uintptr
var controls = map[int]uintptr{}
var choices []window
var state struct {
	sync.Mutex
	active, armed  bool
	cancel         context.CancelFunc
	sink           *winSink
	target         uintptr
	status, action string
	preview        *image.RGBA
}

type wc struct {
	Size, Style                        uint32
	Proc                               uintptr
	ClsExtra, WndExtra                 int32
	Instance, Icon, Cursor, Background uintptr
	Menu, Class                        *uint16
	SmallIcon                          uintptr
}
type msg struct {
	H       uintptr
	ID      uint32
	W, L    uintptr
	Time    uint32
	P       point
	Private uint32
}
type paintStruct struct {
	DC              uintptr
	Erase           int32
	Paint           rect
	Restore, Update int32
	Reserved        [32]byte
}

const (
	endpointID = 101
	modelID    = 102
	keyID      = 103
	windowID   = 104
	refreshID  = 105
	dryID      = 106
	startID    = 107
	stopID     = 108
	statusID   = 109
	actionID   = 110
	proxyID    = 111
)

func text(h uintptr) string {
	n := call(user, "GetWindowTextLengthW", h)
	b := make([]uint16, n+1)
	call(user, "GetWindowTextW", h, uintptr(unsafe.Pointer(&b[0])), n+1)
	return syscall.UTF16ToString(b)
}
func setText(h uintptr, s string) {
	p := utf(s)
	call(user, "SetWindowTextW", h, uintptr(unsafe.Pointer(p)))
}
func child(class, label string, id int, x, y, w, h int32, style uintptr) uintptr {
	c := call(user, "CreateWindowExW", 0, sp(class), sp(label), 0x50000000|style, uintptr(x), uintptr(y), uintptr(w), uintptr(h), mainWindow, uintptr(id), 0, 0)
	call(user, "SendMessageW", c, 0x30, call(gdi, "GetStockObject", 17), 1)
	if id != 0 {
		controls[id] = c
	}
	return c
}
func refresh() {
	choices = windows()
	c := controls[windowID]
	call(user, "SendMessageW", c, 0x14b, 0, 0)
	for _, w := range choices {
		call(user, "SendMessageW", c, 0x143, 0, sp(w.Title))
	}
	if len(choices) > 0 {
		call(user, "SendMessageW", c, 0x14e, 0, 0)
	}
}
func status(s string) { state.Lock(); state.status = s; state.Unlock() }
func stop(reason string) {
	state.Lock()
	cancel, sink := state.cancel, state.sink
	state.armed = false
	state.status = reason
	state.Unlock()
	if cancel != nil {
		cancel()
	}
	if sink != nil {
		sink.Stop()
	}
}
func start() {
	state.Lock()
	busy := state.active
	state.Unlock()
	if busy {
		return
	}
	idx := int32(call(user, "SendMessageW", controls[windowID], 0x147, 0, 0))
	if idx < 0 || int(idx) >= len(choices) {
		status("Choose a target window first")
		return
	}
	cfg := Config{Endpoint: strings.TrimSpace(text(controls[endpointID])), Model: strings.TrimSpace(text(controls[modelID])), APIKey: text(controls[keyID]), Proxy: call(user, "SendMessageW", controls[proxyID], 0xf0, 0, 0) == 1, Dry: call(user, "SendMessageW", controls[dryID], 0xf0, 0, 0) == 1, Window: choices[idx].Handle}
	ctx, cancel := context.WithCancel(context.Background())
	sink := &winSink{target: cfg.Window, held: map[string]bool{}}
	state.Lock()
	state.active = true
	state.armed = false
	state.cancel = cancel
	state.sink = sink
	state.target = cfg.Window
	state.action = ""
	state.Unlock()
	go run(ctx, cfg, sink)
}
func run(ctx context.Context, cfg Config, sink *winSink) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	defer func() {
		sink.Stop()
		state.Lock()
		state.active = false
		state.armed = false
		state.cancel = nil
		state.sink = nil
		state.Unlock()
	}()
	for n := 5; n > 0; n-- {
		status(fmt.Sprintf("Starting in %d. Focus the game now. F8 cancels.", n))
		t := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
	if ctx.Err() != nil {
		return
	}
	if err := ready(cfg.Window); err != nil {
		status(err.Error())
		return
	}
	state.Lock()
	state.armed = true
	state.Unlock()
	// Sample at the dataset's 5Hz, independent of endpoint latency. Keep only
	// the newest frame; slow inference must never build a capture/action backlog.
	type sample struct {
		frame *image.RGBA
		err   error
	}
	samples := make(chan sample, 1)
	go func() {
		ticker := time.NewTicker(FrameDuration)
		defer ticker.Stop()
		for {
			if ctx.Err() != nil {
				return
			}
			f, e := capture(cfg.Window)
			if e == nil {
				state.Lock()
				state.preview = f
				state.Unlock()
			}
			select {
			case <-samples:
			default:
			}
			select {
			case samples <- sample{f, e}:
			case <-ctx.Done():
				return
			}
			if e != nil {
				return
			}
			select {
			case <-ticker.C:
			case <-ctx.Done():
				return
			}
		}
	}()
	client := Client{}
	var ps *ProxySession
	if cfg.Proxy {
		var e error
		if ps, e = NewProxySession(nil, cfg.Endpoint, cfg.APIKey); e != nil {
			status("Stopped: " + e.Error())
			return
		}
		rc, rcancel := context.WithTimeout(ctx, 10*time.Second)
		e = ps.Reset(rc) // fresh session on every start
		rcancel()
		if e != nil {
			status("Stopped: proxy reset failed: " + e.Error())
			return
		}
		defer func() { // stop/failure: never continue uncertain history
			rc, rcancel := context.WithTimeout(context.Background(), 3*time.Second)
			ps.Reset(rc)
			rcancel()
		}()
	}
	for ctx.Err() == nil {
		var snap sample
		select {
		case snap = <-samples:
		case <-ctx.Done():
			return
		}
		if snap.err != nil {
			status(snap.err.Error())
			return
		}
		start := time.Now()
		frame := snap.frame
		var result, actionID string
		var err error
		if ps != nil {
			var sr StepResult
			sr, err = ps.Step(ctx, frame)
			result, actionID = sr.Action, sr.ActionID
		} else {
			result, err = client.Infer(ctx, cfg, frame)
		}
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			status("Stopped: " + err.Error())
			return
		}
		a, err := ParseAction(result)
		state.Lock()
		state.action = result
		state.Unlock()
		if err != nil {
			status("Stopped: " + err.Error())
			return
		}
		inference := time.Since(start)
		if err = ready(cfg.Window); err != nil {
			status(err.Error())
			return
		}
		if !cfg.Dry {
			if err = Play(ctx, a, sink); err != nil {
				if ctx.Err() == nil {
					status("Stopped: " + err.Error())
				}
				return
			}
			if ps != nil {
				ps.MarkExecuted(actionID) // only after real execution
			}
		} else {
			// In dry-run keep the same action horizon, but never construct input events.
			t := time.NewTimer(FrameDuration)
			select {
			case <-ctx.Done():
				t.Stop()
				return
			case <-t.C:
			}
		}
		// Single frame, synchronous inference. Never queue stale predictions or catch up.
		elapsed := time.Since(start)
		mode := "LIVE"
		if cfg.Dry {
			mode = "DRY RUN"
		}
		status(fmt.Sprintf("%s   inference %.0f ms   %.2f decisions/s   F8 stops", mode, float64(inference.Milliseconds()), 1/elapsed.Seconds()))
	}
}
func paint(h uintptr) {
	var p paintStruct
	dc := call(user, "BeginPaint", h, uintptr(unsafe.Pointer(&p)))
	defer call(user, "EndPaint", h, uintptr(unsafe.Pointer(&p)))
	state.Lock()
	im := state.preview
	state.Unlock()
	if im == nil {
		return
	}
	bgra := make([]byte, len(im.Pix))
	for i := 0; i < len(bgra); i += 4 {
		bgra[i] = im.Pix[i+2]
		bgra[i+1] = im.Pix[i+1]
		bgra[i+2] = im.Pix[i]
		bgra[i+3] = 255
	}
	bi := bmpInfo(1280, 720)
	call(gdi, "SetStretchBltMode", dc, 4)
	call(gdi, "StretchDIBits", dc, 20, 276, 800, 450, 0, 0, 1280, 720, uintptr(unsafe.Pointer(&bgra[0])), uintptr(unsafe.Pointer(&bi)), 0, 0x00CC0020)
}
func windowProc(h uintptr, m uint32, w, l uintptr) uintptr {
	switch m {
	case 0x111:
		switch int(w & 65535) {
		case refreshID:
			refresh()
		case startID:
			start()
		case stopID:
			stop("Stopped. All held inputs released.")
		}
		return 0
	case 0x312:
		stop("F8 panic stop. All held inputs released.")
		return 0
	case 0x113:
		state.Lock()
		s, a, active, armed, target := state.status, state.action, state.active, state.armed, state.target
		state.Unlock()
		if armed && foreground() != target {
			stop("Target lost focus. Stopped and released inputs.")
		}
		setText(controls[statusID], s)
		setText(controls[actionID], a)
		enabled := uintptr(1)
		if active {
			enabled = 0
		}
		for _, id := range []int{endpointID, modelID, keyID, windowID, refreshID, dryID, proxyID, startID} {
			call(user, "EnableWindow", controls[id], enabled)
		}
		var r = rect{20, 276, 820, 726}
		call(user, "InvalidateRect", h, uintptr(unsafe.Pointer(&r)), 0)
		return 0
	case 0xf:
		paint(h)
		return 0
	case 0x10:
		stop("Closed")
		call(user, "DestroyWindow", h)
		return 0
	case 2:
		call(user, "UnregisterHotKey", h, 1)
		call(user, "PostQuitMessage", 0)
		return 0
	}
	return call(user, "DefWindowProcW", h, uintptr(m), w, l)
}
func main() {
	runtime.LockOSThread()
	winmm := syscall.NewLazyDLL("winmm.dll")
	call(winmm, "timeBeginPeriod", 1)
	defer call(winmm, "timeEndPeriod", 1)
	// Physical pixels on mixed-DPI desktops.
	if p := user.NewProc("SetProcessDpiAwarenessContext"); p.Find() == nil {
		p.Call(^uintptr(3))
	} else {
		call(user, "SetProcessDPIAware")
	}
	instance := call(kernel, "GetModuleHandleW", 0)
	cls := utf("LumineRunner")
	c := wc{Size: uint32(unsafe.Sizeof(wc{})), Proc: syscall.NewCallback(windowProc), Instance: instance, Class: cls, Cursor: call(user, "LoadCursorW", 0, 32512), Background: 16}
	if call(user, "RegisterClassExW", uintptr(unsafe.Pointer(&c))) == 0 {
		return
	}
	mainWindow = call(user, "CreateWindowExW", 0, uintptr(unsafe.Pointer(cls)), sp("Lumine / model runner"), 0x00ca0000, 100, 100, 860, 820, 0, 0, instance, 0)
	if mainWindow == 0 {
		return
	}
	child("STATIC", "LUMINE   /   MODEL RUNNER", 0, 20, 14, 500, 25, 0)
	child("STATIC", "Endpoint", 0, 20, 49, 90, 22, 0)
	ep := os.Getenv("LUMINE_ENDPOINT")
	if ep == "" {
		ep = "http://127.0.0.1:8000/v1"
	}
	child("EDIT", ep, endpointID, 112, 46, 708, 25, 0x00810080)
	child("STATIC", "Model", 0, 20, 83, 90, 22, 0)
	child("EDIT", "lumine-mini-2b", modelID, 112, 80, 300, 25, 0x00810080)
	child("STATIC", "API key", 0, 430, 83, 70, 22, 0)
	child("EDIT", os.Getenv("LUMINE_API_KEY"), keyID, 500, 80, 320, 25, 0x008100a0)
	child("STATIC", "Target", 0, 20, 117, 90, 22, 0)
	child("COMBOBOX", "", windowID, 112, 114, 600, 350, 0x00210003)
	child("BUTTON", "Refresh", refreshID, 722, 114, 98, 26, 0x10000)
	child("BUTTON", "Dry run (no input)", dryID, 20, 155, 250, 28, 0x10003)
	call(user, "SendMessageW", controls[dryID], 0xf1, 1, 0)
	child("BUTTON", "Session proxy mode", proxyID, 280, 155, 200, 28, 0x10003)
	child("BUTTON", "Start / 5s countdown", startID, 490, 151, 200, 34, 0x10000)
	child("BUTTON", "Stop / F8", stopID, 702, 151, 118, 34, 0x10000)
	child("STATIC", "Ready. Dry run is enabled. Borderless/windowed game must stay visible and focused.", statusID, 20, 196, 800, 32, 0)
	child("EDIT", "", actionID, 20, 232, 800, 32, 0x00800884)
	child("STATIC", "Foreground client preview   /   1280 x 720   /   single-frame stage-1 policy", 0, 20, 735, 800, 22, 0)
	if call(user, "RegisterHotKey", mainWindow, 1, 0x4000, 0x77) == 0 {
		call(user, "MessageBoxW", mainWindow, sp("F8 is already registered. Close the other hotkey app and restart. Runner will not start without its panic hotkey."), sp("Lumine safety check"), 0x10)
		call(user, "DestroyWindow", mainWindow)
		return
	}
	refresh()
	call(user, "SetTimer", mainWindow, 1, 50, 0)
	call(user, "ShowWindow", mainWindow, 1)
	call(user, "UpdateWindow", mainWindow)
	var m msg
	for {
		r := call(user, "GetMessageW", uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		if r == 0 || int32(r) == -1 {
			break
		}
		if call(user, "IsDialogMessageW", mainWindow, uintptr(unsafe.Pointer(&m))) == 0 {
			call(user, "TranslateMessage", uintptr(unsafe.Pointer(&m)))
			call(user, "DispatchMessageW", uintptr(unsafe.Pointer(&m)))
		}
	}
}
