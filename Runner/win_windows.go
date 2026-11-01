//go:build windows && amd64

package main

import (
	"encoding/binary"
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

var user = syscall.NewLazyDLL("user32.dll")
var gdi = syscall.NewLazyDLL("gdi32.dll")
var kernel = syscall.NewLazyDLL("kernel32.dll")

//go:uintptrescapes
func call(d *syscall.LazyDLL, n string, args ...uintptr) uintptr {
	r, _, _ := d.NewProc(n).Call(args...)
	return r
}
func utf(s string) *uint16 { p, _ := syscall.UTF16PtrFromString(s); return p }

var stringsMu sync.Mutex
var nativeStrings = map[string]*uint16{}

func sp(s string) uintptr {
	stringsMu.Lock()
	defer stringsMu.Unlock()
	p := nativeStrings[s]
	if p == nil {
		p = utf(s)
		nativeStrings[s] = p
	}
	return uintptr(unsafe.Pointer(p))
}

type point struct{ X, Y int32 }
type rect struct{ L, T, R, B int32 }

func foreground() uintptr { return call(user, "GetForegroundWindow") }
func geometry(w uintptr) (rect, error) {
	var r rect
	var p point
	if call(user, "IsWindow", w) == 0 || call(user, "IsIconic", w) != 0 || call(user, "IsWindowVisible", w) == 0 {
		return r, fmt.Errorf("target closed, hidden or minimized")
	}
	if call(user, "GetClientRect", w, uintptr(unsafe.Pointer(&r))) == 0 || call(user, "ClientToScreen", w, uintptr(unsafe.Pointer(&p))) == 0 {
		return r, fmt.Errorf("cannot query client rectangle")
	}
	r.R += p.X
	r.B += p.Y
	r.L = p.X
	r.T = p.Y
	if r.R <= r.L || r.B <= r.T {
		return r, fmt.Errorf("empty client")
	}
	if int64(r.R-r.L)*9 != int64(r.B-r.T)*16 {
		return r, fmt.Errorf("target client must be 16:9; use 1280x720 windowed or borderless 16:9")
	}
	return r, nil
}
func ready(w uintptr) error {
	if foreground() != w {
		return fmt.Errorf("target lost foreground focus; stopped")
	}
	_, err := geometry(w)
	return err
}

type window struct {
	Handle uintptr
	Title  string
}

var enumWindowsResult []window
var enumWindowsCallback = syscall.NewCallback(func(w, unused uintptr) uintptr {
	if call(user, "IsWindowVisible", w) == 0 {
		return 1
	}
	var title [512]uint16
	n := call(user, "GetWindowTextW", w, uintptr(unsafe.Pointer(&title[0])), 512)
	if n > 0 && w != mainWindow {
		enumWindowsResult = append(enumWindowsResult, window{w, syscall.UTF16ToString(title[:])})
	}
	return 1
})

func windows() []window {
	enumWindowsResult = nil
	call(user, "EnumWindows", enumWindowsCallback, 0)
	return append([]window(nil), enumWindowsResult...)
}

type bitmapInfo struct {
	Size                   uint32
	Width, Height          int32
	Planes, BitCount       uint16
	Compression, SizeImage uint32
	XPels, YPels           int32
	Used, Important        uint32
}

func bmpInfo(w, h int32) bitmapInfo {
	return bitmapInfo{Size: 40, Width: w, Height: -h, Planes: 1, BitCount: 32}
}

// Windows x64 INPUT is 40 bytes with union at offset 8.
func input(kind uint32, payload []byte) error {
	var data [40]byte
	binary.LittleEndian.PutUint32(data[:4], kind)
	copy(data[8:], payload)
	if call(user, "SendInput", 1, uintptr(unsafe.Pointer(&data[0])), 40) != 1 {
		return fmt.Errorf("SendInput blocked (integrity level/UIPI?)")
	}
	return nil
}
func mouse(x, y int32, data, flags uint32) error {
	var p [32]byte
	binary.LittleEndian.PutUint32(p[0:], uint32(x))
	binary.LittleEndian.PutUint32(p[4:], uint32(y))
	binary.LittleEndian.PutUint32(p[8:], data)
	binary.LittleEndian.PutUint32(p[12:], flags)
	return input(0, p[:])
}
func sendKey(k string, down bool) error {
	if f, ok := map[string][2]uint32{"LB": {2, 4}, "RB": {8, 16}, "MB": {32, 64}}[k]; ok {
		n := 1
		if down {
			n = 0
		}
		return mouse(0, 0, 0, f[n])
	}
	vk, ok := keyCodes[k]
	if !ok {
		return fmt.Errorf("unknown key")
	}
	var p [24]byte
	// Hardware scan codes are required by many games.
	switch vk {
	case 16:
		vk = 0xa0
	case 17:
		vk = 0xa2
	case 18:
		vk = 0xa4
	}
	scan := uint16(call(user, "MapVirtualKeyW", uintptr(vk), 0))
	if scan == 0 {
		return fmt.Errorf("no scan code for key %s", k)
	}
	binary.LittleEndian.PutUint16(p[2:4], scan)
	flags := uint32(2 | 8)
	if down {
		flags = 8
	}
	if vk >= 37 && vk <= 40 {
		flags |= 1
	}
	binary.LittleEndian.PutUint32(p[4:], flags)
	return input(1, p[:])
}

type winSink struct {
	mu      sync.Mutex
	target  uintptr
	held    map[string]bool
	stopped bool
}

// Keep unsuccessful key-ups tracked so a repeated F8/Stop can retry them.
func (s *winSink) releaseLocked() {
	for k := range s.held {
		for tries := 0; tries < 3; tries++ {
			if sendKey(k, false) == nil {
				delete(s.held, k)
				break
			}
		}
	}
}
func (s *winSink) Release() { s.mu.Lock(); defer s.mu.Unlock(); s.releaseLocked() }
func (s *winSink) Stop()    { s.mu.Lock(); defer s.mu.Unlock(); s.stopped = true; s.releaseLocked() }
func (s *winSink) Keys(keys []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return fmt.Errorf("stopped")
	}
	if err := ready(s.target); err != nil {
		return err
	}
	next := map[string]bool{}
	for _, k := range keys {
		next[k] = true
	}
	for k := range s.held {
		if !next[k] {
			if err := sendKey(k, false); err != nil {
				return err
			}
			delete(s.held, k)
		}
	}
	for k := range next {
		if !s.held[k] {
			if err := sendKey(k, true); err != nil {
				return err
			}
			s.held[k] = true
		}
	}
	return nil
}
func (s *winSink) Move(a Action) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return fmt.Errorf("stopped")
	}
	if err := ready(s.target); err != nil {
		return err
	}
	x, y := int32(a.X), int32(a.Y)
	flags := uint32(1)
	if a.Absolute {
		r, e := geometry(s.target)
		if e != nil {
			return e
		}
		px := int64(r.L) + int64(a.X)*int64(r.R-r.L-1)/1279
		py := int64(r.T) + int64(a.Y)*int64(r.B-r.T-1)/719
		vx := int32(call(user, "GetSystemMetrics", 76))
		vy := int32(call(user, "GetSystemMetrics", 77))
		vw := int32(call(user, "GetSystemMetrics", 78))
		vh := int32(call(user, "GetSystemMetrics", 79))
		if vw <= 1 || vh <= 1 {
			return fmt.Errorf("invalid virtual desktop")
		}
		x = int32((px - int64(vx)) * 65535 / int64(vw-1))
		y = int32((py - int64(vy)) * 65535 / int64(vh-1))
		flags |= 0x8000 | 0x4000
	}
	if err := mouse(x, y, 0, flags); err != nil {
		return err
	}
	if a.Wheel != 0 {
		return mouse(0, 0, uint32(int32(a.Wheel*120)), 0x0800)
	}
	return nil
}
