#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <windowsx.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cstdint>

#ifndef QWORD
typedef unsigned long long QWORD;
#endif

std::ofstream g_logFile;
std::mutex g_mutex;
std::atomic<bool> g_running(true);
std::atomic<bool> g_paused(false);
HHOOK g_keyboardHook = nullptr;
bool g_cursorVisible = true;
bool g_isClipped = false;
bool g_cursorStateKnown = false;
int g_lastMouseX = 0;
int g_lastMouseY = 0;

// Backward-compatible CSV, FILETIME timestamps. Each raw mouse event emits
// MOUSE_REL,dx,dy (raw device counts for relative devices) and MOUSE_ABS,x,y
// (physical desktop pixels from GetCursorPos), even when locked or stationary.
// Absolute-input devices use successive cursor pixel differences for REL.
// MOUSE,LOCK|UNLOCK and MOUSE,SHOW|HIDE are initial/resume snapshots plus
// transitions, sampled per raw event and at 30Hz even without mouse movement.
// SCREEN,left,top,width,height describes the recorded primary desktop in pixels.
void CheckCursorState(int64_t timestamp);
void ProcessMouseEvent(const RAWINPUT& raw, int64_t timestamp);

// Key state tracking
bool g_keysHeld[256] = { false };
bool g_mouseHeld[5] = { false }; // LB, RB, MB, XB1, XB2

inline int64_t GetHighResTimestamp() {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    return (static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

std::string KeyCodeToToken(DWORD vkCode) {
    switch (vkCode) {
    case '0': return "zero";
    case '1': return "one";
    case '2': return "two";
    case '3': return "three";
    case '4': return "four";
    case '5': return "five";
    case '6': return "six";
    case '7': return "seven";
    case '8': return "eight";
    case '9': return "nine";
    case 'A': return "A";
    case 'B': return "B";
    case 'C': return "C";
    case 'D': return "D";
    case 'E': return "E";
    case 'F': return "F";
    case 'G': return "G";
    case 'H': return "H";
    case 'I': return "I";
    case 'J': return "J";
    case 'K': return "K";
    case 'L': return "L";
    case 'M': return "M";
    case 'N': return "N";
    case 'O': return "O";
    case 'P': return "P";
    case 'Q': return "Q";
    case 'R': return "R";
    case 'S': return "S";
    case 'T': return "T";
    case 'U': return "U";
    case 'V': return "V";
    case 'W': return "W";
    case 'X': return "X";
    case 'Y': return "Y";
    case 'Z': return "Z";
    case VK_F1: return "One";
    case VK_F2: return "Two";
    case VK_F3: return "Three";
    case VK_F4: return "Four";
    case VK_F5: return "Five";
    case VK_F6: return "Six";
    case VK_F7: return "Seven";
    case VK_F8: return "Eight";
    case VK_F9: return "Nine";
    case VK_F10: return "Ten";
    case VK_F11: return "Eleven";
    case VK_F12: return "Twelve";
    case VK_ESCAPE: return "Esc";
    case VK_TAB: return "Tab";
    case VK_CAPITAL: return "Caps";
    case VK_LSHIFT:
    case VK_RSHIFT:
    case VK_SHIFT: return "Shift";
    case VK_LCONTROL:
    case VK_RCONTROL:
    case VK_CONTROL: return "Ctrl";
    case VK_LMENU:
    case VK_RMENU:
    case VK_MENU: return "Alt";
    case VK_SPACE: return "Space";
    default: return "";
    }
}

LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        KBDLLHOOKSTRUCT* kbStruct = (KBDLLHOOKSTRUCT*)lParam;
        bool isDown = (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN);
        bool isUp = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP);

        // Handle F12 (pause) and F11 (resume) before updating key state
        if (kbStruct->vkCode == VK_F12 && isDown) {
            if (!g_paused) {
                g_paused = true;
                int64_t ts = GetHighResTimestamp();
                std::lock_guard<std::mutex> lock(g_mutex);
                g_logFile << ts << ",PAUSE\n";
                g_logFile.flush();
                std::cout << "Recording PAUSED (F12)" << std::endl;
            }
            // Don't record F12 key in held keys
            return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
        }

        if (kbStruct->vkCode == VK_F11 && isDown) {
            if (g_paused) {
                g_cursorStateKnown = false;
                g_paused = false;
                int64_t ts = GetHighResTimestamp();
                std::lock_guard<std::mutex> lock(g_mutex);
                g_logFile << ts << ",RESUME\n";
                g_logFile.flush();
                std::cout << "Recording RESUMED (F11)" << std::endl;
            }
            // Don't record F11 key in held keys
            return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
        }

        // Don't update key states when paused
        if (!g_paused) {
            if (isDown) g_keysHeld[kbStruct->vkCode] = true;
            if (isUp) g_keysHeld[kbStruct->vkCode] = false;
        }

        if (kbStruct->vkCode == VK_F5 && isDown) {
            g_running = false;
        }
    }
    return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_INPUT) {
        UINT cbSize;
        if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &cbSize, sizeof(RAWINPUTHEADER)) == 0) {
            std::vector<BYTE> buffer(cbSize);
            if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, buffer.data(), &cbSize, sizeof(RAWINPUTHEADER)) == cbSize) {
                RAWINPUT* pRawInput = reinterpret_cast<RAWINPUT*>(buffer.data());
                if (pRawInput->header.dwType == RIM_TYPEMOUSE) {
                    const int64_t timestamp = GetHighResTimestamp();
                    CheckCursorState(timestamp);
                    ProcessMouseEvent(*pRawInput, timestamp);
                }
            }
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void ProcessMouseEvent(const RAWINPUT& raw, int64_t timestamp) {
    if (g_paused) return;

    POINT p = {};
    const bool havePosition = GetCursorPos(&p) != FALSE;
    const bool absolute = (raw.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE) != 0;
    const long dx = absolute ? (havePosition ? p.x - g_lastMouseX : 0) : raw.data.mouse.lLastX;
    const long dy = absolute ? (havePosition ? p.y - g_lastMouseY : 0) : raw.data.mouse.lLastY;
    if (havePosition) {
        g_lastMouseX = p.x;
        g_lastMouseY = p.y;
    }

    USHORT btnFlags = raw.data.mouse.usButtonFlags;
    if (btnFlags & RI_MOUSE_LEFT_BUTTON_DOWN)   g_mouseHeld[0] = true;
    if (btnFlags & RI_MOUSE_LEFT_BUTTON_UP)     g_mouseHeld[0] = false;
    if (btnFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)  g_mouseHeld[1] = true;
    if (btnFlags & RI_MOUSE_RIGHT_BUTTON_UP)    g_mouseHeld[1] = false;
    if (btnFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) g_mouseHeld[2] = true;
    if (btnFlags & RI_MOUSE_MIDDLE_BUTTON_UP)   g_mouseHeld[2] = false;
    if (btnFlags & RI_MOUSE_BUTTON_4_DOWN)      g_mouseHeld[3] = true;
    if (btnFlags & RI_MOUSE_BUTTON_4_UP)        g_mouseHeld[3] = false;
    if (btnFlags & RI_MOUSE_BUTTON_5_DOWN)      g_mouseHeld[4] = true;
    if (btnFlags & RI_MOUSE_BUTTON_5_UP)        g_mouseHeld[4] = false;

    std::lock_guard<std::mutex> lock(g_mutex);
    g_logFile << timestamp << ",MOUSE_REL," << dx << "," << dy << "\n";
    // If the desktop is inaccessible, retain REL/buttons/wheel but do not
    // fabricate an absolute coordinate.
    if (havePosition) {
        g_logFile << timestamp << ",MOUSE_ABS," << p.x << "," << p.y << "\n";
    }
    // usButtonData already IS the signed 16-bit wheel delta, not its HIWORD.
    if (btnFlags & RI_MOUSE_WHEEL) {
        const short wheelDelta = static_cast<short>(raw.data.mouse.usButtonData);
        g_logFile << timestamp << ",MOUSE,WHEEL," << wheelDelta << "\n";
    }
}

void CheckCursorState(int64_t timestamp) {
    if (g_paused) return;
    const bool initial = !g_cursorStateKnown;
    bool changed = false;
    bool haveVisibility = false;
    bool haveClip = false;
    CURSORINFO ci = {sizeof(CURSORINFO)};
    if (GetCursorInfo(&ci)) {
        // CURSOR_SUPPRESSED means hidden by touch/pen, not visible.
        const bool visible = (ci.flags & CURSOR_SHOWING) != 0 &&
                             (ci.flags & CURSOR_SUPPRESSED) == 0;
        haveVisibility = true;
        if (initial || visible != g_cursorVisible) {
            changed = true;
            g_cursorVisible = visible;
            std::lock_guard<std::mutex> lock(g_mutex);
            g_logFile << timestamp << ",MOUSE," << (visible ? "SHOW" : "HIDE") << "\n";
        }
    }

    RECT clipRect;
    if (GetClipCursor(&clipRect)) {
        // An unclipped cursor spans the VIRTUAL desktop, which may have a
        // negative origin or several monitors. Compare all four edges.
        const int left = GetSystemMetrics(SM_XVIRTUALSCREEN);
        const int top = GetSystemMetrics(SM_YVIRTUALSCREEN);
        const int right = left + GetSystemMetrics(SM_CXVIRTUALSCREEN);
        const int bottom = top + GetSystemMetrics(SM_CYVIRTUALSCREEN);
        const bool clipped = clipRect.left > left || clipRect.top > top ||
                             clipRect.right < right || clipRect.bottom < bottom;
        haveClip = true;
        if (initial || clipped != g_isClipped) {
            changed = true;
            g_isClipped = clipped;
            std::lock_guard<std::mutex> lock(g_mutex);
            g_logFile << timestamp << ",MOUSE," << (clipped ? "LOCK" : "UNLOCK") << "\n";
        }
    }
    // On query failure retain state and retry; failure does not mean unlocked.
    if (changed || initial) {
        POINT p;
        if (GetCursorPos(&p)) {
            g_lastMouseX = p.x;
            g_lastMouseY = p.y;
            std::lock_guard<std::mutex> lock(g_mutex);
            if (initial) {
                g_logFile << timestamp << ",SCREEN,0,0,"
                          << GetSystemMetrics(SM_CXSCREEN) << ","
                          << GetSystemMetrics(SM_CYSCREEN) << "\n";
            }
            // A game can warp/show the menu cursor without a raw input event.
            g_logFile << timestamp << ",MOUSE_ABS," << p.x << "," << p.y << "\n";
            if (haveVisibility && haveClip) g_cursorStateKnown = true;
        }
    }
}

BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType) {
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT) {
        std::cout << "\nShutting down..." << std::endl;
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================" << std::endl;
    std::cout << "  KeyRecorder - Keyboard/Mouse Logger" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    std::string outputFile = "input_log.txt";
    if (argc > 1) {
        outputFile = argv[1];
    }

    if (outputFile == "input_log.txt") {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        outputFile = "input_log_" + ss.str() + ".txt";
    }

    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Press F12 to pause recording" << std::endl;
    std::cout << "  Press F11 to resume recording" << std::endl;
    std::cout << "  Press F5 to stop recording" << std::endl;
    std::cout << std::endl;

    // Keep GetCursorPos and SCREEN in physical pixels on high-DPI displays.
    SetProcessDPIAware();
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    g_logFile.open(outputFile, std::ios::out);
    if (!g_logFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return 1;
    }

    g_logFile << "# KeyRecorder Input Log" << std::endl;
    g_logFile << "# Format: timestamp,EVENT_TYPE,data" << std::endl;
    g_logFile << "# timestamp: Windows FILETIME (100-nanosecond intervals since 1601-01-01)" << std::endl;
    g_logFile << "#" << std::endl;
    g_logFile << "# Events:" << std::endl;
    g_logFile << "#   KEY_CHUNK,token1 token2 ..." << std::endl;
    g_logFile << "#   SCREEN,left,top,width,height (primary desktop pixels)" << std::endl;
    g_logFile << "#   MOUSE_ABS,x,y" << std::endl;
    g_logFile << "#   MOUSE_REL,dx,dy" << std::endl;
    g_logFile << "#   MOUSE,WHEEL,delta" << std::endl;
    g_logFile << "#   MOUSE,SHOW|HIDE" << std::endl;
    g_logFile << "#   MOUSE,LOCK|UNLOCK" << std::endl;
    g_logFile << "#   PAUSE" << std::endl;
    g_logFile << "#   RESUME" << std::endl;
    g_logFile << "#" << std::endl;
    g_logFile.flush();

    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = L"KeyRecorderWindow";
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(0, L"KeyRecorderWindow", L"KeyRecorder", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);

    RAWINPUTDEVICE rid;
    rid.usUsagePage = 0x01;
    rid.usUsage = 0x02;
    rid.dwFlags = RIDEV_INPUTSINK;
    rid.hwndTarget = hwnd;
    
    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid))) {
        std::cerr << "Failed to register raw input: " << GetLastError() << std::endl;
    } else {
        std::cout << "Raw input registered successfully" << std::endl;
    }

    HINSTANCE hInstance = GetModuleHandle(NULL);

    g_keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, hInstance, 0);
    if (!g_keyboardHook) {
        std::cerr << "Failed to install keyboard hook: " << GetLastError() << std::endl;
    } else {
        std::cout << "Keyboard hook installed successfully" << std::endl;
    }

    POINT cursorPos;
    if (GetCursorPos(&cursorPos)) {
        g_lastMouseX = cursorPos.x;
        g_lastMouseY = cursorPos.y;
    }

    CheckCursorState(GetHighResTimestamp());

    std::cout << "Recording started... (Press F5 to stop)" << std::endl;

    using namespace std::chrono;
    auto nextTickTime = steady_clock::now();
    const nanoseconds tickDuration(1000000000LL / 30); // Exactly 1/30th of a second
    MSG msg;

    while (g_running) {
        auto nowSteady = steady_clock::now();
        if (nowSteady < nextTickTime) {
            auto sleepTime = duration_cast<milliseconds>(nextTickTime - nowSteady);
            MsgWaitForMultipleObjects(0, NULL, FALSE, (DWORD)sleepTime.count(), QS_ALLINPUT);
        }

        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) g_running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        nowSteady = steady_clock::now();
        if (nowSteady >= nextTickTime) {
            int64_t highResTs = GetHighResTimestamp();
            
            // Poll transitions even when no raw mouse events arrive.
            CheckCursorState(highResTs);
            
            // Skip KEY_CHUNK and cursor state logging when paused
            if (!g_paused) {
                // 2. Poll Key State (including mouse buttons)
                std::stringstream heldKeys;
                
                // Mouse buttons
                if (g_mouseHeld[0]) heldKeys << "LB ";
                if (g_mouseHeld[1]) heldKeys << "RB ";
                if (g_mouseHeld[2]) heldKeys << "MB ";
                if (g_mouseHeld[3]) heldKeys << "XB1 ";
                if (g_mouseHeld[4]) heldKeys << "XB2 ";

                // Keyboard
                for (int i = 0; i < 256; i++) {
                    if (g_keysHeld[i]) {
                        std::string token = KeyCodeToToken(i);
                        if (!token.empty()) {
                            heldKeys << token << " ";
                        }
                    }
                }
                
                std::string keys = heldKeys.str();
                // Trim trailing space
                if (!keys.empty() && keys.back() == ' ') keys.pop_back();

                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_logFile << highResTs << ",KEY_CHUNK," << keys << "\n";
                }
            }

            nextTickTime += tickDuration;
        }
    }

    std::cout << "Cleaning up..." << std::endl;

    if (g_keyboardHook) {
        UnhookWindowsHookEx(g_keyboardHook);
        g_keyboardHook = nullptr;
    }

    if (g_logFile.is_open()) {
        g_logFile.close();
    }

    DestroyWindow(hwnd);
    UnregisterClassW(L"KeyRecorderWindow", GetModuleHandle(NULL));
    
    std::cout << "Recording stopped. Log saved to: " << outputFile << std::endl;
    
    return 0;
}
