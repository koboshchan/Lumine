// Local gameplay training capture. No network, injection, hooks or background capture.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <obs-module.h>
#include <obs-frontend-api.h>
#include <util/platform.h>
#include <util/bmem.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

OBS_DECLARE_MODULE()
MODULE_EXPORT const char *obs_module_description(void)
{
	return "Local keyboard/mouse sidecars for Lumine gameplay recordings";
}
MODULE_EXPORT const char *obs_module_name(void) { return "Lumine Input Logger"; }

namespace {
constexpr UINT wake_message = WM_APP + 37;
constexpr UINT_PTR poll_timer = 1;
constexpr uint64_t chunk_ns = 1000000000ULL / 30;
constexpr size_t max_packets = 16384;
constexpr wchar_t window_class[] = L"LumineOBSInputLogger";

std::string key_token(unsigned vk)
{
	if (vk >= 'A' && vk <= 'Z') return std::string(1, static_cast<char>(vk));
	static const char *digits[] = {"zero", "one", "two", "three", "four",
				      "five", "six", "seven", "eight", "nine"};
	static const char *functions[] = {"One", "Two", "Three", "Four", "Five", "Six",
					 "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"};
	if (vk >= '0' && vk <= '9') return digits[vk - '0'];
	if (vk >= VK_F1 && vk <= VK_F12) return functions[vk - VK_F1];
	switch (vk) {
	case VK_ESCAPE: return "Esc";
	case VK_TAB: return "Tab";
	case VK_CAPITAL: return "Caps";
	case VK_SHIFT: case VK_LSHIFT: case VK_RSHIFT: return "Shift";
	case VK_CONTROL: case VK_LCONTROL: case VK_RCONTROL: return "Ctrl";
	case VK_MENU: case VK_LMENU: case VK_RMENU: return "Alt";
	case VK_SPACE: return "Space";
	default: return "";
	}
}

uint64_t wall_filetime()
{
	FILETIME ft;
	GetSystemTimePreciseAsFileTime(&ft);
	return (uint64_t(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

struct Frame {
	int64_t pts;
	int32_t num, den;
	uint64_t cts, sys_dts_us;
};
struct Command {
	bool pause, stop;
	uint64_t ns;
};

class Capture {
public:
	~Capture() { stop(); }
	bool start(obs_output_t *recording, const std::filesystem::path &video)
	{
		running = true;
		clean_stop = false;
		paused = false;
		cursor_known = false;
		io_failed = false;
		timing_lost = false;
		held.fill(false);
		mouse.fill(false);
		previous_devices.clear();
		frames.clear();
		commands.clear();
		output = recording; // Owned frontend reference, released in stop().
		path = video;
		path.replace_extension(L".log");
		auto existing_marker = path;
		existing_marker += L".complete";
		if (GetFileAttributesW(existing_marker.c_str()) != INVALID_FILE_ATTRIBUTES) {
			blog(LOG_ERROR, "[Lumine] Existing completion artifact; capture refused");
			return false;
		}
		file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
				   CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (file == INVALID_HANDLE_VALUE) {
			blog(LOG_ERROR, "[Lumine] Cannot create sidecar (error %lu); never overwrites logs",
			     GetLastError());
			return false;
		}
		mono_origin = os_gettime_ns();
		wall_origin = wall_filetime();
		frame_origin = obs_get_video_frame_time();
		obs_get_video_info(&video_info);
		std::promise<bool> ready;
		auto future = ready.get_future();
		worker = std::thread([this, ready = std::move(ready)]() mutable {
			run(std::move(ready));
		});
		if (!future.get()) {
			worker.join();
			return false;
		}
		// No file I/O in the encoder thread. Retain only timing, never video data.
		obs_output_add_packet_callback(output, packet_callback, this);
		callback_attached = true;
		return true;
	}

	void pause(bool value)
	{
		enqueue({value, false, os_gettime_ns()});
	}
	void stop()
	{
		// Never hold the queue mutex while disconnecting or joining.
		if (callback_attached) {
			obs_output_remove_packet_callback(output, packet_callback, this);
			callback_attached = false;
		}
		if (worker.joinable()) {
			enqueue({false, true, os_gettime_ns()});
			worker.join();
		}
		if (file != INVALID_HANDLE_VALUE) {
			CloseHandle(file);
			file = INVALID_HANDLE_VALUE;
		}
		if (output) {
			obs_output_release(output);
			output = nullptr;
		}
	}
	bool failed() const { return io_failed.load() || timing_lost.load() || !clean_stop; }

private:
	obs_output_t *output = nullptr;
	std::filesystem::path path;
	HANDLE file = INVALID_HANDLE_VALUE;
	std::thread worker;
	std::atomic<HWND> window{nullptr};
	std::atomic<bool> io_failed{false}, timing_lost{false};
	bool callback_attached = false, running = true, paused = false;
	bool clean_stop = false;
	bool cursor_known = false, visible = false, clipped = false;
	DWORD resume_tick = 0;
	uint64_t mono_origin = 0, wall_origin = 0, frame_origin = 0, next_chunk = 0;
	obs_video_info video_info{};
	std::array<bool, 256> held{};
	std::array<bool, 5> mouse{};
	std::vector<RAWINPUTDEVICE> previous_devices;
	std::mutex queue_mutex;
	std::deque<Frame> frames;
	std::deque<Command> commands;

	uint64_t timestamp(uint64_t ns) const
	{
		return ns >= mono_origin ? wall_origin + (ns - mono_origin) / 100
					: wall_origin - (mono_origin - ns) / 100;
	}
	void write(const std::string &text)
	{
		if (io_failed.load()) return;
		const char *ptr = text.data();
		size_t remaining = text.size();
		while (remaining) {
			DWORD written = 0;
			const DWORD count = static_cast<DWORD>(std::min<size_t>(remaining, MAXDWORD));
			if (!WriteFile(file, ptr, count, &written, nullptr) || !written) {
				io_failed = true;
				blog(LOG_ERROR, "[Lumine] Sidecar write failed; input capture stopping");
				return;
			}
			remaining -= written;
			ptr += written;
		}
	}
	void event(uint64_t ns, const std::string &data)
	{
		write(std::to_string(timestamp(ns)) + "," + data + "\n");
	}
	void enqueue(Command command)
	{
		{
			std::lock_guard<std::mutex> guard(queue_mutex);
			commands.push_back(command);
		}
		if (auto hwnd = window.load()) PostMessageW(hwnd, wake_message, 0, 0);
	}
	static void packet_callback(obs_output_t *, encoder_packet *packet,
				    encoder_packet_time *timing, void *param)
	{
		auto &self = *static_cast<Capture *>(param);
		if (packet->type != OBS_ENCODER_VIDEO || packet->track_idx != 0) return;
		std::lock_guard<std::mutex> guard(self.queue_mutex);
		if (self.frames.size() >= max_packets) {
			self.timing_lost = true;
			return;
		}
		self.frames.push_back({packet->pts, packet->timebase_num, packet->timebase_den,
				       timing ? timing->cts : 0, uint64_t(packet->sys_dts_usec)});
	}
	void drain()
	{
		std::deque<Frame> pending_frames;
		std::deque<Command> pending_commands;
		{
			std::lock_guard<std::mutex> guard(queue_mutex);
			pending_frames.swap(frames);
			pending_commands.swap(commands);
		}
		for (const auto &frame : pending_frames) {
			write("# FRAME," + std::to_string(frame.pts) + "," + std::to_string(frame.num) +
			      "," + std::to_string(frame.den) + "," + std::to_string(frame.cts) +
			      "," + std::to_string(frame.sys_dts_us) + "\n");
		}
		for (const auto &command : pending_commands) {
			const auto observed = os_gettime_ns();
			write("# FRONTEND_EVENT," + std::to_string(command.ns) + "," +
			      (command.stop ? "STOP" : command.pause ? "PAUSE" : "RESUME") + "\n");
			if (command.stop) {
				event(observed, "STOP");
				clean_stop = true;
				running = false;
			} else if (command.pause != paused) {
				paused = command.pause;
				event(observed, paused ? "PAUSE" : "RESUME");
				if (!paused) {
					resume_tick = GetTickCount();
					seed_keys();
					cursor_known = false;
					sample(observed);
					next_chunk = observed + chunk_ns;
				}
			}
		}
	}
	void seed_keys()
	{
		for (unsigned vk = 0; vk < held.size(); ++vk)
			held[vk] = !key_token(vk).empty() && ((GetAsyncKeyState(int(vk)) & 0x8000) != 0);
		// Modifier state is tracked separately per side, never generic + sided together.
		held[VK_SHIFT] = held[VK_CONTROL] = held[VK_MENU] = false;
		const int buttons[] = {VK_LBUTTON, VK_RBUTTON, VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2};
		for (size_t i = 0; i < mouse.size(); ++i)
			mouse[i] = (GetAsyncKeyState(buttons[i]) & 0x8000) != 0;
	}
	void cursor(uint64_t ns)
	{
		CURSORINFO info{};
		info.cbSize = sizeof(info);
		RECT rect{};
		if (!GetCursorInfo(&info) || !GetClipCursor(&rect)) return;
		const bool now_visible = (info.flags & CURSOR_SHOWING) != 0;
		const int left = GetSystemMetrics(SM_XVIRTUALSCREEN);
		const int top = GetSystemMetrics(SM_YVIRTUALSCREEN);
		const bool now_clipped = rect.left > left || rect.top > top ||
			rect.right < left + GetSystemMetrics(SM_CXVIRTUALSCREEN) ||
			rect.bottom < top + GetSystemMetrics(SM_CYVIRTUALSCREEN);
		if (!cursor_known || visible != now_visible)
			event(ns, now_visible ? "MOUSE,SHOW" : "MOUSE,HIDE");
		if (!cursor_known || clipped != now_clipped)
			event(ns, now_clipped ? "MOUSE,LOCK" : "MOUSE,UNLOCK");
		visible = now_visible;
		clipped = now_clipped;
		cursor_known = true;
	}
	void position(uint64_t ns, LONG dx, LONG dy)
	{
		POINT point{};
		event(ns, "MOUSE_REL," + std::to_string(dx) + "," + std::to_string(dy));
		if (GetCursorPos(&point))
			event(ns, "MOUSE_ABS," + std::to_string(point.x) + "," + std::to_string(point.y));
		else
			event(ns, "MOUSE,POSITION_UNAVAILABLE");
	}
	void sample(uint64_t ns)
	{
		if (paused) return;
		cursor(ns);
		position(ns, 0, 0);
		std::string keys;
		const char *buttons[] = {"LB", "RB", "MB", "XB1", "XB2"};
		std::set<std::string> seen;
		auto add = [&](const std::string &token) {
			if (!token.empty() && seen.insert(token).second) {
				if (!keys.empty()) keys += ' ';
				keys += token;
			}
		};
		for (size_t i = 0; i < mouse.size(); ++i) if (mouse[i]) add(buttons[i]);
		for (unsigned vk = 0; vk < held.size(); ++vk) if (held[vk]) add(key_token(vk));
		event(ns, "KEY_CHUNK," + keys);
	}
	void raw(HRAWINPUT handle)
	{
		if (paused || io_failed.load() ||
		    static_cast<LONG>(static_cast<DWORD>(GetMessageTime()) - resume_tick) <= 0) return;
		UINT bytes = 0;
		if (GetRawInputData(handle, RID_INPUT, nullptr, &bytes, sizeof(RAWINPUTHEADER)) != 0 ||
		    bytes < sizeof(RAWINPUTHEADER) || bytes > 65536) return;
		// RAWINPUT needs pointer alignment; byte buffers are not a portable cast target.
		std::vector<uint64_t> storage((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
		if (GetRawInputData(handle, RID_INPUT, storage.data(), &bytes, sizeof(RAWINPUTHEADER)) != bytes)
			return;
		auto *input = reinterpret_cast<const RAWINPUT *>(storage.data());
		const auto ns = os_gettime_ns();
		cursor(ns);
		if (input->header.dwType == RIM_TYPEKEYBOARD) {
			unsigned vk = input->data.keyboard.VKey;
			if (vk == VK_SHIFT)
				vk = MapVirtualKeyW(input->data.keyboard.MakeCode, MAPVK_VSC_TO_VK_EX);
			else if (vk == VK_CONTROL)
				vk = (input->data.keyboard.Flags & RI_KEY_E0) ? VK_RCONTROL : VK_LCONTROL;
			else if (vk == VK_MENU)
				vk = (input->data.keyboard.Flags & RI_KEY_E0) ? VK_RMENU : VK_LMENU;
			if (vk < held.size()) {
				const bool down = !(input->data.keyboard.Flags & RI_KEY_BREAK);
				if (held[vk] != down) {
					held[vk] = down;
					sample(ns); // Preserve short presses between regular 30 Hz samples.
				}
			}
		} else if (input->header.dwType == RIM_TYPEMOUSE) {
			const auto &m = input->data.mouse;
			const USHORT down[] = {RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_DOWN,
				RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_5_DOWN};
			const USHORT up[] = {RI_MOUSE_LEFT_BUTTON_UP, RI_MOUSE_RIGHT_BUTTON_UP,
				RI_MOUSE_MIDDLE_BUTTON_UP, RI_MOUSE_BUTTON_4_UP, RI_MOUSE_BUTTON_5_UP};
			for (size_t i = 0; i < mouse.size(); ++i) {
				if (m.usButtonFlags & down[i]) mouse[i] = true;
				if (m.usButtonFlags & up[i]) mouse[i] = false;
			}
			// Absolute-only HID devices have no honest raw relative delta. Emit zero.
			position(ns, (m.usFlags & MOUSE_MOVE_ABSOLUTE) ? 0 : m.lLastX,
				 (m.usFlags & MOUSE_MOVE_ABSOLUTE) ? 0 : m.lLastY);
			if (m.usButtonFlags & RI_MOUSE_WHEEL)
				event(ns, "MOUSE,WHEEL," + std::to_string(static_cast<SHORT>(m.usButtonData)));
			if (m.usButtonFlags & ~(RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL)) sample(ns);
		}
	}
	static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		auto *self = reinterpret_cast<Capture *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
		if (msg == WM_NCCREATE) {
			self = static_cast<Capture *>(reinterpret_cast<CREATESTRUCTW *>(lp)->lpCreateParams);
			SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
		}
		if (self) {
			if (msg == WM_INPUT) {
				self->drain();
				if (self->running) self->raw(reinterpret_cast<HRAWINPUT>(lp));
			} else if (msg == wake_message || (msg == WM_TIMER && wp == poll_timer)) {
				self->drain();
				const auto ns = os_gettime_ns();
				if (self->running && !self->paused && ns >= self->next_chunk) {
					self->sample(ns);
					self->next_chunk = ns + chunk_ns; // No invented catch-up inputs.
				}
			}
		}
		// WM_INPUT requires DefWindowProc cleanup even for foreground input.
		return DefWindowProcW(hwnd, msg, wp, lp);
	}
	bool register_devices(HWND hwnd)
	{
		UINT count = 0;
		if (GetRegisteredRawInputDevices(nullptr, &count, sizeof(RAWINPUTDEVICE)) == UINT(-1))
			return false;
		std::vector<RAWINPUTDEVICE> registered(count);
		if (count && GetRegisteredRawInputDevices(registered.data(), &count,
							  sizeof(RAWINPUTDEVICE)) == UINT(-1)) return false;
		for (const auto &device : registered) {
			if (device.usUsagePage == 1 && (device.usUsage == 2 || device.usUsage == 6)) {
				blog(LOG_ERROR, "[Lumine] Existing process Raw Input registration conflicts; capture refused");
				return false; // Do not steal events from OBS or another plugin.
			}
		}
		RAWINPUTDEVICE devices[] = {{1, 2, RIDEV_INPUTSINK, hwnd}, {1, 6, RIDEV_INPUTSINK, hwnd}};
		return RegisterRawInputDevices(devices, 2, sizeof(RAWINPUTDEVICE)) != FALSE;
	}
	void restore_devices()
	{
		// Raw Input registration is process-wide. Do not remove a newer owner's target.
		UINT count = 0;
		if (GetRegisteredRawInputDevices(nullptr, &count, sizeof(RAWINPUTDEVICE)) == UINT(-1)) return;
		std::vector<RAWINPUTDEVICE> current(count);
		if (count && GetRegisteredRawInputDevices(current.data(), &count,
							sizeof(RAWINPUTDEVICE)) == UINT(-1)) return;
		for (const auto &device : current) {
			if (device.hwndTarget != window.load() || device.usUsagePage != 1 ||
			    (device.usUsage != 2 && device.usUsage != 6)) continue;
			RAWINPUTDEVICE replacement{1, device.usUsage, RIDEV_REMOVE, nullptr};
			for (const auto &old : previous_devices)
				if (old.usUsage == device.usUsage) replacement = old;
			RegisterRawInputDevices(&replacement, 1, sizeof(replacement));
		}
	}
	void run(std::promise<bool> ready)
	{
		// Thread-only DPI setting: never change OBS's process DPI mode.
		auto previous_dpi = SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
		WNDCLASSW wc{};
		wc.lpfnWndProc = wndproc;
		wc.hInstance = GetModuleHandleW(nullptr);
		wc.lpszClassName = window_class;
		const ATOM registered = RegisterClassW(&wc);
		HWND hwnd = registered ? CreateWindowExW(0, window_class, L"Lumine input capture", 0,
			0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, this) : nullptr;
		window = hwnd;
		const bool devices_ok = hwnd && register_devices(hwnd);
		const bool ok = devices_ok && SetTimer(hwnd, poll_timer, 5, nullptr);
		if (!ok) {
			blog(LOG_ERROR, "[Lumine] Raw Input/window/timer initialization failed (%lu)", GetLastError());
			write("# ERROR,initialization_failed\n");
			ready.set_value(false);
		} else {
			std::ostringstream header;
			header << "# Lumine OBS Input Log v1\n"
			       << "# timestamp: FILETIME 100ns derived from monotonic OBS clock\n"
			       << "# start_filetime," << wall_origin << "\n# start_monotonic_ns," << mono_origin
			       << "\n# obs_video_frame_ns," << frame_origin
			       << "\n# fps," << video_info.fps_num << ',' << video_info.fps_den
			       << "\n# resolution," << video_info.output_width << ',' << video_info.output_height
			       << "\n# base_resolution," << video_info.base_width << ',' << video_info.base_height
			       << "\n# FRAME_SCHEMA,pts,timebase_num,timebase_den,composition_ns,system_dts_us\n"
			       << "# Input timestamps are receipt time, not hardware time. FRAME maps to encoded PTS.\n";
			write(header.str());
			event(mono_origin, "SCREEN,0,0," + std::to_string(GetSystemMetrics(SM_CXSCREEN)) +
			      "," + std::to_string(GetSystemMetrics(SM_CYSCREEN)));
			resume_tick = GetTickCount();
			seed_keys();
			sample(mono_origin);
			next_chunk = mono_origin + chunk_ns;
			ready.set_value(true);
			MSG msg{};
			while (running && !io_failed.load() && !timing_lost.load()) {
				const auto result = GetMessageW(&msg, nullptr, 0, 0);
				if (result <= 0) break;
				TranslateMessage(&msg);
				DispatchMessageW(&msg);
				// WM_TIMER is low priority: high-rate mice must not starve chunks/commands.
				drain();
				const auto now = os_gettime_ns();
				if (running && !paused && now >= next_chunk) {
					sample(now);
					next_chunk = now + chunk_ns;
				}
			}
			drain();
			if (timing_lost.load()) write("# ERROR,frame_timing_queue_overflow\n");
			if (clean_stop && !io_failed.load() && !timing_lost.load())
				write("# COMPLETE\n");
			// Publish a success artifact only AFTER the data flush succeeds.
			if (!FlushFileBuffers(file)) io_failed = true;
			if (clean_stop && !io_failed.load() && !timing_lost.load()) {
				LARGE_INTEGER size{};
				auto marker_path = path;
				marker_path += L".complete";
				HANDLE marker = CreateFileW(marker_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
					nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
				bool success = marker != INVALID_HANDLE_VALUE && GetFileSizeEx(file, &size);
				if (success) {
					const auto text = "LUMINE_COMPLETE_V1 " + std::to_string(size.QuadPart) + "\n";
					DWORD written = 0;
					success = WriteFile(marker, text.data(), static_cast<DWORD>(text.size()), &written,
						nullptr) && written == text.size() && FlushFileBuffers(marker);
				}
				if (marker != INVALID_HANDLE_VALUE) CloseHandle(marker);
				if (!success) {
					io_failed = true;
					if (marker != INVALID_HANDLE_VALUE) DeleteFileW(marker_path.c_str());
				}
			}
		}
		if (hwnd) {
			KillTimer(hwnd, poll_timer);
			restore_devices();
			DestroyWindow(hwnd);
		}
		window = nullptr;
		if (registered) UnregisterClassW(window_class, wc.hInstance);
		if (previous_dpi) SetThreadDpiAwarenessContext(previous_dpi);
	}
};

Capture capture;
bool enabled = false;
bool active = false;

void show_error(const wchar_t *text)
{
	MessageBoxW(static_cast<HWND>(obs_frontend_get_main_window_handle()), text,
		    L"Lumine input logger", MB_OK | MB_ICONWARNING);
}
void tools_toggle(void *)
{
	if (active || obs_frontend_recording_active()) {
		show_error(L"Change input logging between recordings. Stop recording first.");
		return;
	}
	const auto answer = MessageBoxW(static_cast<HWND>(obs_frontend_get_main_window_handle()),
		enabled
			? L"Input logging is ON for the next recording.\n\nKeep it enabled?\nYes = on, No = off.\n\nIt records your keyboard and mouse system-wide only while OBS records."
			: L"Enable input logging for the next recording?\nYes = on, No = off.\n\nIt records your keyboard and mouse system-wide only while OBS records. Close private chats and password fields first. Logs stay beside the video.",
		L"Lumine input logger", MB_YESNOCANCEL | MB_ICONINFORMATION);
	if (answer != IDCANCEL) enabled = answer == IDYES;
	blog(LOG_INFO, "[Lumine] Input logging %s (session only)", enabled ? "enabled" : "disabled");
}
void frontend_event(obs_frontend_event event, void *)
{
	switch (event) {
	case OBS_FRONTEND_EVENT_RECORDING_STARTED: {
		if (!enabled || active) break;
		obs_output_t *output = obs_frontend_get_recording_output();
		char *path = obs_frontend_get_current_record_output_path();
		if (!output || !path || !*path) {
			if (output) obs_output_release(output);
			bfree(path);
			show_error(L"No local recording path. Input capture was not started.");
			break;
		}
		const auto video_path = std::filesystem::u8path(path);
		bfree(path);
		try {
			active = capture.start(output, video_path);
		} catch (const std::exception &error) {
			blog(LOG_ERROR, "[Lumine] Capture initialization exception: %s", error.what());
			active = false;
		}
		if (!active) {
			capture.stop();
			show_error(L"Could not start input logging. Check the OBS log. Existing .log files are never overwritten.");
		}
		break;
	}
	case OBS_FRONTEND_EVENT_RECORDING_PAUSED:
		if (active) capture.pause(true);
		break;
	case OBS_FRONTEND_EVENT_RECORDING_UNPAUSED:
		if (active) capture.pause(false);
		break;
	case OBS_FRONTEND_EVENT_RECORDING_STOPPING:
		if (active) capture.pause(true); // Do not capture inputs during muxer drain.
		break;
	case OBS_FRONTEND_EVENT_RECORDING_STOPPED:
		if (active) {
			capture.stop();
			active = false;
			if (capture.failed()) show_error(L"Input logging failed. This sidecar is incomplete; do not use it for training.");
		}
		break;
	case OBS_FRONTEND_EVENT_EXIT:
		capture.stop();
		active = false;
		break;
	default: break;
	}
}
} // namespace

bool obs_module_load(void)
{
	obs_frontend_add_tools_menu_item("Lumine input logging...", tools_toggle, nullptr);
	obs_frontend_add_event_callback(frontend_event, nullptr);
	blog(LOG_INFO, "[Lumine] Loaded. Input logging is OFF until explicitly enabled in Tools.");
	return true;
}
void obs_module_unload(void)
{
	obs_frontend_remove_event_callback(frontend_event, nullptr);
	capture.stop();
}
