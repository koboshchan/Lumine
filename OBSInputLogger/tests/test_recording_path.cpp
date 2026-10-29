#include <cassert>
#include <filesystem>
#include <string>
#include <iostream>
struct obs_data_t { std::string path; };
struct obs_output_t { std::string id; obs_data_t settings; };
static int releases = 0;
const char *obs_output_get_id(obs_output_t *o) { return o->id.c_str(); }
obs_data_t *obs_output_get_settings(obs_output_t *o) { return &o->settings; }
const char *obs_data_get_string(obs_data_t *s, const char *) { return s->path.c_str(); }
void obs_data_release(obs_data_t *) { ++releases; }
#include "../src/recording-path.hpp"
int main() {
	auto dir = std::filesystem::temp_directory_path() / "lumine-path-regression";
	std::filesystem::create_directories(dir);
	std::filesystem::path result;
	obs_output_t out{"ffmpeg_muxer", {(dir / "recording-one.mkv").u8string()}};
	assert(!lumine_recording_path(nullptr, result));
	for (auto id : {"ffmpeg_muxer", "mp4_output", "mov_output"}) {
		out.id = id;
		out.settings.path = (dir / "recording-one.mkv").u8string();
		assert(lumine_recording_path(&out, result));
		assert(result.filename() == "recording-one.mkv");
		auto first = result; first.replace_extension(".log");
		assert(first.parent_path() == dir);
		out.settings.path = (dir / "recording-two.mkv").u8string();
		assert(lumine_recording_path(&out, result));
		result.replace_extension(".log");
		assert(result != first);
	}
	for (auto path : {"", "relative.mkv", "https://example.test/video.mkv"}) {
		out.settings.path = path;
		assert(!lumine_recording_path(&out, result));
	}
	out.settings.path = dir.u8string();
	assert(!lumine_recording_path(&out, result));
	auto dotted_dir = dir / "folder.mkv";
	std::filesystem::create_directory(dotted_dir);
	out.settings.path = dotted_dir.u8string();
	assert(!lumine_recording_path(&out, result));
	out.id = "ffmpeg_output";
	out.settings.path = (dir / "recording.mkv").u8string();
	assert(!lumine_recording_path(&out, result));
	assert(releases == 11);
	std::filesystem::remove(dotted_dir);
	std::filesystem::remove(dir);
	std::cout << "recording path regression passed\n";
}
