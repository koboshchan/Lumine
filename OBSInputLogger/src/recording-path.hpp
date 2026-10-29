#pragma once
#include <filesystem>
#include <string>

// Include after obs.h (or test doubles). The frontend "current record output
// path" API returns the configured DIRECTORY, not the active video filename.
inline bool lumine_recording_path(obs_output_t *output, std::filesystem::path &video)
{
	if (!output) return false;
	const std::string id = obs_output_get_id(output);
	if (id != "ffmpeg_muxer" && id != "mp4_output" && id != "mov_output")
		return false;
	auto *settings = obs_output_get_settings(output);
	if (!settings) return false;
	const std::string filename = obs_data_get_string(settings, "path");
	obs_data_release(settings);
	if (filename.empty()) return false;
	auto candidate = std::filesystem::u8path(filename);
	std::error_code error;
	// Fail closed on directories, relative paths and missing filename extensions.
	// The muxer may not have created the file yet at RECORDING_STARTED.
	if (!candidate.is_absolute() || !candidate.has_extension() ||
	    candidate.filename().empty() || std::filesystem::is_directory(candidate, error))
		return false;
	video = std::move(candidate);
	return true;
}
