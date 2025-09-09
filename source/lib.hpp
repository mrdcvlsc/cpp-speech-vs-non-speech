#pragma once

#include <string>
#include <vector>

#include <SFML/Audio.hpp>
#include <torch/script.h>

#ifdef _WIN32
#  include <stdlib.h>
#endif

/**
 * @brief The core implementation of the executable
 *
 * This class makes up the library part of the executable, which means that the
 * main logic is implemented here. This kind of separation makes it easy to
 * test the implementation for the executable, because the logic is nicely
 * separated from the command-line logic implemented in the main function.
 */
struct library
{
  /**
   * @brief Simply initializes the name member to the name of the project
   */
  library();

  std::string name;
};

class silero_vad
{
  std::string m_model_filename;
  float m_threshold = 0.5f;  // decision thresholdss
  torch::jit::script::Module m_model;
  int m_model_sample_rate_khz;
  int m_model_frame_size;

public:
  explicit silero_vad(const std::string& model_filename,
                      int sample_rate_khz = 16);

  auto get_model_filename() const -> const std::string;

  auto has_speech(const std::string& audio_file) -> bool;
};

auto to_mono_normalized_f32(const int16_t* samples,
                            size_t size,
                            size_t channels) -> std::vector<float>;

auto up_sampler(const std::vector<float>& y_samples, std::size_t new_size)
    -> std::vector<float>;

// Simple linear resampler
auto resample_linear(const std::vector<float>& input_samples,
                     std::size_t input_sample_rate,
                     std::size_t output_sample_rate) -> std::vector<float>;

// Load audio with SFML and return mono float data resampled to
// `target_sample_rate`.
auto load_audio_file_as_mono_f32(const std::string& path,
                                 std::vector<float>& out_waveform,
                                 int target_sample_rate) -> bool;

// Build batched frames from waveform.
// frame_size: required length per model frame (e.g. 512 for sr=16000).
// hop: sliding hop (set = frame_size for non-overlap, or frame_size/2 for 50%
// overlap). Returns frames_data contiguous: [ frame0..., frame1..., ... ] and
// num_frames.
auto build_frames(const std::vector<float>& waveform,
                  int silero_vad_sample_window_size,
                  int hop,
                  std::vector<float>& frames_data,
                  int& num_frames) -> void;

// Utility to stringify tensor shape
auto tensor_shape_to_string(const torch::Tensor& input_tensor) -> std::string;

// Run model on batched frames and return per-frame probabilities in `out_probs`
// sr must be the integer sample rate (e.g. 16000)
auto run_model_on_frames(torch::jit::script::Module& module,
                         const std::vector<float>& frames_data,
                         int num_frames,
                         int frame_size,
                         int sample_rate,
                         std::vector<float>& out_probs) -> bool;

// Aggregate per-frame probabilities into a single decision and print
// diagnostics. Strategy: compute max prob, mean prob, and count frames above
// threshold.
auto aggregate_and_report(const std::vector<float>& probs, float threshold)
    -> bool;
