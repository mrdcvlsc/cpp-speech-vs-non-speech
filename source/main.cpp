// silero_vad_classifier_chunked.cpp
// Updated to chunk waveform into frames expected by silero-vad JIT model.
// Build with CMake linking libtorch + SFML 3.0.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/Parallel.h>
#include <SFML/Audio.hpp>
#include <torch/script.h>

#include "lib.hpp"

// Target sample rate we resample to and pass to the model
constexpr int TARGET_SAMPLE_RATE = 16000;
const int SAMPLE_16KHZ = 16000;

// Default decision threshold (tweak for your model/data)
constexpr float DEFAULT_THRESHOLD = 0.5F;

auto main(int argc, char** argv) -> int
{
#ifdef _WIN32
  _putenv_s("OMP_NUM_THREADS", "1");
  _putenv_s("MKL_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
#endif

  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  std::vector<std::string> files = {"audio-samples/speech.ogg",
                                    "audio-samples/noise.ogg"};
  if (argc > 1) {
    files.clear();
    for (int i = 1; i < argc; ++i) {
      files.emplace_back(argv[i]);
    }
  }

  const float threshold = DEFAULT_THRESHOLD;

  //   std::string mode_path = "silero-vad.pt";
  std::string mode_path = "silero_vad_v6.jit";
  torch::jit::script::Module module;

  try {
    module = torch::jit::load(mode_path);
    module.eval();
    std::cout << "Loaded model: " << mode_path << '\n';
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load model '" << mode_path << "': " << e.what()
              << '\n';
    return 1;
  }

  for (const auto& audio_file : files) {
    std::cout << "Processing: " << audio_file << " ...\n";
    std::vector<float> wave;

    if (!load_audio_file_as_mono_f32(audio_file, wave, TARGET_SAMPLE_RATE)) {
      std::cerr << "Skipping file due to load error: " << audio_file << "\n";
      continue;
    }

    // Determine model frame size based on sr
    int frame_size = (TARGET_SAMPLE_RATE == SAMPLE_16KHZ) ? 512 : 256;
    int hop = frame_size;  // non-overlapping; change to frame_size/2 for
                           // overlap if preferred

    std::vector<float> frames_data;
    int num_frames = 0;
    build_frames(wave, frame_size, hop, frames_data, num_frames);

    if (num_frames <= 0) {
      std::cerr << "  no frames created from waveform; skipping\n";
      continue;
    }
    
    std::vector<float> probs;

    bool is_ok = run_model_on_frames(
        module, frames_data, num_frames, frame_size, TARGET_SAMPLE_RATE, probs);

    if (!is_ok) {
      std::cerr << "  model run failed on frames -> DECISION: does NOT contain "
                   "human speech\n";
      continue;
    }

    if (static_cast<int>(probs.size()) > num_frames) {
      probs.resize(static_cast<size_t>(num_frames));
    }

    aggregate_and_report(probs, threshold, audio_file);
  }

  return 0;
}
