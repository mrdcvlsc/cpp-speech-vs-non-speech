// silero_vad_classifier_chunked.cpp
// Updated to chunk waveform into frames expected by silero-vad JIT model.
// Build with CMake linking libtorch + SFML 3.0.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lib.hpp"

auto main(int argc, char** argv) -> int
{
  std::cout << "############## PROGRAM START #################\n";

  std::vector<std::filesystem::path> files = {
      std::filesystem::path(get_exe_path()) / "audio-samples/speech.ogg",
      std::filesystem::path(get_exe_path()) / "audio-samples/noise.ogg"};

  if (argc > 1) {
    files.clear();
    for (int i = 1; i < argc; ++i) {
      files.emplace_back(argv[i]);
    }
  }

  auto mode_path = std::filesystem::path(get_exe_path()) / "silero_vad_v6.jit";
  silero_vad model(mode_path.string());

  for (const auto& audio_file : files) {
    std::cout << "\n\nfinal result : " << model.has_speech(audio_file.string())
              << '\n';
  }

  return 0;
}
