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

auto main(int argc, char** argv) -> int
{
  std::vector<std::string> files = {"audio-samples/speech.ogg",
                                    "audio-samples/noise.ogg"};
  if (argc > 1) {
    files.clear();
    for (int i = 1; i < argc; ++i) {
      files.emplace_back(argv[i]);
    }
  }

  std::string mode_path = "silero_vad_v6.jit";
  silero_vad model(mode_path);

  for (const auto& audio_file : files) {
    std::cout << "\n\nfinal result : " << model.has_speech(audio_file) << '\n';
  }

  return 0;
}
