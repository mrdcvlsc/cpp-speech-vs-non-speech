#include "lib.hpp"

library::library()
    : name {"speech-vs-non-speech"}
{
}

auto to_mono_f32(const int16_t* samples, size_t size, size_t channels)
    -> std::vector<float>
{
  if (channels == 0) {
    channels = 1;
  }

  size_t frames = size / channels;

  std::vector<float> mono;
  mono.reserve(frames);

  for (size_t sample_frame = 0; sample_frame < frames; ++sample_frame) {
    int sum = 0;

    for (size_t channel = 0; channel < channels; ++channel) {
      sum += samples[(sample_frame * channels) + channel];
    }

    float avg = static_cast<float>(sum) / static_cast<float>(channels);

    mono.push_back(avg / 32768.F);
  }

  return mono;
}

auto resample_linear(const std::vector<float>& input_samples,
                     int in_rate,
                     int out_rate) -> std::vector<float>
{
  if (in_rate == out_rate) {
    return input_samples;
  }

  if (input_samples.empty()) {
    return {};
  }

  double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);

  auto out_len = static_cast<size_t>(
      std::round(static_cast<double>(input_samples.size()) * ratio));

  if (out_len == 0) {
    out_len = 1;
  }

  std::vector<float> out(out_len);

  for (size_t i = 0; i < out_len; ++i) {
    double src_pos = i / ratio;
    auto idx = static_cast<size_t>(std::floor(src_pos));
    double frac = src_pos - idx;
    float a = input_samples[std::min(idx, input_samples.size() - 1)];
    float b = input_samples[std::min(idx + 1, input_samples.size() - 1)];
    out[i] = static_cast<float>((1.0 - frac) * a + frac * b);
  }
  return out;
}

auto load_audio_file_as_mono_f32(const std::string& path,
                                 std::vector<float>& out_waveform,
                                 int target_sample_rate) -> bool
{
  sf::SoundBuffer buffer;

  if (!buffer.loadFromFile(path)) {
    std::cerr << "Failed to load audio file: " << path << '\n';
    return false;
  }

  auto mono = to_mono_f32(
      buffer.getSamples(), buffer.getSampleCount(), buffer.getChannelCount());

  auto sample_rate = static_cast<int>(buffer.getSampleRate());

  if (sample_rate != target_sample_rate) {
    out_waveform = resample_linear(mono, sample_rate, target_sample_rate);
  } else {
    out_waveform = std::move(mono);
  }

  return true;
}

auto build_frames(const std::vector<float>& waveform,
                  int frame_size,
                  int hop,
                  std::vector<float>& frames_data,
                  int& num_frames) -> void
{
  if (frame_size <= 0) {
    num_frames = 0;
    return;
  }
  if (waveform.empty()) {
    num_frames = 0;
    return;
  }

  // compute number of frames (ceil)
  num_frames = static_cast<int>((waveform.size() + hop - 1) / hop);
  if (num_frames <= 0) {
    num_frames = 1;
  }

  frames_data.clear();
  frames_data.reserve(static_cast<size_t>(num_frames) * frame_size);

  for (int i = 0; i < num_frames; ++i) {
    size_t start = static_cast<size_t>(i) * static_cast<size_t>(hop);
    for (int k = 0; k < frame_size; ++k) {
      size_t idx = start + static_cast<size_t>(k);
      float v = (idx < waveform.size()) ? waveform[idx]
                                        : 0.0f;  // zero-pad last frame
      frames_data.push_back(v);
    }
  }
}

auto tensor_shape_to_string(const torch::Tensor& input_tensor) -> std::string
{
  if (!input_tensor.defined()) {
    return "<undefined>";
  }

  std::ostringstream oss;
  oss << "[";

  for (int64_t dimension = 0; dimension < input_tensor.dim(); ++dimension) {
    if (static_cast<bool>(dimension)) {
      oss << ", ";
    }

    oss << input_tensor.size(dimension);
  }

  oss << "]";

  return oss.str();
}

auto run_model_on_frames(torch::jit::script::Module& module,
                         const std::vector<float>& frames_data,
                         int num_frames,
                         int frame_size,
                         int sample_rate,
                         std::vector<float>& out_probs) -> bool
{
  out_probs.clear();
  if (num_frames <= 0 || frame_size <= 0) {
    return false;
  }

  if (frames_data.size() != static_cast<size_t>(num_frames * frame_size)) {
    return false;
  }

  auto options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

  torch::Tensor t = torch::from_blob(const_cast<float*>(frames_data.data()),
                                     {static_cast<int64_t>(num_frames),
                                      static_cast<int64_t>(frame_size)},
                                     options)
                        .clone();

  torch::Tensor out;
  try {
    out = module.forward({t, sample_rate}).toTensor();
  } catch (const c10::Error& e1) {
    try {
      // try adding channel dimension -> [num_frames, 1, frame_size]
      out = module.forward({t.unsqueeze(1), sample_rate}).toTensor();
    } catch (const c10::Error& e2) {
      std::cerr << "Model forward error: " << e2.what() << '\n';
      return false;
    }
  }

  if (!out.defined()) {
    return false;
  }

  // Squeeze and convert to CPU float tensor
  out = out.squeeze().to(torch::kCPU);
  // Now out should have shape [num_frames] or [num_frames, 1] or similar
  int64_t n = out.numel();
  if (n == 0) {
    return false;
  }

  // If number elements != num_frames, try to reshape / interpret carefully
  if (n == num_frames) {
    // good
  } else if (out.dim() == 2 && out.size(0) == num_frames && out.size(1) == 1) {
    // okay too
  } else {
    // Unexpected shape — we'll attempt to flatten and take first num_frames
    // values (defensive) But warn user
    std::cerr << "Warning: model output has unexpected size "
              << tensor_shape_to_string(out)
              << "; expected per-frame probs length " << num_frames
              << ". Flattening.\n";
  }

  // copy outputs into vector<float>
  out_probs.resize(static_cast<size_t>(n));
  auto out_accessor = out.flatten();  // contiguous flattened
  // Use data_ptr to copy
  std::memcpy(out_probs.data(),
              out_accessor.contiguous().data_ptr(),
              sizeof(float) * static_cast<size_t>(n));

  return true;
}

auto aggregate_and_report(const std::vector<float>& probs,
                          float threshold,
                          const std::string& filename) -> void
{
  if (probs.empty()) {
    std::cout
        << "  no frames produced -> DECISION: does NOT contain human speech\n";
    return;
  }
  float maxp = 0.0f;
  double sum = 0.0;
  int cnt = 0;
  for (float p : probs) {
    maxp = std::max(p, maxp);
    sum += p;
    if (p >= threshold) {
      ++cnt;
    }
  }
  double meanp = sum / static_cast<double>(probs.size());
  std::cout << "  frames: " << probs.size() << ", max=" << maxp
            << ", mean=" << meanp << ", frames>=threshold(" << threshold
            << ")=" << cnt << "\n";

  // Decision rule (simple): if any frame >= threshold OR average >=
  // (threshold/2)
  bool contains_speech = (maxp >= threshold) || (meanp >= (threshold * 0.5));
  std::cout << "  => DECISION: "
            << (contains_speech ? "contains human speech"
                                : "does NOT contain human speech")
            << "\n";
}
