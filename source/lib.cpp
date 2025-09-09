#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <type_traits>

#include "lib.hpp"

#include <ATen/Parallel.h>

library::library()
    : name {"speech-vs-non-speech"}
{
}

auto to_mono_normalized_f32(const int16_t* samples,
                            size_t size,
                            size_t channels) -> std::vector<float>
{
  //   std::cout << "called: to_mono_normalized_f32 \n";

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

auto up_sampler(const std::vector<float>& y_samples, std::size_t new_size)
    -> std::vector<float>
{
  //   std::cout << "called: up_sampler \n";

  if (y_samples.empty() || new_size <= 0) {
    return {};
  }

  if (new_size == static_cast<int>(y_samples.size())) {
    return y_samples;
  }

  if (y_samples.size() == 1) {
    return std::vector<float>(new_size, y_samples[0]);
  }

  std::vector<float> y_upsampled(new_size);

  auto scale = static_cast<float>(y_samples.size() - 1)
      / static_cast<float>(y_upsampled.size() - 1);

  for (std::size_t i = 0; i < y_upsampled.size(); ++i) {
    // interpolate x in input-sample coordinates
    auto x = static_cast<float>(i) * scale;

    auto x0 = static_cast<std::size_t>(std::floor(x));
    auto t = x - static_cast<float>(x0);  // since x1 = x0 + 1, t = (x - x0) /
                                          // (x1 - x0) simplifies to x-x0

    // clamp / edge case when x0 is last sample
    if (x0 >= y_samples.size() - 1) {
      y_upsampled[i] = y_samples.back();
    } else {
      float y0 = y_samples[x0];
      float y1 = y_samples[x0 + 1];

      // linear interpolation: y = (1 - t) * y0 + t * y1
      float y = (1.0f - t) * y0 + t * y1;
      y_upsampled[i] = y;
    }
  }

  return y_upsampled;
}

std::vector<float> downsampler(const std::vector<float>& input,
                               std::size_t out_size)
{
  //   std::cout << "called: downsampler \n";

  std::size_t in_size = input.size();
  if (out_size == 0 || in_size == 0) {
    return {};
  }
  if (out_size == in_size) {
    return input;
  }

  // Map input domain length = in_size, output domain split into out_size bins
  double in_len = static_cast<double>(in_size);
  double bin_width = in_len / static_cast<double>(out_size);

  std::vector<float> out(out_size, 0.0);

  for (std::size_t k = 0; k < out_size; ++k) {
    double bin_start = k * bin_width;
    double bin_end = (k + 1) * bin_width;
    double acc = 0.0;
    // only check input indices that can overlap the bin
    std::size_t j0 = static_cast<std::size_t>(
        std::max(0, static_cast<int>(std::floor(bin_start))));
    std::size_t j1 = static_cast<std::size_t>(std::min(
        static_cast<int>(in_size) - 1,
        static_cast<int>(std::floor(bin_end))  // floor of right edge may be
                                               // equal to in_size, clamp below
        ));

    // sometimes bin_end == in_size -> floor(in_size) == in_size -> clamp j1
    if (j1 >= in_size) {
      j1 = in_size - 1;
    }

    for (std::size_t j = j0; j <= j1; ++j) {
      double sample_start = static_cast<double>(j);
      double sample_end = static_cast<double>(j + 1);
      double left = std::max(bin_start, sample_start);
      double right = std::min(bin_end, sample_end);
      double overlap = right - left;
      if (overlap > 0) {
        acc += input[j] * overlap;
      }
    }

    out[k] = acc / bin_width;
  }

  return out;
}

auto resample_linear(const std::vector<float>& input_samples,
                     std::size_t input_sample_rate,
                     std::size_t output_sample_rate) -> std::vector<float>
{
  //   std::cout << "called: resample_linear \n";

  if (input_sample_rate == output_sample_rate) {
    return input_samples;
  }

  if (input_samples.empty() || output_sample_rate <= 0
      || input_sample_rate <= 0)
  {
    return {};
  }

  double ratio = static_cast<double>(output_sample_rate)
      / static_cast<double>(input_sample_rate);

  auto new_sample_size = static_cast<std::size_t>(
      std::round(static_cast<double>(input_samples.size()) * ratio));

  std::vector<float> output(new_sample_size);

  return up_sampler(input_samples, new_sample_size);
}

auto load_audio_file_as_mono_f32(const std::string& path,
                                 std::vector<float>& out_waveform,
                                 int target_sample_rate) -> bool
{
  //   std::cout << "called: load_audio_file_as_mono_f32 \n";

  sf::SoundBuffer buffer;

  if (!buffer.loadFromFile(path)) {
    std::cerr << "Failed to load audio file: " << path << '\n';
    return false;
  }

  std::cout << "file loaded : " << path << '\n';
  std::cout << "samples     : " << buffer.getSampleCount() << '\n';
  std::cout << "channels    : " << buffer.getChannelCount() << '\n';
  std::cout << "sample rate : " << buffer.getSampleRate() << '\n';
  std::cout << "seconds     : " << buffer.getDuration().asSeconds() << '\n';
  std::cout << '\n';

  auto mono = to_mono_normalized_f32(
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
                  int silero_vad_sample_window_size,
                  int hop,
                  std::vector<float>& frames_data,
                  int& num_frames) -> void
{
  //   std::cout << "called: build_frames \n";

  if (silero_vad_sample_window_size <= 0 || waveform.empty()) {
    num_frames = 0;
    return;
  }

  // compute number of frames (ceil)
  num_frames = static_cast<int>((waveform.size() + hop - 1) / hop);

  if (num_frames <= 0) {
    num_frames = 1;
  }

  frames_data.clear();
  frames_data.reserve(static_cast<size_t>(num_frames)
                      * silero_vad_sample_window_size);

  for (std::size_t i = 0; i < num_frames; ++i) {
    auto start = i * static_cast<std::size_t>(hop);

    for (std::size_t k = 0; k < silero_vad_sample_window_size; ++k) {
      auto idx = start + k;

      // zero-pad last frame
      float value = (idx < waveform.size()) ? waveform[idx] : 0.F;
      frames_data.push_back(value);
    }
  }
}

auto tensor_shape_to_string(const torch::Tensor& input_tensor) -> std::string
{
  //   std::cout << "called: tensor_shape_to_string \n";

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
  //   std::cout << "called: run_model_on_frames \n";

  out_probs.clear();
  if (num_frames <= 0 || frame_size <= 0) {
    return false;
  }

  if (frames_data.size()
      != (static_cast<std::size_t>(num_frames)
          * static_cast<std::size_t>(frame_size)))
  {
    return false;
  }

  auto options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

  torch::Tensor tnsr = torch::from_blob(const_cast<float*>(frames_data.data()),
                                        {static_cast<int64_t>(num_frames),
                                         static_cast<int64_t>(frame_size)},
                                        options)
                           .clone();  // clone might not be needed here.

  torch::Tensor out;
  try {
    out = module.forward({tnsr, sample_rate}).toTensor();
  } catch (const c10::Error& e1) {
    try {
      // try adding channel dimension -> [num_frames, 1, frame_size]
      out = module.forward({tnsr.unsqueeze(1), sample_rate}).toTensor();
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
  int64_t n = out.numel();  // numel - number of elements in a tensor.
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

auto aggregate_and_report(const std::vector<float>& probs, float threshold)
    -> bool
{
  //   std::cout << "called: aggregate_and_report \n";

  if (probs.empty()) {
    std::cout
        << "  no frames produced -> DECISION: does NOT contain human speech\n";
    return false;
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
  bool has_speech = (maxp >= threshold) || (meanp >= (threshold * 0.5));
  std::cout << "  => DECISION: "
            << (has_speech ? "contains human speech"
                           : "does NOT contain human speech")
            << "\n";

  return has_speech;
}

silero_vad::silero_vad(const std::string& model_filename, int sample_rate_khz)
    : m_model_filename(model_filename)
    , m_model_sample_rate_khz(sample_rate_khz * 1000)
    , m_model_frame_size((sample_rate_khz == 16) ? 512 : 256)
{
  //   std::cout << "called: silero_vad \n";

#ifdef _WIN32
  _putenv_s("OMP_NUM_THREADS", "1");
  _putenv_s("MKL_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
#endif

  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  m_model = torch::jit::load(model_filename);
  m_model.eval();
  std::cout << "Loaded model: " << model_filename << '\n';
}

auto silero_vad::has_speech(const std::string& audio_file) -> bool
{
  //   std::cout << "called: has_speech \n";

  std::vector<float> wave;
  if (!load_audio_file_as_mono_f32(audio_file, wave, m_model_sample_rate_khz)) {
    std::cerr << "Skipping file due to load error: " << audio_file << "\n";
    exit(1);
  }

  std::vector<float> frames_data;
  int num_frames = 0;

  build_frames(
      wave, m_model_frame_size, m_model_frame_size, frames_data, num_frames);

  if (num_frames <= 0) {
    std::cerr << "  no frames created from waveform; skipping\n";
    exit(2);
  }

  std::vector<float> probs;

  bool is_ok = run_model_on_frames(m_model,
                                   frames_data,
                                   num_frames,
                                   m_model_frame_size,
                                   m_model_sample_rate_khz,
                                   probs);

  if (!is_ok) {
    std::cerr << "  model run failed on frames -> DECISION: does NOT contain "
                   "human speech\n";
    exit(3);
  }

  if (!is_ok) {
    std::cerr << "  model run failed on frames -> DECISION: does NOT contain "
                   "human speech\n";
    exit(4);
  }

  if (static_cast<int>(probs.size()) > num_frames) {
    probs.resize(static_cast<size_t>(num_frames));
  }

  return aggregate_and_report(probs, m_threshold);
}
