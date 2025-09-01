#include <iostream>
#include <memory>
#include <vector>

#include <torch/script.h>  // Include the necessary header for LibTorch

// Function to load and run inference with the TorchScript model
auto main(int argc, const char* argv[]) -> int
{
  // 1. Define the file path to the saved TorchScript model
  const std::string model_path = "audio_classifier_scripted.pt";

  // 2. Use torch::jit::load() to load the TorchScript model
  std::shared_ptr<torch::jit::Module> module;
  
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
    std::cout << "TorchScript model loaded successfully from " << model_path
              << '\n';
  } catch (const c10::Error& e) {
    // Include error handling for the loading process.
    std::cerr << "Error loading the model\n";
    std::cerr << e.what() << '\n';
    return -1;
  }

  // Set the model to evaluation mode (optional but good practice)
  module->eval();
  std::cout << "Model set to evaluation mode." << '\n';

  // 4. Create a dummy input tensor in C++
  // The model expects a tensor of shape {1, segment_len_frames}
  // where segment_len_frames is 80000 (for 5 seconds at 16kHz)
  int segment_len_frames = 80000;
  // torch::randn({batch_size, sequence_length})
  torch::Tensor input_tensor = torch::randn({1, segment_len_frames});
  std::cout << "Created dummy input tensor with shape: " << input_tensor.sizes()
            << '\n';

  // 5. Prepare the input tensor for inference (move to device if needed)
  // This example focuses on CPU inference for simplicity.
  // If using GPU, you would typically do:
  // input_tensor = input_tensor.to(at::kCUDA);
  // module->to(at::kCUDA); // Move model to GPU

  // 6. Perform inference by passing the input tensor to the loaded model
  // object.
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(input_tensor);

  std::cout << "Performing inference..." << '\n';
  torch::Tensor output = module->forward(inputs).toTensor();
  std::cout << "Inference complete." << '\n';
  // Raw output tensor: contains logits
  std::cout << "Raw output tensor: " << output << '\n';

  // 7. Process the output tensor
  // Apply softmax to get probabilities
  // For binary classification, softmax over the last dimension
  torch::Tensor probabilities = torch::softmax(output, /*dim=*/1);
  std::cout << "Class probabilities: " << probabilities << '\n';

  // Find the index of the maximum value for the predicted class
  torch::Tensor predicted_class_index = torch::argmax(probabilities, /*dim=*/1);

  // 8. Print the results of the inference
  std::cout << "Predicted class index: " << predicted_class_index.item<int>()
            << '\n';

  // Map index to label (assuming 0: speech, 1: non-speech)
  const std::vector<std::string> label_map = {"speech", "non-speech"};
  int pred_idx = predicted_class_index.item<int>();
  std::string predicted_label = (pred_idx >= 0 && pred_idx < label_map.size())
      ? label_map[pred_idx]
      : "unknown";
  std::cout << "Predicted label: " << predicted_label << '\n';
}
