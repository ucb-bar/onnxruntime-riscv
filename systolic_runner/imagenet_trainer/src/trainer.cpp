// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/bfc_arena.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"

#include "stb_image.h"

#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"

#include <systolic/systolic_provider_factory.h>

#ifdef FOR_FIRESIM
#include <sys/mman.h>
#endif

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Systolic(int use_arena, char accelerator_mode);
}

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

typedef vector<uint8_t> Image;

#define IMG_SIZE 224 * 224 * 3
#define NUM_CLASS 1000
const static vector<int64_t> IMAGE_DIMS = {3, 224, 224};
const static vector<int64_t> LABEL_DIMS = {1000};

// You may wonder why we have this wrapper thing. It's because
// it's not very simple to just copy the ParseResult
// https://github.com/jarro2783/cxxopts/issues/146
// Besides, since we are assigning to TrainingRunner::Parameters
// directly we may as well follow the same pattern
struct MnistParameters : public TrainingRunner::Parameters {
  int debug;
  int num_samples;
  TransformerLevel optimization_level;
};

Status ParseArguments(int argc, char* argv[], MnistParameters& params) {
  cxxopts::Options options("POC Training", "Main Program to train on MNIST");
  // clang-format off
  options
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("train_data_txt", "The value of the batch txt file.",
        cxxopts::value<std::string>()->default_value("mnist_data"))
      ("num_samples", "Number of samples to use from file.",
        cxxopts::value<int>()->default_value("-1"))
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value(""))
      ("use_profiler", "Collect runtime profile data during this training run.", cxxopts::value<bool>()->default_value("false"))
      ("use_gist", "Whether to use GIST encoding/decoding.")
      ("gist_op", "Opearator type(s) to which GIST is applied.", cxxopts::value<int>()->default_value("0"))
      ("gist_compr", "Compression type used for GIST", cxxopts::value<std::string>()->default_value("GistPack8"))
      ("x,execution", "Systolic execution mode. Either 0, 1, or 2 (CPU, OS, WS).", cxxopts::value<int>(), "[0/1/2]")
      ("O,optimization_level", "Optimization level. NHWC transformation is applied at -O 99.",
                            cxxopts::value<int>()->default_value("1"), "[0 (none) / 1 (basic) / 2 (extended) / 99 (all)]")
      ("d,debug", "Debug level", cxxopts::value<int>()->default_value("2"), "[0-4, with 0 being most verbose]")
      ("num_train_steps", "Number of training steps.", cxxopts::value<int>()->default_value("2000"))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>()->default_value("100"))
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>()->default_value("100"))
      ("learning_rate", "The initial learning rate for Adam.", cxxopts::value<float>()->default_value("0.01"))
      ("display_loss_steps", "How often to dump loss into tensorboard", cxxopts::value<size_t>()->default_value("10"))
      ("data_parallel_size", "Data parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("horizontal_parallel_size", "Horizontal model parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("pipeline_parallel_size", "Number of pipeline stages.", cxxopts::value<int>()->default_value("1"))
      ("cut_group_info", "Specify the cutting info for graph partition (pipeline only). An example of a cut_group_info of "
      "size two is: 1393:407-1463/1585/1707,2369:407-2439/2561/2683. Here, the cut info is split by ',', with the first "
      "cut_info equal to 1393:407-1463/1585/1707, and second cut_info equal to 2369:407-2439/2561/2683. Each CutEdge is "
      "seperated by ':'. If consumer nodes need to be specified, specify them after producer node with a '-' delimiter and "
      "separate each consumer node with a '/'. ", cxxopts::value<std::vector<std::string>>()->default_value(""))
      ("evaluation_period", "How many training steps to make before making an evaluation.",
        cxxopts::value<size_t>()->default_value("1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    // Set parameters
    params.model_name = flags["model_name"].as<std::string>();
    params.use_gist = flags.count("use_gist") > 0;
    params.lr_params.initial_lr = flags["learning_rate"].as<float>();
    params.num_train_steps = flags["num_train_steps"].as<int>();
    params.batch_size = flags["train_batch_size"].as<int>();
    params.gist_config.op_type = flags["gist_op"].as<int>();
    params.gist_config.compr_type = flags["gist_compr"].as<std::string>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.batch_size;
    }
    params.evaluation_period = flags["evaluation_period"].as<size_t>();

    params.shuffle_data = false;
    params.is_perf_test = false;

    auto train_data_dir = flags["train_data_txt"].as<std::string>();
    auto log_dir = flags["log_dir"].as<std::string>();
    params.train_data_dir.assign(train_data_dir.begin(), train_data_dir.end());
    params.log_dir.assign(log_dir.begin(), log_dir.end());
    params.use_profiler = flags.count("use_profiler") > 0;
    params.display_loss_steps = flags["display_loss_steps"].as<size_t>();
    params.data_parallel_size = flags["data_parallel_size"].as<int>();
    params.horizontal_parallel_size = flags["horizontal_parallel_size"].as<int>();
    // pipeline_parallel_size controls the number of pipeline's stages.
    // pipeline_parallel_size=1 means no model partition, which means all processes run
    // the same model. We only partition model when pipeline_parallel_size > 1.
    params.pipeline_parallel_size = flags["pipeline_parallel_size"].as<int>();
    ORT_RETURN_IF_NOT(params.data_parallel_size > 0, "data_parallel_size must > 0");
    ORT_RETURN_IF_NOT(params.horizontal_parallel_size > 0, "horizontal_parallel_size must > 0");
    ORT_RETURN_IF_NOT(params.pipeline_parallel_size > 0, "pipeline_parallel_size must > 0");

    // If user doesn't provide partitioned model files, a cut list should be provided for ORT to do partition
    // online. If the pipeline contains n stages, the cut list should be of length (n-1), in order to cut the
    // graph into n partitions.
    if (params.pipeline_parallel_size > 1) {
      auto cut_info_groups = flags["cut_group_info"].as<std::vector<std::string>>();

      ORT_RETURN_IF_NOT(static_cast<int>(cut_info_groups.size() + 1) == params.pipeline_parallel_size,
                        "cut_info length plus one must match pipeline parallel size");

      auto process_with_delimiter = [](std::string& input_str, const std::string& delimiter) {
        std::vector<std::string> result;
        size_t pos = 0;
        std::string token;
        while ((pos = input_str.find(delimiter)) != std::string::npos) {
          token = input_str.substr(0, pos);
          result.emplace_back(token);
          input_str.erase(0, pos + delimiter.length());
        }
        // push the last split of substring into result.
        result.emplace_back(input_str);
        return result;
      };

      auto process_cut_info = [&](std::string& cut_info_string) {
        TrainingSession::TrainingConfiguration::CutInfo cut_info;
        const std::string edge_delimiter = ":";
        const std::string consumer_delimiter = "/";
        const std::string producer_consumer_delimiter = "-";

        auto cut_edges = process_with_delimiter(cut_info_string, edge_delimiter);
        for (auto& cut_edge : cut_edges) {
          auto process_edge = process_with_delimiter(cut_edge, producer_consumer_delimiter);
          if (process_edge.size() == 1) {
            TrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0]};
            cut_info.emplace_back(edge);
          } else {
            ORT_ENFORCE(process_edge.size() == 2);
            auto consumer_list = process_with_delimiter(process_edge[1], consumer_delimiter);

            TrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0], consumer_list};
            cut_info.emplace_back(edge);
          }
        }
        return cut_info;
      };

      for (auto& cut_info : cut_info_groups) {
        TrainingSession::TrainingConfiguration::CutInfo cut = process_cut_info(cut_info);
        params.pipeline_partition_cut_list.emplace_back(cut);
      }
    }

    params.debug = flags["debug"].as<int>();
    params.num_samples = flags["num_samples"].as<int>();
    int optimization_level = flags["optimization_level"].as<int>();
    params.optimization_level = optimization_level > (int)TransformerLevel::MaxLevel ? TransformerLevel::MaxLevel : (TransformerLevel)optimization_level;

    if (flags.count("execution") > 0) {
      params.providers.emplace(kSystolicExecutionProvider, CreateExecutionProviderFactory_Systolic(/*use_arena=*/1, /*accelerator_mode=*/(char)flags["execution"].as<int>()));
    }
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return Status(ONNXRUNTIME, FAIL, msg);
  }
  return Status::OK();
}

// NOTE: these variables need to be alive when the error_function is called.
int true_count = 0;
float total_loss = 0.0f;

void setup_training_params(MnistParameters& params) {
  params.model_path = ToPathString(params.model_name);
  params.model_with_loss_func_path = ToPathString(params.model_name) + ORT_TSTR("_with_cost.onnx");
  params.model_with_training_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw.onnx");
  params.model_actual_running_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw_running.onnx");
  params.model_with_gist_nodes_path = ToPathString(params.model_name) + ORT_TSTR("_with_gist.onnx");
  params.output_dir = ORT_TSTR(".");

  auto output_label = "gpu_0/softmax_1";
  //Gist encode
  params.loss_func_info = LossFunctionInfo(OpDef("SoftmaxCrossEntropy", kMSDomain, 1),
                                           "loss",
                                           {output_label, "labels"});
  params.fetch_names = {output_label, "loss"};

  params.training_optimizer_name = "SGDOptimizer";

  params.error_function = [](const std::vector<std::string>& /*feed_names*/,
                             const std::vector<OrtValue>& feeds,
                             const std::vector<std::string>& /*fetch_names*/,
                             const std::vector<OrtValue>& fetches,
                             size_t /*step*/) {
    const Tensor& label_t = feeds[1].Get<Tensor>();
    const Tensor& predict_t = fetches[0].Get<Tensor>();
    const Tensor& loss_t = fetches[1].Get<Tensor>();

    const float* prediction_data = predict_t.template Data<float>();
    const float* label_data = label_t.template Data<float>();
    const float* loss_data = loss_t.template Data<float>();

    const TensorShape predict_shape = predict_t.Shape();
    const TensorShape label_shape = label_t.Shape();
    const TensorShape loss_shape = loss_t.Shape();
    ORT_ENFORCE(predict_shape == label_shape);

    int64_t batch_size = predict_shape[0];
    for (int n = 0; n < batch_size; ++n) {
      auto max_class_index = std::distance(prediction_data,
                                           std::max_element(prediction_data, prediction_data + NUM_CLASS));
      printf("Actual label: %d\n", (int) max_class_index);
      printf("Expected label: %d\n", (int) std::distance(label_data,
                                           std::max_element(label_data, label_data + NUM_CLASS)));
      if (static_cast<int>(label_data[max_class_index]) == 1) {
        true_count++;
      }

      prediction_data += predict_shape.SizeFromDimension(1);
      label_data += label_shape.SizeFromDimension(1);
    }
    total_loss += *loss_data;
  };

  std::shared_ptr<EventWriter> tensorboard;
  if (!params.log_dir.empty() && MPIContext::GetInstance().GetWorldRank() == 0)
    tensorboard = std::make_shared<EventWriter>(params.log_dir);

  params.post_evaluation_callback = [tensorboard](size_t num_samples, size_t step, const std::string /*tag*/) {
    float precision = float(true_count) / num_samples;
    float average_loss = total_loss / float(num_samples);
    if (tensorboard != nullptr) {
      tensorboard->AddScalar("precision", precision, step);
      tensorboard->AddScalar("loss", average_loss, step);
    }
    printf("Step: %zu, #examples: %d, #correct: %d, precision: %0.04f, loss: %0.04f \n\n",
           step,
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };
}

std::vector<float> preprocessImage(const std::string& path) {

  int dimX, dimY, numChannels;
  unsigned char* data = stbi_load(path.c_str(), &dimX, &dimY, &numChannels, 0);
  unsigned char* orig_data = data;

  if (data == nullptr) {
    printf("Could not load image\n");
    exit(-1);
  }
  if (numChannels != 3) {
    printf("Loaded image has more than 3 channels. Use JPG instead of PNG\n");
    exit(-1);
  }

  std::vector<float> input_tensor_values(IMG_SIZE);

  for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {
      unsigned char r = *(data++);
      unsigned char g = *(data++);
      unsigned char b = *(data++);
      // TODO: ADD A CASE FOR PYTORCH MODEL EXPORT
      input_tensor_values[(0 * 224 + i) * 224 + j] = (b / 255.0 - 0.406) / 0.225;
      input_tensor_values[(1 * 224 + i) * 224 + j] = (g / 255.0 - 0.456) / 0.224;
      input_tensor_values[(2 * 224 + i) * 224 + j] = (r / 255.0 - 0.485) / 0.229;
    }
  }
  stbi_image_free(orig_data);
  return input_tensor_values;
}

int getLabelOfBatchImage(const std::string &path) {
  size_t lastidx = path.find_last_of("/\\");
  size_t secondlastidx = path.find_last_of("/\\", lastidx - 1);
  return std::stoi(path.substr(secondlastidx + 1, (lastidx - secondlastidx - 1)));
}

int getTotalNumberLines(string file) {
    std::ifstream myfile(file);
    myfile.unsetf(std::ios_base::skipws);
    return std::count(
        std::istream_iterator<char>(myfile),
        std::istream_iterator<char>(), 
        '\n');
}

pair<vector<vector<float>>, vector<vector<float>>> getImageAndLabels(string batch_txt, int start, int upTo) {
    std::ifstream batch_in(batch_txt);
    std::string path;

    vector<vector<float>> normalized_images;
    normalized_images.reserve(upTo);
    vector<vector<float>> one_hot_labels;
    one_hot_labels.reserve(upTo);

    int startLine = start;
    int endLine = upTo;
    int curLine = 0;
    while (std::getline(batch_in, path)) {
      curLine += 1;
      if (curLine < startLine || (endLine != -1 && curLine > endLine)) {
        continue;
      }
      normalized_images.push_back(preprocessImage(path));
      int expected_label = getLabelOfBatchImage(path);
      vector<float> one_hot(NUM_CLASS);
      ORT_ENFORCE(expected_label < NUM_CLASS, "Expected label idx out of bounds");
      one_hot[expected_label] = 1;
      one_hot_labels.push_back(one_hot);
    }
    return std::make_pair(normalized_images, one_hot_labels);
}

void ConvertData(const vector<vector<float>>& images,
                 const vector<vector<float>>& labels,
                 const vector<int64_t>& image_dims,
                 const vector<int64_t>& label_dims,
                 DataSet& data_set,
                 size_t shard_index = 0,
                 size_t total_shard = 1) {
  for (size_t i = 0; i < images.size(); ++i) {
    if (i % total_shard == shard_index) {
      MLValue imageMLValue;
      TrainingUtil::CreateCpuMLValue(image_dims, images[i], &imageMLValue);
      MLValue labelMLValue;
      TrainingUtil::CreateCpuMLValue(label_dims, labels[i], &labelMLValue);

      data_set.AddData(make_unique<vector<MLValue>>(vector<MLValue>{imageMLValue, labelMLValue}));
    }
  }
}

int main(int argc, char* args[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");

#ifdef FOR_FIRESIM
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  } else {
    printf("Finished mlockall\n");
  }
#endif

  MnistParameters params;
  RETURN_IF_FAIL(ParseArguments(argc, args, params));

  printf("Setting up logger\n");
  // setup logger
  string default_logger_id{"Default"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                                           static_cast<logging::Severity>(params.debug),
                                                                           false,
                                                                           logging::LoggingManager::InstanceType::Default,
                                                                           &default_logger_id);

  // setup onnxruntime env
  printf("Setting up env\n");
  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(std::move(logging_manager), env).IsOK());

  // setup training params
  printf("Setting up training params\n");
  setup_training_params(params);

  // setup data
  printf("Setting up data\n");

  int totalNumLines = params.num_samples == -1 ? getTotalNumberLines(params.train_data_dir) : params.num_samples;
  ORT_ENFORCE(totalNumLines >= 2, "Must have at least two total images");
  printf("Total number of lines in batch txt %d\n", totalNumLines);
  auto training_images_and_labels = getImageAndLabels(params.train_data_dir, 1, totalNumLines/2);
  auto testing_images_and_labels = getImageAndLabels(params.train_data_dir, totalNumLines/2 + 1, totalNumLines);
  printf("Training batch size %zd\n", training_images_and_labels.first.size());
  printf("Testing batch size %zd\n", testing_images_and_labels.first.size());
  printf("First few image values %f %f %f\n", training_images_and_labels.first[0][0], training_images_and_labels.first[0][1], training_images_and_labels.first[0][2]);
  printf("First label %d\n", (int) std::distance(training_images_and_labels.second[0].begin(),
                                           std::max_element(training_images_and_labels.second[0].begin(), training_images_and_labels.second[0].end())));


  auto device_count = MPIContext::GetInstance().GetWorldSize();
  std::vector<string> feeds{"gpu_0/data_0", "labels"};
  auto trainingData = std::make_shared<DataSet>(feeds);
  auto testData = std::make_shared<DataSet>(feeds);


  size_t shard_index = MPIContext::GetInstance().GetWorldRank();
  size_t total_shard = device_count;
  ORT_ENFORCE(shard_index < total_shard, "shard_index must be 0~", total_shard - 1);

  ConvertData(training_images_and_labels.first, training_images_and_labels.second, IMAGE_DIMS, LABEL_DIMS, *trainingData,
              shard_index, total_shard);
  ConvertData(testing_images_and_labels.first, testing_images_and_labels.second, IMAGE_DIMS, LABEL_DIMS, *testData,
              shard_index, total_shard);

  if (testData->NumSamples() == 0) {
    printf("Warning: No data loaded - run cancelled.\n");
    return -1;
  }

  SessionOptions session_options;
  session_options.intra_op_param = {1};
  session_options.graph_optimization_level = (TransformerLevel)params.optimization_level;
  session_options.execution_order = ExecutionOrder::PRIORITY_BASED;

  printf("Creating training runner\n");
  auto training_data_loader = std::make_shared<SingleDataLoader>(trainingData, feeds);
  auto test_data_loader = std::make_shared<SingleDataLoader>(testData, feeds);
  auto runner = std::make_unique<TrainingRunner>(params, *env, session_options);
  printf("Initializing training runner\n");
  RETURN_IF_FAIL(runner->Initialize());
  printf("Starting training\n");
  RETURN_IF_FAIL(runner->Run(training_data_loader.get(), test_data_loader.get()));
  RETURN_IF_FAIL(runner->EndTraining(test_data_loader.get()));
}
