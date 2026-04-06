#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

struct BatchOptions {
  fs::path input_dir = "ABC";
  fs::path output_dir = "quadResult";
  int jobs = 0;
  int threads_per_job = 0;
  bool show_help = false;
};

struct RunResult {
  std::string model_name;
  fs::path model_path;
  fs::path log_path;
  fs::path result_dir;
  int exit_code = -1;
  bool success = false;
};

static void print_usage(const char *exe_name) {
  std::cout
      << "Usage:\n"
      << "  " << exe_name
      << " [--input-dir ABC] [--output-dir quadResult] [--jobs N] [--threads-per-job N]\n\n"
      << "Options:\n"
      << "  --input-dir DIR        Folder containing OBJ/OFF models. Default: ABC\n"
      << "  --output-dir DIR       Output root for logs and final results. Default: quadResult\n"
      << "  --jobs N               Number of models processed in parallel.\n"
      << "  --threads-per-job N    Threads used by each quadcover_main process.\n"
      << "  -h, --help             Show this help message.\n";
}

static bool parse_args(int argc, char **argv, BatchOptions &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Error: missing value for " << flag << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      opts.show_help = true;
      return true;
    }
    if (arg == "--input-dir") {
      const char *value = need_value("--input-dir");
      if (!value) return false;
      opts.input_dir = value;
      continue;
    }
    if (arg == "--output-dir") {
      const char *value = need_value("--output-dir");
      if (!value) return false;
      opts.output_dir = value;
      continue;
    }
    if (arg == "--jobs") {
      const char *value = need_value("--jobs");
      if (!value) return false;
      opts.jobs = std::max(0, std::stoi(value));
      continue;
    }
    if (arg == "--threads-per-job") {
      const char *value = need_value("--threads-per-job");
      if (!value) return false;
      opts.threads_per_job = std::max(0, std::stoi(value));
      continue;
    }
    std::cerr << "Error: unknown option " << arg << "\n";
    return false;
  }
  return true;
}

static std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\"'\"'";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

static bool is_supported_mesh(const fs::path &path) {
  if (!fs::is_regular_file(path)) return false;
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext == ".obj" || ext == ".off";
}

static int default_jobs_for_machine() {
  const unsigned int hw = std::thread::hardware_concurrency();
  if (hw == 0) return 1;
  return std::max(1u, std::min(4u, hw));
}

int main(int argc, char **argv) {
  BatchOptions options;
  if (!parse_args(argc, argv, options)) {
    print_usage((argc > 0) ? argv[0] : "quadcover_batch");
    return 1;
  }
  if (options.show_help) {
    print_usage((argc > 0) ? argv[0] : "quadcover_batch");
    return 0;
  }

  fs::path input_dir = fs::absolute(options.input_dir);
  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    std::cerr << "IOError: input directory " << input_dir << " does not exist.\n";
    return 1;
  }

  fs::path output_dir = fs::absolute(options.output_dir);
  fs::path log_dir = output_dir / "quadCover";
  fs::create_directories(output_dir);
  fs::create_directories(log_dir);

  std::vector<fs::path> model_paths;
  for (const auto &entry : fs::recursive_directory_iterator(input_dir)) {
    if (is_supported_mesh(entry.path())) {
      model_paths.push_back(fs::absolute(entry.path()));
    }
  }
  std::sort(model_paths.begin(), model_paths.end());

  if (model_paths.empty()) {
    std::cerr << "IOError: no OBJ/OFF models found in " << input_dir << ".\n";
    return 1;
  }

  const unsigned int hw = std::thread::hardware_concurrency();
  const int jobs = options.jobs > 0
                       ? options.jobs
                       : std::min<int>(default_jobs_for_machine(),
                                       (int)model_paths.size());
  const int threads_per_job =
      options.threads_per_job > 0
          ? options.threads_per_job
          : std::max(1, (int)(hw > 0 ? hw : 1) / std::max(1, jobs));

  fs::path quadcover_main_path =
      fs::weakly_canonical(fs::path(argv[0])).parent_path() / "quadcover_main";
  if (!fs::exists(quadcover_main_path)) {
    std::cerr << "IOError: cannot locate quadcover_main next to " << argv[0]
              << ".\n";
    return 1;
  }

  std::cout << "Batch start: models=" << model_paths.size()
            << " jobs=" << jobs
            << " threads-per-job=" << threads_per_job << std::endl;

  std::vector<RunResult> results(model_paths.size());
  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  workers.reserve(std::max(1, jobs));

  auto worker = [&]() {
    while (true) {
      const size_t index = next_index.fetch_add(1);
      if (index >= model_paths.size()) break;

      const fs::path &model_path = model_paths[index];
      const std::string model_name = model_path.stem().string();
      const fs::path model_result_dir = output_dir / model_name;
      const fs::path log_path = log_dir / (model_name + ".log");
      fs::create_directories(model_result_dir);

      std::string command =
          shell_quote(quadcover_main_path.string()) + " --surface " +
          shell_quote(model_path.string()) + " --input " +
          shell_quote(model_path.string()) + " --name " +
          shell_quote(model_name) + " --output " +
          shell_quote(model_result_dir.string()) + " --threads " +
          std::to_string(threads_per_job) + " --final-only > " +
          shell_quote(log_path.string()) + " 2>&1";

      RunResult result;
      result.model_name = model_name;
      result.model_path = model_path;
      result.log_path = log_path;
      result.result_dir = model_result_dir;
      result.exit_code = std::system(command.c_str());
      result.success = (result.exit_code == 0);
      results[index] = std::move(result);
    }
  };

  for (int i = 0; i < std::max(1, jobs); ++i) {
    workers.emplace_back(worker);
  }
  for (auto &thread : workers) {
    thread.join();
  }

  const fs::path summary_path = output_dir / "summary.csv";
  std::ofstream summary(summary_path, std::ios::out | std::ios::trunc);
  summary << "model,status,exit_code,model_path,log_path,result_dir\n";

  int failures = 0;
  for (const auto &result : results) {
    if (!result.success) {
      ++failures;
    }
    summary << result.model_name << ","
            << (result.success ? "success" : "failed") << ","
            << result.exit_code << ","
            << result.model_path.string() << ","
            << result.log_path.string() << ","
            << result.result_dir.string() << "\n";
  }

  std::cout << "Batch finished: success=" << (results.size() - failures)
            << " failed=" << failures
            << " summary=" << summary_path << std::endl;
  return failures == 0 ? 0 : 2;
}
