#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>

#include <BGAL/Model/ManifoldModel.h>
#include <BGAL/QuadCoverLike/QuadCover.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static std::filesystem::path guess_data_root(const std::filesystem::path &exe_path) {
  namespace fs = std::filesystem;
  const fs::path cwd = fs::current_path();
  const fs::path exe_dir = fs::canonical(exe_path).parent_path();

  if (fs::exists(cwd / "data")) return fs::canonical(cwd / "data");
  if (fs::exists(exe_dir.parent_path() / "data")) return fs::canonical(exe_dir.parent_path() / "data");
  if (fs::exists(exe_dir / "data")) return fs::canonical(exe_dir / "data");
  return {};
}

struct CliOptions {
  std::filesystem::path workdir;
  std::string model = "block";
  std::filesystem::path input_obj;
  bool show_help = false;
};

static void print_usage(const char *exe_name) {
  std::cout
      << "Usage:\n"
      << "  " << exe_name
      << " [--workdir DIR] [--model NAME] [--input FILE]\n\n"
      << "Options:\n"
      << "  --workdir DIR   Change working directory before resolving paths.\n"
      << "  --model NAME    Use data/NAME.obj as the target surface. Default: block\n"
      << "  --input FILE    Use FILE as the initialization OBJ.\n"
      << "  -h, --help      Show this help message.\n\n"
      << "Legacy positional form is still supported:\n"
      << "  " << exe_name << " [workdir] [model] [input_obj]\n";
}

static bool parse_args(int argc, char **argv, CliOptions &opts) {
  std::vector<std::string> positional;

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
    if (arg == "--workdir") {
      const char *value = need_value("--workdir");
      if (!value) return false;
      opts.workdir = value;
      continue;
    }
    if (arg == "--model") {
      const char *value = need_value("--model");
      if (!value) return false;
      opts.model = value;
      continue;
    }
    if (arg == "--input") {
      const char *value = need_value("--input");
      if (!value) return false;
      opts.input_obj = value;
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "Error: unknown option " << arg << "\n";
      return false;
    }
    positional.push_back(arg);
  }

  if (positional.size() > 3) {
    std::cerr << "Error: too many positional arguments.\n";
    return false;
  }
  if (!positional.empty() && opts.workdir.empty()) opts.workdir = positional[0];
  if (positional.size() >= 2 && opts.model == "block") opts.model = positional[1];
  if (positional.size() >= 3 && opts.input_obj.empty()) opts.input_obj = positional[2];
  return true;
}

int main(int argc, char **argv) {
  namespace fs = std::filesystem;

#ifdef _OPENMP
  omp_set_dynamic(0);
  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const int max_threads = std::max<int>(1, hw_threads > 0 ? (int)hw_threads
                                                          : omp_get_num_procs());
  omp_set_num_threads(max_threads);
  Eigen::setNbThreads(max_threads);
#endif

  CliOptions options;
  if (!parse_args(argc, argv, options)) {
    print_usage((argc > 0) ? argv[0] : "quadcover_main");
    return 1;
  }
  if (options.show_help) {
    print_usage((argc > 0) ? argv[0] : "quadcover_main");
    return 0;
  }

  if (!options.workdir.empty()) {
    std::error_code ec;
    fs::current_path(options.workdir, ec);
    if (ec) {
      std::cerr << "Warning: failed to chdir to " << options.workdir
                << " (" << ec.message() << ")\n";
    }
  }

  const std::string &model = options.model;
  fs::path input_obj_override = options.input_obj;

  fs::path data_root = guess_data_root((argc > 0) ? fs::path(argv[0]) : fs::path());
  if (data_root.empty()) {
    std::cerr << "IOError: cannot locate 'data/' directory near CWD or executable.\n";
    return 1;
  }
  const fs::path project_root = data_root.parent_path();

  fs::path obj_path = data_root / (model + ".obj");
  if (!fs::exists(obj_path)) {
    std::cerr << "IOError: " << obj_path << " does not exist.\n";
    return 1;
  }

  fs::path init_obj_path = input_obj_override.empty() ? obj_path : input_obj_override;
  if (!init_obj_path.is_absolute()) {
    init_obj_path = fs::current_path() / init_obj_path;
  }
  if (!fs::exists(init_obj_path)) {
    std::cerr << "IOError: init obj " << init_obj_path << " does not exist.\n";
    return 1;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if (!igl::readOBJ(obj_path.string(), V, F)) {
    std::cerr << "IOError: " << obj_path << " could not be opened (igl::readOBJ failed).\n";
    return 1;
  }

  Eigen::MatrixXd init_V;
  Eigen::MatrixXi init_F;
  if (!igl::readOBJ(init_obj_path.string(), init_V, init_F)) {
    std::cerr << "IOError: " << init_obj_path << " could not be opened (igl::readOBJ failed).\n";
    return 1;
  }
  if (init_V.rows() == 0) {
    std::cerr << "IOError: " << init_obj_path << " has no vertices.\n";
    return 1;
  }

  std::vector<BGAL::_Point3> init_sites;
  init_sites.reserve(init_V.rows());
  for (int i = 0; i < init_V.rows(); ++i) {
    init_sites.emplace_back(init_V(i, 0), init_V(i, 1), init_V(i, 2));
  }

  const fs::path temp_off_path = project_root / "Temp.off";
  const fs::path temp_obj_path = project_root / "Temp.obj";
  igl::writeOFF(temp_off_path.string(), V, F);
  igl::writeOBJ(temp_obj_path.string(), V, F);

  BGAL::_ManifoldModel model_mesh(temp_obj_path.string());
  BGAL::_QuadCover3D::_Parameter para;
  para.is_show = true;
  para.export_each_iteration = true;
  para.max_outer_iterations = 800;
  para.max_line_search = 10;
  para.active_eps = 1e-8;
  para.step_cap_scale = 0.02;

  BGAL::_QuadCover3D solver(model_mesh, para);
  solver.set_outpath(project_root.generic_string());
  solver.calculate_(init_sites, model);
  return 0;
}
