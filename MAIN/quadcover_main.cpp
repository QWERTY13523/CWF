#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>

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
  std::filesystem::path surface_path;
  std::string model = "block";
  std::filesystem::path input_obj;
  std::filesystem::path output_dir;
  std::string model_name_override;
  int threads = 0;
  bool final_only = false;
  bool show_help = false;
};

static void print_usage(const char *exe_name) {
  std::cout
      << "Usage:\n"
      << "  " << exe_name
      << " [--workdir DIR] [--model NAME] [--surface FILE] [--input FILE]\n"
      << "                 [--output DIR] [--name NAME] [--threads N] [--final-only]\n\n"
      << "Options:\n"
      << "  --workdir DIR   Change working directory before resolving paths.\n"
      << "  --model NAME    Use data/NAME.obj as the target surface. Default: block\n"
      << "  --surface FILE  Use FILE as the target surface mesh.\n"
      << "  --input FILE    Use FILE as the initialization OBJ.\n"
      << "  --output DIR    Directory for QuadCover result files.\n"
      << "  --name NAME     Output basename. Defaults to the surface stem.\n"
      << "  --threads N     Number of threads used inside one run.\n"
      << "  --final-only    Export only the final QuadCover result.\n"
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
    if (arg == "--surface") {
      const char *value = need_value("--surface");
      if (!value) return false;
      opts.surface_path = value;
      continue;
    }
    if (arg == "--input") {
      const char *value = need_value("--input");
      if (!value) return false;
      opts.input_obj = value;
      continue;
    }
    if (arg == "--output") {
      const char *value = need_value("--output");
      if (!value) return false;
      opts.output_dir = value;
      continue;
    }
    if (arg == "--name") {
      const char *value = need_value("--name");
      if (!value) return false;
      opts.model_name_override = value;
      continue;
    }
    if (arg == "--threads") {
      const char *value = need_value("--threads");
      if (!value) return false;
      opts.threads = std::max(0, std::stoi(value));
      continue;
    }
    if (arg == "--final-only") {
      opts.final_only = true;
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

static bool load_triangle_mesh_file(const std::filesystem::path &path,
                                    Eigen::MatrixXd &V,
                                    Eigen::MatrixXi &F) {
  if (!igl::read_triangle_mesh(path.string(), V, F)) {
    return false;
  }
  return V.rows() > 0 && F.rows() > 0;
}

static std::string to_lower_copy(std::string value) {
  for (char &c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

static bool load_xyz_sites_file(const std::filesystem::path &path,
                                std::vector<BGAL::_Point3> &sites) {
  std::ifstream input(path);
  if (!input) return false;

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(iss >> x >> y >> z)) continue;
    sites.emplace_back(x, y, z);
  }
  return !sites.empty();
}

static bool load_obj_sites_file(const std::filesystem::path &path,
                                std::vector<BGAL::_Point3> &sites) {
  std::ifstream input(path);
  if (!input) return false;

  std::string line;
  while (std::getline(input, line)) {
    if (line.size() < 2 || line[0] != 'v' ||
        !std::isspace(static_cast<unsigned char>(line[1]))) {
      continue;
    }
    std::istringstream iss(line.substr(1));
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(iss >> x >> y >> z)) continue;
    sites.emplace_back(x, y, z);
  }
  return !sites.empty();
}

static bool load_off_sites_file(const std::filesystem::path &path,
                                std::vector<BGAL::_Point3> &sites) {
  std::ifstream input(path);
  if (!input) return false;

  std::string token;
  input >> token;
  if (token != "OFF" && token != "COFF") return false;

  int num_vertices = 0;
  int num_faces = 0;
  int num_edges = 0;
  if (!(input >> num_vertices >> num_faces >> num_edges) || num_vertices <= 0) {
    return false;
  }

  sites.reserve(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(input >> x >> y >> z)) return false;
    if (token == "COFF") {
      std::string rest_of_line;
      std::getline(input, rest_of_line);
    }
    sites.emplace_back(x, y, z);
  }
  return !sites.empty();
}

static bool load_init_sites_file(const std::filesystem::path &path,
                                 std::vector<BGAL::_Point3> &sites) {
  const std::string ext = to_lower_copy(path.extension().string());
  if (ext == ".xyz" || ext == ".txt" || ext == ".pts") {
    return load_xyz_sites_file(path, sites);
  }
  if (ext == ".obj") {
    return load_obj_sites_file(path, sites);
  }
  if (ext == ".off") {
    return load_off_sites_file(path, sites);
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if (!igl::read_triangle_mesh(path.string(), V, F) || V.rows() == 0) {
    return false;
  }
  sites.reserve(V.rows());
  for (int i = 0; i < V.rows(); ++i) {
    sites.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  }
  return !sites.empty();
}

static BGAL::_ManifoldModel build_manifold_model(const Eigen::MatrixXd &V,
                                                 const Eigen::MatrixXi &F) {
  std::vector<BGAL::_Point3> vertices;
  vertices.reserve(V.rows());
  for (int i = 0; i < V.rows(); ++i) {
    vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  }

  std::vector<BGAL::_Model::_MFace> faces;
  faces.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i) {
    if (F.cols() != 3) {
      throw std::runtime_error("surface mesh is not triangulated.");
    }
    faces.emplace_back(F(i, 0), F(i, 1), F(i, 2));
  }
  return BGAL::_ManifoldModel(vertices, faces);
}

int main(int argc, char **argv) {
  namespace fs = std::filesystem;

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

  const std::string model =
      options.model_name_override.empty()
          ? (!options.surface_path.empty() ? options.surface_path.stem().string()
                                           : options.model)
          : options.model_name_override;
  fs::path input_obj_override = options.input_obj;

  fs::path surface_path;
  fs::path output_dir = options.output_dir;
  std::optional<fs::path> project_root;

  if (!options.surface_path.empty()) {
    surface_path = options.surface_path;
    if (!surface_path.is_absolute()) {
      surface_path = fs::current_path() / surface_path;
    }
    if (output_dir.empty()) {
      output_dir = fs::current_path() / "data" / "QuadCover";
    }
  } else {
    fs::path data_root =
        guess_data_root((argc > 0) ? fs::path(argv[0]) : fs::path());
    if (data_root.empty()) {
      std::cerr << "IOError: cannot locate 'data/' directory near CWD or executable.\n";
      return 1;
    }
    project_root = data_root.parent_path();
    surface_path = data_root / (options.model + ".obj");
    if (output_dir.empty()) {
      output_dir = *project_root / "data" / "QuadCover";
    }
  }

  if (!fs::exists(surface_path)) {
    std::cerr << "IOError: surface mesh " << surface_path << " does not exist.\n";
    return 1;
  }

  fs::path init_obj_path = input_obj_override.empty() ? surface_path : input_obj_override;
  if (!init_obj_path.is_absolute()) {
    init_obj_path = fs::current_path() / init_obj_path;
  }
  if (!fs::exists(init_obj_path)) {
    std::cerr << "IOError: init obj " << init_obj_path << " does not exist.\n";
    return 1;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if (!load_triangle_mesh_file(surface_path, V, F)) {
    std::cerr << "IOError: " << surface_path
              << " could not be opened (igl::read_triangle_mesh failed).\n";
    return 1;
  }

  std::vector<BGAL::_Point3> init_sites;
  if (input_obj_override.empty()) {
    init_sites.reserve(V.rows());
    for (int i = 0; i < V.rows(); ++i) {
      init_sites.emplace_back(V(i, 0), V(i, 1), V(i, 2));
    }
  } else if (!load_init_sites_file(init_obj_path, init_sites)) {
    std::cerr << "IOError: " << init_obj_path
              << " could not be parsed as init sites (.xyz/.obj/.off supported).\n";
    return 1;
  }
  if (init_sites.empty()) {
    std::cerr << "IOError: " << init_obj_path << " has no valid init sites.\n";
    return 1;
  }

#ifdef _OPENMP
  omp_set_dynamic(0);
  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const int max_threads =
      options.threads > 0
          ? options.threads
          : std::max<int>(1, hw_threads > 0 ? (int)hw_threads
                                            : omp_get_num_procs());
  omp_set_num_threads(max_threads);
  Eigen::setNbThreads(max_threads);
#endif

  BGAL::_ManifoldModel model_mesh = build_manifold_model(V, F);
  BGAL::_QuadCover3D::_Parameter para;
  para.is_show = true;
  para.export_initial_state = !options.final_only;
  para.export_each_iteration = !options.final_only;
  para.max_outer_iterations = 800;
  para.max_line_search = 10;
  para.active_eps = 1e-8;
  para.step_cap_scale = 0.02;

  BGAL::_QuadCover3D solver(model_mesh, para);
  solver.set_outpath(output_dir.string());
  solver.calculate_(init_sites, model);
  return 0;
}
