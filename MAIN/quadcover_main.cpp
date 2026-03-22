#include <filesystem>
#include <iostream>
#include <vector>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>

#include <BGAL/Model/ManifoldModel.h>
#include <BGAL/QuadCoverLike/QuadCover.h>

static std::filesystem::path guess_data_root(const std::filesystem::path &exe_path) {
  namespace fs = std::filesystem;
  const fs::path cwd = fs::current_path();
  const fs::path exe_dir = fs::canonical(exe_path).parent_path();

  if (fs::exists(cwd / "data")) return fs::canonical(cwd / "data");
  if (fs::exists(exe_dir.parent_path() / "data")) return fs::canonical(exe_dir.parent_path() / "data");
  if (fs::exists(exe_dir / "data")) return fs::canonical(exe_dir / "data");
  return {};
}

int main(int argc, char **argv) {
  namespace fs = std::filesystem;

  std::string model = (argc >= 3) ? argv[2] : "block";
  int N = (argc >= 4) ? std::max(1, std::atoi(argv[3])) :8000;

  if (argc >= 2) {
    std::error_code ec;
    fs::current_path(fs::path(argv[1]), ec);
    if (ec) {
      std::cerr << "Warning: failed to chdir to " << argv[1] << " (" << ec.message() << ")\n";
    }
  }

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

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if (!igl::readOBJ(obj_path.string(), V, F)) {
    std::cerr << "IOError: " << obj_path << " could not be opened (igl::readOBJ failed).\n";
    return 1;
  }

  const fs::path temp_off_path = project_root / "Temp.off";
  const fs::path temp_obj_path = project_root / "Temp.obj";
  igl::writeOFF(temp_off_path.string(), V, F);
  igl::writeOBJ(temp_obj_path.string(), V, F);

  BGAL::_ManifoldModel model_mesh(temp_obj_path.string());
  BGAL::_QuadCover3D::_Parameter para;
  para.is_show = true;
  para.export_each_iteration = true;
  para.max_outer_iterations = 500;
  para.max_line_search = 10;
  para.active_eps = 1e-8;
  para.step_cap_scale = 0.02;

  BGAL::_QuadCover3D solver(model_mesh, para);
  solver.set_outpath(project_root.generic_string());
  solver.calculate_(N, (char *)model.c_str(),(char*)"/home/yiming/research/CWF/data/Block/Ours_8000_block_Iter52_Points.xyz");
  return 0;
}
