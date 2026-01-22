#include <iostream>
#include <filesystem>
#include <random>
#include <functional>

#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>

#include <BGAL/Optimization/LBFGS/LBFGS.h>
#include <BGAL/Integral/Integral.h>
#include <BGAL/Model/ManifoldModel.h>
#include <BGAL/Model/Model_Iterator.h>
#include <BGAL/Tessellation3D/Tessellation3D.h>
#include <BGAL/CVTLike/CVT.h>

// ---- 小工具：找 data 目录，并生成 Temp 路径 ----
static std::filesystem::path guess_data_root(const std::filesystem::path& exe_path)
{
    namespace fs = std::filesystem;
    const fs::path cwd = fs::current_path();
    const fs::path exe_dir = fs::canonical(exe_path).parent_path();

    // 1) CWD/data
    if (fs::exists(cwd / "data")) return fs::canonical(cwd / "data");
    // 2) <exe_dir>/../data  （通常 bin/MAIN 的上一级是工程根）
    if (fs::exists(exe_dir.parent_path() / "data")) return fs::canonical(exe_dir.parent_path() / "data");
    // 3) <exe_dir>/data
    if (fs::exists(exe_dir / "data")) return fs::canonical(exe_dir / "data");

    return {}; // not found
}

void CWF3DTest(const std::filesystem::path& exe_path, int Nums = 8000, std::string file = "block")
{
    namespace fs = std::filesystem;

    std::cout << "CWD: " << fs::current_path() << "\n";
    std::cout << "====================CWF3DTest\n";
    std::cout << "Now file: " << file << "\n";
    std::cout << Nums << "   " << file << "   \n";

    // 1) 找 data 目录
    fs::path data_root = guess_data_root(exe_path);
    if (data_root.empty()) {
        std::cerr << "IOError: cannot locate 'data/' directory near CWD or executable.\n";
        return;
    }

    // 2) 组成 OBJ 路径并校验
    fs::path obj_path = data_root / (file + ".obj");
    if (!fs::exists(obj_path)) {
        std::cerr << "IOError: " << obj_path << " does not exist.\n";
        return;
    }
    std::cout << "OBJ: " << obj_path << "\n";   

    // 3) 读取 OBJ
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(obj_path.string(), V, F)) {
        std::cerr << "IOError: " << obj_path << " could not be opened (igl::readOBJ failed).\n";
        return;
    }

    igl::writeOFF("Temp.off", V, F);
    igl::writeOBJ("Temp.obj", V, F);

    // 5) 模型与 CVT
    BGAL::_ManifoldModel model("Temp.obj");

    std::function<double(BGAL::_Point3&)> rho = [](BGAL::_Point3&) {
        return 1.0;
    };

    BGAL::_LBFGS::_Parameter para;
    para.is_show = true;
    para.epsilon = 1e-30;
    para.max_iteration = 50;

    BGAL::_CVT3D cvt(model, rho, para);

    int num = Nums;
    cvt.calculate_(num, (char*)file.c_str());
}

int main(int argc, char** argv)
{
    // 允许从命令行指定：  MAIN [data_root] [model_name] [num_sites]
    std::string model = (argc >= 3) ? argv[2] : "block";
    int N = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 8000;

    // 如果用户显式传 data_root，就先把 CWD 切过去（这样 Temp/输出更直观）
    if (argc >= 2) {
        std::error_code ec;
        std::filesystem::current_path(std::filesystem::path(argv[1]), ec);
        if (ec) {
            std::cerr << "Warning: failed to chdir to " << argv[1] << " (" << ec.message() << ")\n";
        }
    }
    CWF3DTest((argc > 0) ? std::filesystem::path(argv[0]) : "", N, model);
    return 0;
}
