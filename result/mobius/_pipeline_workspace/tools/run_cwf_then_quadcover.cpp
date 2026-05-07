#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "BGAL/BaseShape/Point.h"
#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Model/ManifoldModel.h"
#include "BGAL/Optimization/LBFGS/LBFGS.h"
#include "BGAL/QuadCoverLike/QuadCover.h"

namespace {

std::vector<BGAL::_Point3> load_xyz_points(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in) {
        throw std::runtime_error("failed to open xyz file: " + filepath);
    }
    std::vector<BGAL::_Point3> pts;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(iss >> x >> y >> z)) continue;
        pts.emplace_back(x, y, z);
    }
    return pts;
}

void save_xyz_points(const std::string& filepath, const std::vector<BGAL::_Point3>& pts) {
    std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
    std::ofstream out(filepath);
    if (!out) {
        throw std::runtime_error("failed to open output xyz file: " + filepath);
    }
    out.setf(std::ios::fixed);
    out.precision(17);
    for (const auto& p : pts) {
        out << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " processed.obj init.xyz output_dir model_name [cwf_iters=50] [quad_outer_iters=750] [export_each_iter=0] [export_interval=50] [cwf_export_process=0]\n";
        return 1;
    }

    const std::string processed_obj = argv[1];
    const std::string init_xyz = argv[2];
    const std::string output_dir = argv[3];
    const std::string model_name = argv[4];
    const int cwf_iters = (argc >= 6) ? std::stoi(argv[5]) : 50;
    const int quad_outer_iters = (argc >= 7) ? std::stoi(argv[6]) : 750;
    const bool export_each_iter = (argc >= 8) ? (std::stoi(argv[7]) != 0) : false;
    const int export_interval = (argc >= 9) ? std::stoi(argv[8]) : 50;
    const bool cwf_export_process = (argc >= 10) ? (std::stoi(argv[9]) != 0) : false;

    try {
        std::filesystem::create_directories(output_dir);

        BGAL::_ManifoldModel model(processed_obj);
        auto init_sites = load_xyz_points(init_xyz);
        if (init_sites.empty()) {
            throw std::runtime_error("init xyz contains no valid points: " + init_xyz);
        }

        std::function<double(BGAL::_Point3&)> rho = [] (BGAL::_Point3& p) {
            (void)p;
            return 1.0;
        };

        BGAL::_LBFGS::_Parameter cvt_para;
        cvt_para.is_show = true;
        cvt_para.epsilon = 1e-30;
        cvt_para.max_linearsearch = 20;
        cvt_para.max_iteration = cwf_iters;

        BGAL::_CVT3D cvt(model, rho, cvt_para);
        cvt.set_outpath(output_dir + "/");
        cvt.calculate_(init_sites, model_name, cwf_export_process);
        const auto& cwf_sites = cvt.get_sites();

        const std::string cwf_xyz = (std::filesystem::path(output_dir) / (model_name + "_cwf_points.xyz")).string();
        save_xyz_points(cwf_xyz, cwf_sites);
        std::cout << "Saved CWF points to: " << cwf_xyz << "\n";

        BGAL::_QuadCover3D::_Parameter quad_para;
        quad_para.is_show = true;
        quad_para.export_initial_state = true;
        quad_para.export_each_iteration = export_each_iter;
        quad_para.export_interval = export_interval;
        quad_para.use_cwf_warm_start = false;
        quad_para.show_cwf_progress = false;
        quad_para.cwf_max_iterations = 0;
        quad_para.max_outer_iterations = quad_outer_iters;

        BGAL::_QuadCover3D quad(model, quad_para);
        quad.set_outpath(output_dir);
        quad.calculate_(cwf_sites, model_name);

        const auto& final_sites = quad.get_sites();
        const std::string quad_xyz = (std::filesystem::path(output_dir) / (model_name + "_quadcover_points.xyz")).string();
        save_xyz_points(quad_xyz, final_sites);
        std::cout << "Saved QuadCover points to: " << quad_xyz << "\n";

        const auto& hist = quad.get_history();
        std::cout << "QuadCover iterations recorded: " << hist.size() << "\n";
        if (!hist.empty()) {
            const auto& last = hist.back();
            std::cout << "Last iter summary | iter=" << last.iteration
                      << " quads=" << last.num_quads
                      << " active=" << last.active_quads
                      << " min_margin=" << last.min_margin
                      << " step=" << last.accepted_step << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[run_cwf_then_quadcover] Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
