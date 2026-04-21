#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从原始 OBJ 开始，自动执行：
1) 导出“处理好后的模型”（通过 BGAL::_ManifoldModel 读入并 save_obj_file_ 导出）
2) 按用户提供的 vcglib 采样方法，对处理后的模型采样 k 个点（输出 xyz+normal）
3) 执行 CWF (_CVT3D)
4) 执行 QEM + hinge loss (_QuadCover3D)

说明：
- 这是一个 Python 驱动脚本，但它会自动生成并编译 3 个小的 C++ helper：
  a) export_processed_mesh
  b) poisson_sample_xyz
  c) run_cwf_then_quadcover
- 这样做的原因是你的核心逻辑都在 C++ 库里，Python 这里负责把完整流程串起来。
- 采样部分复用了你给的 vcglib Montecarlo + Poisson pruning 思路。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List


EXPORT_PROCESSED_MESH_CPP = r'''
#include <filesystem>
#include <iostream>
#include <string>

#include "BGAL/Model/ManifoldModel.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.obj output.obj\n";
        return 1;
    }

    const std::string input_obj = argv[1];
    const std::string output_obj = argv[2];

    try {
        std::filesystem::create_directories(std::filesystem::path(output_obj).parent_path());
        BGAL::_ManifoldModel model(input_obj);
        model.save_obj_file_(output_obj);
        std::cout << "Saved processed mesh to: " << output_obj << "\n";
        std::cout << "Vertices: " << model.number_vertices_() << "\n";
        std::cout << "Faces   : " << model.number_faces_() << "\n";
        std::cout << "Has nonmanifold topology flag: " << model.has_nonmanifold_topology_() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[export_processed_mesh] Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
'''


POISSON_SAMPLE_XYZ_CPP = r'''
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/bounding.h>

#include <wrap/io_trimesh/import_obj.h>

class MyVertex;
class MyFace;
class MyEdge;

struct MyUsedTypes : public vcg::UsedTypes<
    vcg::Use<MyVertex>::AsVertexType,
    vcg::Use<MyEdge>::AsEdgeType,
    vcg::Use<MyFace>::AsFaceType> {};

class MyVertex : public vcg::Vertex<
    MyUsedTypes,
    vcg::vertex::Coord3f,
    vcg::vertex::Normal3f,
    vcg::vertex::BitFlags> {};

class MyFace : public vcg::Face<
    MyUsedTypes,
    vcg::face::VertexRef,
    vcg::face::Normal3f,
    vcg::face::BitFlags> {};

class MyEdge : public vcg::Edge<
    MyUsedTypes> {};

class MyMesh : public vcg::tri::TriMesh<
    std::vector<MyVertex>,
    std::vector<MyFace>,
    std::vector<MyEdge>> {};

namespace {

size_t CountLiveVertices(const MyMesh& mesh) {
    size_t count = 0;
    for (const auto& v : mesh.vert) {
        if (!v.IsD()) {
            ++count;
        }
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " input.obj output.xyz sample_num [seed]\n\n"
                  << "Example:\n"
                  << "  " << argv[0]
                  << " bunny.obj samples.xyz 10000 12345\n";
        return 1;
    }

    const std::string input_obj  = argv[1];
    const std::string output_xyz = argv[2];
    const int sample_num         = std::atoi(argv[3]);
    const unsigned int seed      = (argc >= 5) ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 0u;

    if (sample_num <= 0) {
        std::cerr << "sample_num must be > 0\n";
        return 1;
    }

    MyMesh mesh;

    int load_mask = 0;
    int err = vcg::tri::io::ImporterOBJ<MyMesh>::Open(mesh, input_obj.c_str(), load_mask);
    if (err != 0) {
        std::cerr << "Failed to load OBJ: " << input_obj << "\n";
        std::cerr << "Error code: " << err << "\n";
        std::cerr << "Error msg : " << vcg::tri::io::ImporterOBJ<MyMesh>::ErrorMsg(err) << "\n";
        return 1;
    }

    if (mesh.fn == 0 || mesh.vn == 0) {
        std::cerr << "Mesh is empty or has no valid triangles.\n";
        return 1;
    }

    vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalizedPerFaceNormalized(mesh);
    vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

    MyMesh montecarlo_mesh;
    MyMesh sample_mesh;

    using MeshSampler = vcg::tri::MeshSampler<MyMesh>;
    using SurfaceSampler = vcg::tri::SurfaceSampling<MyMesh, MeshSampler>;

    MeshSampler mc_sampler(montecarlo_mesh);
    MeshSampler output_sampler(sample_mesh);

    const int fast_candidate_num = std::max(10000, sample_num * 12);
    const int fallback_candidate_num = std::max(10000, sample_num * 40);
    const float pruning_tolerance = 0.04f;
    const size_t min_acceptable_samples = static_cast<size_t>(
        std::ceil(sample_num * (1.0f - pruning_tolerance))
    );

    MyMesh::ScalarType radius = vcg::tri::ComputePoissonDiskRadius(mesh, sample_num);
    SurfaceSampler::PoissonDiskParam pp;
    pp.pds.sampleNum = sample_num;
    pp.randomSeed = seed;

    if (seed) {
        SurfaceSampler::SamplingRandomGenerator().initialize(seed);
    }

    SurfaceSampler::Montecarlo(mesh, mc_sampler, fast_candidate_num);
    vcg::tri::UpdateBounding<MyMesh>::Box(montecarlo_mesh);

    bool fast_path_succeeded = false;
    size_t produced_samples = 0;

    for (int iter = 0; iter < 3; ++iter) {
        output_sampler.reset();
        SurfaceSampler::PoissonDiskPruning(output_sampler, montecarlo_mesh, radius, pp);
        produced_samples = CountLiveVertices(sample_mesh);

        if (produced_samples >= min_acceptable_samples) {
            fast_path_succeeded = true;
            break;
        }

        if (produced_samples == 0) {
            radius *= 0.5f;
            continue;
        }

        const float ratio = std::sqrt(
            static_cast<float>(produced_samples) / static_cast<float>(sample_num)
        );
        radius *= std::max(0.5f, std::min(0.95f, ratio * 0.98f));
    }

    if (!fast_path_succeeded) {
        montecarlo_mesh.Clear();
        sample_mesh.Clear();

        SurfaceSampler::Montecarlo(mesh, mc_sampler, fallback_candidate_num);
        vcg::tri::UpdateBounding<MyMesh>::Box(montecarlo_mesh);

        radius = 0.0f;
        SurfaceSampler::PoissonDiskPruningByNumber(
            output_sampler,
            montecarlo_mesh,
            sample_num,
            radius,
            pp,
            pruning_tolerance
        );
    }

    std::ofstream ofs(output_xyz);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_xyz << "\n";
        return 1;
    }

    ofs.setf(std::ios::fixed);
    ofs.precision(8);

    size_t written_samples = 0;
    for (const auto& v : sample_mesh.vert) {
        if (v.IsD()) {
            continue;
        }

        if (written_samples >= static_cast<size_t>(sample_num)) {
            break;
        }

        auto n = v.cN();
        if (n.SquaredNorm() > 0) {
            n.Normalize();
        }

        ofs << v.cP()[0] << ' ' << v.cP()[1] << ' ' << v.cP()[2] << ' '
            << n[0] << ' ' << n[1] << ' ' << n[2] << '\n';
        ++written_samples;
    }

    ofs.close();

    std::cout << "Input mesh     : " << input_obj << "\n";
    std::cout << "Vertices       : " << mesh.vn << "\n";
    std::cout << "Faces          : " << mesh.fn << "\n";
    std::cout << "Sample number  : " << written_samples << "\n";
    std::cout << "Estimated r    : " << radius << "\n";
    std::cout << "Saved to       : " << output_xyz << "\n";

    return 0;
}
'''


RUN_CWF_THEN_QUADCOVER_CPP = r'''
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
                  << " processed.obj init.xyz output_dir model_name [cwf_iters=50] [quad_outer_iters=30] [export_each_iter=0] [export_interval=50] [cwf_export_process=0]\n";
        return 1;
    }

    const std::string processed_obj = argv[1];
    const std::string init_xyz = argv[2];
    const std::string output_dir = argv[3];
    const std::string model_name = argv[4];
    const int cwf_iters = (argc >= 6) ? std::stoi(argv[5]) : 50;
    const int quad_outer_iters = (argc >= 7) ? std::stoi(argv[6]) : 30;
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
'''


TOPLEVEL_CMAKELISTS = r'''
cmake_minimum_required(VERSION 3.17)
project(BGALPipelineTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED BGAL_SRC_DIR)
  message(FATAL_ERROR "BGAL_SRC_DIR is not set")
endif()

if(NOT DEFINED VCGLIB_DIR)
  message(FATAL_ERROR "VCGLIB_DIR is not set")
endif()

find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(libigl CONFIG QUIET)
if(NOT TARGET igl::igl_core)
  add_library(igl_core_stub INTERFACE)
  add_library(igl::igl_core ALIAS igl_core_stub)
endif()

# 让下游 BGAL 子模块里使用的 ${PROJECT_SOURCE_DIR}/include 指到当前 workspace/include。
# 这个 include 目录由 Python 脚本预先从实际工程复制/同步过来。
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/include")
  message(FATAL_ERROR "workspace/include does not exist")
endif()

add_subdirectory("${BGAL_SRC_DIR}/src" "${CMAKE_BINARY_DIR}/bgal_src_build")

add_executable(export_processed_mesh tools/export_processed_mesh.cpp)
target_link_libraries(export_processed_mesh PRIVATE Model)
target_include_directories(export_processed_mesh PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_executable(poisson_sample_xyz tools/poisson_sample_xyz.cpp)
target_include_directories(poisson_sample_xyz PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${VCGLIB_DIR}"
  "${VCGLIB_DIR}/eigenlib"
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(poisson_sample_xyz PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(run_cwf_then_quadcover tools/run_cwf_then_quadcover.cpp)
target_link_libraries(run_cwf_then_quadcover PRIVATE QuadCoverLike CVTLike Model)
target_include_directories(run_cwf_then_quadcover PRIVATE "${PROJECT_SOURCE_DIR}/include")

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(run_cwf_then_quadcover PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()
'''


def run(cmd: List[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def copy_include_tree(project_root: Path, workspace: Path, force: bool) -> None:
    src_include = project_root / "include"
    dst_include = workspace / "include"
    if not src_include.exists():
        raise FileNotFoundError(f"BGAL include dir not found: {src_include}")

    if force and dst_include.exists():
        shutil.rmtree(dst_include)

    if not dst_include.exists():
        shutil.copytree(src_include, dst_include)


def exe_path(build_dir: Path, name: str) -> Path:
    if os.name == "nt":
        release = build_dir / "Release" / f"{name}.exe"
        if release.exists():
            return release
        return build_dir / f"{name}.exe"
    return build_dir / name


def configure_and_build(
    workspace: Path,
    project_root: Path,
    vcglib_dir: Path,
    cmake_prefix_paths: List[str],
    extra_cmake_args: List[str],
    build_type: str,
    force_configure: bool,
) -> Path:
    copy_include_tree(project_root, workspace, force=force_configure)
    write_text(workspace / "CMakeLists.txt", TOPLEVEL_CMAKELISTS)
    write_text(workspace / "tools" / "export_processed_mesh.cpp", EXPORT_PROCESSED_MESH_CPP)
    write_text(workspace / "tools" / "poisson_sample_xyz.cpp", POISSON_SAMPLE_XYZ_CPP)
    write_text(workspace / "tools" / "run_cwf_then_quadcover.cpp", RUN_CWF_THEN_QUADCOVER_CPP)

    build_dir = workspace / "build"
    cmake_args = [
        "cmake",
        "-S", str(workspace),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DBGAL_SRC_DIR={project_root}",
        f"-DVCGLIB_DIR={vcglib_dir}",
    ]
    if cmake_prefix_paths:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={';'.join(cmake_prefix_paths)}")
    cmake_args.extend(extra_cmake_args)

    if force_configure or not (build_dir / "CMakeCache.txt").exists():
        run(cmake_args)
    run(["cmake", "--build", str(build_dir), "--config", build_type, "-j"])
    return build_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="处理模型 -> 采样 k 点 -> CWF -> QEM+hinge 的完整测试流程")
    p.add_argument("--project-root", required=True, help="BGAL 工程根目录（里面应有 include/ 和 src/）")
    p.add_argument("--vcglib-dir", required=True, help="vcglib 根目录")
    p.add_argument("--input-obj", required=True, help="原始输入 OBJ")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--k", type=int, required=True, help="采样点数")
    p.add_argument("--model-name", default="model", help="输出命名前缀")
    p.add_argument("--seed", type=int, default=0, help="采样随机种子")
    p.add_argument("--cwf-iters", type=int, default=50, help="CWF 迭代次数")
    p.add_argument("--quad-iters", type=int, default=30, help="QEM+hinge 外层迭代次数")
    p.add_argument("--export-each-iter", action="store_true", help="QuadCover 每若干步导出一次")
    p.add_argument("--export-interval", type=int, default=50, help="QuadCover 导出间隔")
    p.add_argument("--cwf-export-process", action="store_true", help="是否导出 CWF 过程")
    p.add_argument("--processed-obj", default="", help="若已知处理后的模型路径，可直接传入并跳过导出")
    p.add_argument("--workspace", default="", help="构建 helper 的工作目录；默认 output-dir/_pipeline_workspace")
    p.add_argument("--build-type", default="Release", choices=["Release", "RelWithDebInfo", "Debug", "MinSizeRel"])
    p.add_argument("--cmake-prefix-path", action="append", default=[], help="追加到 CMAKE_PREFIX_PATH 的路径，可重复传多次")
    p.add_argument("--cmake-arg", action="append", default=[], help="额外透传给 cmake configure 的参数，可重复传多次")
    p.add_argument("--force-configure", action="store_true", help="强制重新同步 include/ 并重新 configure")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    vcglib_dir = Path(args.vcglib_dir).resolve()
    input_obj = Path(args.input_obj).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = Path(args.workspace).resolve() if args.workspace else (output_dir / "_pipeline_workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    build_dir = configure_and_build(
        workspace=workspace,
        project_root=project_root,
        vcglib_dir=vcglib_dir,
        cmake_prefix_paths=args.cmake_prefix_path,
        extra_cmake_args=args.cmake_arg,
        build_type=args.build_type,
        force_configure=args.force_configure,
    )

    export_exe = exe_path(build_dir, "export_processed_mesh")
    sampler_exe = exe_path(build_dir, "poisson_sample_xyz")
    pipeline_exe = exe_path(build_dir, "run_cwf_then_quadcover")

    processed_obj = Path(args.processed_obj).resolve() if args.processed_obj else (output_dir / f"{args.model_name}_processed.obj")
    sampled_xyz = output_dir / f"{args.model_name}_sampled_{args.k}.xyz"

    if not args.processed_obj:
        run([str(export_exe), str(input_obj), str(processed_obj)])
    else:
        print(f"[INFO] skip export, use processed mesh: {processed_obj}")

    run([str(sampler_exe), str(processed_obj), str(sampled_xyz), str(args.k), str(args.seed)])

    run([
        str(pipeline_exe),
        str(processed_obj),
        str(sampled_xyz),
        str(output_dir),
        str(args.model_name),
        str(args.cwf_iters),
        str(args.quad_iters),
        "1" if args.export_each_iter else "0",
        str(args.export_interval),
        "1" if args.cwf_export_process else "0",
    ])

    manifest = {
        "input_obj": str(input_obj),
        "processed_obj": str(processed_obj),
        "sampled_xyz": str(sampled_xyz),
        "cwf_points_xyz": str(output_dir / f"{args.model_name}_cwf_points.xyz"),
        "quadcover_points_xyz": str(output_dir / f"{args.model_name}_quadcover_points.xyz"),
        "output_dir": str(output_dir),
        "workspace": str(workspace),
        "build_dir": str(build_dir),
    }
    manifest_path = output_dir / f"{args.model_name}_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[DONE] pipeline finished.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] subprocess failed with return code {e.returncode}", file=sys.stderr)
        raise
