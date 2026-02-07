#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>

#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Integral/Integral.h"
#include "BGAL/Sphere/Sphere.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/squared_distance_3.h>
#include <cmath>
#include <Eigen/Sparse>
#include <memory>
#include <igl/gaussian_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/principal_curvature.h>
#include <igl/adjacency_list.h>
#include <igl/avg_edge_length.h>
#include <numeric>

#include "BGAL/BaseShape/KDTree.h"

typedef CGAL::Simple_cartesian<double> K_T;
typedef K_T::FT FT;
typedef K_T::Point_3 Point_T;

typedef K_T::Segment_3 Segment;
typedef CGAL::Polyhedron_3<K_T> Polyhedron;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits_3<K_T, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;
double kgammaTol = 0.00000000000001;
struct MyPoint {
  MyPoint(Eigen::Vector3d a) { p = a; }

  MyPoint(double a, double b, double c) {
    p.x() = a;
    p.y() = b;
    p.z() = c;
  }
  Eigen::Vector3d p;

  bool operator<(const MyPoint &a) const {

    double dis = (p - a.p).norm();
    if (dis < kgammaTol) {
      return false;
    }

    if ((p.x() - a.p.x()) < 0.00000000001 &&
        (p.x() - a.p.x()) > -0.00000000001) {
      if ((p.y() - a.p.y()) < 0.00000000001 &&
          (p.y() - a.p.y()) > -0.00000000001) {
        return (p.z() < a.p.z());
      }
      return (p.y() < a.p.y());
    }
    return (p.x() < a.p.x());
  }
  bool operator==(const MyPoint &a) const {
    if ((p.x() - a.p.x()) < 0.00000000001 &&
        (p.x() - a.p.x()) > -0.00000000001) {
      if ((p.y() - a.p.y()) < 0.00000000001 &&
          (p.y() - a.p.y()) > -0.00000000001) {
        if ((p.z() - a.p.z()) < 0.00000000001 &&
            (p.z() - a.p.z()) > -0.00000000001) {
          return true;
        }
      }
    }
    return false;
  }
};

struct MyFace {
  MyFace(Eigen::Vector3i a) { p = a; }
  MyFace(int a, int b, int c) {
    p.x() = a;
    p.y() = b;
    p.z() = c;
  }
  Eigen::Vector3i p;
  bool operator<(const MyFace &a) const {
    if (p.x() == a.p.x()) {
      if (p.y() == a.p.y()) {
        return p.z() > a.p.z();
      }
      return p.y() > a.p.y();
    }
    return p.x() > a.p.x();
  }
};

namespace BGAL {
void OutputMesh(std::vector<_Point3> &sites, _Restricted_Tessellation3D RVD,
                int num, std::string outpath, std::string modelname, int step) {
  const std::vector<std::vector<std::tuple<int, int, int>>> &cells =
      RVD.get_cells_();
  std::string filepath =
      outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
  if (step == 2) {
    filepath =
        outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
  }

  if (step > 2) {
    filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname +
               "_Iter" + std::to_string(step - 3) + "_RVD.obj";
  }
  std::cout << "filepath = " << filepath << std::endl;
  std::ofstream out(filepath);
  out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar"
      << std::endl;
  for (int i = 0; i < RVD.number_vertices_(); ++i) {
    out << "v " << RVD.vertex_(i) << std::endl;
  }
  double totarea = 0, parea = 0;
  for (int i = 0; i < cells.size(); ++i) {
    double area = 0;
    for (int j = 0; j < cells[i].size(); ++j) {
      BGAL::_Point3 p1 = RVD.vertex_(std::get<0>(cells[i][j]));
      BGAL::_Point3 p2 = RVD.vertex_(std::get<1>(cells[i][j]));
      BGAL::_Point3 p3 = RVD.vertex_(std::get<2>(cells[i][j]));
      area += (p2 - p1).cross_(p3 - p1).length_() / 2;
    }
    totarea += area;

    auto color = (double)BGAL::_BOC::rand_();
    if (i > cells.size() / 3) {
      if (step == 1) {
        color = 0;
      }
      //
    } else {
      parea += area;
    }

    out << "vt " << color << " 0" << std::endl;

    for (int j = 0; j < cells[i].size(); ++j) {
      out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1 << " "
          << std::get<1>(cells[i][j]) + 1 << "/" << i + 1 << " "
          << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
    }
  }
  out.close();

  filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname +
             "_Points.xyz";
  if (step == 2) {
    filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname +
               "_Points.xyz";
  }

  if (step > 2) {
    filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname +
               "_Iter" + std::to_string(step - 3) + "_Points.xyz";
  }

  std::ofstream outP(filepath);

  int outnum = sites.size();
  if (step == 1)
    outnum = sites.size() / 3;

  for (int i = 0; i < outnum; ++i) {
    outP << sites[i] << std::endl;
  }
  outP.close();

  if (step >= 2) {
    std::string filepath = outpath + "/Ours_" + std::to_string(num) + "_" +
                           modelname + "_Remesh.obj";

    std::string filepath1 = outpath + "/Ours_" + std::to_string(num) + "_" +
                            modelname + "_Iter" + std::to_string(step - 3) +
                            "_Remesh.obj";
    std::ofstream outRDT(filepath);
    std::ofstream outRDT1(filepath1);

    auto Vs = sites;
    auto Edges = RVD.get_edges_();
    std::set<std::pair<int, int>> RDT_Edges;
    std::vector<std::set<int>> neibors;
    neibors.resize(Vs.size());
    for (int i = 0; i < Edges.size(); i++) {
      for (const auto ee : Edges[i]) {
        RDT_Edges.insert(
            std::make_pair(std::min(i, ee.first), std::max(i, ee.first)));
        neibors[i].insert(ee.first);
        neibors[ee.first].insert(i);
        // std::cout << ee.first << std::endl;
      }
    }

    for (auto v : Vs) {
      if (step >= 2)
        outRDT << "v " << v << std::endl;
      outRDT1 << "v " << v << std::endl;
    }

    std::set<MyFace> rdtFaces;

    for (auto e : RDT_Edges) {
      for (int pid : neibors[e.first]) {
        if (RDT_Edges.find(std::make_pair(std::min(pid, e.first),
                                          std::max(pid, e.first))) !=
            RDT_Edges.end()) {
          if (RDT_Edges.find(std::make_pair(std::min(pid, e.second),
                                            std::max(pid, e.second))) !=
              RDT_Edges.end()) {
            int f1 = pid, f2 = e.first, f3 = e.second;

            int mid;
            if (f1 != std::max(f1, std::max(f2, f3)) &&
                f1 != std::min(f1, min(f2, f3))) {
              mid = f1;
            }
            if (f2 != std::max(f1, std::max(f2, f3)) &&
                f2 != std::min(f1, std::min(f2, f3))) {
              mid = f2;
            }
            if (f3 != std::max(f1, max(f2, f3)) &&
                f3 != std::min(f1, min(f2, f3))) {
              mid = f3;
            }
            rdtFaces.insert(MyFace(std::max(f1, std::max(f2, f3)), mid,
                                   std::min(f1, std::min(f2, f3))));
          }
        }
      }
    }
    for (auto f : rdtFaces) {
      if (step >= 2)
        outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " "
               << f.p.z() + 1 << std::endl;
      outRDT1 << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1
              << std::endl;
    }

    outRDT.close();
    outRDT1.close();
  }
}

_CVT3D::_CVT3D(const _ManifoldModel &model)
    : _model(model), _RVD(model), _RVD2(model), _para() {
  _para.is_show = true;
  _para.epsilon = 1e-30;
  _para.max_linearsearch = 20;
  // 默认密度函数，稍后会被覆盖
  _rho = [this](BGAL::_Point3 &p) { return 1; };

  std::shared_ptr<std::vector<double>> density_field =
      std::make_shared<std::vector<double>>();
  std::shared_ptr<_KDTree> kdtree;

  {
    const int nv = _model.number_vertices_();
    const int nf = _model.number_faces_();

    Eigen::MatrixXd V(nv, 3);
    Eigen::MatrixXi F(nf, 3);
    std::vector<_Point3> pts;
    pts.reserve(nv);

    for (int i = 0; i < nv; ++i) {
      const auto &p = _model.vertex_(i);
      V(i, 0) = p.x();
      V(i, 1) = p.y();
      V(i, 2) = p.z();
      pts.push_back(p);
    }
    for (int fi = 0; fi < nf; ++fi) {
      const auto &f = _model.face_(fi);
      F(fi, 0) = f[0];
      F(fi, 1) = f[1];
      F(fi, 2) = f[2];
    }

    // -------------------------------------------------------------
    // 基于面法向量夹角的特征检测
    // 对每个顶点，计算其相邻面法向量之间的最大夹角
    // 在特征边(棱边/角点)处，相邻面法向量变化剧烈，夹角大
    // 在平坦区域，相邻面法向量几乎相同，夹角接近0
    // -------------------------------------------------------------

    // 1. 计算每个面的法向量
    std::vector<Eigen::Vector3d> face_normals(nf);
    for (int fi = 0; fi < nf; ++fi) {
      Eigen::Vector3d v0 = V.row(F(fi, 0));
      Eigen::Vector3d v1 = V.row(F(fi, 1));
      Eigen::Vector3d v2 = V.row(F(fi, 2));
      Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
      double len = n.norm();
      if (len > 1e-15) n /= len;
      face_normals[fi] = n;
    }

    // 2. 建立顶点到相邻面的映射
    std::vector<std::vector<int>> vert_faces(nv);
    for (int fi = 0; fi < nf; ++fi) {
      vert_faces[F(fi, 0)].push_back(fi);
      vert_faces[F(fi, 1)].push_back(fi);
      vert_faces[F(fi, 2)].push_back(fi);
    }

    // 3. 对每个顶点，计算相邻面法向量之间的最大夹角
    std::vector<double> max_angle(nv, 0.0);
    for (int i = 0; i < nv; ++i) {
      const auto &adj_faces = vert_faces[i];
      for (int a = 0; a < (int)adj_faces.size(); ++a) {
        for (int b = a + 1; b < (int)adj_faces.size(); ++b) {
          double cos_val = face_normals[adj_faces[a]].dot(face_normals[adj_faces[b]]);
          cos_val = std::max(-1.0, std::min(1.0, cos_val));
          double angle = std::acos(cos_val); // [0, π]
          if (angle > max_angle[i]) max_angle[i] = angle;
        }
      }
    }

    // 4. 构建密度场
    // base_density: 平坦区域的基础密度
    // feature_weight: 特征边密度倍数
    // 阈值 angle_thresh: 小于此角度视为平坦
    density_field->resize(nv);
    double base_density = 1.0;
    double feature_weight = 10.0; // [可调] 特征边比平坦区域密多少倍
    double angle_thresh = 0.1;    // ~6度，低于此角度视为平坦

    for (int i = 0; i < nv; ++i) {
      double a = max_angle[i];
      if (a < angle_thresh) a = 0.0;
      // 用角度的平方，使特征区域与平坦区域区分更明显
      double w = a / M_PI; // 归一化到 [0, 1]
      (*density_field)[i] = base_density + feature_weight * (w * w);
    }

    kdtree = std::make_shared<_KDTree>(pts);
  }

  const double eps = _para.epsilon;
  // 更新 lambda 函数，使用处理好的 density_field
  _rho = std::function<double(BGAL::_Point3&)>([density_field, kdtree, eps](BGAL::_Point3 &p) {
    const int vid = kdtree->search_(p);
    if (vid < 0 || vid >= (int)density_field->size()) {
      return eps; // Fallback
    }
    // 注意：这里返回的值直接作为 CVT 的密度权重
    return (*density_field)[vid] + eps;
  });
}
_CVT3D::_CVT3D(const _ManifoldModel &model,
               std::function<double(_Point3 &p)> &rho, _LBFGS::_Parameter para)
    : _model(model), _RVD(model), _RVD2(model), _rho(rho), _para(para) {}
void _CVT3D::calculate_(int num_sites, char *modelNamee, char *pointsName) {

  double allTime = 0, RVDtime = 0;
  clock_t start, end;
  clock_t startRVD, endRVD;
  std::cout << std::filesystem::current_path() << std::endl;

  double PI = 3.14159265359;
  std::string modelname = modelNamee;
  Polyhedron polyhedron;
  std::ifstream input("Temp.off");
  input >> polyhedron;
  Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);
  std::unordered_map<const void *, int> face_id_map;
  {
    int idx = 0;
    for (auto fit = polyhedron.facets_begin(); fit != polyhedron.facets_end();
         ++fit, ++idx) {
      face_id_map[static_cast<const void *>(&*fit)] = idx; // 0-based
    }
  }
  double Movement = 0.01;
  std::string inPointsName;
  namespace fs = std::filesystem;
  fs::path obj("/home/yiming/research/CWF/data/block.obj");
  fs::path base = obj.parent_path();
  if (pointsName == nullptr) {
    inPointsName = base / ("n" + std::to_string(num_sites) + "_" + modelname +
                           "_inputPoints.xyz");
  } else {
    inPointsName = pointsName;
  }
  std::ifstream inPoints(inPointsName.c_str());
  std::vector<Eigen::Vector3d> Pts, Nors;

  int count = 0;
  double x, y, z, nx, ny, nz; // if xyz file has normal
  while (inPoints >> x >> y >> z >> nx >> ny >> nz) {
    Pts.push_back(Eigen::Vector3d(x, y, z));
    Nors.push_back(
        Eigen::Vector3d(nx, ny, nz)); // Nors here is useless, if do not have
                                      // normal, just set it to (1,0,0)
    ++count;
  }
  inPoints.close();
  std::cout << "Pts.size(): " << Pts.size() << std::endl;

  if (pointsName != nullptr) {
    num_sites = static_cast<int>(Pts.size());
    ;
  }
  // begin step 1.
  int num = Pts.size();

  std::vector<Eigen::Vector3d> Pts3;
  std::cout << "\nBegin CWF.\n" << std::endl;

  int Fnum = 4;
  double alpha = 1.0, eplison = 1,
         lambda = 0; // eplison is CVT weight,  lambda is qe weight.
  double decay = 1;
  std::vector<int> FaceIDs;
  FaceIDs.assign(num, -1);
  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm2 =
      [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
        eplison = eplison * decay;
        double lossCVT = 0, lossQE = 0, loss = 0;
        double lossCenter = 0.0;

        startRVD = clock();
        std::vector<Sphere::Sphere> spheres(num);
        for (int i = 0; i < num; ++i) {
          Point_T query(X(i * 3), X(i * 3 + 1),
                        X(i * 3 + 2)); // project to base surface
          Point_T closest = tree.closest_point(query);
          auto tri = tree.closest_point_and_primitive(query);
          Polyhedron::Face_handle f = tri.second;
          int fid = -1;
          auto it = face_id_map.find(static_cast<const void *>(&*f));
          if (it != face_id_map.end())
            fid = it->second;
          FaceIDs[i] = fid;
          auto p1 = f->halfedge()->vertex()->point();
          auto p2 = f->halfedge()->next()->vertex()->point();
          auto p3 = f->halfedge()->next()->next()->vertex()->point();
          Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
          Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
          Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
          Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
          N.normalize();
          Nors[i] = N;
          BGAL::_Point3 p(closest.x(), closest.y(), closest.z());
          _sites[i] = p;
        }
        _RVD.calculate_(_sites);
        Fnum++;
        if (Fnum % 1 == 0) {
          OutputMesh(_sites, _RVD, num_sites,
                     std::filesystem::current_path() / "data" / "Block",
                     modelname, Fnum);
        }
        endRVD = clock();
        RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

        const std::vector<std::vector<std::tuple<int, int, int>>> &cells =
            _RVD.get_cells_();
        const std::vector<std::map<int, std::vector<std::pair<int, int>>>>
            &edges = _RVD.get_edges_();

        double energy = 0.0;
        g.setZero();
        std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());

        omp_set_num_threads(30); // change to your CPU core numbers
#pragma omp parallel for reduction(+ : lossCVT, loss, lossCenter)
        for (int i = 0; i < num; ++i) {
          _Point3 site = _sites[i];
          Eigen::Vector3d xi(site.x(), site.y(), site.z());

          // ----------------------
          // 积分与梯度（原逻辑不变）
          // ----------------------
          for (int j = 0; j < (int)cells[i].size(); ++j) {
            auto [a, b, c] = cells[i][j];
            _Point3 pa = _RVD.vertex_(a), pb = _RVD.vertex_(b),
                    pc = _RVD.vertex_(c);

            Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
                [&](BGAL::_Point3 p) {
                  Eigen::VectorXd r(5);

                  BGAL::_Point3 NorTriM =
                      (_RVD.vertex_(std::get<1>(cells[i][j])) -
                       _RVD.vertex_(std::get<0>(cells[i][j])))
                          .cross_(_RVD.vertex_(std::get<2>(cells[i][j])) -
                                  _RVD.vertex_(std::get<0>(cells[i][j])));

                  // 确保真正归一化（若 normalized_
                  // 为返回新向量的版本，可改为：NorTriM =
                  // NorTriM.normalized_();）
                  NorTriM.normalized_();

                  double rho_val = _rho(p);
                  r(0) = (eplison * rho_val *
                          ((_sites[i] - p).sqlength_())); // CVT
                  r(1) = lambda * (NorTriM.dot_(p - _sites[i])) *
                             (NorTriM.dot_(p - _sites[i])) +
                         eplison * rho_val *
                             ((p - _sites[i])
                                  .sqlength_()); // QE + CVT with density

                  r(2) = lambda * -2 * NorTriM.x() *
                             (NorTriM.dot_(p - _sites[i])) +
                         eplison * rho_val * -2 * (p - _sites[i]).x(); // gx
                  r(3) = lambda * -2 * NorTriM.y() *
                             (NorTriM.dot_(p - _sites[i])) +
                         eplison * rho_val * -2 * (p - _sites[i]).y(); // gy
                  r(4) = lambda * -2 * NorTriM.z() *
                             (NorTriM.dot_(p - _sites[i])) +
                         eplison * rho_val * -2 * (p - _sites[i]).z(); // gz

                  return r;
                },
                pa, pb, pc);

            lossCVT += alpha * inte(0);
            loss += alpha * inte(1);
            gi[i].x() += alpha * inte(2);
            gi[i].y() += alpha * inte(3);
            gi[i].z() += alpha * inte(4);
          }

          // -----------------------------------------
          // 求“定心外接球”的最远点：仅在 cell 边界顶点集合上搜索
          // -----------------------------------------
          int best_vid = -1;
          double x = site.x(), y = site.y(),
                 z = site.z(); // 默认回退为 site 本身

          std::unordered_set<int> bnd;
          if (i < (int)edges.size()) {
            for (const auto &kv : edges[i]) {
              for (const auto &e : kv.second) {
                bnd.insert(e.first);
                bnd.insert(e.second);
              }
            }
          }

          std::vector<BGAL::_Point3> boundary_pts;
          boundary_pts.reserve(bnd.size() + cells[i].size() * 3);

          if (!bnd.empty()) {
            for (int vid : bnd) {
              const auto &pv = _RVD.vertex_(vid);
              boundary_pts.emplace_back(pv.x(), pv.y(), pv.z());
            }
          }
          // auto consider_vid = [&](int vid) {
          //     const auto& pv = _RVD.vertex_(vid);
          //     const double dx = site.x() - pv.x();
          //     const double dy = site.y() - pv.y();
          //     const double dz = site.z() - pv.z();
          //     const double d2 = dx*dx + dy*dy + dz*dz; // 用距离平方比较稳定
          //     if (d2 > best_d2 || (d2 == best_d2 && vid < best_vid)) {
          //         best_d2 = d2; best_vid = vid;
          //         x = pv.x(); y = pv.y(); z = pv.z();
          //     }
          // };
          //
          // if (!bnd.empty())
          // {
          //  for (int vid : bnd) consider_vid(vid);
          // }
          // } else {
          //     // 罕见兜底：边界集合缺失时，退回到 cell 的顶点集合（去重）
          //     std::unordered_set<int> used;
          //     used.reserve(cells[i].size()*3);
          //     for (int j = 0; j < (int)cells[i].size(); ++j) {
          //         int a,b,c; std::tie(a,b,c) = cells[i][j];
          //         if (used.insert(a).second) consider_vid(a);
          //         if (used.insert(b).second) consider_vid(b);
          //         if (used.insert(c).second) consider_vid(c);
          //     }
          // }
          // if (Fnum >= 37 && !boundary_pts.empty())
          // {
          // 	// 1. 计算当前 cell 边界点的最小外接球
          // 	BGAL::MEBall B = BGAL::minimum_enclosing_ball(boundary_pts);
          //
          // 	// 2. 把最小外接球球心投影到原始三角网表面
          // 	//    Point_T 就是你前面用来投影 site 的 CGAL 点类型
          // 	Point_T bc_query(B.c.x(), B.c.y(), B.c.z());
          // 	Point_T bc_proj  = tree.closest_point(bc_query);
          //
          // 	// 3. 能量：site 到这个"投影点"的距离平方
          // 	double dx = site.x() - bc_proj.x();
          // 	double dy = site.y() - bc_proj.y();
          // 	double dz = site.z() - bc_proj.z();
          // 	double Ei_center = dx*dx + dy*dy + dz*dz;
          //
          // 	// 4. 累加能量和梯度（把投影点当常量，和你之前对 B.c
          // 的处理一致） 	lossCenter += eplison * Ei_center; gi[i].x()
          // += 2.0 * eplison * dx; 	gi[i].y()  += 2.0 * eplison * dy;
          // gi[i].z()  += 2.0 * eplison * dz;
          // }
          BGAL::_Point3 farp = site;
          double best_d2 = -1.0;
          for (const auto &p : boundary_pts) {
            double d2 = (p - site).length_();
            if (d2 > best_d2) {
              best_d2 = d2;
              farp = p;
            }
          }

          spheres[i].c = decltype(spheres[i].c)(site.x(), site.y(), site.z());
          spheres[i].r = best_d2;
          spheres[i].max_point =
              decltype(spheres[i].max_point)(farp.x(), farp.y(), farp.z());

        } // end omp for

        for (int i = 0; i < num; i++) {
          gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
          g(i * 3) += gi[i].x();
          g(i * 3 + 1) += gi[i].y();
          g(i * 3 + 2) += gi[i].z();
        }
        energy += loss;

        std::cout << std::setprecision(7) << "energy: " << energy
                  << " LossCVT: " << lossCVT / eplison
                  << " LossQE: " << (loss - lossCVT) / lambda
                  << " Lambda_CVT: " << eplison << std::endl;

        namespace fs = std::filesystem;
        fs::path file =
            fs::absolute(fs::current_path() / "data" / "Block" /
                         ("Sphere_6000_" + std::to_string(Fnum) + ".csv"));
        // //fs::path file1 =
        //     fs::absolute(fs::current_path() / "data" / "Block" / "Max_point" /
        //                  ("MaxPoint_6000_" + std::to_string(Fnum) + ".xyz"));
        std::cerr << "[io] cwd   = " << fs::current_path().string() << "\n";
        std::cerr << "[io] write = " << file.string() << "\n";

        std::ofstream out(file, std::ios::out | std::ios::trunc);
        if (!out) {
          std::cerr << "[io] open failed, errno=" << errno << " ("
                    << std::strerror(errno) << ")\n";
        } else {
          out << std::setprecision(17);
          for (int i = 0; i < (int)spheres.size(); ++i) {
            Eigen::Vector3d n = Nors[i].normalized();
            out << spheres[i].c.x() << "," << spheres[i].c.y() << ","
                << spheres[i].c.z() << "," << spheres[i].r << "," << FaceIDs[i]
                << "," << n.x() << "," << n.y() << "," << n.z() << "\n";
          }
          out.close();
          if (!out.good())
            std::cerr << "[io] write failed (badbit/failbit set)\n";
          else
            std::cerr << "[io] ok\n";
        }

        // fs::path edge_dir = fs::current_path() / "data" / "Block" / "Edges";
        // fs::create_directories(edge_dir);
        // fs::path edge_obj = fs::absolute(
        //     edge_dir / ("Center2Max_6000_" + std::to_string(Fnum) + ".obj"));
        // std::cerr << "[io] write = " << edge_obj.string() << "\n";

        // {
        //   std::ofstream eout(edge_obj, std::ios::out | std::ios::trunc);
        //   if (!eout) {
        //     std::cerr << "[io] open failed (OBJ), errno=" << errno << " ("
        //               << std::strerror(errno) << ")\n";
        //   } else {
        //     eout << std::setprecision(17);
        //     for (int i = 0; i < (int)spheres.size(); ++i) {
        //       eout << "v " << spheres[i].c.x() << " " << spheres[i].c.y() << " "
        //            << spheres[i].c.z() << "\n";
        //       eout << "v " << spheres[i].max_point.x() << " "
        //            << spheres[i].max_point.y() << " "
        //            << spheres[i].max_point.z() << "\n";
        //     }
        //     for (int i = 0; i < (int)spheres.size(); ++i) {
        //       eout << "l " << (2 * i + 1) << " " << (2 * i + 2) << "\n";
        //     }
        //     eout.close();
        //     if (!eout.good())
        //       std::cerr << "[io] write failed (OBJ)\n";
        //     else
        //       std::cerr << "[io] ok (OBJ)\n";
        //   }
        // }

        // ……(后续 coverage 采样代码保持不变，如需可继续保留)
        return energy;
      };

  std::vector<Eigen::Vector3d> Pts2;

  Pts2 = Pts;
  num = Pts2.size();
  std::cout << Pts2.size() << "  " << num << std::endl;
  _sites.resize(num);
  _para.max_linearsearch = 20;
  BGAL::_LBFGS lbfgs2(_para);
  Eigen::VectorXd iterX2(num * 3);
  for (int i = 0; i < num; ++i) {
    iterX2(i * 3) = Pts2[i].x();
    iterX2(i * 3 + 1) = Pts2[i].y();
    iterX2(i * 3 + 2) = Pts2[i].z();
    _sites[i] = BGAL::_Point3(Pts2[i](0), Pts2[i](1), Pts2[i](2));
  }
  _RVD.calculate_(_sites);
  start = clock();
  lbfgs2.minimize(fgm2, iterX2);
  end = clock();
  allTime += (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << "allTime: " << allTime << " RVDtime: " << RVDtime
            << " L-BFGS time: " << allTime - RVDtime << std::endl;
  for (int i = 0; i < num; ++i) {
    // Point_T query(x0[i * 3], x0[i * 3+1], x0[i * 3+2]);
    Point_T query(iterX2(i * 3), iterX2(i * 3 + 1), iterX2(i * 3 + 2));
    Point_T closest = tree.closest_point(query);
    auto tri = tree.closest_point_and_primitive(query);

    Polyhedron::Face_handle f = tri.second;
    int fid = -1;
    auto it = face_id_map.find(static_cast<const void *>(&*tri.second));
    if (it != face_id_map.end())
      fid = it->second;
    FaceIDs[i] = fid;
    auto p1 = f->halfedge()->vertex()->point();
    auto p2 = f->halfedge()->next()->vertex()->point();
    auto p3 = f->halfedge()->next()->next()->vertex()->point();
    Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
    Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
    Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
    Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
    N.normalize();
    Nors[i] = N;

    _sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
  }
  _RVD.calculate_(_sites);

  OutputMesh(_sites, _RVD, num_sites,
             std::filesystem::current_path() / "data" / "Block", modelname, 2);

  // // --- Calculate Displacement Statistics ---
  // {
  //   std::cout << "[INFO] Calculating displacement on feature edges vs faces..."
  //             << std::endl;
  //   std::vector<Segment> feature_edges;
  //   double angle_threshold = 30.0 * 3.14159265358979323846 / 180.0;

  //   // 1. Identify feature edges
  //   int edge_idx = 0;
  //   for (auto eb = polyhedron.edges_begin(); eb != polyhedron.edges_end();
  //        ++eb) {
  //     auto h = eb;
  //     bool is_feature = false;
  //     if (h->is_border() || h->opposite()->is_border()) {
  //       is_feature = true;
  //     } else {
  //       auto p1 = h->vertex()->point();
  //       auto p2 = h->opposite()->vertex()->point(); // The edge vertices
  //       // Triangle 1: p1, p2, p3
  //       auto p3 = h->next()->vertex()->point();
  //       // Triangle 2: p2, p1, p4 (from opposite view)
  //       auto p4 = h->opposite()->next()->vertex()->point();

  //       auto v1 = p2 - p1;
  //       auto v2 = p3 - p1;
  //       auto n1 = CGAL::cross_product(v1, v2);
  //       // Approximate normalization
  //       double l1 = std::sqrt(n1.squared_length());
  //       if (l1 > 1e-10)
  //         n1 = n1 / l1;

  //       auto v1_opp = p1 - p2;
  //       auto v2_opp = p4 - p2;
  //       auto n2 = CGAL::cross_product(v1_opp, v2_opp);
  //       double l2 = std::sqrt(n2.squared_length());
  //       if (l2 > 1e-10)
  //         n2 = n2 / l2;

  //       double dot = n1 * n2;
  //       if (dot < -1.0)
  //         dot = -1.0;
  //       if (dot > 1.0)
  //         dot = 1.0;
  //       double angle = std::acos(dot);

  //       if (angle > angle_threshold)
  //         is_feature = true;
  //     }

  //     if (is_feature) {
  //       feature_edges.emplace_back(h->vertex()->point(),
  //                                  h->opposite()->vertex()->point());
  //     }
  //     edge_idx++;
  //   }
  //   std::cout << "[INFO] Identified " << feature_edges.size()
  //             << " feature edges." << std::endl;

  //   // 2. Compute stats with dynamic threshold & median

  //   // Calculate Bounding Box of Pts to determine scale
  //   double min_x = 1e30, min_y = 1e30, min_z = 1e30;
  //   double max_x = -1e30, max_y = -1e30, max_z = -1e30;
  //   for (const auto &p : Pts) {
  //     if (p.x() < min_x)
  //       min_x = p.x();
  //     if (p.y() < min_y)
  //       min_y = p.y();
  //     if (p.z() < min_z)
  //       min_z = p.z();
  //     if (p.x() > max_x)
  //       max_x = p.x();
  //     if (p.y() > max_y)
  //       max_y = p.y();
  //     if (p.z() > max_z)
  //       max_z = p.z();
  //   }
  //   double diag =
  //       std::sqrt(std::pow(max_x - min_x, 2) + std::pow(max_y - min_y, 2) +
  //                 std::pow(max_z - min_z, 2));
  //   double dist_threshold = diag * 0.01; // 1% of diagonal
  //   std::cout << "[INFO] BBox Diagonal: " << diag
  //             << ", Using dist_threshold: " << dist_threshold << std::endl;

  //   std::vector<double> disps_edge;
  //   std::vector<double> disps_face;

  //   // Assume Pts (initial) and _sites (final) are aligned
  //   size_t n = _sites.size();
  //   if (n > Pts.size())
  //     n = Pts.size();

  //   for (size_t i = 0; i < n; ++i) {
  //     double dx = _sites[i].x() - Pts[i].x();
  //     double dy = _sites[i].y() - Pts[i].y();
  //     double dz = _sites[i].z() - Pts[i].z();
  //     double disp = std::sqrt(dx * dx + dy * dy + dz * dz);

  //     Point_T pt(_sites[i].x(), _sites[i].y(), _sites[i].z());
  //     double min_dist_sq = 1e30;

  //     // Brute force check distance to feature edges
  //     for (const auto &seg : feature_edges) {
  //       double d2 = CGAL::squared_distance(pt, seg);
  //       if (d2 < min_dist_sq)
  //         min_dist_sq = d2;
  //     }

  //     if (min_dist_sq < dist_threshold * dist_threshold) {
  //       disps_edge.push_back(disp);
  //     } else {
  //       disps_face.push_back(disp);
  //     }
  //   }

  //   auto compute_median = [](std::vector<double> &v) -> double {
  //     if (v.empty())
  //       return 0.0;
  //     size_t n = v.size();
  //     std::sort(v.begin(), v.end());
  //     if (n % 2 == 0)
  //       return (v[n / 2 - 1] + v[n / 2]) / 2.0;
  //     return v[n / 2];
  //   };

  //   auto compute_sum = [](const std::vector<double> &v) -> double {
  //     double s = 0;
  //     for (double d : v)
  //       s += d;
  //     return s;
  //   };

  //   double median_edge = compute_median(disps_edge);
  //   double sum_edge = compute_sum(disps_edge);
  //   double mean_edge = disps_edge.empty() ? 0.0 : sum_edge / disps_edge.size();

  //   double median_face = compute_median(disps_face);
  //   double sum_face = compute_sum(disps_face);
  //   double mean_face = disps_face.empty() ? 0.0 : sum_face / disps_face.size();

  //   std::cout << "Feature Edge Sites: " << disps_edge.size()
  //             << " | Mean: " << mean_edge << " | Median: " << median_edge
  //             << " | Total: " << sum_edge << std::endl;

  //   // ... (existing logging code) ...
  //   std::cout << "Face Sites: " << disps_face.size() << " | Mean: " << mean_face
  //             << " | Median: " << median_face << " | Total: " << sum_face
  //             << std::endl;

  //   // --- Export Statistics to CSV ---
  //   namespace fs = std::filesystem;
  //   fs::path base_path = fs::current_path() / "data" / "Mobius";

  //   // 1. Export Feature Edge Displacements
  //   {
  //     fs::path csv_path =
  //         base_path / ("Displacement_FeatureEdges_" +
  //                      std::to_string(num_sites) + "_" + modelname + ".csv");
  //     std::ofstream out_edge(csv_path);
  //     if (out_edge.is_open()) {
  //       out_edge << "Displacement\n";
  //       for (double d : disps_edge) {
  //         out_edge << d << "\n";
  //       }
  //       out_edge.close();
  //       std::cout << "[INFO] Written feature edge displacements to " << csv_path
  //                 << std::endl;
  //     } else {
  //       std::cerr << "[ERROR] Failed to open " << csv_path << " for writing."
  //                 << std::endl;
  //     }
  //   }

  //   // 2. Export Face Displacements
  //   {
  //     fs::path csv_path =
  //         base_path / ("Displacement_Faces_" + std::to_string(num_sites) + "_" +
  //                      modelname + ".csv");
  //     std::ofstream out_face(csv_path);
  //     if (out_face.is_open()) {
  //       out_face << "Displacement\n";
  //       for (double d : disps_face) {
  //         out_face << d << "\n";
  //       }
  //       out_face.close();
  //       std::cout << "[INFO] Written face displacements to " << csv_path
  //                 << std::endl;
  //     } else {
  //       std::cerr << "[ERROR] Failed to open " << csv_path << " for writing."
  //                 << std::endl;
  //     }
  //   }
  // } // end block

  // // --- Power Diagram Partitioning (Added) ---
  // {
  //   std::cout << "\n[INFO] Starting Power Diagram Partitioning..." << std::endl;
  //   namespace fs = std::filesystem;
  //   fs::path base_path = fs::current_path() / "data" / "Mobius";

  //   std::vector<double> all_displacements;
  //   all_displacements.reserve(_sites.size());

  //   // 1. Calculate all displacements
  //   size_t n = _sites.size();
  //   if (n > Pts.size())
  //     n = Pts.size(); // Safety check

  //   for (size_t i = 0; i < n; ++i) {
  //     double dx = _sites[i].x() - Pts[i].x();
  //     double dy = _sites[i].y() - Pts[i].y();
  //     double dz = _sites[i].z() - Pts[i].z();
  //     double disp = (dx * dx + dy * dy + dz * dz);
  //     all_displacements.push_back(disp);
  //   }

  //   // 2. Compute Box Plot Statistics
  //   std::vector<double> sorted_disp = all_displacements;
  //   std::sort(sorted_disp.begin(), sorted_disp.end());

  //   double Q1 = 0, Q3 = 0;
  //   if (!sorted_disp.empty()) {
  //     Q1 = sorted_disp[sorted_disp.size() / 4];
  //     Q3 = sorted_disp[sorted_disp.size() * 3 / 4];
  //   }
  //   double IQR = Q3 - Q1;
  //   double UpperFence = Q3 + 1.5 * IQR;

  //   std::cout << "[INFO] Stats - Q1: " << Q1 << ", Q3: " << Q3
  //             << ", IQR: " << IQR << ", Upper Fence: " << UpperFence
  //             << std::endl;

  //   // 3. Clamp Weights (Disabled per user request, using squared distance
  //   // directly)
  //   std::vector<double> weights;
  //   weights.reserve(n);
  //   for (double d_sq : all_displacements) {
  //     // 'd_sq' is squared displacement
  //     double d_linear = std::sqrt(d_sq);

  //     // Apply thresholds
  //     if (d_linear > 0.68) {
  //       d_linear = 0.68;
  //     }
  //     if (d_linear < 0.003) {
  //       d_linear = 0.0;
  //     }

  //     // Weight is squared clamped distance
  //     double w = d_linear * d_linear;
  //     weights.push_back(w);
  //   }

  //   // 4. Generate Power Diagram
  //   _RVD.calculate_(_sites, weights);

  //   // 5. Output
  //   OutputMesh(_sites, _RVD, num_sites, base_path, modelname, 2500);

  //   std::cout
  //       << "[INFO] Power Diagram generation complete. Output with index 2500."
  //       << std::endl;

  //   // 6. Calculate Site-Centered Max-Radius Sphere for each cell
  //   std::cout
  //       << "[INFO] Calculating Site-Centered Max-Radius Spheres using Edges..."
  //       << std::endl;
  //   std::vector<Sphere::Sphere> meb_spheres;
  //   const std::vector<std::map<int, std::vector<std::pair<int, int>>>> &edges =
  //       _RVD.get_edges_();

  //   // Resize to match number of sites
  //   meb_spheres.resize(n);

  //   for (int i = 0; i < (int)n; ++i) {
  //     double max_dist_sq = 0.0;
  //     bool has_vertices = false;

  //     // Check if site i has valid edges
  //     if (i < edges.size()) {
  //       // Iterate over all neighbors (adjacent sites or -1 for boundary)
  //       for (auto const &[neighbor_id, edge_segments] : edges[i]) {
  //         // Iterate over segments of the edge between i and neighbor
  //         for (auto const &seg : edge_segments) {
  //           // seg.first and seg.second are indices into _RVD.vertex_()
  //           int v_idx1 = seg.first;
  //           int v_idx2 = seg.second;

  //           // Check vertex 1
  //           BGAL::_Point3 v1 = _RVD.vertex_(v_idx1);
  //           double d2_1 = (v1 - _sites[i]).sqlength_();
  //           if (d2_1 > max_dist_sq)
  //             max_dist_sq = d2_1;

  //           // Check vertex 2
  //           BGAL::_Point3 v2 = _RVD.vertex_(v_idx2);
  //           double d2_2 = (v2 - _sites[i]).sqlength_();
  //           if (d2_2 > max_dist_sq)
  //             max_dist_sq = d2_2;

  //           has_vertices = true;
  //         }
  //       }
  //     }

  //     meb_spheres[i].c = decltype(meb_spheres[i].c)(
  //         _sites[i].x(), _sites[i].y(), _sites[i].z());

  //     if (!has_vertices) {
  //       meb_spheres[i].r = 0;
  //     } else {
  //       meb_spheres[i].r = std::sqrt(max_dist_sq);
  //     }
  //   }

  //   // 7. Output MEB to CSV
  //   fs::path meb_csv = base_path / ("MEB_PowerDiagram_2500.csv");
  //   std::ofstream out_meb(meb_csv);
  //   if (out_meb.is_open()) {
  //     out_meb << std::setprecision(17);
  //     for (int i = 0; i < (int)n; ++i) {
  //       // Format: cx,cy,cz,r,FaceID,nx,ny,nz
  //       Eigen::Vector3d nor = Nors[i].normalized();
  //       out_meb << meb_spheres[i].c.x() << "," << meb_spheres[i].c.y() << ","
  //               << meb_spheres[i].c.z() << "," << meb_spheres[i].r << ","
  //               << FaceIDs[i] << "," << nor.x() << "," << nor.y() << ","
  //               << nor.z() << "\n";
  //     }
  //     out_meb.close();
  //     std::cout << "[INFO] Written MEB to " << meb_csv << std::endl;
  //   }
  // }
}
} // namespace BGAL
