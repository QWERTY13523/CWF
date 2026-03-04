#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

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

namespace {
struct FeatureSegment {
  BGAL::_Point3 a;
  BGAL::_Point3 b;
};

double squared_distance_point_to_segment(const BGAL::_Point3 &p,
                                         const BGAL::_Point3 &a,
                                         const BGAL::_Point3 &b) {
  const BGAL::_Point3 ab = b - a;
  const double ab2 = ab.sqlength_();
  if (ab2 <= 1e-30) {
    return (p - a).sqlength_();
  }
  double t = (p - a).dot_(ab) / ab2;
  t = std::max(0.0, std::min(1.0, t));
  const BGAL::_Point3 proj = a + ab * t;
  return (p - proj).sqlength_();
}

std::vector<FeatureSegment> collect_feature_segments(
    const BGAL::_ManifoldModel &model, const double sharp_angle_deg) {
  std::vector<FeatureSegment> segments;
  segments.reserve(std::max(1, model.number_edges_() / 8));

  const double sharp_cos =
      std::cos(sharp_angle_deg * 3.14159265358979323846 / 180.0);

  for (int eid = 0; eid < model.number_edges_(); ++eid) {
    const auto e = model.edge_(eid);
    if (e._id_reverse_edge < 0 || eid > e._id_reverse_edge) {
      continue;
    }

    const auto re = model.edge_(e._id_reverse_edge);
    bool is_feature = false;
    if (e._id_face == -1 || re._id_face == -1) {
      is_feature = true;
    } else {
      double dot = model.normal_face_(e._id_face).dot_(model.normal_face_(re._id_face));
      dot = std::max(-1.0, std::min(1.0, dot));
      if (dot < sharp_cos) {
        is_feature = true;
      }
    }

    if (is_feature) {
      segments.push_back(
          FeatureSegment{model.vertex_(e._id_left_vertex),
                         model.vertex_(e._id_right_vertex)});
    }
  }
  return segments;
}
} // namespace

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
  _para.max_iteration = 75;

  // constexpr double kBaseDensity = 1.0;
  // constexpr double kFeatureSparsityRatio =
  //     1.05; // strong sparsification near feature edges
  // constexpr double kSharpAngleDeg = 30.0;
  // constexpr double kFeatureBandScale = 0.03;

  // auto feature_segments =
  //     std::make_shared<std::vector<FeatureSegment>>(
  //         collect_feature_segments(_model, kSharpAngleDeg));

  // const auto bbox = _model.bounding_box_();
  // const double bbox_diag = (bbox.second - bbox.first).length_();
  // const double sigma = std::max(1e-12, kFeatureBandScale * bbox_diag);
  // const double inv_two_sigma2 = 0.5 / (sigma * sigma);

  // _rho = [feature_segments, inv_two_sigma2, kBaseDensity,
  //         kFeatureSparsityRatio](BGAL::_Point3 &p) -> double {
  //   if (feature_segments->empty()) {
  //     return kBaseDensity;
  //   }

  //   double min_d2 = std::numeric_limits<double>::max();
  //   for (const auto &seg : *feature_segments) {
  //     const double d2 = squared_distance_point_to_segment(p, seg.a, seg.b);
  //     if (d2 < min_d2) {
  //       min_d2 = d2;
  //     }
  //   }

  //   const double w = std::exp(-min_d2 * inv_two_sigma2);
  //   return kBaseDensity * (1.0 - (1.0 - kFeatureSparsityRatio) * w);
  // };

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
  fs::path obj("./data/block.obj");
  fs::path base = obj.parent_path();
  if (pointsName == nullptr) {
    inPointsName = (base / ("n" + std::to_string(num_sites) + "_" + modelname +
                           "_inputPoints.xyz")).string();
  } else {
    inPointsName = pointsName;
  }
  std::ifstream inPoints(inPointsName.c_str());
  std::vector<Eigen::Vector3d> Pts, Nors;

  int count = 0;
  double x, y, z, nx, ny, nz;
  while (inPoints >> x >> y >> z >> nx >> ny >> nz) {
    Pts.push_back(Eigen::Vector3d(x, y, z));
    Nors.push_back(
        Eigen::Vector3d(nx, ny, nz));
    ++count;
  }
  inPoints.close();
  std::cout << "Pts.size(): " << Pts.size() << std::endl;

  if (pointsName != nullptr) {
    num_sites = static_cast<int>(Pts.size());
  }
  // begin step 1.
  int num = Pts.size();

  std::vector<Eigen::Vector3d> Pts3;
  std::cout << "\nBegin CWF.\n" << std::endl;

  int Fnum = 4;
  double alpha = 1.0, eplison = 1,
         lambda = 0.85;
  double decay = 0.985;
  const double quality_weight = 0.1;
  const double feature_sharp_angle_deg = 30.0;
  const double feature_band_scale = 0.03;
  const auto feature_segments =
      collect_feature_segments(_model, feature_sharp_angle_deg);
  const auto bbox = _model.bounding_box_();
  const double bbox_diag = (bbox.second - bbox.first).length_();
  const double sigma = std::max(1e-12, feature_band_scale * bbox_diag);
  const double inv_two_sigma2 = 0.5 / (sigma * sigma);
  std::vector<int> FaceIDs;
  FaceIDs.assign(num, -1);
  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm2 =
      [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
        eplison = eplison * decay;
        double lossCVT = 0, lossQE = 0, loss = 0;
        double lossQuality = 0.0;
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
          this->_sites[i] = p;
        }
        this->_RVD.calculate_(this->_sites);
        Fnum++;
        if (Fnum % 1 == 0) {
          OutputMesh(this->_sites, this->_RVD, num_sites,
                     (std::filesystem::current_path() / "data" / "Block").string(),
                     modelname, Fnum);
        }
        endRVD = clock();
        RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

        const std::vector<std::vector<std::tuple<int, int, int>>> &cells =
            this->_RVD.get_cells_();
        const std::vector<std::map<int, std::vector<std::pair<int, int>>>>
            &edges = this->_RVD.get_edges_();

        double energy = 0.0;
        g.setZero();
        std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());
        std::vector<double> feature_weights(num, 1.0);
        if (!feature_segments.empty()) {
          for (int i = 0; i < num; ++i) {
            const BGAL::_Point3 p = this->_sites[i];
            double min_d2 = std::numeric_limits<double>::max();
            for (const auto &seg : feature_segments) {
              const double d2 = squared_distance_point_to_segment(p, seg.a, seg.b);
              if (d2 < min_d2) {
                min_d2 = d2;
              }
            }
            const double w = 1.0 - std::exp(-min_d2 * inv_two_sigma2);
            feature_weights[i] = w;
          }
        }

        omp_set_num_threads(30); // change to your CPU core numbers
#pragma omp parallel for reduction(+ : lossCVT, loss, lossCenter)
        for (int i = 0; i < num; ++i) {
          BGAL::_Point3 site = this->_sites[i];
          Eigen::Vector3d xi(site.x(), site.y(), site.z());

          for (int j = 0; j < (int)cells[i].size(); ++j) {
            auto [a, b, c] = cells[i][j];
            BGAL::_Point3 pa = this->_RVD.vertex_(a), pb = this->_RVD.vertex_(b),
                    pc = this->_RVD.vertex_(c);

            Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
                [&, site](BGAL::_Point3 p) {
                  Eigen::VectorXd r(5);

                  BGAL::_Point3 NorTriM =
                      (pb - pa).cross_(pc - pa);

                  NorTriM.normalized_();

                  // ===========================================================
                  // 【核心恢复】：完美地通过 this->_rho(p) 进行调用
                  // 因为你在构造函数中已经对它进行了初始化，所以这里不再需要做判断
                  // ===========================================================
                  double rho_val = this->_rho(p);
                  
                  r(0) = (eplison * rho_val *
                          ((site - p).sqlength_())); 
                  r(1) = lambda * (NorTriM.dot_(p - site)) *
                             (NorTriM.dot_(p - site)) +
                         eplison * rho_val *
                             ((p - site)
                                  .sqlength_()); 

                  r(2) = lambda * -2 * NorTriM.x() *
                             (NorTriM.dot_(p - site))+
                         eplison * rho_val * -2 * (p - site).x();
                  r(3) = lambda * -2 * NorTriM.y() *
                             (NorTriM.dot_(p - site)) +
                         eplison * rho_val * -2 * (p - site).y(); 
                  r(4) = lambda * -2 * NorTriM.z() *
                             (NorTriM.dot_(p - site)) +
                         eplison * rho_val * -2 * (p - site).z();

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
                 z = site.z(); // 默认回退至 site 本身

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
              BGAL::_Point3 pv = this->_RVD.vertex_(vid);
              boundary_pts.emplace_back(pv.x(), pv.y(), pv.z());
            }
          }
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

        if (quality_weight > 0.0) {
          double sum_len = 0.0;
          size_t edge_count = 0;
          for (int i = 0; i < (int)edges.size(); ++i) {
            for (const auto &kv : edges[i]) {
              const int j = kv.first;
              if (j <= i || j >= num) {
                continue;
              }
              const BGAL::_Point3 pi = this->_sites[i];
              const BGAL::_Point3 pj = this->_sites[j];
              const Eigen::Vector3d d(pi.x() - pj.x(), pi.y() - pj.y(),
                                      pi.z() - pj.z());
              const double len = d.norm();
              if (len > 1e-12) {
                sum_len += len;
                ++edge_count;
              }
            }
          }
          if (edge_count > 0) {
            const double L_avg = sum_len / static_cast<double>(edge_count);
            for (int i = 0; i < (int)edges.size(); ++i) {
              for (const auto &kv : edges[i]) {
                const int j = kv.first;
                if (j <= i || j >= num) {
                  continue;
                }
                const BGAL::_Point3 pi = this->_sites[i];
                const BGAL::_Point3 pj = this->_sites[j];
                const Eigen::Vector3d d(pi.x() - pj.x(), pi.y() - pj.y(),
                                        pi.z() - pj.z());
                const double len = d.norm();
                if (len <= 1e-12) {
                  continue;
                }
                const double wij =
                    quality_weight * feature_weights[i] * feature_weights[j];
                const double diff = len - L_avg;
                lossQuality += wij * diff * diff;
                const double scale = (2.0 * wij * diff) / len;
                const Eigen::Vector3d grad = scale * d;
                gi[i] += grad;
                gi[j] -= grad;
              }
            }
          }
        }

        for (int i = 0; i < num; i++) {
          gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
          g(i * 3) += gi[i].x();
          g(i * 3 + 1) += gi[i].y();
          g(i * 3 + 2) += gi[i].z();
        }
        energy += loss + lossQuality;

        std::cout << std::setprecision(7) << "energy: " << energy
                  << " LossCVT: " << lossCVT / eplison
                  << " LossQE: " << (loss - lossCVT) / lambda
                  << " LossQuality: " << lossQuality
                  << " Lambda_CVT: " << eplison << std::endl;

        namespace fs = std::filesystem;
        fs::path file =
            fs::absolute(fs::current_path() / "data" / "Block" /
                         ("Sphere_8000_" + std::to_string(Fnum) + ".csv"));
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
        }
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
             (std::filesystem::current_path() / "data" / "Block").string(), modelname, 2);

}
} // namespace BGAL
