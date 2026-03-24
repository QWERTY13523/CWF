#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <cstdint>
#include <numeric>

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
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <memory>
#include <igl/gaussian_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/principal_curvature.h>
#include <igl/adjacency_list.h>
#include <igl/avg_edge_length.h>

#include "BGAL/BaseShape/KDTree.h"
#include <omp.h>

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
    p.x() = a; p.y() = b; p.z() = c;
  }
  Eigen::Vector3d p;

  bool operator<(const MyPoint &a) const {
    double dis = (p - a.p).norm();
    if (dis < kgammaTol) return false;

    if ((p.x() - a.p.x()) < 1e-11 && (p.x() - a.p.x()) > -1e-11) {
      if ((p.y() - a.p.y()) < 1e-11 && (p.y() - a.p.y()) > -1e-11) {
        return (p.z() < a.p.z());
      }
      return (p.y() < a.p.y());
    }
    return (p.x() < a.p.x());
  }

  bool operator==(const MyPoint &a) const {
    if ((p.x() - a.p.x()) < 1e-11 && (p.x() - a.p.x()) > -1e-11) {
      if ((p.y() - a.p.y()) < 1e-11 && (p.y() - a.p.y()) > -1e-11) {
        if ((p.z() - a.p.z()) < 1e-11 && (p.z() - a.p.z()) > -1e-11) {
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
    p.x() = a; p.y() = b; p.z() = c;
  }
  Eigen::Vector3i p;
  bool operator<(const MyFace &a) const {
    if (p.x() == a.p.x()) {
      if (p.y() == a.p.y()) return p.z() > a.p.z();
      return p.y() > a.p.y();
    }
    return p.x() > a.p.x();
  }
};

namespace {
using Vec3 = Eigen::Vector3d;

struct FeatureSegment {
  BGAL::_Point3 a;
  BGAL::_Point3 b;
};

static inline Vec3 to_eigen(const BGAL::_Point3 &p) {
  return Vec3(p.x(), p.y(), p.z());
}

struct DynamicRadiusEval {
  double total_loss = 0.0;
  int active_quads = 0;
  double min_g = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd grads;

  explicit DynamicRadiusEval(int n = 0, bool need_grad = false)
      : grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()) {}
};

struct QuadKeyLess {
  bool operator()(const std::array<int, 4> &a,
                  const std::array<int, 4> &b) const {
    return a < b;
  }
};

static inline double squared_distance_point_to_segment(const BGAL::_Point3 &p,
                                                       const BGAL::_Point3 &a,
                                                       const BGAL::_Point3 &b) {
  const BGAL::_Point3 ab = b - a;
  const double ab2 = ab.sqlength_();
  if (ab2 <= 1e-30) return (p - a).sqlength_();
  double t = (p - a).dot_(ab) / ab2;
  t = std::max(0.0, std::min(1.0, t));
  const BGAL::_Point3 proj = a + ab * t;
  return (p - proj).sqlength_();
}

static inline std::vector<FeatureSegment> collect_feature_segments(
    const BGAL::_ManifoldModel &model, const double sharp_angle_deg) {
  std::vector<FeatureSegment> segments;
  segments.reserve(std::max(1, model.number_edges_() / 8));
  const double sharp_cos = std::cos(sharp_angle_deg * 3.14159265358979323846 / 180.0);
  for (int eid = 0; eid < model.number_edges_(); ++eid) {
    const auto e = model.edge_(eid);
    if (e._id_reverse_edge < 0 || eid > e._id_reverse_edge) continue;
    const auto re = model.edge_(e._id_reverse_edge);
    bool is_feature = false;
    if (e._id_face == -1 || re._id_face == -1) {
      is_feature = true;
    } else {
      double dot = model.normal_face_(e._id_face).dot_(model.normal_face_(re._id_face));
      dot = std::max(-1.0, std::min(1.0, dot));
      if (dot < sharp_cos) is_feature = true;
    }
    if (is_feature) {
      segments.push_back(
          FeatureSegment{model.vertex_(e._id_left_vertex), model.vertex_(e._id_right_vertex)});
    }
  }
  return segments;
}

static inline bool intersect_three_spheres(
    const Eigen::Vector3d& c1, double r1,
    const Eigen::Vector3d& c2, double r2,
    const Eigen::Vector3d& c3, double r3,
    Eigen::Vector3d& s_plus,
    Eigen::Vector3d& s_minus) {
  const double eps = 1e-12;
  Eigen::Vector3d ex = c2 - c1;
  double d = ex.norm();
  if (d < eps) return false;
  ex /= d;
  Eigen::Vector3d c3c1 = c3 - c1;
  double i = ex.dot(c3c1);
  Eigen::Vector3d tmp = c3c1 - i * ex;
  double tmpn = tmp.norm();
  if (tmpn < eps) return false;
  Eigen::Vector3d ey = tmp / tmpn;
  Eigen::Vector3d ez = ex.cross(ey);
  double j = ey.dot(c3c1);
  if (std::abs(j) < eps) return false;
  double x = (r1*r1 - r2*r2 + d*d) / (2.0 * d);
  double y = (r1*r1 - r3*r3 + i*i + j*j - 2.0*i*x) / (2.0 * j);
  double z2 = r1*r1 - x*x - y*y;
  if (z2 < 0.0) return false;
  double z = std::sqrt(std::max(0.0, z2));
  Eigen::Vector3d base = c1 + x*ex + y*ey;
  s_plus  = base + z*ez;
  s_minus = base - z*ez;
  return true;
}

static inline void build_spheres_from_rvd(
    const std::vector<BGAL::_Point3> &sites,
    const BGAL::_Restricted_Tessellation3D &rvd,
    std::vector<Sphere::Sphere> &spheres) {
  const auto &edges = rvd.get_edges_();
  spheres.assign(sites.size(), Sphere::Sphere());
  for (int i = 0; i < (int)sites.size(); ++i) {
    const BGAL::_Point3 site = sites[i];
    std::unordered_set<int> bnd;
    if (i < (int)edges.size()) {
      for (const auto &kv : edges[i]) {
        for (const auto &e : kv.second) {
          bnd.insert(e.first);
          bnd.insert(e.second);
        }
      }
    }
    BGAL::_Point3 farp = site;
    double best_dist = 0.0;
    for (int vid : bnd) {
      const BGAL::_Point3 pv = rvd.vertex_(vid);
      const double dist = (pv - site).length_();
      if (dist > best_dist) {
        best_dist = dist;
        farp = pv;
      }
    }
    spheres[i].c = decltype(spheres[i].c)(site.x(), site.y(), site.z());
    spheres[i].r = best_dist;
    spheres[i].max_point =
        decltype(spheres[i].max_point)(farp.x(), farp.y(), farp.z());
  }
}

static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_edges(
    int num_sites,
    const std::vector<std::map<int, std::vector<std::pair<int, int>>>> &edges) {
  std::set<std::pair<int, int>> rdt_edges;
  std::vector<std::set<int>> neighbors(num_sites);
  for (int i = 0; i < (int)edges.size(); ++i) {
    for (const auto &ee : edges[i]) {
      const int j = ee.first;
      if (j < 0 || j >= num_sites || j == i) continue;
      rdt_edges.insert(std::make_pair(std::min(i, j), std::max(i, j)));
      neighbors[i].insert(j);
      neighbors[j].insert(i);
    }
  }
  std::set<MyFace> rdt_faces;
  for (const auto &e : rdt_edges) {
    for (int pid : neighbors[e.first]) {
      if (rdt_edges.find(std::make_pair(std::min(pid, e.first), std::max(pid, e.first))) == rdt_edges.end()) continue;
      if (rdt_edges.find(std::make_pair(std::min(pid, e.second), std::max(pid, e.second))) == rdt_edges.end()) continue;
      const int hi = std::max(pid, std::max(e.first, e.second));
      const int lo = std::min(pid, std::min(e.first, e.second));
      const int mid = pid + e.first + e.second - hi - lo;
      rdt_faces.insert(MyFace(hi, mid, lo));
    }
  }
  std::vector<Eigen::Vector3i> tris;
  tris.reserve(rdt_faces.size());
  for (const auto &f : rdt_faces) tris.emplace_back(f.p.x(), f.p.y(), f.p.z());
  return tris;
}

static inline void build_quads_from_search_faces(
    const std::vector<Eigen::Vector3i> &faces,
    const std::vector<BGAL::_Point3> &sites,
    const std::vector<Sphere::Sphere> &spheres,
    std::vector<std::array<int, 4>> &quads) {
  quads.clear();
  const int n = (int)sites.size();
  if (n <= 0 || faces.empty()) return;

  double global_rmax = 0.0;
  for (int i = 0; i < n; ++i) {
    global_rmax = std::max(global_rmax, std::max(0.0, (double)spheres[i].r));
  }

  BGAL::_KDTree kdtree(sites);
  std::vector<double> R_cached(n, 0.0);
  for (int i = 0; i < n; ++i) {
    R_cached[i] = std::max(0.0, (double)spheres[i].r) + global_rmax;
  }

  std::vector<int> face_pivots(faces.size(), -1);
  std::vector<char> pivot_used(n, 0);
  for (int f_idx = 0; f_idx < (int)faces.size(); ++f_idx) {
    const auto &f = faces[f_idx];
    const int i = f.x(); const int j = f.y(); const int k = f.z();
    if (i < 0 || j < 0 || k < 0 || i >= n || j >= n || k >= n) continue;
    if (i == j || i == k || j == k) continue;

    int pivot = i;
    if (R_cached[j] < R_cached[pivot]) pivot = j;
    if (R_cached[k] < R_cached[pivot]) pivot = k;
    face_pivots[f_idx] = pivot;
    pivot_used[pivot] = 1;
  }

  std::vector<std::vector<int>> pivot_candidates(n);
  for (int pivot = 0; pivot < n; ++pivot) {
    if (!pivot_used[pivot]) continue;
    pivot_candidates[pivot] = kdtree.rsearch_(sites[pivot], R_cached[pivot]);
  }

  std::vector<std::array<int, 4>> all_quads;

  for (int f_idx = 0; f_idx < (int)faces.size(); ++f_idx) {
    const auto &f = faces[f_idx];
    const int i = f.x(); const int j = f.y(); const int k = f.z();
    if (i < 0 || j < 0 || k < 0 || i >= n || j >= n || k >= n) continue;
    if (i == j || i == k || j == k) continue;

    const double Ri = R_cached[i];
    const double Rj = R_cached[j];
    const double Rk = R_cached[k];
    const int pivot = face_pivots[f_idx];
    if (pivot < 0) continue;
    const std::vector<int> &cand = pivot_candidates[pivot];

    const double Ri2 = Ri * Ri, Rj2 = Rj * Rj, Rk2 = Rk * Rk;
    const double tol = 1e-12 * std::max({1.0, Ri2, Rj2, Rk2});

    for (int l : cand) {
      if (l == i || l == j || l == k) continue;
      if ((sites[l] - sites[i]).sqlength_() > Ri2 + tol) continue;
      if ((sites[l] - sites[j]).sqlength_() > Rj2 + tol) continue;
      if ((sites[l] - sites[k]).sqlength_() > Rk2 + tol) continue;
      std::array<int, 4> q{i, j, k, l};
      std::sort(q.begin(), q.end());
      all_quads.push_back(q);
    }
  }
  std::sort(all_quads.begin(), all_quads.end());
  all_quads.erase(std::unique(all_quads.begin(), all_quads.end()), all_quads.end());
  quads = std::move(all_quads);
}

static inline DynamicRadiusEval evaluate_dynamic_radius_surrogate(
    const std::vector<BGAL::_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<Vec3> &normals,
    const std::vector<std::array<int, 4>> &quads,
    double eps,
    bool need_grad) {
  DynamicRadiusEval eval((int)sites.size(), need_grad);

  for (const auto &q : quads) {
    Vec3 p_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) p_bar += to_eigen(sites[q[k]]);
    p_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 pi = to_eigen(sites[sid]);
      const double ri2 = (pi - frozen_poles[sid]).squaredNorm();
      g_val += (p_bar - pi).squaredNorm() - ri2;
    }

    eval.min_g = std::min(eval.min_g, g_val);
    const double violation = -(g_val + eps);
    if (violation <= 0.0) continue;

    eval.total_loss += violation * violation;
    eval.active_quads++;

    if (!need_grad) continue;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      eval.grads.row(sid) += (-4.0 * violation * (frozen_poles[sid] - p_bar)).transpose();
    }
  }

  if (need_grad && !normals.empty()) {
    for (int i = 0; i < eval.grads.rows(); ++i) {
      if (eval.grads.row(i).squaredNorm() < 1e-24) continue;
      Vec3 g = eval.grads.row(i).transpose();
      const Vec3 n = normals[i];
      const double nn = n.squaredNorm();
      if (nn > 1e-30) {
        g -= n * (g.dot(n) / nn);
      }
      eval.grads.row(i) = g.transpose();
    }
  }

  if (!std::isfinite(eval.min_g)) eval.min_g = 0.0;
  return eval;
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

  std::ofstream out(filepath);
  out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar"
      << std::endl;

  for (int i = 0; i < RVD.number_vertices_(); ++i) {
    out << "v " << RVD.vertex_(i) << std::endl;
  }

  double totarea = 0, parea = 0;
  for (int i = 0; i < (int)cells.size(); ++i) {
    double area = 0;
    for (int j = 0; j < (int)cells[i].size(); ++j) {
      BGAL::_Point3 p1 = RVD.vertex_(std::get<0>(cells[i][j]));
      BGAL::_Point3 p2 = RVD.vertex_(std::get<1>(cells[i][j]));
      BGAL::_Point3 p3 = RVD.vertex_(std::get<2>(cells[i][j]));
      area += (p2 - p1).cross_(p3 - p1).length_() / 2;
    }
    totarea += area;

    auto color = (double)BGAL::_BOC::rand_();
    if (i > (int)cells.size() / 3) {
      if (step == 1) color = 0;
    } else {
      parea += area;
    }

    out << "vt " << color << " 0" << std::endl;

    for (int j = 0; j < (int)cells[i].size(); ++j) {
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
  int outnum = (int)sites.size();
  if (step == 1) outnum = (int)sites.size() / 3;
  for (int i = 0; i < outnum; ++i) outP << sites[i] << std::endl;
  outP.close();

  if (step >= 2) {
    std::string filepath = outpath + "/Ours_" + std::to_string(num) + "_" +
                           modelname + "_Remesh.obj";
    std::string filepath1 = outpath + "/Ours_" + std::to_string(num) + "_" +
                            modelname + "_Iter" + std::to_string(step - 3) +
                            "_Remesh.obj";
    std::ofstream outRDT(filepath);
    std::ofstream outRDT1(filepath1);

    const auto rdt_faces =
        build_rdt_faces_from_edges((int)sites.size(), RVD.get_edges_());

    for (auto v : sites) {
      outRDT << "v " << v << std::endl;
      outRDT1 << "v " << v << std::endl;
    }

    for (const auto &f : rdt_faces) {
      outRDT << "f " << f.x() + 1 << " " << f.y() + 1 << " "
             << f.z() + 1 << std::endl;
      outRDT1 << "f " << f.x() + 1 << " " << f.y() + 1 << " "
              << f.z() + 1 << std::endl;
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
  _para.max_iteration = 65;

  constexpr double kBaseDensity = 1.0;
  constexpr double kFeatureSparsityRatio = 1.15;
  constexpr double kSharpAngleDeg = 30.0;
  constexpr double kFeatureBandScale = 0.03;

  auto feature_segments =
      std::make_shared<std::vector<FeatureSegment>>(
          collect_feature_segments(_model, kSharpAngleDeg));

  const auto bbox = _model.bounding_box_();
  const double bbox_diag = (bbox.second - bbox.first).length_();
  const double sigma = std::max(1e-12, kFeatureBandScale * bbox_diag);
  const double inv_two_sigma2 = 0.5 / (sigma * sigma);

  _rho =[feature_segments, inv_two_sigma2, kBaseDensity,
          kFeatureSparsityRatio](BGAL::_Point3 &p) -> double {
    if (feature_segments->empty()) return kBaseDensity;

    double min_d2 = std::numeric_limits<double>::max();
    for (const auto &seg : *feature_segments) {
      const double d2 = squared_distance_point_to_segment(p, seg.a, seg.b);
      if (d2 < min_d2) min_d2 = d2;
    }

    const double w = std::exp(-min_d2 * inv_two_sigma2);
    return kBaseDensity * (1.0 - (1.0 - kFeatureSparsityRatio) * w);
  };
}

_CVT3D::_CVT3D(const _ManifoldModel &model,
               std::function<double(_Point3 &p)> &rho, _LBFGS::_Parameter para)
    : _model(model), _RVD(model), _RVD2(model), _rho(rho), _para(para) {}

void _CVT3D::calculate_(int num_sites, char *modelNamee, char *pointsName) {

  double allTime = 0, RVDtime = 0;
  clock_t start, end;
  clock_t startRVD, endRVD;
  std::cout << std::filesystem::current_path() << std::endl;

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
      face_id_map[static_cast<const void *>(&*fit)] = idx;
    }
  }

  std::string inPointsName;
  namespace fs = std::filesystem;
  fs::path obj("./data/block.obj");
  fs::path base = obj.parent_path();
  if (pointsName == nullptr) {
    inPointsName = (base / ("n" + std::to_string(num_sites) + "_" + modelname + "_inputPoints.xyz")).string();
  } else {
    inPointsName = pointsName;
  }

  std::ifstream inPoints(inPointsName.c_str());
  std::vector<Eigen::Vector3d> Pts, Nors;
  std::string line;
  while (std::getline(inPoints, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(ss >> x >> y >> z)) continue;
    double nx = 0.0, ny = 0.0, nz = 0.0;
    const bool has_normal = static_cast<bool>(ss >> nx >> ny >> nz);
    Pts.push_back(Eigen::Vector3d(x, y, z));
    if (has_normal) Nors.push_back(Eigen::Vector3d(nx, ny, nz));
    else Nors.push_back(Eigen::Vector3d::Zero());
  }
  inPoints.close();
  std::cout << "Pts.size(): " << Pts.size() << std::endl;

  if (pointsName != nullptr) num_sites = static_cast<int>(Pts.size());
  int num = (int)Pts.size();
  std::cout << "\n=============================================\n"
            << "     Begin CWF (ADMM Version) Optimization\n"
            << "=============================================\n" << std::endl;

  _sites.resize(num);
  _para.max_linearsearch = 20;
  Eigen::VectorXd iterX2(num * 3);
  for (int i = 0; i < num; ++i) {
    iterX2(i * 3 + 0) = Pts[i].x();
    iterX2(i * 3 + 1) = Pts[i].y();
    iterX2(i * 3 + 2) = Pts[i].z();
    _sites[i] = BGAL::_Point3(Pts[i].x(), Pts[i].y(), Pts[i].z());
  }

  // ======= 参数 =======
  double alpha = 1.0, eplison = 1.0, lambda = 1.0;
  double decay = 0.95;

  // ======= ADMM 参数 =======
  Eigen::VectorXd iterP = iterX2;
  Eigen::VectorXd iterZ = iterX2;
  Eigen::VectorXd iterU = Eigen::VectorXd::Zero(num * 3);

  double rho_admm = 1e3;              // 初始耦合权重
  const double rho_admm_max = 1e7;    // 权重上限
  const double admm_tau = 1.15;       // 外层rho增长速率
  const double mu_cover = 1e6;        // Z-Update 中四球约束惩罚权重

  const int total_lagrange_iterations = 70;
  const int cover_start_iteration = 25;
  const int lagrange_block_max_iteration = 20;
  std::vector<int> FaceIDs(num, -1);
  int qem_iterations_done = 0;

  // 增加记录 L-BFGS 内层迭代步数的计数器，供详细输出使用
  int eval_P_count = 0;
  int eval_Z_count = 0;

  // 用于在控制台输出日志的统计量
  int last_constraint_active_quads = 0;
  double last_constraint_loss_raw = 0.0;
  int last_constraint_num_quads = 0;

  auto face_handle_to_normal = [&](Polyhedron::Face_handle f) -> Eigen::Vector3d {
    auto p1 = f->halfedge()->vertex()->point();
    auto p2 = f->halfedge()->next()->vertex()->point();
    auto p3 = f->halfedge()->next()->next()->vertex()->point();
    Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
    Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
    Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
    Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
    const double nrm = N.norm();
    if (nrm > 1e-30) N /= nrm;
    return N;
  };

  auto sync_sites_from_iterates = [&](Eigen::VectorXd &X) {
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
      Point_T query(X(i * 3 + 0), X(i * 3 + 1), X(i * 3 + 2));
      Point_T closest = tree.closest_point(query);
      X(i * 3 + 0) = closest.x();
      X(i * 3 + 1) = closest.y();
      X(i * 3 + 2) = closest.z();
    }
  };

  // ==========================================
  // ADMM P-Update : min_P E_CWF(P) + (rho/2)||P - Z + U||^2
  // ==========================================
  std::function<double(const Eigen::VectorXd &, Eigen::VectorXd &)> fgm_P = [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
      eval_P_count++;
      eplison = eplison * decay;
      double lossCVT = 0, loss = 0;

      startRVD = clock();
      #pragma omp parallel for
      for (int i = 0; i < num; ++i) {
        Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2));
        Point_T closest = tree.closest_point(query);
        auto tri = tree.closest_point_and_primitive(query);

        int fid = -1;
        auto it = face_id_map.find(static_cast<const void *>(&*tri.second));
        if (it != face_id_map.end()) fid = it->second;
        FaceIDs[i] = fid;
        Nors[i] = face_handle_to_normal(tri.second);
        this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
      }

      this->_RVD.calculate_(this->_sites);
      endRVD = clock();
      RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

      const auto &cells = this->_RVD.get_cells_();
      g.setZero();
      std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());

      omp_set_num_threads(128);
      #pragma omp parallel for reduction(+ : lossCVT, loss)
      for (int i = 0; i < num; ++i) {
        BGAL::_Point3 site = this->_sites[i];
        for (int j = 0; j < (int)cells[i].size(); ++j) {
          auto[a, b, c] = cells[i][j];
          BGAL::_Point3 pa = this->_RVD.vertex_(a);
          BGAL::_Point3 pb = this->_RVD.vertex_(b);
          BGAL::_Point3 pc = this->_RVD.vertex_(c);

          Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D([&, site](BGAL::_Point3 p) {
                const double rho_val = this->_rho(p);
                Eigen::VectorXd r(5);
                BGAL::_Point3 NorTriM = (pb - pa).cross_(pc - pa);
                NorTriM.normalized_();

                r(0) = (eplison * rho_val * ((site - p).sqlength_()));
                r(1) = lambda * (NorTriM.dot_(p - site)) * (NorTriM.dot_(p - site))+
                       eplison * rho_val * ((p - site).sqlength_());

                r(2) = lambda * -2 * NorTriM.x() * (NorTriM.dot_(p - site)) +
                       eplison * rho_val * -2 * (p - site).x();
                r(3) = lambda * -2 * NorTriM.y() * (NorTriM.dot_(p - site)) +
                       eplison * rho_val * -2 * (p - site).y();
                r(4) = lambda * -2 * NorTriM.z() * (NorTriM.dot_(p - site)) +
                       eplison * rho_val * -2 * (p - site).z();
                return r;
              }, pa, pb, pc);

          lossCVT += alpha * inte(0);
          loss += alpha * inte(1);
          gi[i].x() += alpha * inte(2);
          gi[i].y() += alpha * inte(3);
          gi[i].z() += alpha * inte(4);
        }
      }

      double coupling_energy = 0.0;
      for (int i = 0; i < num; ++i) {
        gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
        
        if (rho_admm > 0.0) {
            Eigen::Vector3d p(X(i*3), X(i*3+1), X(i*3+2));
            Eigen::Vector3d z(iterZ(i*3), iterZ(i*3+1), iterZ(i*3+2));
            Eigen::Vector3d u(iterU(i*3), iterU(i*3+1), iterU(i*3+2));
            Eigen::Vector3d diff = p - z + u;
            
            coupling_energy += 0.5 * rho_admm * diff.squaredNorm();
            
            Eigen::Vector3d grad_admm = rho_admm * diff;
            grad_admm -= Nors[i] * (grad_admm.dot(Nors[i]) / Nors[i].squaredNorm());
            gi[i] += grad_admm;
        }

        g(i * 3 + 0) = gi[i].x();
        g(i * 3 + 1) = gi[i].y();
        g(i * 3 + 2) = gi[i].z();
      }

      double total_P_energy = loss + coupling_energy;
      
      std::cout << "[P-Eval " << std::setw(2) << std::setfill('0') << eval_P_count << "] "
                << std::setfill(' ')
                << "E_tot: "  << std::setprecision(8) << total_P_energy
                << " | E_cvt: " << lossCVT / eplison
                << " | E_qe: " << (loss - lossCVT)
                << " | E_coup: " << coupling_energy
                << " | |g|: " << g.norm() << std::endl;

      return total_P_energy;
  };

  // ==========================================
  // ADMM Z-Update : min_Z \mu * H(Z) + (rho/2)||P - Z + U||^2
  // ==========================================
  std::function<double(const Eigen::VectorXd &, Eigen::VectorXd &)> fgm_Z = [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
      eval_Z_count++;
      startRVD = clock();
      #pragma omp parallel for
      for (int i = 0; i < num; ++i) {
          Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2));
          Point_T closest = tree.closest_point(query);
          auto tri = tree.closest_point_and_primitive(query);
          Nors[i] = face_handle_to_normal(tri.second);
          this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
      }
      this->_RVD.calculate_(this->_sites);
      endRVD = clock();
      RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

      std::vector<Sphere::Sphere> spheres;
      build_spheres_from_rvd(this->_sites, this->_RVD, spheres);

      std::vector<Vec3> frozen_poles(num, Vec3::Zero());
      double max_r = 0.0;
      for (int i = 0; i < num; ++i) {
        frozen_poles[i] = to_eigen(spheres[i].max_point);
        max_r = std::max(max_r, std::max(0.0, (double)spheres[i].r));
      }

      const auto rdt_faces = build_rdt_faces_from_edges(num, this->_RVD.get_edges_());
      std::vector<std::array<int, 4>> search_quads;
      build_quads_from_search_faces(rdt_faces, this->_sites, spheres, search_quads);

      const double eps_cover = 1e-4 * std::max(1e-12, max_r) * std::max(1e-12, max_r);
      const DynamicRadiusEval constraint_eval = evaluate_dynamic_radius_surrogate(
          this->_sites, frozen_poles, Nors, search_quads, eps_cover, true);

      double cover_loss_raw = constraint_eval.total_loss;
      double lossCover = mu_cover * cover_loss_raw;

      g.setZero();
      double coupling_energy = 0.0;
      
      for (int i = 0; i < num; ++i) {
          Eigen::Vector3d p(iterP(i*3), iterP(i*3+1), iterP(i*3+2));
          Eigen::Vector3d z(X(i*3), X(i*3+1), X(i*3+2));
          Eigen::Vector3d u(iterU(i*3), iterU(i*3+1), iterU(i*3+2));
          
          Eigen::Vector3d diff = p - z + u;
          coupling_energy += 0.5 * rho_admm * diff.squaredNorm();
          
          Eigen::Vector3d grad_admm = -rho_admm * diff;
          Eigen::Vector3d grad_cover = mu_cover * constraint_eval.grads.row(i).transpose();
          
          Eigen::Vector3d gi = grad_cover + grad_admm;
          
          const Eigen::Vector3d n = Nors[i];
          const double nn = n.squaredNorm();
          if (nn > 1e-30) {
              gi -= n * (gi.dot(n) / nn);
          }
          
          g(i*3+0) = gi.x();
          g(i*3+1) = gi.y();
          g(i*3+2) = gi.z();
      }

      last_constraint_active_quads = constraint_eval.active_quads;
      last_constraint_loss_raw = cover_loss_raw;
      last_constraint_num_quads = (int)search_quads.size();

      double total_Z_energy = lossCover + coupling_energy;

      std::cout << "[Z-Eval " << std::setw(2) << std::setfill('0') << eval_Z_count << "] "
                << std::setfill(' ')
                << "E_tot: "  << std::setprecision(4) << total_Z_energy
                << " | E_cover: " << lossCover
                << " | E_coup: " << coupling_energy
                << " | Quads: " << constraint_eval.active_quads << "/" << search_quads.size()
                << " | min_g: " << std::fixed << std::setprecision(6) << constraint_eval.min_g
                << " | |g|: " << std::scientific << std::setprecision(4) << g.norm() << std::endl;

      return total_Z_energy;
  };

  _RVD.calculate_(_sites);
  const std::string final_outdir =
      (std::filesystem::current_path() / "data" / "Block_New").string();

  auto flush_outputs = [&](int step) {
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
      Point_T query(iterX2(i * 3 + 0), iterX2(i * 3 + 1), iterX2(i * 3 + 2));
      Point_T closest = tree.closest_point(query);
      auto tri = tree.closest_point_and_primitive(query);

      int fid = -1;
      auto it = face_id_map.find(static_cast<const void *>(&*tri.second));
      if (it != face_id_map.end()) fid = it->second;
      FaceIDs[i] = fid;
      Nors[i] = face_handle_to_normal(tri.second);

      _sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
    }

    _RVD.calculate_(_sites);
    std::vector<Sphere::Sphere> final_spheres;
    build_spheres_from_rvd(_sites, _RVD, final_spheres);

    std::string sphere_path =
        final_outdir + "/Sphere_" + modelname + ".csv";
    if (step > 2) {
      sphere_path = final_outdir + "/Sphere_" + modelname + "_Iter" +
                    std::to_string(step - 3) + ".csv";
    }

    std::ofstream out(sphere_path, std::ios::out | std::ios::trunc);
    if (!out) {
      std::cerr << "[io] open failed, errno=" << errno << " (" << std::strerror(errno) << ")\n";
    } else {
      out << std::setprecision(17);
      for (int i = 0; i < (int)final_spheres.size(); ++i) {
        Eigen::Vector3d n = Nors[i].normalized();
        out << final_spheres[i].c.x() << "," << final_spheres[i].c.y() << ","
            << final_spheres[i].c.z() << "," << final_spheres[i].r << ","
            << FaceIDs[i] << "," << n.x() << "," << n.y() << "," << n.z()
            << "\n";
      }
    }
    OutputMesh(_sites, _RVD, num_sites, final_outdir, modelname, step);
  };

  start = clock();
  bool constraint_satisfied = false;

  while (qem_iterations_done < total_lagrange_iterations) {
    eplison *= decay; 
    
    bool cover_enabled_this_round = (qem_iterations_done >= cover_start_iteration);

    if (!cover_enabled_this_round) {
        std::cout << "\n--------------------------------------------------------------\n"
                  << "[Warmup Phase] Outer Iteration " << qem_iterations_done << " / " << cover_start_iteration << "\n"
                  << "--------------------------------------------------------------\n";

        double old_rho = rho_admm;
        rho_admm = 0.0; 
        
        BGAL::_LBFGS::_Parameter p_para(_para);
        p_para.max_iteration = lagrange_block_max_iteration;
        BGAL::_LBFGS lagrange_lbfgs(p_para);
        
        eval_P_count = 0; 
        int steps_done = lagrange_lbfgs.minimize(fgm_P, iterP);
        sync_sites_from_iterates(iterP);

        std::cout << "-> Block finished in " << steps_done << " inner L-BFGS steps.\n";

        rho_admm = old_rho; 
        iterZ = iterP;      
        iterU.setZero();

        qem_iterations_done += std::max(steps_done, 1);
        iterX2 = iterP;     
        flush_outputs(qem_iterations_done + 2);

    } else {
        std::cout << "\n--------------------------------------------------------------\n"
                  << "[ADMM Phase] Outer Iteration " << qem_iterations_done << " / " << total_lagrange_iterations << "\n"
                  << " Current ADMM Penalty (rho): " << std::scientific << std::setprecision(4) << rho_admm << "\n"
                  << "--------------------------------------------------------------\n";

        BGAL::_LBFGS::_Parameter admm_para(_para);
        admm_para.max_iteration = lagrange_block_max_iteration;
        
        std::cout << "  >>> Start P-Update (Optimize Geometry + Uniformity) <<<\n";
        eval_P_count = 0;
        BGAL::_LBFGS lbfgs_P(admm_para);
        int p_steps = lbfgs_P.minimize(fgm_P, iterP);
        sync_sites_from_iterates(iterP);

        std::cout << "  >>> Start Z-Update (Enforce Topology Constraint) <<<\n";
        Eigen::VectorXd old_Z = iterZ; 
        eval_Z_count = 0;
        BGAL::_LBFGS lbfgs_Z(admm_para);
        int z_steps = lbfgs_Z.minimize(fgm_Z, iterZ);
        sync_sites_from_iterates(iterZ);

        iterU += (iterP - iterZ);

        double num_points = static_cast<double>(num * 3);
        double primal_res = (iterP - iterZ).norm() / std::sqrt(num_points);
        double dual_res   = rho_admm * (iterZ - old_Z).norm() / std::sqrt(num_points);

        rho_admm = std::min(rho_admm * admm_tau, rho_admm_max);

        qem_iterations_done += std::max(p_steps, 1);
        iterX2 = iterP; 
        flush_outputs(qem_iterations_done + 2);

        std::cout << "\n-> ADMM Block Summary:\n"
                  << "   P_Steps: " << p_steps << ", Z_Steps: " << z_steps << "\n"
                  << "   Primal Residual ||P - Z|| (avg): " << std::scientific << std::setprecision(4) << primal_res << "\n"
                  << "   Dual Residual rho||Z - Z_old|| (avg): " << dual_res << "\n"
                  << "   Violating Quads: " << last_constraint_active_quads << " / " << last_constraint_num_quads << "\n";

        if (last_constraint_active_quads == 0 && primal_res < 1e-4) {
            constraint_satisfied = true;
            break;
        }
    }
  }

  if (!constraint_satisfied) {
    std::cout << "\n[ADMM] Optimization stopped without strictly finding a feasible set. \n"
              << "Last Violating Quads=" << last_constraint_active_quads 
              << " / " << last_constraint_num_quads << std::endl;
  } else {
    std::cout << "\n[ADMM] Successfully satisfied topology constraints! " 
              << "Violations = 0, Primal Residual converged." << std::endl;
  }

  flush_outputs(2);

  end = clock();
  allTime += (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << "\n--- Optimization Finished ---\n"
            << "Total Time: " << std::fixed << std::setprecision(2) << allTime << " s\n"
            << "RVD Calculation Time: " << RVDtime << " s\n"
            << "L-BFGS Solver Time: " << allTime - RVDtime << " s\n" << std::endl;

}
} // namespace BGAL
