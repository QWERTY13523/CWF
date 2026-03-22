#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <fstream>
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

  const double sharp_cos =
      std::cos(sharp_angle_deg * 3.14159265358979323846 / 180.0);

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
          FeatureSegment{model.vertex_(e._id_left_vertex),
                         model.vertex_(e._id_right_vertex)});
    }
  }
  return segments;
}

static inline double stable_sigmoid(double x) {
  if (x >= 0.0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    double z = std::exp(x);
    return z / (1.0 + z);
  }
}

static inline double stable_softplus(double x) {
  if (x > 50.0) return x;
  if (x < -50.0) return std::exp(x);
  return std::log1p(std::exp(x));
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

static inline Eigen::Vector3d solve_damped_pseudoinverse(
    const Eigen::Matrix3d &A, const Eigen::Vector3d &b, double damp) {
  Eigen::Matrix3d H = A.transpose() * A;
  double d2 = std::max(1e-12, damp * damp);
  H(0, 0) += d2;
  H(1, 1) += d2;
  H(2, 2) += d2;
  return H.inverse() * (A.transpose() * b);
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
      if (rdt_edges.find(std::make_pair(std::min(pid, e.first),
                                        std::max(pid, e.first))) ==
          rdt_edges.end()) {
        continue;
      }
      if (rdt_edges.find(std::make_pair(std::min(pid, e.second),
                                        std::max(pid, e.second))) ==
          rdt_edges.end()) {
        continue;
      }

      const int hi = std::max(pid, std::max(e.first, e.second));
      const int lo = std::min(pid, std::min(e.first, e.second));
      const int mid = pid + e.first + e.second - hi - lo;
      rdt_faces.insert(MyFace(hi, mid, lo));
    }
  }

  std::vector<Eigen::Vector3i> tris;
  tris.reserve(rdt_faces.size());
  for (const auto &f : rdt_faces) {
    tris.emplace_back(f.p.x(), f.p.y(), f.p.z());
  }
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
  std::set<std::array<int, 4>, QuadKeyLess> uniq;

  for (const auto &f : faces) {
    const int i = f.x();
    const int j = f.y();
    const int k = f.z();
    if (i < 0 || j < 0 || k < 0 || i >= n || j >= n || k >= n) continue;
    if (i == j || i == k || j == k) continue;

    const double ri = std::max(0.0, (double)spheres[i].r);
    const double rj = std::max(0.0, (double)spheres[j].r);
    const double rk = std::max(0.0, (double)spheres[k].r);

    const double Ri = ri + global_rmax;
    const double Rj = rj + global_rmax;
    const double Rk = rk + global_rmax;

    int pivot = i;
    double Rpivot = Ri;
    if (Rj < Rpivot) {
      pivot = j;
      Rpivot = Rj;
    }
    if (Rk < Rpivot) {
      pivot = k;
      Rpivot = Rk;
    }

    std::vector<BGAL::_Point3> query(1, sites[pivot]);
    std::vector<int> cand = kdtree.rsearch_(query, Rpivot);

    const double Ri2 = Ri * Ri;
    const double Rj2 = Rj * Rj;
    const double Rk2 = Rk * Rk;
    const double tol = 1e-12 * std::max({1.0, Ri2, Rj2, Rk2});

    for (int l : cand) {
      if (l == i || l == j || l == k) continue;
      if ((sites[l] - sites[i]).sqlength_() > Ri2 + tol) continue;
      if ((sites[l] - sites[j]).sqlength_() > Rj2 + tol) continue;
      if ((sites[l] - sites[k]).sqlength_() > Rk2 + tol) continue;

      std::array<int, 4> q{i, j, k, l};
      std::sort(q.begin(), q.end());
      uniq.insert(q);
    }
  }

  quads.reserve(uniq.size());
  for (const auto &q : uniq) quads.push_back(q);
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
      eval.grads.row(sid) +=
          (-4.0 * violation * (frozen_poles[sid] - p_bar)).transpose();
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

// 专用于加速优化的全局 Adam 状态结构体
struct AdamState {
    Eigen::VectorXd m;
    Eigen::VectorXd v;
    int t = 0;

    // 记录 cover 阶段 active 区域梯度 RMS 的指数滑动平均
    double cover_grad_rms_ema = 0.0;

    void reset(int size) {
        m = Eigen::VectorXd::Zero(size);
        v = Eigen::VectorXd::Zero(size);
        t = 0;
        cover_grad_rms_ema = 0.0;
    }
};

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
  //fs::path obj("./data/bunny.obj");
  fs::path base = obj.parent_path();
  if (pointsName == nullptr) {
    inPointsName = (base / ("n" + std::to_string(num_sites) + "_" + modelname +
                           "_inputPoints.xyz")).string();
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
    if (has_normal) {
      Nors.push_back(Eigen::Vector3d(nx, ny, nz));
    } else {
      Nors.push_back(Eigen::Vector3d::Zero());
    }
  }
  inPoints.close();
  std::cout << "Pts.size(): " << Pts.size() << std::endl;

  if (pointsName != nullptr) {
    num_sites = static_cast<int>(Pts.size());
  }

  int num = (int)Pts.size();
  std::cout << "\nBegin CWF.\n" << std::endl;

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
  int Fnum = 4;
  double alpha = 1.0, eplison = 1.0, lambda = 1.0;
  double decay = 0.95;

  const double cover_weight = 10.0;
#if 0
  const auto bbox = _model.bounding_box_();
  const double bbox_diag = (bbox.second - bbox.first).length_();
  const double cover_tau_m_scale = 0.02; 
  const double cover_tau_h_scale = 0.01; 
  const double cover_margin_scale = 0.02; 
  const double cover_push_out_scale = 0.0002; 
  const double cover_damp   = 1e-6;
#endif
  double lagrange_multiplier = 0.0;  // ALM multiplier mu for H(P)=0
  double rho_cover = 10.0;           // ALM penalty parameter rho
  double prev_cover_energy = std::numeric_limits<double>::infinity();
  const double eta_cover = 0.5;
  const double beta_cover = 2.0;
  const double rho_cover_max = 1e8;
  const double cover_target_ratio = 0.2;
  const double cover_target_floor = 1e-3;
  const double cover_scale_min = 0.05;
  const double cover_scale_max = 1e6;
  const double cover_scale_ema_decay = 0.8;
  double cover_scale_ema = 1.0;
  bool cover_scale_initialized = false;
  const int total_lagrange_iterations = 70;
  const int cover_start_iteration = 30;
  const int lagrange_block_max_iteration = 20;
  std::vector<int> FaceIDs(num, -1);
  bool suppress_intermediate_output = false;
  int qem_iterations_done = 0;
  bool cover_enabled_for_current_block = false;
  bool cover_phase_initialized = false;

  int cover_invalid_seed_count = -1;
  double cover_last_energy = 0.0;
  double cover_last_max_violation = 0.0;
  double cover_last_covered_ratio = 0.0;
  double cover_last_score = 0.0;
  int last_constraint_active_quads = 0;
  double last_constraint_loss = 0.0;
  double last_constraint_loss_raw = 0.0;
  double last_constraint_loss_scale = 1.0;
  double last_constraint_min_g = 0.0;
  double last_constraint_r = 1.0;
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

  auto cover_progress_score = [&](double max_violation, double avg_radius, double covered_ratio) -> double {
    const double normalized_violation = max_violation / std::max(avg_radius, 1e-12);
    return 0.2 * normalized_violation + 0.8 * covered_ratio;
  };

  auto sync_sites_from_iterates = [&](const Eigen::VectorXd &X) {
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
      Point_T query(X(i * 3 + 0), X(i * 3 + 1), X(i * 3 + 2));
      Point_T closest = tree.closest_point(query);
      this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
      iterX2(i * 3 + 0) = closest.x();
      iterX2(i * 3 + 1) = closest.y();
      iterX2(i * 3 + 2) = closest.z();
    }
  };


  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm2 =[&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
        if (qem_iterations_done < total_lagrange_iterations) {
          eplison *= decay;
        }
        double lossCVT = 0, loss = 0;
        double lossCover = 0;

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

        if (!suppress_intermediate_output) {
          Fnum++;
          OutputMesh(this->_sites, this->_RVD, num_sites,
                     (std::filesystem::current_path() / "data" / "Block_New").string(),
                     modelname, Fnum);
          // OutputMesh(this->_sites, this->_RVD, num_sites,
          //            (std::filesystem::current_path() / "data" / "Bunny").string(),
          //            modelname, Fnum);
        }

        endRVD = clock();
        RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

        const auto &cells = this->_RVD.get_cells_();
        double energy = 0.0;
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
                },
                pa, pb, pc);

            lossCVT += alpha * inte(0);
            loss += alpha * inte(1);
            gi[i].x() += alpha * inte(2);
            gi[i].y() += alpha * inte(3);
            gi[i].z() += alpha * inte(4);
          }
        }

        for (int i = 0; i < num; ++i) {
          gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
          g(i * 3 + 0) += gi[i].x();
          g(i * 3 + 1) += gi[i].y();
          g(i * 3 + 2) += gi[i].z();
        }

        std::vector<Sphere::Sphere> spheres;
        build_spheres_from_rvd(this->_sites, this->_RVD, spheres);

        std::vector<Vec3> frozen_poles(num, Vec3::Zero());
        double max_r = 0.0;
        for (int i = 0; i < num; ++i) {
          frozen_poles[i] = to_eigen(spheres[i].max_point);
          max_r = std::max(max_r, std::max(0.0, (double)spheres[i].r));
        }

        const auto rdt_faces =
            build_rdt_faces_from_edges(num, this->_RVD.get_edges_());
        std::vector<std::array<int, 4>> search_quads;
        build_quads_from_search_faces(rdt_faces, this->_sites, spheres, search_quads);

        const double eps_cover =
            1e-4 * std::max(1e-12, max_r) * std::max(1e-12, max_r);
        const DynamicRadiusEval constraint_eval =
            evaluate_dynamic_radius_surrogate(this->_sites, frozen_poles, Nors,
                                             search_quads, eps_cover, true);

        const bool cover_enabled = cover_enabled_for_current_block;
        const double cover_loss_raw = constraint_eval.total_loss;
        const double base_energy_current = std::max(1e-12, loss);
        const double cover_target_energy =
            std::max(cover_target_floor, cover_target_ratio * base_energy_current);
        const double cover_loss_scale_raw =
            cover_target_energy / std::max(cover_loss_raw, 1e-18);
        const double cover_loss_scale_clamped = std::clamp(
            cover_loss_scale_raw,
            cover_scale_min,
            cover_scale_max);
        double cover_loss_scale = cover_loss_scale_clamped;
        if (cover_enabled) {
          if (cover_scale_initialized) {
            cover_loss_scale =
                cover_scale_ema_decay * cover_scale_ema +
                (1.0 - cover_scale_ema_decay) * cover_loss_scale_clamped;
          } else {
            cover_scale_initialized = true;
          }
          cover_scale_ema = cover_loss_scale;
        }
        lossCover = cover_loss_raw * cover_loss_scale;
        if (cover_enabled) {
          const double cover_coeff =
              (lagrange_multiplier + rho_cover * lossCover) * cover_loss_scale;
          for (int i = 0; i < num; ++i) {
            g.segment<3>(i * 3) +=
                cover_coeff * constraint_eval.grads.row(i).transpose();
          }
        }

        last_constraint_active_quads = constraint_eval.active_quads;
        last_constraint_loss = lossCover;
        last_constraint_loss_raw = cover_loss_raw;
        last_constraint_loss_scale = cover_loss_scale;
        last_constraint_min_g = constraint_eval.min_g;
        last_constraint_r = std::max(1e-12, max_r);
        last_constraint_num_quads = (int)search_quads.size();
        cover_invalid_seed_count = constraint_eval.active_quads;
        cover_last_energy = lossCover;
        cover_last_max_violation = std::max(0.0, -(constraint_eval.min_g + eps_cover));
        cover_last_covered_ratio =
            (search_quads.empty())
                ? 0.0
                : (double)constraint_eval.active_quads /
                      (double)std::max(1, (int)search_quads.size());
        cover_last_score = cover_progress_score(cover_last_max_violation,
                                                last_constraint_r,
                                                cover_last_covered_ratio);

        energy += loss;
        if (cover_enabled) {
          energy += lagrange_multiplier * lossCover
                 + 0.5 * rho_cover * lossCover * lossCover;
        }

        std::cout << std::setprecision(7)
                  << "energy: " << energy
                  << " LossCVT: " << lossCVT / eplison
                  << " LossQE: "  << (loss - lossCVT) / lambda
                  << " LossCover(raw): " << cover_loss_raw
                  << " LossCover(scaled): " << lossCover
                  << " CoverScale: " << cover_loss_scale
                  << " CoverOn: " << (cover_enabled ? 1 : 0)
                  << " ActiveQuads: " << constraint_eval.active_quads
                  << " Quads: " << search_quads.size()
                  << " r: " << last_constraint_r
                  << " Lambda_CVT: " << eplison
                  << " Lambda_Cover: " << lagrange_multiplier
                  << " Rho_Cover: " << rho_cover << std::endl;

        return energy;
      };


  // 只保留“前 50 次拉格朗日乘子法”主流程；
  // 其余旧的 cover/Adam 优化代码先整体注释掉。
#if 0
  struct CoverSeedConstraint {
    int i = -1;
    int j = -1;
    int k = -1;
    int seed_index = 0; 
    std::vector<int> candidate_ids;
  };

  struct CoverActiveProblem {
    Eigen::VectorXd baseX;
    std::vector<Eigen::Vector3d> C;
    std::vector<double> R;
    std::vector<int> active_ids;
    std::vector<int> global_to_local;
    std::vector<CoverSeedConstraint> constraints;
    std::vector<Eigen::Vector3d> base_normals;
    double avg_radius = 0.0;
    double violation_scale = 1e-12;
    double push_out_dist = 0.0; 
    double cover_delta = 0.0;
    double tau_m = 1e-12;
    double tau_h = 1e-12;
    double cover_margin = 0.0;
    double global_energy = 0.0;
    double global_max_violation = 0.0;
    int global_invalid_seed_count = 0;
    int global_total_seed_count = 0;
    double global_covered_ratio = 0.0;
    double global_score = 0.0;
  };

  auto project_all_sites_to_surface_and_update_state = [&](Eigen::VectorXd &X) {
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

      X(i * 3 + 0) = closest.x();
      X(i * 3 + 1) = closest.y();
      X(i * 3 + 2) = closest.z();
      this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
    }
  };

  auto project_active_sites_to_surface_and_update_state =
    [&](Eigen::VectorXd &X, const std::vector<int> &active_ids) {
      #pragma omp parallel for
      for (int local_idx = 0; local_idx < (int)active_ids.size(); ++local_idx) {
        const int i = active_ids[local_idx];

        Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2));
        Point_T closest = tree.closest_point(query);
        auto tri = tree.closest_point_and_primitive(query);

        int fid = -1;
        auto it = face_id_map.find(static_cast<const void *>(&*tri.second));
        if (it != face_id_map.end()) fid = it->second;
        FaceIDs[i] = fid;
        Nors[i] = face_handle_to_normal(tri.second);

        X(i * 3 + 0) = closest.x();
        X(i * 3 + 1) = closest.y();
        X(i * 3 + 2) = closest.z();
        this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
      }
    };


  auto build_cover_active_problem = [&](Eigen::VectorXd &X,
                                     CoverActiveProblem &problem,
                                     bool update_cover_globals = true,
                                     bool verbose_output = true,
                                     bool need_project_all = true) -> bool {
  constexpr int kCoverNeighborK = 24;

  if (need_project_all) {
    project_all_sites_to_surface_and_update_state(X);
  }

  startRVD = clock();
  this->_RVD.calculate_(this->_sites);
  endRVD = clock();
  RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

  std::vector<Sphere::Sphere> spheres(num);
  const auto &edges = this->_RVD.get_edges_();
  build_spheres_from_rvd(this->_sites, this->_RVD, spheres);
  const std::vector<Eigen::Vector3i> rdt_tris = build_rdt_faces_from_edges(num, edges);

  problem = CoverActiveProblem();
  problem.baseX = X;
  problem.C.resize(num);
  problem.R.resize(num, 0.0);
  problem.global_to_local.assign(num, -1);

  std::vector<BGAL::_Point3> center_points(num);
  double radius_sum = 0.0;
  for (int i = 0; i < num; ++i) {
    problem.C[i] = Eigen::Vector3d(this->_sites[i].x(), this->_sites[i].y(), this->_sites[i].z());
    problem.R[i] = std::max(0.0, (double)spheres[i].r);
    center_points[i] = this->_sites[i];
    radius_sum += problem.R[i];
  }

  problem.avg_radius = num > 0 ? radius_sum / static_cast<double>(num) : 0.0;
  problem.violation_scale = std::max(1e-12, problem.avg_radius);
  problem.push_out_dist = cover_push_out_scale * problem.avg_radius;
  problem.tau_m = std::max(1e-12, cover_tau_m_scale);
  problem.tau_h = std::max(1e-12, cover_tau_h_scale);
  problem.cover_margin = std::max(1e-12, cover_margin_scale);

  BGAL::_KDTree center_tree(center_points);

  std::vector<std::vector<int>> adjacency(num);
  for (const auto &t : rdt_tris) {
    const int a = t.x(); const int b = t.y(); const int c = t.z();
    adjacency[a].push_back(b); adjacency[a].push_back(c);
    adjacency[b].push_back(a); adjacency[b].push_back(c);
    adjacency[c].push_back(a); adjacency[c].push_back(b);
  }

  std::unordered_set<int> active_seed_vertices;
  std::unordered_set<int> active_cover_vertices;

  #pragma omp parallel
  {
    struct CandidateInfo {
      int id; double raw_v; double norm_v;
    };
    std::vector<CandidateInfo> candidates;
    candidates.reserve(24);

    std::vector<CoverSeedConstraint> local_constraints;
    std::unordered_set<int> local_seed_vertices;
    std::unordered_set<int> local_cover_vertices;
    double local_global_energy = 0.0;
    double local_max_violation = 0.0;
    int local_invalid_seed_count = 0;
    int local_total_seed_count = 0;

    #pragma omp for nowait
    for (int t_idx = 0; t_idx < (int)rdt_tris.size(); ++t_idx) {
      const auto &t = rdt_tris[t_idx];
      const int i = t.x();
      const int j = t.y();
      const int k = t.z();
      if (problem.R[i] <= 0 || problem.R[j] <= 0 || problem.R[k] <= 0) continue;

      Eigen::Vector3d s1, s2;
      if (!intersect_three_spheres(problem.C[i], problem.R[i], problem.C[j], problem.R[j],
                                   problem.C[k], problem.R[k], s1, s2)) {
        continue;
      }

      Eigen::Vector3d seeds[2] = {s1, s2};
      for (int si = 0; si < 2; ++si) {
        ++local_total_seed_count;
        const Eigen::Vector3d s = seeds[si];
        candidates.clear();

        const std::vector<int> nearest_ids = center_tree.nsearch_(
            {BGAL::_Point3(s.x(), s.y(), s.z())}, std::min(num, kCoverNeighborK + 3));

        double max_raw_v = -std::numeric_limits<double>::infinity();
        double max_norm_v = -std::numeric_limits<double>::infinity();

        for (int l : nearest_ids) {
          if (l < 0 || l >= num || l == i || l == j || l == k || problem.R[l] <= 0) continue;
          const Eigen::Vector3d d = s - problem.C[l];
          const double dist = d.norm();

          const double raw_v = (problem.R[l]) - dist;
          const double norm_v = raw_v / problem.violation_scale;

          candidates.push_back({l, raw_v, norm_v});
          max_raw_v = std::max(max_raw_v, raw_v);
          max_norm_v = std::max(max_norm_v, norm_v);
        }

        if (candidates.empty() || max_norm_v <= 0.0) continue;

        ++local_invalid_seed_count;
        local_max_violation = std::max(local_max_violation, max_raw_v);

        double exp_sum = 0.0;
        for (const auto &cand : candidates) {
          exp_sum += std::exp((cand.norm_v - max_norm_v) / problem.tau_m);
        }
        const double m_val = max_norm_v + problem.tau_m * std::log(std::max(exp_sum, 1e-300));
        const double z = (m_val + problem.cover_margin) / problem.tau_h;
        const double phi = problem.tau_h * stable_softplus(z);
        local_global_energy += cover_weight * phi;

        CoverSeedConstraint constraint;
        constraint.i = i; constraint.j = j; constraint.k = k;
        constraint.seed_index = si;
        constraint.candidate_ids.reserve(candidates.size());
        for (const auto &cand : candidates) {
          constraint.candidate_ids.push_back(cand.id);
          if (cand.raw_v > 0.0) local_cover_vertices.insert(cand.id);
        }
        local_constraints.push_back(std::move(constraint));

        local_seed_vertices.insert(i);
        local_seed_vertices.insert(j);
        local_seed_vertices.insert(k);
      }
    }

    #pragma omp critical
    {
      problem.global_energy += local_global_energy;
      problem.global_invalid_seed_count += local_invalid_seed_count;
      problem.global_total_seed_count += local_total_seed_count;
      problem.global_max_violation = std::max(problem.global_max_violation, local_max_violation);
      problem.constraints.insert(problem.constraints.end(), local_constraints.begin(), local_constraints.end());
      active_seed_vertices.insert(local_seed_vertices.begin(), local_seed_vertices.end());
      active_cover_vertices.insert(local_cover_vertices.begin(), local_cover_vertices.end());
    }
  }

  problem.global_covered_ratio =
      (problem.global_total_seed_count > 0)
          ? static_cast<double>(problem.global_invalid_seed_count) /
                static_cast<double>(problem.global_total_seed_count)
          : 0.0;
  problem.global_score = cover_progress_score(problem.global_max_violation,
                                             problem.avg_radius,
                                             problem.global_covered_ratio);

  if (problem.global_invalid_seed_count == 0) {
    if (update_cover_globals) {
      cover_invalid_seed_count = 0;
      cover_last_energy = 0.0;
      cover_last_max_violation = 0.0;
      cover_last_covered_ratio = 0.0;
      cover_last_score = 0.0;
    }
    if (verbose_output) {
      std::cout << std::setprecision(7)
                << "cover-only energy: 0 LossCover: 0 maxViolation: 0 invalidSeeds: 0" << std::endl;
    }
    return true;
  }

  std::unordered_set<int> active_set = active_seed_vertices;
  active_set.insert(active_cover_vertices.begin(), active_cover_vertices.end());
  const std::vector<int> seed_snapshot(active_set.begin(), active_set.end());
  for (int id : seed_snapshot) {
    for (int nb : adjacency[id]) active_set.insert(nb);
  }

  problem.active_ids.assign(active_set.begin(), active_set.end());
  std::sort(problem.active_ids.begin(), problem.active_ids.end());
  for (int local_idx = 0; local_idx < (int)problem.active_ids.size(); ++local_idx) {
    problem.global_to_local[problem.active_ids[local_idx]] = local_idx;
  }

  problem.base_normals.resize(problem.active_ids.size());
  for (int local_idx = 0; local_idx < (int)problem.active_ids.size(); ++local_idx) {
    problem.base_normals[local_idx] = Nors[problem.active_ids[local_idx]];
  }

  if (update_cover_globals) {
    cover_invalid_seed_count = problem.global_invalid_seed_count;
    cover_last_energy = problem.global_energy;
    cover_last_max_violation = problem.global_max_violation;
    cover_last_covered_ratio = problem.global_covered_ratio;
    cover_last_score = problem.global_score;
  }

  if (verbose_output) {
    std::cout << std::setprecision(7)
              << "cover-only energy: " << problem.global_energy
              << " maxViolation: " << problem.global_max_violation
              << " invalidSeeds: " << problem.global_invalid_seed_count << std::endl;
  }

  return true;
};

  auto evaluate_cover_local_surrogate =
      [&](const CoverActiveProblem &problem,
          const Eigen::VectorXd &XA,
          Eigen::VectorXd &gA) -> double {
        // 由于我们在 Adam 中始终使用投影后的基准点，此时无须重新执行 AABB Tree 精确投影，直接复用当前法向量即可
        std::vector<Eigen::Vector3d> Ccur = problem.C;
        for (int local_idx = 0; local_idx < (int)problem.active_ids.size(); ++local_idx) {
          const int global_id = problem.active_ids[local_idx];
          Ccur[global_id] = Eigen::Vector3d(XA(local_idx * 3 + 0),
                                            XA(local_idx * 3 + 1),
                                            XA(local_idx * 3 + 2));
        }

        gA.setZero(XA.size());
        std::vector<Eigen::Vector3d> localGi(problem.active_ids.size(), Eigen::Vector3d::Zero());
        double lossCover = 0.0;

        struct CoverCandidate {
          int id; Eigen::Vector3d d; double raw_v; double norm_v; double dist;
        };
        std::vector<CoverCandidate> candidates;
        candidates.reserve(64);

        for (const auto &constraint : problem.constraints) {
          const int i = constraint.i; const int j = constraint.j; const int k = constraint.k;
          if (problem.R[i] <= 0 || problem.R[j] <= 0 || problem.R[k] <= 0) continue;

          Eigen::Vector3d s1, s2;
          if (!intersect_three_spheres(Ccur[i], problem.R[i], Ccur[j], problem.R[j],
                                       Ccur[k], problem.R[k], s1, s2)) {
            continue;
          }

          const Eigen::Vector3d s = (constraint.seed_index == 0) ? s1 : s2;
          candidates.clear();

          double max_raw_v = -std::numeric_limits<double>::infinity();
          double max_norm_v = -std::numeric_limits<double>::infinity();
          for (int l : constraint.candidate_ids) {
            if (l < 0 || l >= num || l == i || l == j || l == k || problem.R[l] <= 0) continue;
            const Eigen::Vector3d d = s - Ccur[l];
            const double dist = d.norm();
            
            // 【修改4】：同步更改为加上外推距离
            const double raw_v = (problem.R[l]) - dist;
            const double norm_v = raw_v / problem.violation_scale;
            
            candidates.push_back({l, d, raw_v, norm_v, dist});
            max_raw_v = std::max(max_raw_v, raw_v);
            max_norm_v = std::max(max_norm_v, norm_v);
          }
          if (candidates.empty() || max_norm_v <= 0.0) continue;

          double exp_sum = 0.0;
          for (const auto &cand : candidates) {
            exp_sum += std::exp((cand.norm_v - max_norm_v) / problem.tau_m);
          }
          const double m_val = max_norm_v + problem.tau_m * std::log(std::max(exp_sum, 1e-300));
          const double softplus_arg = (m_val + problem.cover_margin) / problem.tau_h;
          const double phi = problem.tau_h * stable_softplus(softplus_arg);
          const double sig = stable_sigmoid(softplus_arg);
          const double dE_dm = cover_weight * sig;

          lossCover += cover_weight * phi;

          Eigen::Vector3d seed_grad_s = Eigen::Vector3d::Zero();
          for (const auto &cand : candidates) {
            const double softmax_weight = std::exp((cand.norm_v - m_val) / problem.tau_m);
            const double dE_draw_v = (dE_dm * softmax_weight) / problem.violation_scale;

            Eigen::Vector3d d_normed = Eigen::Vector3d::Zero();
            if (cand.dist > 1e-12) d_normed = cand.d / cand.dist;

            seed_grad_s += dE_draw_v * (-d_normed);
            const int local_l = problem.global_to_local[cand.id];
            if (local_l >= 0) localGi[local_l] += dE_draw_v * d_normed;
          }

          const Eigen::Vector3d di = s - Ccur[i];
          const Eigen::Vector3d dj = s - Ccur[j];
          const Eigen::Vector3d dk = s - Ccur[k];
          Eigen::Matrix3d JT;
          JT.col(0) = 2.0 * di;
          JT.col(1) = 2.0 * dj;
          JT.col(2) = 2.0 * dk;

          const Eigen::Vector3d yvec = solve_damped_pseudoinverse(JT, seed_grad_s, cover_damp);

          const int local_i = problem.global_to_local[i];
          const int local_j = problem.global_to_local[j];
          const int local_k = problem.global_to_local[k];
          if (local_i >= 0) localGi[local_i] += 2.0 * di * yvec(0);
          if (local_j >= 0) localGi[local_j] += 2.0 * dj * yvec(1);
          if (local_k >= 0) localGi[local_k] += 2.0 * dk * yvec(2);
        }

        for (int local_idx = 0; local_idx < (int)problem.active_ids.size(); ++local_idx) {
          const Eigen::Vector3d &n = problem.base_normals[local_idx];
          const double nn = n.squaredNorm();
          if (nn > 1e-30) {
            localGi[local_idx] -= n * (localGi[local_idx].dot(n) / nn);
          }
          gA(local_idx * 3 + 0) = localGi[local_idx].x();
          gA(local_idx * 3 + 1) = localGi[local_idx].y();
          gA(local_idx * 3 + 2) = localGi[local_idx].z();
        }

        return lossCover;
      };

  struct CoverStepResult {
    int step_count = 0;  
    bool feasible_done = false;
    bool move_too_small = false;
    double max_move = 0.0;
    int invalid_seeds = 0; 
  };

  // --------------------------------------------------------------------------
  // 全新的 Adam 优化器 (专职负责极其坑洼的 Softplus Cover能量)
  // --------------------------------------------------------------------------
 auto cover_adam_step = [&](Eigen::VectorXd &X, AdamState &adam) -> CoverStepResult {
  CoverStepResult result;
  CoverActiveProblem problem;

  // 只在 outer step 开头做一次全局 build
  if (!build_cover_active_problem(X, problem, true, false, true)) {
    return result;
  }

  result.invalid_seeds = problem.global_invalid_seed_count;

  if (problem.global_invalid_seed_count == 0 || problem.active_ids.empty()) {
    result.feasible_done = true;

    // 打印一次真实状态
    build_cover_active_problem(X, problem, true, true, false);
    result.invalid_seeds = problem.global_invalid_seed_count;
    return result;
  }

  const int active_n = (int)problem.active_ids.size();
  const double avg_r = std::max(1e-12, problem.avg_radius);

  // ----------------------------
  // 一个 outer step 内做多个 inner Adam 小步
  // ----------------------------
  const int inner_cover_steps = 5;

  double max_move_all = 0.0;
  int inner_done = 0;

  for (int inner = 0; inner < inner_cover_steps; ++inner) {
    Eigen::VectorXd XA(active_n * 3);
    for (int local_idx = 0; local_idx < active_n; ++local_idx) {
      const int global_id = problem.active_ids[local_idx];
      XA(local_idx * 3 + 0) = X(global_id * 3 + 0);
      XA(local_idx * 3 + 1) = X(global_id * 3 + 1);
      XA(local_idx * 3 + 2) = X(global_id * 3 + 2);
    }

    Eigen::VectorXd gA = Eigen::VectorXd::Zero(XA.size());
    evaluate_cover_local_surrogate(problem, XA, gA);

    adam.t++;

    double grad_sq_sum = 0.0;
    double grad_max = 0.0;
    for (int local_idx = 0; local_idx < active_n; ++local_idx) {
      const double gn = gA.segment<3>(local_idx * 3).norm();
      grad_sq_sum += gn * gn;
      grad_max = std::max(grad_max, gn);
    }

    const double grad_rms =
        std::sqrt(grad_sq_sum / std::max(1, active_n));

    const double ema_before =
        (adam.cover_grad_rms_ema > 0.0) ? adam.cover_grad_rms_ema : grad_rms;

    adam.cover_grad_rms_ema =
        (adam.cover_grad_rms_ema > 0.0)
            ? (0.90 * adam.cover_grad_rms_ema + 0.10 * grad_rms)
            : grad_rms;

    const double grad_ratio =
        grad_rms / std::max(1e-12, ema_before);

    const double grad_spikiness =
        grad_max / std::max(1e-12, grad_rms);

    // 比之前更激进一点，减少总迭代数
    const double base_lr = 0.06 * avg_r;

    double lr_scale = 1.0 / std::sqrt(std::max(1.0, grad_ratio));
    lr_scale = std::clamp(lr_scale, 0.40, 1.25);

    double spike_scale = 1.0 / std::sqrt(std::max(1.0, grad_spikiness));
    spike_scale = std::clamp(spike_scale, 0.60, 1.0);

    const double lr =
        std::max(1e-12, base_lr * lr_scale * spike_scale);

    const double beta1 = 0.30;
    const double beta2 = 0.99;
    const double eps = 1e-8;

    // cap 略放宽一点，减少总 inner step 数
    const double cap =
        std::clamp(1.5 * problem.global_max_violation,
                   0.003 * avg_r,
                   0.14  * avg_r);

    double max_move_inner = 0.0;

    for (int local_idx = 0; local_idx < active_n; ++local_idx) {
      const int global_id = problem.active_ids[local_idx];
      const Eigen::Vector3d g = gA.segment<3>(local_idx * 3);

      Eigen::Vector3d m_i = adam.m.segment<3>(global_id * 3);
      Eigen::Vector3d v_i = adam.v.segment<3>(global_id * 3);

      m_i = beta1 * m_i + (1.0 - beta1) * g;
      v_i = beta2 * v_i + (1.0 - beta2) * g.cwiseProduct(g);

      adam.m.segment<3>(global_id * 3) = m_i;
      adam.v.segment<3>(global_id * 3) = v_i;

      const Eigen::Vector3d m_hat =
          m_i / (1.0 - std::pow(beta1, adam.t));
      const Eigen::Vector3d v_hat =
          v_i / (1.0 - std::pow(beta2, adam.t));

      Eigen::Vector3d step;
      step.x() = lr * m_hat.x() / (std::sqrt(v_hat.x()) + eps);
      step.y() = lr * m_hat.y() / (std::sqrt(v_hat.y()) + eps);
      step.z() = lr * m_hat.z() / (std::sqrt(v_hat.z()) + eps);

      // 切平面投影
      const Eigen::Vector3d n = problem.base_normals[local_idx];
      const double nn = n.squaredNorm();
      if (nn > 1e-30) {
        step -= n * (step.dot(n) / nn);
      }

      const double step_len = step.norm();
      if (step_len > cap) {
        step *= (cap / step_len);
      }

      X.segment<3>(global_id * 3) -= step;
      max_move_inner = std::max(max_move_inner, step.norm());
    }

    // 只投影 active 点
    project_active_sites_to_surface_and_update_state(X, problem.active_ids);

    max_move_all = std::max(max_move_all, max_move_inner);
    ++inner_done;

    // inner step 已经很小，就没必要继续做后面的 inner 了
    if (max_move_inner < 1e-5 * avg_r) {
      break;
    }
  }

  // inner 若干步结束后，再做一次真实问题重建并更新日志/全局状态
  CoverActiveProblem problem_after;
  build_cover_active_problem(X, problem_after, true, true, false);

  result.step_count = inner_done;
  result.invalid_seeds = problem_after.global_invalid_seed_count;
  result.max_move = max_move_all;
  result.move_too_small = (max_move_all < 1e-5 * avg_r);

  if (problem_after.global_invalid_seed_count == 0) {
    result.feasible_done = true;
  }

  return result;
};

  // --------------------------------------------------------------------------
  // QEM 阶段：使用 Adam 做一次平滑更新
  // --------------------------------------------------------------------------
  auto qem_cvt_adam_step = [&](Eigen::VectorXd &X, AdamState &adam) -> int {
    Eigen::VectorXd g = Eigen::VectorXd::Zero(X.size());
    fgm2(X, g);

    adam.t++;
    const double avg_spacing = bbox_diag / std::sqrt(std::max(1, num));
    const double lr = std::max(1e-12, 0.05 * avg_spacing);
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;

    Eigen::VectorXd X_new = X;

    for (int i = 0; i < num; ++i) {
      const Eigen::Vector3d g_i = g.segment<3>(i * 3);

      Eigen::Vector3d m_i = adam.m.segment<3>(i * 3);
      Eigen::Vector3d v_i = adam.v.segment<3>(i * 3);

      m_i = beta1 * m_i + (1.0 - beta1) * g_i;
      v_i = beta2 * v_i + (1.0 - beta2) * g_i.cwiseProduct(g_i);

      adam.m.segment<3>(i * 3) = m_i;
      adam.v.segment<3>(i * 3) = v_i;

      const Eigen::Vector3d m_hat = m_i / (1.0 - std::pow(beta1, adam.t));
      const Eigen::Vector3d v_hat = v_i / (1.0 - std::pow(beta2, adam.t));

      Eigen::Vector3d step;
      step.x() = lr * m_hat.x() / (std::sqrt(v_hat.x()) + eps);
      step.y() = lr * m_hat.y() / (std::sqrt(v_hat.y()) + eps);
      step.z() = lr * m_hat.z() / (std::sqrt(v_hat.z()) + eps);

      const Eigen::Vector3d n = Nors[i];
      const double nn = n.squaredNorm();
      if (nn > 1e-30) {
        step -= n * (step.dot(n) / nn);
      }

      const double step_len = step.norm();
      const double cap = std::max(1e-12, 0.20 * avg_spacing);
      if (step_len > cap) {
        step *= (cap / step_len);
      }

      X_new.segment<3>(i * 3) -= step;
    }

    X = X_new;
    sync_sites_from_iterates(X);
    return 1;
  };
#endif

  _RVD.calculate_(_sites);
  const std::string final_outdir =
      (std::filesystem::current_path() / "data" / "Block_New").string();
  // const std::string final_outdir =
  //     (std::filesystem::current_path() / "data" / "Bunny").string();
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

  suppress_intermediate_output = false;

  bool constraint_satisfied = false;
  while (qem_iterations_done < total_lagrange_iterations) {
    const int remaining_before_cover =
        std::max(0, cover_start_iteration - qem_iterations_done);
    const int this_round_max_iteration =
        (remaining_before_cover > 0)
            ? std::min(lagrange_block_max_iteration, remaining_before_cover)
            : lagrange_block_max_iteration;

    BGAL::_LBFGS::_Parameter lagrange_para(_para);
    lagrange_para.max_iteration = std::max(1, this_round_max_iteration);
    BGAL::_LBFGS lagrange_lbfgs(lagrange_para);

    const bool cover_enabled_this_round =
        (qem_iterations_done >= cover_start_iteration);
    cover_enabled_for_current_block = cover_enabled_this_round;

    if (cover_enabled_this_round && !cover_phase_initialized) {
      lagrange_multiplier =
          std::max(1.0, rho_cover * std::max(prev_cover_energy, 1e-6));
      cover_phase_initialized = true;
    }

    const int steps_done = lagrange_lbfgs.minimize(fgm2, iterX2);
    sync_sites_from_iterates(iterX2);
    qem_iterations_done += std::max(steps_done, 1);
    suppress_intermediate_output = true;
    flush_outputs(qem_iterations_done + 2);

    if (cover_enabled_this_round) {
      lagrange_multiplier += rho_cover * last_constraint_loss;
      if (std::isfinite(prev_cover_energy) &&
          last_constraint_loss > eta_cover * prev_cover_energy) {
        rho_cover = std::min(rho_cover * beta_cover, rho_cover_max);
      }
      prev_cover_energy = last_constraint_loss;
    } else {
      prev_cover_energy = last_constraint_loss;
    }

    std::cout << "[Lagrange] iter=" << qem_iterations_done
              << " active_quads=" << last_constraint_active_quads
              << "/" << last_constraint_num_quads
              << " loss_raw=" << last_constraint_loss_raw
              << " loss_scaled=" << last_constraint_loss
              << " cover_scale=" << last_constraint_loss_scale
              << " min_g=" << last_constraint_min_g
              << " r=" << last_constraint_r
              << " lambda_cover=" << lagrange_multiplier
              << " rho_cover=" << rho_cover
              << " cover_enabled=" << (cover_enabled_this_round ? 1 : 0)
              << std::endl;

    if (cover_enabled_this_round &&
        last_constraint_active_quads == 0 && last_constraint_loss < 1e-14) {
      constraint_satisfied = true;
      break;
    }

    if (steps_done == 0) {
      if (!cover_enabled_this_round && qem_iterations_done < cover_start_iteration) {
        qem_iterations_done = cover_start_iteration;
        continue;
      }
      break;
    }
  }

  if (!constraint_satisfied) {
    std::cout << "[Lagrange] stop with active_quads="
              << last_constraint_active_quads << "/" << last_constraint_num_quads
              << " loss_raw=" << last_constraint_loss_raw
              << " loss_scaled=" << last_constraint_loss
              << " lambda_cover=" << lagrange_multiplier
              << " rho_cover=" << rho_cover << std::endl;
  }

  flush_outputs(2);

  end = clock();

  allTime += (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << "allTime: " << allTime << " RVDtime: " << RVDtime
            << " Optimizer time: " << allTime - RVDtime << std::endl;

  flush_outputs(2);
}

} // namespace BGAL
