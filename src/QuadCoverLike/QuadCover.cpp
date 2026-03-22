#include "BGAL/QuadCoverLike/QuadCover.h"
#include "BGAL/BaseShape/KDTree.h"
#include "BGAL/Algorithm/BOC/BOC.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {

using BGAL::_Point3;
using BGAL::_Restricted_Tessellation3D;
using BGAL::_QuadCover3D;
using Vec3 = Eigen::Vector3d;

static inline Vec3 to_eigen(const _Point3 &p) {
  return Vec3(p.x(), p.y(), p.z());
}

static inline _Point3 to_point(const Vec3 &v) {
  return _Point3(v.x(), v.y(), v.z());
}

static inline _Point3 project_to_surface(const BGAL::_ManifoldModel &model,
                                         const _Point3 &p) {
  auto ret = const_cast<BGAL::_ManifoldModel &>(model).nearest_point_(p);
  return std::get<0>(ret);
}

struct CellGeom {
  std::vector<int> vertex_ids;
  std::vector<_Point3> vertex_pos;
  double r2 = 0.0;
  _Point3 far_v;
};

// 严谨的动态半径评估体
struct DynamicRadiusEval {
  double total_loss = 0.0;
  int active_quads = 0;
  double min_g = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd grads;

  explicit DynamicRadiusEval(int n = 0, bool need_grad = false)
      : grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()) {}
};

struct LBFGSPair {
  Eigen::VectorXd s;
  Eigen::VectorXd y;
  double rho = 0.0;
};

struct SAMetric {
  int active = 0;
  int quads = 0;
  double loss = 0.0;
  double merit = 0.0;
};

static inline SAMetric make_sa_metric(int active,
                                      int quads,
                                      double loss,
                                      double avg_r) {
  SAMetric m;
  m.active = active;
  m.quads = quads;
  m.loss = loss;

  const double r = std::max(avg_r, 1e-12);
  const double r4 = r * r * r * r;

  const double active_ratio =
      (quads > 0) ? (double)active / (double)quads : 0.0;
  const double mean_norm_loss =
      (active > 0) ? loss / ((double)active * r4) : 0.0;

  // 主目标仍然是减少活跃违规四元组；loss 只作为同层次 tie-breaker。
  m.merit = 8.0 * active_ratio + mean_norm_loss;
  return m;
}

static inline Eigen::VectorXd flatten_matrix(const Eigen::MatrixXd &M) {
  Eigen::VectorXd x(M.rows() * 3);
  for (int i = 0; i < M.rows(); ++i) {
    x.segment<3>(3 * i) = M.row(i).transpose();
  }
  return x;
}

static inline void update_surface_normals(
    const BGAL::_ManifoldModel &model,
    const std::vector<_Point3> &sites,
    std::vector<Vec3> &surface_normals) {
  surface_normals.assign(sites.size(), Vec3(0.0, 0.0, 1.0));
  for (int i = 0; i < (int)sites.size(); ++i) {
    auto nearest = const_cast<BGAL::_ManifoldModel &>(model).nearest_point_(sites[i]);
    const int face_id = std::get<2>(nearest);
    if (face_id >= 0 && face_id < model.number_faces_()) {
      _Point3 n_p = model.normal_face_(face_id);
      n_p.normalized_();
      surface_normals[i] = to_eigen(n_p);
    }
  }
}

static inline Eigen::MatrixXd unflatten_vector(const Eigen::VectorXd &x, int n) {
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; ++i) {
    M.row(i) = x.segment<3>(3 * i).transpose();
  }
  return M;
}

static inline double max_row_norm(const Eigen::MatrixXd &M) {
  double ret = 0.0;
  for (int i = 0; i < M.rows(); ++i) {
    ret = std::max(ret, M.row(i).norm());
  }
  return ret;
}

// 核心评估函数：动态半径 + 切面投影
static inline DynamicRadiusEval evaluate_dynamic_radius_surrogate(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<Vec3> &normals, // <--- 传入法线信息用于切面投影
    const std::vector<std::array<int, 4>> &quads,
    double eps,
    bool need_grad) {
  DynamicRadiusEval eval((int)sites.size(), need_grad);

  for (const auto &q : quads) {
    Vec3 P_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) P_bar += to_eigen(sites[q[k]]);
    P_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 Pi = to_eigen(sites[sid]);
      // 坚持动态半径：r_i 随 P_i 严格动态更新
      const double ri2 = (Pi - frozen_poles[sid]).squaredNorm();
      g_val += (P_bar - Pi).squaredNorm() - ri2;
    }

    eval.min_g = std::min(eval.min_g, g_val);
    const double violation = -(g_val + eps);
    if (violation > 0.0) {
      eval.total_loss += violation * violation;
      eval.active_quads++;
      if (need_grad) {
        for (int k = 0; k < 4; ++k) {
          const int sid = q[k];
          const Vec3 &Vi = frozen_poles[sid];
          eval.grads.row(sid) += (-4.0 * violation * (Vi - P_bar)).transpose();
        }
      }
    }
  }

  // 【防震荡绝对核心】：把计算出的三维空间梯度，完美剥离掉法向分量，只保留贴合曲面的切向分量
  if (need_grad && !normals.empty()) {
    for (int i = 0; i < eval.grads.rows(); ++i) {
      if (eval.grads.row(i).squaredNorm() > 1e-24) {
        Vec3 g = eval.grads.row(i).transpose();
        // 梯度切面投影：g_tangent = g - (g · n) * n
        g -= g.dot(normals[i]) * normals[i];
        eval.grads.row(i) = g.transpose();
      }
    }
  }

  if (!std::isfinite(eval.min_g)) eval.min_g = 0.0;
  return eval;
}

static inline std::vector<_Point3> apply_projected_step(
    const std::vector<_Point3> &sites,
    const Eigen::MatrixXd &step,
    const BGAL::_ManifoldModel &model) {
  std::vector<_Point3> trial = sites;
  for (int i = 0; i < (int)sites.size(); ++i) {
    if (step.row(i).squaredNorm() < 1e-30) continue;
    const Vec3 p = to_eigen(sites[i]) + step.row(i).transpose();
    trial[i] = project_to_surface(model, to_point(p));
  }
  return trial;
}

static inline Eigen::MatrixXd realized_step_matrix(
    const std::vector<_Point3> &from,
    const std::vector<_Point3> &to) {
  Eigen::MatrixXd step((int)from.size(), 3);
  for (int i = 0; i < (int)from.size(); ++i) {
    step.row(i) = (to_eigen(to[i]) - to_eigen(from[i])).transpose();
  }
  return step;
}

static inline Eigen::VectorXd lbfgs_two_loop_direction(
    const Eigen::VectorXd &grad,
    const std::vector<LBFGSPair> &history) {
  if (grad.norm() < 1e-30) return -grad;
  if (history.empty()) return -grad;

  Eigen::VectorXd q = grad;
  std::vector<double> alpha(history.size(), 0.0);

  for (int i = (int)history.size() - 1; i >= 0; --i) {
    alpha[i] = history[i].rho * history[i].s.dot(q);
    q.noalias() -= alpha[i] * history[i].y;
  }

  double gamma = 1.0;
  {
    const auto &last = history.back();
    const double yy = last.y.dot(last.y);
    const double sy = last.s.dot(last.y);
    if (yy > 1e-30 && sy > 1e-30) gamma = sy / yy;
  }

  Eigen::VectorXd r = gamma * q;
  for (int i = 0; i < (int)history.size(); ++i) {
    const double beta = history[i].rho * history[i].y.dot(r);
    r.noalias() += (alpha[i] - beta) * history[i].s;
  }

  Eigen::VectorXd dir = -r;
  if (!(dir.dot(grad) < -1e-18)) dir = -grad;
  return dir;
}

static inline void push_lbfgs_pair(std::vector<LBFGSPair> &history,
                                   const Eigen::VectorXd &s,
                                   const Eigen::VectorXd &y,
                                   int memory_limit) {
  const double sy = s.dot(y);
  if (!(sy > 1e-18)) return;

  LBFGSPair pair;
  pair.s = s;
  pair.y = y;
  pair.rho = 1.0 / sy;

  if ((int)history.size() >= memory_limit) {
    history.erase(history.begin());
  }
  history.push_back(std::move(pair));
}


static inline std::vector<int> collect_active_sites(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<std::array<int, 4>> &quads,
    double eps) {
  std::vector<char> mark(sites.size(), 0);

  for (const auto &q : quads) {
    Vec3 P_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) P_bar += to_eigen(sites[q[k]]);
    P_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 Pi = to_eigen(sites[sid]);
      const double ri2 = (Pi - frozen_poles[sid]).squaredNorm();
      g_val += (P_bar - Pi).squaredNorm() - ri2;
    }

    if (-(g_val + eps) > 0.0) {
      for (int k = 0; k < 4; ++k) mark[q[k]] = 1;
    }
  }

  std::vector<int> ids;
  ids.reserve(sites.size());
  for (int i = 0; i < (int)mark.size(); ++i) {
    if (mark[i]) ids.push_back(i);
  }
  return ids;
}

struct QuadKeyLess {
  bool operator()(const std::array<int, 4> &a, const std::array<int, 4> &b) const {
    return a < b;
  }
};

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

  std::set<Eigen::Vector3i, std::function<bool(const Eigen::Vector3i&, const Eigen::Vector3i&)>> faces([](const Eigen::Vector3i &a, const Eigen::Vector3i &b) {
        if (a.x() != b.x()) return a.x() > b.x();
        if (a.y() != b.y()) return a.y() > b.y();
        return a.z() > b.z();
      });

  for (const auto &e : rdt_edges) {
    for (int pid : neighbors[e.first]) {
      if (rdt_edges.find(std::make_pair(std::min(pid, e.first), std::max(pid, e.first))) == rdt_edges.end()) continue;
      if (rdt_edges.find(std::make_pair(std::min(pid, e.second), std::max(pid, e.second))) == rdt_edges.end()) continue;

      const int hi = std::max(pid, std::max(e.first, e.second));
      const int lo = std::min(pid, std::min(e.first, e.second));
      const int mid = pid + e.first + e.second - hi - lo;
      faces.insert(Eigen::Vector3i(hi, mid, lo));
    }
  }

  std::vector<Eigen::Vector3i> tris;
  tris.reserve(faces.size());
  for (const auto &f : faces) tris.push_back(f);
  return tris;
}

static inline void build_quads_from_search_faces(
    const std::vector<Eigen::Vector3i> &faces,
    const std::vector<_Point3> &sites,
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
    if (Rj < Rpivot) { pivot = j; Rpivot = Rj; }
    if (Rk < Rpivot) { pivot = k; Rpivot = Rk; }

    std::vector<_Point3> query(1, sites[pivot]);
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

static inline void compute_cell_geom_and_spheres(
    const std::vector<_Point3> &sites,
    const _Restricted_Tessellation3D &rvd,
    std::vector<CellGeom> &cells,
    std::vector<Sphere::Sphere> &spheres) {
  const auto &cell_tris = rvd.get_cells_();
  cells.assign(sites.size(), CellGeom());
  spheres.assign(sites.size(), Sphere::Sphere());

  for (int i = 0; i < (int)cell_tris.size(); ++i) {
    std::set<int> uniq;
    for (const auto &tri : cell_tris[i]) {
      uniq.insert(std::get<0>(tri));
      uniq.insert(std::get<1>(tri));
      uniq.insert(std::get<2>(tri));
    }

    cells[i].vertex_ids.assign(uniq.begin(), uniq.end());
    cells[i].vertex_pos.reserve(cells[i].vertex_ids.size());

    double best_d2 = 0.0;
    _Point3 best_v = sites[i];

    for (int vid : cells[i].vertex_ids) {
      const _Point3 pv = rvd.vertex_(vid);
      cells[i].vertex_pos.push_back(pv);
      const double d2 = (pv - sites[i]).sqlength_();
      if (d2 > best_d2) {
        best_d2 = d2;
        best_v = pv;
      }
    }
    cells[i].r2 = best_d2;
    cells[i].far_v = best_v;

    spheres[i].c = sites[i];
    spheres[i].r = std::sqrt(best_d2) + 1e-6;
    spheres[i].max_point = best_v;
  }
}

static inline void output_mesh(const std::vector<_Point3> &sites,
                               const _Restricted_Tessellation3D &rvd,
                               int num,
                               const std::string &outpath,
                               const std::string &modelname,
                               int step) {
  const std::vector<std::vector<std::tuple<int, int, int>>> &cells = rvd.get_cells_();

  namespace fs = std::filesystem;
  fs::create_directories(outpath);

  std::string filepath = outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
  if (step > 0) {
    filepath = outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
               "_Iter" + std::to_string(step) + "_RVD.obj";
  }

  std::ofstream out(filepath);
  out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar\n";

  for (int i = 0; i < rvd.number_vertices_(); ++i) {
    out << "v " << rvd.vertex_(i) << "\n";
  }

  for (int i = 0; i < (int)cells.size(); ++i) {
    const double color = (double)BGAL::_BOC::rand_();
    out << "vt " << color << " 0\n";
    for (const auto &tri : cells[i]) {
      out << "f " << std::get<0>(tri) + 1 << "/" << i + 1 << " "
          << std::get<1>(tri) + 1 << "/" << i + 1 << " "
          << std::get<2>(tri) + 1 << "/" << i + 1 << "\n";
    }
  }
  out.close();

  std::string points_path = outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
  if (step > 0) {
    points_path = outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
                  "_Iter" + std::to_string(step) + "_Points.xyz";
  }
  std::ofstream outP(points_path);
  for (const auto &s : sites) outP << s << "\n";
  outP.close();
}

static inline void write_spheres_csv(const std::string &outdir,
                                     int num_sites,
                                     const std::string &model_name,
                                     int iter,
                                     const std::vector<Sphere::Sphere> &spheres,
                                     const BGAL::_ManifoldModel &model) {
  namespace fs = std::filesystem;
  fs::create_directories(outdir);
  std::string filename = outdir + "/QuadCover_" + std::to_string(num_sites) + "_" + model_name + "_Spheres.csv";
  if (iter > 0) {
    filename = outdir + "/QuadCover_" + std::to_string(num_sites) + "_" + model_name +
               "_Iter" + std::to_string(iter) + "_Spheres.csv";
  }

  std::ofstream out(filename, std::ios::out | std::ios::trunc);
  if (!out) return;

  out << std::setprecision(17);
  for (const auto &sphere : spheres) {
    auto nearest = const_cast<BGAL::_ManifoldModel &>(model).nearest_point_(sphere.c);
    const int face_id = std::get<2>(nearest);

    _Point3 normal(0.0, 0.0, 0.0);
    if (face_id >= 0 && face_id < model.number_faces_()) {
      normal = model.normal_face_(face_id);
      normal.normalized_();
    }

    out << sphere.c.x() << "," << sphere.c.y() << "," << sphere.c.z() << ","
        << sphere.r << "," << face_id << "," << normal.x() << ","
        << normal.y() << "," << normal.z() << "\n";
  }
}

static inline bool load_sites_from_xyz(const std::string &filename,
                                       std::vector<_Point3> &sites) {
  std::ifstream in(filename.c_str());
  if (!in.is_open()) return false;
  sites.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(ss >> x >> y >> z)) continue;
    sites.emplace_back(x, y, z);
  }
  return !sites.empty();
}

} // namespace

namespace BGAL {

_QuadCover3D::_QuadCover3D(const _ManifoldModel &model)
    : _model(model), _RVD(model), _para() {
  const_cast<_ManifoldModel &>(_model).initialization_PQP_();
}

_QuadCover3D::_QuadCover3D(const _ManifoldModel &model, const _Parameter &para)
    : _model(model), _RVD(model), _para(para) {
  const_cast<_ManifoldModel &>(_model).initialization_PQP_();
}

void _QuadCover3D::calculate_(int site_num, char *modelNamee, char *pointsName) {
  std::string modelname = (modelNamee == nullptr) ? std::string("model") : std::string(modelNamee);
  std::vector<_Point3> init_sites;

  namespace fs = std::filesystem;
  fs::path obj("./data/man.obj");
  fs::path base = obj.parent_path();
  std::string inPointsName = (pointsName == nullptr) ? 
    (base / ("n" + std::to_string(site_num) + "_" + modelname + "_inputPoints.xyz")).string() : pointsName;

  if (!load_sites_from_xyz(inPointsName, init_sites)) {
    throw std::runtime_error("[QuadCoverLike] failed to load init points from: " + inPointsName);
  }
  calculate_(init_sites, modelname);
}

void _QuadCover3D::calculate_(const std::vector<_Point3> &init_sites,
                              const std::string &model_name) {
  _history.clear();
  _sites = init_sites;
  if (_sites.empty()) {
    throw std::runtime_error("[QuadCoverLike] empty init_sites.");
  }

  const int n = (int)_sites.size();

  // ================= 局部 basin-hopping / 弱模拟退火 =================
  std::vector<_Point3> current_sites = init_sites;
  std::vector<_Point3> best_sites = init_sites;

  SAMetric current_metric;
  bool has_current_metric = false;

  double best_final_loss = std::numeric_limits<double>::infinity();
  int best_active = std::numeric_limits<int>::max();

  std::vector<int> current_hot_sites;
  std::vector<Vec3> current_normals;

  // 温度只控制“是否接受略差 basin”的概率，不再直接驱动大尺度破坏性扰动。
  double T = 0.15;
  const double T_min = 1e-4;
  const double alpha_cooling =
      std::pow(T_min / T, 1.0 / std::max(1, _para.max_outer_iterations - 1));

  std::mt19937 rng(1337);
  std::normal_distribution<double> gaussian(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  double last_avg_r = 1.0;

  for (int outer = 0; outer < _para.max_outer_iterations; ++outer) {
    // -----------------------------------------------------------------
    // 1. 从当前已接受 basin 出发，只对热点点做很小的切向 kick
    // -----------------------------------------------------------------
    _sites = current_sites;
    if (outer > 0) {
      std::vector<int> ids = current_hot_sites;
      if (ids.empty()) {
        ids.resize(n);
        std::iota(ids.begin(), ids.end(), 0);
      }

      std::shuffle(ids.begin(), ids.end(), rng);

      const int kick_num = std::max(1, (int)std::ceil(0.08 * ids.size()));
      const double T0 = 0.15;
      const double noise_scale =
          0.015 * last_avg_r * std::sqrt(std::max(T, 1e-12) / T0);

      for (int kk = 0; kk < kick_num; ++kk) {
        const int i = ids[kk];

        Vec3 nrm = (i < (int)current_normals.size())
                       ? current_normals[i]
                       : Vec3(0.0, 0.0, 1.0);
        if (nrm.squaredNorm() < 1e-24) nrm = Vec3(0.0, 0.0, 1.0);
        nrm.normalize();

        Vec3 t1 = nrm.unitOrthogonal();
        if (t1.squaredNorm() < 1e-24) t1 = Vec3(1.0, 0.0, 0.0);
        t1.normalize();
        Vec3 t2 = nrm.cross(t1);
        if (t2.squaredNorm() < 1e-24) t2 = nrm.unitOrthogonal();
        t2.normalize();

        const double a = gaussian(rng) * noise_scale;
        const double b = gaussian(rng) * noise_scale;
        const Vec3 p = to_eigen(_sites[i]) + a * t1 + b * t2;
        _sites[i] = project_to_surface(_model, to_point(p));
      }
    }

    // -----------------------------------------------------------------
    // 2. 重建当前拓扑与几何，形成当前 surrogate
    // -----------------------------------------------------------------
    _RVD.calculate_(_sites);

    std::vector<CellGeom> cells;
    compute_cell_geom_and_spheres(_sites, _RVD, cells, _spheres);

    double sum_r = 0.0;
    std::vector<Vec3> frozen_poles(n, Vec3::Zero());
    for (int i = 0; i < n; ++i) {
      frozen_poles[i] = to_eigen(_spheres[i].max_point);
      sum_r += (to_eigen(_sites[i]) - frozen_poles[i]).norm();
    }

    const double avg_r = sum_r / std::max(1, n);
    last_avg_r = avg_r;
    const double eps = 1e-4 * avg_r * avg_r;

    std::vector<Vec3> surface_normals;
    update_surface_normals(_model, _sites, surface_normals);

    std::vector<Eigen::Vector3i> rdt_faces =
        build_rdt_faces_from_edges(n, _RVD.get_edges_());
    std::vector<std::array<int, 4>> search_quads;
    build_quads_from_search_faces(rdt_faces, _sites, _spheres, search_quads);

    _quads.clear();
    for (const auto &q : search_quads) _quads.push_back({q});

    if (_para.export_each_iteration || outer == 0) {
      output_mesh(_sites, _RVD, n, outpath, model_name, outer);
      write_spheres_csv(outpath, n, model_name, outer, _spheres, _model);
    }

    DynamicRadiusEval current_eval = evaluate_dynamic_radius_surrogate(
        _sites, frozen_poles, surface_normals, search_quads, eps, true);

    const double initial_loss = current_eval.total_loss;
    const int initial_active_quads = current_eval.active_quads;

    // -----------------------------------------------------------------
    // 3. 外层结束判据：看“本轮 L-BFGS 之前”的当前状态，而不是 proposal/final 状态
    // -----------------------------------------------------------------
    const double outer_stop_tol = 1e-12;
    if (initial_active_quads == 0 && initial_loss < outer_stop_tol) {
      best_sites = _sites;
      best_active = initial_active_quads;
      best_final_loss = initial_loss;

      _IterationInfo info;
      info.iteration = outer + 1;
      info.num_quads = search_quads.size();
      info.active_quads = initial_active_quads;
      info.min_margin = current_eval.min_g;
      info.accepted_step = 0.0;
      _history.push_back(info);

      if (_para.is_show) {
        std::cout << "[QuadCoverLike][SA-local] iter=" << info.iteration
                  << " | T=" << std::setprecision(4) << T
                  << " | Quads=" << info.num_quads
                  << " | Act=" << initial_active_quads
                  << " | Loss=" << std::scientific << std::setprecision(3)
                  << initial_loss
                  << " | STOP(Before L-BFGS)"
                  << std::endl;
      }
      break;
    }

    // -----------------------------------------------------------------
    // 4. 内层 L-BFGS：固定当前 surrogate 向下滑到 basin 底部
    // -----------------------------------------------------------------
    const int inner_iters = 8;
    const int max_linesearch = 10;
    const int lbfgs_memory = 6;
    std::vector<LBFGSPair> history;

    const double base_step_cap = 0.03 * avg_r;
    const double c1 = 1e-4;
    double last_accepted_step = 0.0;

    for (int inner = 0; inner < inner_iters; ++inner) {
      if (current_eval.active_quads == 0) break;

      const Eigen::VectorXd grad = flatten_matrix(current_eval.grads);
      if (grad.norm() < 1e-14) break;

      Eigen::VectorXd dir = lbfgs_two_loop_direction(grad, history);
      Eigen::MatrixXd dir_mat = unflatten_vector(dir, n);
      const double dir_max_row = max_row_norm(dir_mat);
      if (dir_max_row < 1e-16) break;

      double alpha = 1.0;
      if (dir_max_row > base_step_cap) {
        alpha = base_step_cap / dir_max_row;
      }

      bool accepted = false;
      std::vector<_Point3> accepted_sites;
      Eigen::MatrixXd accepted_realized_step;

      for (int ls = 0; ls < max_linesearch; ++ls) {
        const Eigen::MatrixXd proposed_step = alpha * dir_mat;
        std::vector<_Point3> trial_sites =
            apply_projected_step(_sites, proposed_step, _model);
        Eigen::MatrixXd realized_step =
            realized_step_matrix(_sites, trial_sites);

        const double realized_max_row = max_row_norm(realized_step);
        if (realized_max_row < 1e-16) {
          alpha *= 0.5;
          continue;
        }

        const Eigen::VectorXd real_step_vec = flatten_matrix(realized_step);
        const double real_dot_grad = grad.dot(real_step_vec);
        if (!(real_dot_grad < -1e-18)) {
          alpha *= 0.5;
          continue;
        }

        const DynamicRadiusEval trial_eval =
            evaluate_dynamic_radius_surrogate(trial_sites, frozen_poles,
                                             surface_normals, search_quads,
                                             eps, false);

        const double delta = trial_eval.total_loss - current_eval.total_loss;
        const double expected_desc = c1 * real_dot_grad;

        if (delta <= expected_desc + 1e-14) {
          accepted = true;
          accepted_sites.swap(trial_sites);
          accepted_realized_step = std::move(realized_step);
          last_accepted_step = realized_max_row;
          break;
        }
        alpha *= 0.5;
      }

      if (!accepted) {
        history.clear();
        break;
      }

      const Eigen::VectorXd old_grad = grad;
      _sites.swap(accepted_sites);
      update_surface_normals(_model, _sites, surface_normals);

      DynamicRadiusEval new_eval = evaluate_dynamic_radius_surrogate(
          _sites, frozen_poles, surface_normals, search_quads, eps, true);

      const Eigen::VectorXd s = flatten_matrix(accepted_realized_step);
      const Eigen::VectorXd y = flatten_matrix(new_eval.grads) - old_grad;
      const double sy = s.dot(y);
      const double s2 = s.squaredNorm();

      if (sy <= 1e-10 * std::max(1.0, s2) || alpha < 0.25) {
        history.clear();
      } else {
        push_lbfgs_pair(history, s, y, lbfgs_memory);
      }

      current_eval = std::move(new_eval);
    }

    const double final_loss = current_eval.total_loss;
    const int final_active = current_eval.active_quads;
    const SAMetric proposal_metric =
        make_sa_metric(final_active, (int)search_quads.size(), final_loss, avg_r);

    bool accept = false;
    std::string sa_status;

    // -----------------------------------------------------------------
    // 5. 接受判定：比较“优化后 basin 的质量”，而不是比较 raw initial loss
    // -----------------------------------------------------------------
    if (!has_current_metric) {
      accept = true;
      sa_status = "INIT";
    } else {
      const double delta = proposal_metric.merit - current_metric.merit;
      if (delta <= 0.0) {
        continue;
      } else {
        const double p = std::exp(-delta / std::max(T, 1e-12));
        if (uniform(rng) < p) {
          accept = true;
          sa_status = "ACCEPT(MH)";
        } else {
          sa_status = "REJECT";
        }
      }
    }

    if (accept) {
      current_sites = _sites;
      current_metric = proposal_metric;
      has_current_metric = true;
      current_hot_sites =
          collect_active_sites(_sites, frozen_poles, search_quads, eps);
      current_normals = surface_normals;
    }

    // -----------------------------------------------------------------
    // 6. 全局最优追踪：仍然看最终 basin 的结果
    // -----------------------------------------------------------------
    if (final_active < best_active ||
        (final_active == best_active && final_loss < best_final_loss)) {
      best_active = final_active;
      best_final_loss = final_loss;
      best_sites = _sites;
    }

    _IterationInfo info;
    info.iteration = outer + 1;
    info.num_quads = search_quads.size();
    info.active_quads = final_active;
    info.min_margin = current_eval.min_g;
    info.accepted_step = last_accepted_step;
    _history.push_back(info);

    if (_para.is_show) {
      std::cout << "[QuadCoverLike][SA-local] iter=" << info.iteration
                << " | T=" << std::setprecision(4) << T
                << " | Quads=" << info.num_quads
                << " | Act=" << initial_active_quads << "->" << final_active
                << " (Best:" << best_active << ")"
                << " | Loss=" << std::scientific << std::setprecision(3)
                << initial_loss << " -> " << final_loss
                << " | Merit=" << std::fixed << std::setprecision(6)
                << proposal_metric.merit
                << " | " << sa_status
                << std::endl;
    }

    T *= alpha_cooling;
  }

  // -----------------------------------------------------------------
  // 7. Final Output (恢复具有最小 Final Loss 的绝对全局最优)
  // -----------------------------------------------------------------
  _sites = best_sites;
  _RVD.calculate_(_sites);
  
  std::vector<CellGeom> final_cells;
  compute_cell_geom_and_spheres(_sites, _RVD, final_cells, _spheres);
  output_mesh(_sites, _RVD, n, outpath, model_name, (int)_history.size() + 1);
  write_spheres_csv(outpath, n, model_name, (int)_history.size() + 1, _spheres, _model);
}

} // namespace BGAL