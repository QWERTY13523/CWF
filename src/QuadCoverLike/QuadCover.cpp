#include "BGAL/QuadCoverLike/QuadCover.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>

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
#include <set>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <omp.h> // 引入 OpenMP

namespace {

using BGAL::_Point3;
using BGAL::_Restricted_Tessellation3D;
using BGAL::_QuadCover3D;
using Vec3 = Eigen::Vector3d;

static inline double wall_seconds_since(double start) {
  return omp_get_wtime() - start;
}

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

static inline Eigen::VectorXd flatten_matrix(const Eigen::MatrixXd &M) {
  Eigen::VectorXd x(M.rows() * 3);
  // Flattening is fast, but we can parallelize if sizes are huge
  #pragma omp parallel for
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
  
  // 并行化寻找表面法线
  #pragma omp parallel for
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
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    M.row(i) = x.segment<3>(3 * i).transpose();
  }
  return M;
}

static inline double max_row_norm(const Eigen::MatrixXd &M) {
  double ret = 0.0;
  // 对归约操作进行多线程加速
  #pragma omp parallel for reduction(max:ret)
  for (int i = 0; i < M.rows(); ++i) {
    ret = std::max(ret, M.row(i).norm());
  }
  return ret;
}

// 核心评估函数：动态半径 + 切面投影
static inline DynamicRadiusEval evaluate_dynamic_radius_surrogate(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<Vec3> &normals,
    const std::vector<std::array<int, 4>> &quads,
    double eps,
    bool need_grad) {
    
  const int n_sites = sites.size();
  DynamicRadiusEval eval(n_sites, need_grad);
  const int num_quads = quads.size();

  // 采用 Thread-local accumulator 避免梯度写入冲突
  #pragma omp parallel
  {
    double local_total_loss = 0.0;
    int local_active_quads = 0;
    double local_min_g = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd local_grads = need_grad ? Eigen::MatrixXd::Zero(n_sites, 3) : Eigen::MatrixXd();

    #pragma omp for nowait
    for (int i = 0; i < num_quads; ++i) {
      const auto &q = quads[i];
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

      local_min_g = std::min(local_min_g, g_val);
      const double violation = -(g_val + eps);
      if (violation > 0.0) {
        local_total_loss += violation * violation;
        local_active_quads++;
        if (need_grad) {
          for (int k = 0; k < 4; ++k) {
            const int sid = q[k];
            const Vec3 &Vi = frozen_poles[sid];
            local_grads.row(sid) += (-4.0 * violation * (Vi - P_bar)).transpose();
          }
        }
      }
    }

    // 线程数据汇聚
    #pragma omp critical
    {
      eval.total_loss += local_total_loss;
      eval.active_quads += local_active_quads;
      eval.min_g = std::min(eval.min_g, local_min_g);
      if (need_grad) {
        eval.grads += local_grads;
      }
    }
  }

  // 并行切面投影
  if (need_grad && !normals.empty()) {
    #pragma omp parallel for
    for (int i = 0; i < eval.grads.rows(); ++i) {
      if (eval.grads.row(i).squaredNorm() > 1e-24) {
        Vec3 g = eval.grads.row(i).transpose();
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
  
  // 并行计算预测步并投影
  #pragma omp parallel for
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
  #pragma omp parallel for
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


static inline Vec3 orthogonal_unit(const Vec3 &n) {
  Vec3 a = (std::abs(n.x()) < 0.9) ? Vec3::UnitX() : Vec3::UnitY();
  Vec3 t = n.cross(a);
  double len = t.norm();
  if (len < 1e-12) {
    t = n.cross(Vec3::UnitZ());
    len = t.norm();
  }
  if (len < 1e-12) return Vec3::UnitX();
  return t / len;
}

static inline std::vector<char> build_active_site_mask(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<std::array<int, 4>> &quads,
    double eps) {
  std::vector<char> mask(sites.size(), 0);

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
      for (int k = 0; k < 4; ++k) mask[q[k]] = 1;
    }
  }
  return mask;
}

static inline int count_active_mask(const std::vector<char> &mask) {
  return (int)std::count(mask.begin(), mask.end(), (char)1);
}

static inline void zero_out_inactive_rows(Eigen::MatrixXd &M,
                                          const std::vector<char> &active_mask) {
  const int rows = std::min<int>(M.rows(), active_mask.size());
  #pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    if (!active_mask[i]) M.row(i).setZero();
  }
}


static inline bool has_any_active(const std::vector<char> &mask) {
  return std::any_of(mask.begin(), mask.end(), [](char v) { return v != 0; });
}

static inline std::vector<char> make_all_active_mask(int n) {
  return std::vector<char>(n, 1);
}

static inline std::vector<char> expand_active_mask_one_ring(
    const std::vector<char> &seed_mask,
    const std::vector<Eigen::Vector3i> &faces,
    int num_sites,
    int rings) {
  if (num_sites <= 0) return {};
  std::vector<char> mask = seed_mask;
  if ((int)mask.size() != num_sites) mask.assign(num_sites, 0);
  if (!has_any_active(mask)) return make_all_active_mask(num_sites);

  for (int r = 0; r < rings; ++r) {
    std::vector<char> next = mask;
    for (const auto &f : faces) {
      const int a = f.x(), b = f.y(), c = f.z();
      if (a < 0 || a >= num_sites || b < 0 || b >= num_sites || c < 0 || c >= num_sites) {
        continue;
      }
      if (mask[a] || mask[b] || mask[c]) {
        next[a] = next[b] = next[c] = 1;
      }
    }
    mask.swap(next);
  }
  return mask;
}

static inline Eigen::VectorXd project_to_simplex(const Eigen::VectorXd &v) {
  const int n = (int)v.size();
  if (n <= 0) return Eigen::VectorXd();
  if (n == 1) {
    Eigen::VectorXd ret(1);
    ret[0] = 1.0;
    return ret;
  }

  std::vector<double> u(n);
  for (int i = 0; i < n; ++i) u[i] = v[i];
  std::sort(u.begin(), u.end(), std::greater<double>());

  double cssv = 0.0;
  int rho = -1;
  for (int i = 0; i < n; ++i) {
    cssv += u[i];
    const double t = (cssv - 1.0) / double(i + 1);
    if (u[i] - t > 0.0) rho = i;
  }

  double theta = 0.0;
  if (rho >= 0) {
    cssv = 0.0;
    for (int i = 0; i <= rho; ++i) cssv += u[i];
    theta = (cssv - 1.0) / double(rho + 1);
  }

  Eigen::VectorXd w(n);
  for (int i = 0; i < n; ++i) w[i] = std::max(v[i] - theta, 0.0);
  const double sum_w = w.sum();
  if (sum_w <= 1e-30) {
    w.setConstant(1.0 / double(n));
  } else {
    w /= sum_w;
  }
  return w;
}

static inline Eigen::VectorXd solve_min_norm_convex_combination(
    const std::vector<Eigen::VectorXd> &grads,
    int max_iters = 200,
    double tol = 1e-10) {
  const int m = (int)grads.size();
  if (m <= 0) return Eigen::VectorXd();
  if (m == 1) {
    Eigen::VectorXd lambda(1);
    lambda[0] = 1.0;
    return lambda;
  }

  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(m, m);
  for (int i = 0; i < m; ++i) {
    G(i, i) = grads[i].dot(grads[i]);
    for (int j = i + 1; j < m; ++j) {
      const double gij = grads[i].dot(grads[j]);
      G(i, j) = gij;
      G(j, i) = gij;
    }
  }

  double L = 0.0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(G);
  if (eig.info() == Eigen::Success) {
    L = std::max(1e-12, eig.eigenvalues().maxCoeff());
  } else {
    for (int i = 0; i < m; ++i) {
      L = std::max(L, G.row(i).cwiseAbs().sum());
    }
    L = std::max(L, 1e-12);
  }

  Eigen::VectorXd lambda = Eigen::VectorXd::Constant(m, 1.0 / double(m));
  const double step = 1.0 / L;

  for (int it = 0; it < max_iters; ++it) {
    const Eigen::VectorXd grad_qp = G * lambda;
    const Eigen::VectorXd next = project_to_simplex(lambda - step * grad_qp);
    if ((next - lambda).norm() <= tol * std::max(1.0, lambda.norm())) {
      lambda = next;
      break;
    }
    lambda = next;
  }
  return lambda;
}

static inline Eigen::VectorXd convex_combine_gradients(
    const std::vector<Eigen::VectorXd> &grads,
    const Eigen::VectorXd &lambda) {
  if (grads.empty()) return Eigen::VectorXd();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(grads.front().size());
  for (int i = 0; i < (int)grads.size(); ++i) {
    g.noalias() += lambda[i] * grads[i];
  }
  return g;
}

static inline std::vector<_Point3> perturb_sites_for_next_proposal(
    const std::vector<_Point3> &base_sites,
    const std::vector<Vec3> &surface_normals,
    const std::vector<char> &active_mask,
    double sigma,
    const BGAL::_ManifoldModel &model,
    std::mt19937 &rng) {
  std::vector<_Point3> perturbed = base_sites;
  if (!(sigma > 0.0) || !std::isfinite(sigma)) return perturbed;

  std::normal_distribution<double> normal01(0.0, 1.0);
  std::uniform_real_distribution<double> unif01(0.0, 1.0);

  int num_active = 0;
  for (char v : active_mask) num_active += (v != 0);

  for (int i = 0; i < (int)base_sites.size(); ++i) {
    const bool use_site = (num_active > 0) ? (active_mask[i] != 0)
                                           : (unif01(rng) < 0.1);
    if (!use_site) continue;

    Vec3 n = (i < (int)surface_normals.size()) ? surface_normals[i]
                                               : Vec3(0.0, 0.0, 1.0);
    const double n_norm = n.norm();
    if (n_norm < 1e-12) n = Vec3(0.0, 0.0, 1.0);
    else n /= n_norm;

    Vec3 t1 = orthogonal_unit(n);
    Vec3 t2 = n.cross(t1);
    const double t2_norm = t2.norm();
    if (t2_norm < 1e-12) {
      t2 = orthogonal_unit(t1);
    } else {
      t2 /= t2_norm;
    }

    const double a = normal01(rng);
    const double b = normal01(rng);
    Vec3 tangent_step = sigma * (a * t1 + b * t2);
    const double max_len = 3.0 * sigma;
    const double len = tangent_step.norm();
    if (len > max_len && len > 1e-16) tangent_step *= (max_len / len);

    const Vec3 p = to_eigen(base_sites[i]) + tangent_step;
    perturbed[i] = project_to_surface(model, to_point(p));
  }
  return perturbed;
}


// 【速度核弹级提升】使用邻接表交集法代替原来的 std::set，提取 Delaunay 表面结构
static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_edges(
    int num_sites,
    const std::vector<std::map<int, std::vector<std::pair<int, int>>>> &edges) {
        
  std::vector<std::vector<int>> adj(num_sites);
  for (int i = 0; i < (int)edges.size(); ++i) {
    for (const auto &ee : edges[i]) {
      const int j = ee.first;
      if (j >= 0 && j < num_sites && j != i) {
        adj[i].push_back(j);
      }
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < num_sites; ++i) {
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
  }

  std::vector<Eigen::Vector3i> tris;
  // 避免过度锁争用，采用局部收集
  #pragma omp parallel
  {
      std::vector<Eigen::Vector3i> local_tris;
      #pragma omp for schedule(dynamic)
      for (int u = 0; u < num_sites; ++u) {
        for (int v : adj[u]) {
          if (u >= v) continue;
          
          // 求两个有序数组的交集来找到共同邻居 w
          int i = 0, j = 0;
          while (i < adj[u].size() && j < adj[v].size()) {
            if (adj[u][i] == adj[v][j]) {
                int w = adj[u][i];
                if (v < w) {
                   local_tris.emplace_back(u, v, w);
                }
                i++; j++;
            } else if (adj[u][i] < adj[v][j]) {
                i++;
            } else {
                j++;
            }
          }
        }
      }
      #pragma omp critical
      {
          tris.insert(tris.end(), local_tris.begin(), local_tris.end());
      }
  }
  return tris;
}

namespace {

using CGALKernel = CGAL::Simple_cartesian<double>;
using CGALPoint3 = CGALKernel::Point_3;
using CGALTreeTraits = CGAL::Search_traits_3<CGALKernel>;
using CGALKDTree = CGAL::Kd_tree<CGALTreeTraits>;
using CGALFuzzySphere = CGAL::Fuzzy_sphere<CGALTreeTraits>;
using PointKey = std::array<double, 3>;

static inline PointKey make_point_key(const _Point3 &p) {
  return PointKey{p.x(), p.y(), p.z()};
}

static inline PointKey make_point_key(const CGALPoint3 &p) {
  return PointKey{p.x(), p.y(), p.z()};
}

// =========================================================================
// 核心逻辑：CGAL kd-tree 半径查询 + 候选四元组枚举
// =========================================================================
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
    const double ri = (double)spheres[i].r;
    if (std::isfinite(ri) && ri >= 0.0) {
      global_rmax = std::max(global_rmax, ri);
    }
  }

  std::vector<CGALPoint3> kd_points;
  kd_points.reserve(n);
  std::map<PointKey, std::vector<int>> point_to_ids;
  for (int i = 0; i < n; ++i) {
    const CGALPoint3 p(sites[i].x(), sites[i].y(), sites[i].z());
    kd_points.push_back(p);
    point_to_ids[make_point_key(p)].push_back(i);
  }

  CGALKDTree tree(kd_points.begin(), kd_points.end());

  std::vector<CGALPoint3> hits;
  hits.reserve(256);
  std::vector<std::array<int, 4>> local_quads;
  local_quads.reserve(std::min<std::size_t>(faces.size() * 4ull, 1ull << 20));

  for (int f_idx = 0; f_idx < (int)faces.size(); ++f_idx) {
    const auto &f = faces[f_idx];
    const int i = f.x(), j = f.y(), k = f.z();
    if (i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n) continue;

    const double ri = (double)spheres[i].r;
    const double rj = (double)spheres[j].r;
    const double rk = (double)spheres[k].r;
    if (!std::isfinite(ri) || !std::isfinite(rj) || !std::isfinite(rk) ||
        ri < 0.0 || rj < 0.0 || rk < 0.0) {
      continue;
    }

    int anchor = i;
    double rmin = ri;
    if (rj < rmin) {
      rmin = rj;
      anchor = j;
    }
    if (rk < rmin) {
      rmin = rk;
      anchor = k;
    }

    const double query_radius = rmin + global_rmax;
    if (!std::isfinite(query_radius) || query_radius < 0.0) continue;

    hits.clear();
    const CGALPoint3 query_center(sites[anchor].x(), sites[anchor].y(),
                                  sites[anchor].z());
    tree.search(std::back_inserter(hits),
                CGALFuzzySphere(query_center, query_radius));

    for (const auto &hp : hits) {
      auto it = point_to_ids.find(make_point_key(hp));
      if (it == point_to_ids.end()) continue;

      for (int l : it->second) {
        if (l == i || l == j || l == k) continue;
        if (l < 0 || l >= n) continue;

        const double rl = (double)spheres[l].r;
        if (!std::isfinite(rl) || rl < 0.0) continue;

        std::array<int, 4> q{i, j, k, l};
        if (q[0] > q[1]) std::swap(q[0], q[1]);
        if (q[1] > q[2]) std::swap(q[1], q[2]);
        if (q[2] > q[3]) std::swap(q[2], q[3]);
        if (q[0] > q[1]) std::swap(q[0], q[1]);
        if (q[1] > q[2]) std::swap(q[1], q[2]);
        if (q[0] > q[1]) std::swap(q[0], q[1]);
        local_quads.push_back(q);
      }
    }
  }

  quads.swap(local_quads);
  std::sort(quads.begin(), quads.end());
  quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
}

} // namespace 闭合
static inline void compute_cell_geom_and_spheres(
    const std::vector<_Point3> &sites,
    const _Restricted_Tessellation3D &rvd,
    std::vector<CellGeom> &cells,
    std::vector<Sphere::Sphere> &spheres) {
        
  const auto &cell_tris = rvd.get_cells_();
  cells.assign(sites.size(), CellGeom());
  spheres.assign(sites.size(), Sphere::Sphere());

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int)cell_tris.size(); ++i) {
    // 采用局部定长数组或者小 vector 替代 std::set，加速极其明显
    std::vector<int> uniq;
    uniq.reserve(cell_tris[i].size() * 3);
    for (const auto &tri : cell_tris[i]) {
      uniq.push_back(std::get<0>(tri));
      uniq.push_back(std::get<1>(tri));
      uniq.push_back(std::get<2>(tri));
    }
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    cells[i].vertex_ids = std::move(uniq);
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

  std::string remesh_path =
      outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
      "_Remesh.obj";
  std::ofstream out_remesh(remesh_path);
  std::ofstream out_remesh_iter;
  if (step > 0) {
    std::string remesh_iter_path =
        outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
        "_Iter" + std::to_string(step) + "_Remesh.obj";
    out_remesh_iter.open(remesh_iter_path, std::ios::out | std::ios::trunc);
  }

  if (out_remesh) {
    const auto rdt_faces = build_rdt_faces_from_edges((int)sites.size(), rvd.get_edges_());

    for (const auto &s : sites) {
      out_remesh << "v " << s << "\n";
      if (out_remesh_iter) {
        out_remesh_iter << "v " << s << "\n";
      }
    }

    for (const auto &f : rdt_faces) {
      out_remesh << "f " << f.x() + 1 << " " << f.y() + 1 << " "
                 << f.z() + 1 << "\n";
      if (out_remesh_iter) {
        out_remesh_iter << "f " << f.x() + 1 << " " << f.y() + 1 << " "
                        << f.z() + 1 << "\n";
      }
    }
  }
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
  
  // IO串行，但我们可以预先并行计算表面法向
  std::vector<std::pair<int, _Point3>> cached_normals(spheres.size());
  #pragma omp parallel for
  for (int i = 0; i < spheres.size(); ++i) {
      auto nearest = const_cast<BGAL::_ManifoldModel &>(model).nearest_point_(spheres[i].c);
      int face_id = std::get<2>(nearest);
      _Point3 normal(0.0, 0.0, 0.0);
      if (face_id >= 0 && face_id < model.number_faces_()) {
        normal = model.normal_face_(face_id);
        normal.normalized_();
      }
      cached_normals[i] = {face_id, normal};
  }

  for (size_t i = 0; i < spheres.size(); ++i) {
    out << spheres[i].c.x() << "," << spheres[i].c.y() << "," << spheres[i].c.z() << ","
        << spheres[i].r << "," << cached_normals[i].first << "," 
        << cached_normals[i].second.x() << ","
        << cached_normals[i].second.y() << "," 
        << cached_normals[i].second.z() << "\n";
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
  const double total_start = omp_get_wtime();
  _history.clear();
  _sites = init_sites;
  if (_sites.empty()) {
    throw std::runtime_error("[QuadCoverLike] empty init_sites.");
  }

  const int n = (int)_sites.size();
  std::string outpath =
      (std::filesystem::current_path() / "data" / "QuadCover").string();

  double total_rebuild_time = 0.0;
  double total_rvd_time = 0.0;
  double total_sample_time = 0.0;
  double total_line_search_time = 0.0;
  double total_recovery_time = 0.0;
  double total_output_time = 0.0;

  struct RebuiltState {
    std::vector<_Point3> sites;
    std::vector<Sphere::Sphere> spheres;
    std::vector<Vec3> frozen_poles;
    std::vector<Vec3> surface_normals;
    std::vector<Eigen::Vector3i> rdt_faces;
    std::vector<std::array<int, 4>> search_quads;
    DynamicRadiusEval eval;
    double avg_r = 0.0;
    double eps = 0.0;

    explicit RebuiltState(int n_sites = 0) : eval(n_sites, false) {}
  };

  auto rebuild_state = [&](const std::vector<_Point3> &sites_in,
                           bool need_grad,
                           _Restricted_Tessellation3D &rvd,
                           double &rebuild_time_acc,
                           double &rvd_time_acc) -> RebuiltState {
    const double rebuild_start = omp_get_wtime();
    RebuiltState state(need_grad ? n : 0);
    state.sites = sites_in;

    const double rvd_start = omp_get_wtime();
    rvd.calculate_(state.sites);
    const double rvd_elapsed = wall_seconds_since(rvd_start);
    rvd_time_acc += rvd_elapsed;

    std::vector<CellGeom> cells;
    compute_cell_geom_and_spheres(state.sites, rvd, cells, state.spheres);

    state.frozen_poles.assign(n, Vec3::Zero());
    double sum_r = 0.0;
    #pragma omp parallel for reduction(+:sum_r)
    for (int i = 0; i < n; ++i) {
      state.frozen_poles[i] = to_eigen(state.spheres[i].max_point);
      sum_r += (to_eigen(state.sites[i]) - state.frozen_poles[i]).norm();
    }

    state.avg_r = sum_r / std::max(1, n);
    state.eps = 1e-4 * state.avg_r * state.avg_r;

    update_surface_normals(_model, state.sites, state.surface_normals);
    state.rdt_faces = build_rdt_faces_from_edges(n, rvd.get_edges_());
    build_quads_from_search_faces(state.rdt_faces, state.sites, state.spheres,
                                  state.search_quads);

    state.eval = evaluate_dynamic_radius_surrogate(
        state.sites, state.frozen_poles, state.surface_normals,
        state.search_quads, state.eps, need_grad);
    rebuild_time_acc += std::max(0.0, wall_seconds_since(rebuild_start) - rvd_elapsed);
    return state;
  };

  auto rebuilt_energy = [&](const RebuiltState &state) -> double {
    return state.eval.total_loss;
  };

  auto count_active_sites = [&](const RebuiltState &state) -> int {
    std::vector<char> active_mask = build_active_site_mask(
        state.sites, state.frozen_poles, state.search_quads, state.eps);
    return std::max(1, count_active_mask(active_mask));
  };

  auto is_better_candidate = [&](const RebuiltState &cand,
                                 const RebuiltState &ref,
                                 double cand_E,
                                 double ref_E) -> bool {
    if (cand.eval.active_quads != ref.eval.active_quads) {
      return cand.eval.active_quads < ref.eval.active_quads;
    }
    if (std::abs(cand.eval.min_g - ref.eval.min_g) > 1e-15) {
      return cand.eval.min_g > ref.eval.min_g;
    }
    return cand_E < ref_E;
  };

  const uint32_t base_seed = 123456789u;
  RebuiltState current_state =
      rebuild_state(_sites, true, _RVD, total_rebuild_time, total_rvd_time);
  double current_rebuilt_E = rebuilt_energy(current_state);
  const double initial_rebuilt_E = std::max(current_rebuilt_E, 1e-30);

  std::vector<_Point3> best_sites = current_state.sites;
  double best_rebuilt_E = current_rebuilt_E;

  double step_radius =
      std::max(1e-6, 0.03 * std::max(current_state.avg_r, 1e-8) *
                         std::sqrt((double)count_active_sites(current_state)));
  int stagnation_streak = 0;

  std::vector<LBFGSPair> lbfgs_history;
  const int lbfgs_memory = 5;
  const double lbfgs_descent_eta = 1e-3;
  const double alpha_clear_threshold = 0.05;
  const int recovery_stag_trigger = 2;
  const double recovery_min_step_ratio = 1e-3;

  const double outer_stop_tol = 1e-12;
  const double c1 = 1e-4;

  for (int outer = 0; outer < _para.max_outer_iterations; ++outer) {
    const double iter_start = omp_get_wtime();
    const double rebuild_time_before = total_rebuild_time;
    const double rvd_time_before = total_rvd_time;
    const double sample_time_before = total_sample_time;
    const double line_search_time_before = total_line_search_time;
    const double recovery_time_before = total_recovery_time;
    const double output_time_before = total_output_time;

    _sites = current_state.sites;
    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    if (_para.export_each_iteration || outer == 0) {
      const double output_rvd_start = omp_get_wtime();
      _RVD.calculate_(_sites);
      total_rvd_time += wall_seconds_since(output_rvd_start);
      const double output_start = omp_get_wtime();
      output_mesh(_sites, _RVD, n, outpath, model_name, outer);
      write_spheres_csv(outpath, n, model_name, outer, current_state.spheres,
                        _model);
      total_output_time += wall_seconds_since(output_start);
    }

    if (current_state.eval.active_quads == 0 &&
        current_rebuilt_E < outer_stop_tol) {
      _IterationInfo info;
      info.iteration = outer + 1;
      info.num_quads = (int)current_state.search_quads.size();
      info.active_quads = current_state.eval.active_quads;
      info.min_margin = current_state.eval.min_g;
      info.accepted_step = 0.0;
      _history.push_back(info);

      if (_para.is_show) {
        const double iter_time = wall_seconds_since(iter_start);
        std::cout << "[QuadCoverLike][GS] iter=" << info.iteration
                  << " | Quads=" << info.num_quads
                  << " | Act=" << info.active_quads
                  << " | E(rebuilt)=" << std::scientific
                  << std::setprecision(3) << current_rebuilt_E
                  << " | iterT=" << std::fixed << std::setprecision(3)
                  << iter_time << "s"
                  << " | stepR=" << step_radius
                  << " | STOP (Constraint Satisfied)" << std::endl;
      }
      break;
    }

    const int prev_num_quads = (int)current_state.search_quads.size();
    const int prev_active_quads = current_state.eval.active_quads;
    const double prev_min_g = current_state.eval.min_g;
    const double prev_E = current_rebuilt_E;

    std::vector<char> active_mask = build_active_site_mask(
        current_state.sites, current_state.frozen_poles,
        current_state.search_quads, current_state.eps);
    std::vector<char> opt_mask = expand_active_mask_one_ring(
        active_mask, current_state.rdt_faces, n, 1);
    if (!has_any_active(opt_mask)) opt_mask = make_all_active_mask(n);
    const int active_site_count = std::max(1, count_active_mask(opt_mask));

    Eigen::MatrixXd center_grad_mat = current_state.eval.grads;
    zero_out_inactive_rows(center_grad_mat, opt_mask);
    Eigen::VectorXd center_grad_vec = flatten_matrix(center_grad_mat);

    const bool late_stage =
        (current_state.eval.active_quads <= 96) ||
        (active_site_count <= 192) ||
        (current_rebuilt_E <= 1e-3 * std::max(initial_rebuilt_E, 1e-30));

    const int num_samples = 6;
    const double sigma_base = std::max(
        1e-6,
        (late_stage ? 0.020 : 0.035) * std::max(current_state.avg_r, 1e-8) *
            (1.0 + 0.15 * std::min(stagnation_streak, 8)));

    std::vector<Eigen::VectorXd> gs_grads;
    gs_grads.reserve(num_samples + 1);
    gs_grads.push_back(center_grad_vec);

    RebuiltState best_sample_state = current_state;
    double best_sample_E = prev_E;

    std::vector<RebuiltState> sample_states(num_samples);
    std::vector<Eigen::VectorXd> sample_grad_vecs(num_samples);
    std::vector<double> sample_energies(
        num_samples, std::numeric_limits<double>::infinity());

    const double sample_start = omp_get_wtime();
    #pragma omp parallel
    {
      _Restricted_Tessellation3D local_rvd(_model);
      double local_rebuild_time = 0.0;
      double local_rvd_time = 0.0;

      #pragma omp for schedule(dynamic, 1)
      for (int s = 0; s < num_samples; ++s) {
        std::seed_seq seq{base_seed, static_cast<uint32_t>(outer + 1),
                          static_cast<uint32_t>(s + 1), 0xA511E9B3u};
        std::mt19937 local_rng(seq);

        std::vector<_Point3> sample_sites = perturb_sites_for_next_proposal(
            current_state.sites, current_state.surface_normals, opt_mask,
            sigma_base, _model, local_rng);
        RebuiltState sample_state = rebuild_state(
            sample_sites, true, local_rvd, local_rebuild_time, local_rvd_time);

        Eigen::MatrixXd gmat = sample_state.eval.grads;
        zero_out_inactive_rows(gmat, opt_mask);
        sample_grad_vecs[s] = flatten_matrix(gmat);
        sample_energies[s] = rebuilt_energy(sample_state);
        sample_states[s] = std::move(sample_state);
      }

      #pragma omp critical
      {
        total_rebuild_time += local_rebuild_time;
        total_rvd_time += local_rvd_time;
      }
    }
    total_sample_time += wall_seconds_since(sample_start);

    for (int s = 0; s < num_samples; ++s) {
      gs_grads.push_back(std::move(sample_grad_vecs[s]));
      const double sample_E = sample_energies[s];
      if (sample_E < best_sample_E ||
          is_better_candidate(sample_states[s], best_sample_state, sample_E,
                              best_sample_E)) {
        best_sample_E = sample_E;
        best_sample_state = std::move(sample_states[s]);
      }
    }

    const Eigen::VectorXd lambda = solve_min_norm_convex_combination(
        gs_grads, late_stage ? 192 : 96, 1e-10);
    Eigen::VectorXd gs_grad_vec = convex_combine_gradients(gs_grads, lambda);
    Eigen::MatrixXd gs_grad_mat = unflatten_vector(gs_grad_vec, n);
    zero_out_inactive_rows(gs_grad_mat, opt_mask);
    gs_grad_vec = flatten_matrix(gs_grad_mat);

    const double stationarity = gs_grad_vec.norm();
    bool used_lbfgs_metric = false;
    Eigen::VectorXd dir_vec = -gs_grad_vec;
    if (!lbfgs_history.empty() && stationarity > 1e-14) {
      Eigen::VectorXd metric_dir = lbfgs_two_loop_direction(gs_grad_vec, lbfgs_history);
      Eigen::MatrixXd metric_dir_mat = unflatten_vector(metric_dir, n);
      zero_out_inactive_rows(metric_dir_mat, opt_mask);
      metric_dir = flatten_matrix(metric_dir_mat);

      const double metric_norm = metric_dir.norm();
      const double metric_dirderiv = gs_grad_vec.dot(metric_dir);
      if (std::isfinite(metric_norm) && metric_norm > 1e-14 &&
          std::isfinite(metric_dirderiv) &&
          metric_dirderiv <= -lbfgs_descent_eta * gs_grad_vec.squaredNorm()) {
        dir_vec = metric_dir;
        used_lbfgs_metric = true;
      }
    }

    Eigen::MatrixXd dir_mat = unflatten_vector(dir_vec, n);
    zero_out_inactive_rows(dir_mat, opt_mask);
    dir_vec = flatten_matrix(dir_mat);
    const double dir_norm = dir_vec.norm();

    bool accepted = false;
    bool accepted_sample = false;
    bool forced_accept_clear = false;
    bool recovery_accept = false;
    double accepted_step = 0.0;
    double alpha_used = 0.0;
    double last_alpha_tried = 0.0;
    bool memory_cleared = false;
    RebuiltState candidate_state = current_state;
    double cand_E = prev_E;

    const double improve_E_tol =
        std::max(1e-8, 1e-4 * std::max(prev_E, 1e-12));
    const double improve_g_tol =
        std::max(1e-6, 5e-3 * std::max(std::abs(prev_min_g), 1e-3));

    const double local_step_min =
        std::max(1e-12, 1e-4 * std::max(current_state.avg_r, 1e-12));
    const double local_step_max = std::max(
        local_step_min * 10.0,
        0.10 * std::max(current_state.avg_r, 1e-12) *
            std::sqrt((double)std::max(1, n)));
    step_radius = std::min(local_step_max, std::max(local_step_min, step_radius));

    if (stationarity > 1e-14 && dir_norm > 1e-14) {
      const double line_search_start = omp_get_wtime();
      double alpha = std::min(1.0, step_radius / std::max(dir_norm, 1e-16));
      const int max_backtracking = late_stage ? 12 : 10;

      for (int bt = 0; bt < max_backtracking; ++bt) {
        last_alpha_tried = alpha;
        Eigen::MatrixXd proposed_step = alpha * dir_mat;
        std::vector<_Point3> trial_sites =
            apply_projected_step(current_state.sites, proposed_step, _model);
        Eigen::MatrixXd realized =
            realized_step_matrix(current_state.sites, trial_sites);
        zero_out_inactive_rows(realized, opt_mask);
        Eigen::VectorXd realized_vec = flatten_matrix(realized);
        const double realized_norm = realized_vec.norm();
        if (!(realized_norm > 1e-16)) {
          alpha *= 0.5;
          continue;
        }

        const double dirderiv = gs_grad_vec.dot(realized_vec);
        if (!(dirderiv < -1e-18)) {
          alpha *= 0.5;
          continue;
        }

        RebuiltState trial_state = rebuild_state(
            trial_sites, false, _RVD, total_rebuild_time, total_rvd_time);
        const double trial_E = rebuilt_energy(trial_state);
        const double armijo_rhs = prev_E + c1 * dirderiv;

        if (trial_E <= armijo_rhs + 1e-14 ||
            trial_E < prev_E - std::max(1e-8, 1e-4 * std::max(prev_E, 1e-12))) {
          accepted = true;
          candidate_state = rebuild_state(
              trial_sites, true, _RVD, total_rebuild_time, total_rvd_time);
          cand_E = rebuilt_energy(candidate_state);
          accepted_step = max_row_norm(realized_step_matrix(current_state.sites,
                                                            candidate_state.sites));
          alpha_used = alpha;
          break;
        }
        alpha *= 0.5;
      }
      total_line_search_time += wall_seconds_since(line_search_start);
    }

    if (!accepted && best_sample_E < prev_E) {
      accepted = true;
      accepted_sample = true;
      candidate_state = std::move(best_sample_state);
      cand_E = best_sample_E;
      accepted_step = max_row_norm(
          realized_step_matrix(current_state.sites, candidate_state.sites));
      alpha_used = 0.0;
    }

    const bool provisional_improved = accepted &&
        (cand_E < prev_E - improve_E_tol ||
         candidate_state.eval.active_quads < prev_active_quads ||
         candidate_state.eval.min_g > prev_min_g + improve_g_tol);
    const bool tiny_motion_stall = accepted && !accepted_sample &&
        accepted_step <= recovery_min_step_ratio * std::max(step_radius, 1e-16) &&
        !provisional_improved;
    const bool should_try_recovery =
        ((!accepted && last_alpha_tried > 0.0 && last_alpha_tried < alpha_clear_threshold) ||
         (stagnation_streak >= recovery_stag_trigger && (!accepted || tiny_motion_stall)));

    if (should_try_recovery) {
      const double recovery_start = omp_get_wtime();
      lbfgs_history.clear();
      memory_cleared = true;

      const int rescue_samples = late_stage ? 6 : 4;
      double rescue_sigma = std::max(
          2.5 * sigma_base,
          0.05 * std::max(current_state.avg_r, 1e-8) *
              (1.0 + 0.20 * std::min(stagnation_streak, 10)));

      auto recovery_admissible = [&](const RebuiltState &cand, double cand_eval_E) {
        const bool act_better = cand.eval.active_quads < prev_active_quads;
        const bool margin_better = cand.eval.min_g > prev_min_g + improve_g_tol;
        const bool energy_better = cand_eval_E < prev_E - improve_E_tol;
        const bool all_worse =
            (cand.eval.active_quads > prev_active_quads) &&
            (cand.eval.min_g < prev_min_g - improve_g_tol) &&
            (cand_eval_E > prev_E + improve_E_tol);
        return !all_worse && (act_better || margin_better || energy_better);
      };

      auto better_recovery = [&](const RebuiltState &cand, double cand_eval_E,
                                 const RebuiltState &ref, double ref_eval_E) {
        if (cand.eval.active_quads != ref.eval.active_quads) {
          return cand.eval.active_quads < ref.eval.active_quads;
        }
        if (std::abs(cand.eval.min_g - ref.eval.min_g) > improve_g_tol) {
          return cand.eval.min_g > ref.eval.min_g;
        }
        return cand_eval_E < ref_eval_E - improve_E_tol;
      };

      RebuiltState best_recovery_state = current_state;
      double best_recovery_E = prev_E;
      bool found_recovery = false;

      for (int pass = 0; pass < 1; ++pass) {
        std::vector<RebuiltState> recovery_states(rescue_samples);
        std::vector<double> recovery_energies(
            rescue_samples, std::numeric_limits<double>::infinity());
        std::vector<char> recovery_valid(rescue_samples, 0);

        #pragma omp parallel
        {
          _Restricted_Tessellation3D local_rvd(_model);
          double local_rebuild_time = 0.0;
          double local_rvd_time = 0.0;

          #pragma omp for schedule(dynamic, 1)
          for (int s = 0; s < rescue_samples; ++s) {
            std::seed_seq seq{base_seed, static_cast<uint32_t>(outer + 1),
                              static_cast<uint32_t>(pass + 1),
                              static_cast<uint32_t>(s + 1), 0x85EBCA6Bu};
            std::mt19937 local_rng(seq);

            std::vector<_Point3> sample_sites = perturb_sites_for_next_proposal(
                current_state.sites, current_state.surface_normals, opt_mask,
                rescue_sigma, _model, local_rng);
            Eigen::MatrixXd realized =
                realized_step_matrix(current_state.sites, sample_sites);
            zero_out_inactive_rows(realized, opt_mask);
            const double realized_step = max_row_norm(realized);
            if (!(realized_step > 1e-10)) continue;

            RebuiltState sample_state = rebuild_state(
                sample_sites, false, local_rvd, local_rebuild_time, local_rvd_time);
            const double sample_E = rebuilt_energy(sample_state);
            if (!recovery_admissible(sample_state, sample_E)) continue;

            recovery_valid[s] = 1;
            recovery_energies[s] = sample_E;
            recovery_states[s] = std::move(sample_state);
          }

          #pragma omp critical
          {
            total_rebuild_time += local_rebuild_time;
            total_rvd_time += local_rvd_time;
          }
        }

        for (int s = 0; s < rescue_samples; ++s) {
          if (!recovery_valid[s]) continue;
          const double sample_E = recovery_energies[s];
          if (!found_recovery ||
              better_recovery(recovery_states[s], sample_E, best_recovery_state,
                              best_recovery_E)) {
            best_recovery_state = std::move(recovery_states[s]);
            best_recovery_E = sample_E;
            found_recovery = true;
          }
        }
        if (found_recovery) break;
        rescue_sigma *= 2.0;
      }

      if (found_recovery) {
        accepted = true;
        accepted_sample = true;
        forced_accept_clear = true;
        recovery_accept = true;
        candidate_state = rebuild_state(
            best_recovery_state.sites, true, _RVD, total_rebuild_time,
            total_rvd_time);
        cand_E = rebuilt_energy(candidate_state);
        accepted_step = max_row_norm(
            realized_step_matrix(current_state.sites, candidate_state.sites));
        alpha_used = 0.0;
      }
      total_recovery_time += wall_seconds_since(recovery_start);
    }

    if (accepted) {
      std::vector<char> next_active_mask = build_active_site_mask(
          candidate_state.sites, candidate_state.frozen_poles,
          candidate_state.search_quads, candidate_state.eps);
      std::vector<char> next_opt_mask = expand_active_mask_one_ring(
          next_active_mask, candidate_state.rdt_faces, n, 1);
      if (!has_any_active(next_opt_mask)) next_opt_mask = make_all_active_mask(n);

      std::vector<char> hist_mask(n, 0);
      for (int i = 0; i < n; ++i) {
        hist_mask[i] = (opt_mask[i] != 0 || next_opt_mask[i] != 0) ? 1 : 0;
      }

      Eigen::MatrixXd prev_hist_grad = current_state.eval.grads;
      Eigen::MatrixXd next_hist_grad = candidate_state.eval.grads;
      zero_out_inactive_rows(prev_hist_grad, hist_mask);
      zero_out_inactive_rows(next_hist_grad, hist_mask);

      Eigen::MatrixXd s_mat =
          realized_step_matrix(current_state.sites, candidate_state.sites);
      zero_out_inactive_rows(s_mat, hist_mask);

      const Eigen::VectorXd s_vec = flatten_matrix(s_mat);
      const Eigen::VectorXd y_vec =
          flatten_matrix(next_hist_grad) - flatten_matrix(prev_hist_grad);
      if (!forced_accept_clear && s_vec.norm() > 1e-16) {
        push_lbfgs_pair(lbfgs_history, s_vec, y_vec, lbfgs_memory);
      }

      current_state = std::move(candidate_state);
      current_rebuilt_E = cand_E;
      _sites = current_state.sites;
      if (current_rebuilt_E < best_rebuilt_E) {
        best_rebuilt_E = current_rebuilt_E;
        best_sites = current_state.sites;
      }

      if (recovery_accept && accepted_step > 0.0) {
        step_radius = std::min(local_step_max,
                               std::max(1.10 * step_radius, 1.75 * accepted_step));
      } else if (!accepted_sample && alpha_used >= 0.8 && accepted_step >= 0.8 * step_radius) {
        step_radius = std::min(local_step_max,
                               std::max(1.35 * step_radius, 1.25 * accepted_step));
      } else if (accepted_step > 0.0) {
        step_radius = std::min(local_step_max,
                               std::max(0.95 * step_radius, 1.50 * accepted_step));
      }
    } else {
      current_rebuilt_E = prev_E;
      step_radius = std::max(local_step_min, 0.5 * step_radius);
    }

    const bool improved = accepted &&
        (cand_E < prev_E - improve_E_tol ||
         current_state.eval.active_quads < prev_active_quads ||
         current_state.eval.min_g > prev_min_g + improve_g_tol);
    if (improved) stagnation_streak = 0;
    else ++stagnation_streak;

    if (accepted && !accepted_sample && !forced_accept_clear && !improved &&
        alpha_used > 0.0 && alpha_used < alpha_clear_threshold) {
      lbfgs_history.clear();
      memory_cleared = true;
    }

    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    _IterationInfo info;
    info.iteration = outer + 1;
    info.num_quads = (int)current_state.search_quads.size();
    info.active_quads = current_state.eval.active_quads;
    info.min_margin = current_state.eval.min_g;
    info.accepted_step = accepted ? accepted_step : 0.0;
    _history.push_back(info);

    if (_para.is_show) {
      const double iter_time = wall_seconds_since(iter_start);
      const double iter_rebuild_time = total_rebuild_time - rebuild_time_before;
      const double iter_rvd_time = total_rvd_time - rvd_time_before;
      const double iter_sample_time = total_sample_time - sample_time_before;
      const double iter_line_search_time =
          total_line_search_time - line_search_time_before;
      const double iter_recovery_time =
          total_recovery_time - recovery_time_before;
      const double iter_output_time = total_output_time - output_time_before;
      std::cout << "[QuadCoverLike][GS] iter=" << info.iteration
                << " | Quads(prev/curr)=" << prev_num_quads << "/"
                << info.num_quads
                << " | Act(prev/curr)=" << prev_active_quads << "/"
                << info.active_quads
                << " | E(rebuilt)=" << std::scientific << std::setprecision(3)
                << prev_E << " -> " << current_rebuilt_E
                << " | min_g(prev/curr)=" << prev_min_g << "/"
                << info.min_margin
                << " | samples=" << num_samples
                << " | sigma=" << sigma_base
                << " | ||g_GS||=" << stationarity
                << " | ||d||=" << dir_norm
                << " | metric=" << (used_lbfgs_metric ? "LBFGS" : "GS")
                << " | mem=" << lbfgs_history.size()
                << " | memReset=" << (memory_cleared ? 1 : 0)
                << " | step=" << accepted_step
                << " | alpha=" << alpha_used
                << " | sampleAccept=" << (accepted_sample ? 1 : 0)
                << " | recoverAccept=" << (recovery_accept ? 1 : 0)
                << " | stag=" << stagnation_streak
                << " | iterT=" << std::fixed << std::setprecision(3)
                << iter_time << "s"
                << " | rebuildT=" << iter_rebuild_time << "s"
                << " | rvdT=" << iter_rvd_time << "s"
                << " | sampleT=" << iter_sample_time << "s"
                << " | lineT=" << iter_line_search_time << "s"
                << " | recoverT=" << iter_recovery_time << "s"
                << " | outputT=" << iter_output_time << "s"
                << " | stepR=" << step_radius
                << " | " << (accepted ? "ACCEPT" : "REJECT")
                << std::endl;
    }
  }

  _sites = best_sites;
  const double final_rvd_start = omp_get_wtime();
  _RVD.calculate_(_sites);
  total_rvd_time += wall_seconds_since(final_rvd_start);
  std::vector<CellGeom> final_cells;
  compute_cell_geom_and_spheres(_sites, _RVD, final_cells, _spheres);
  const double final_output_start = omp_get_wtime();
  output_mesh(_sites, _RVD, n, outpath, model_name,
              (int)_history.size() + 1);
  write_spheres_csv(outpath, n, model_name, (int)_history.size() + 1,
                    _spheres, _model);
  total_output_time += wall_seconds_since(final_output_start);

  if (_para.is_show) {
    const double total_time = wall_seconds_since(total_start);
    std::cout << "\n--- QuadCover Finished ---\n"
              << "Total Time: " << std::fixed << std::setprecision(3)
              << total_time << " s\n"
              << "Rebuild State Time: " << total_rebuild_time << " s\n"
              << "RVD Calculation Time: " << total_rvd_time << " s\n"
              << "Gradient Sampling Time: " << total_sample_time << " s\n"
              << "Line Search Time: " << total_line_search_time << " s\n"
              << "Recovery Search Time: " << total_recovery_time << " s\n"
              << "Output Time: " << total_output_time << " s\n"
              << std::endl;
  }
}

} // namespace BGAL
