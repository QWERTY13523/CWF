#include "BGAL/QuadCoverLike/QuadCover.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/CVTLike/CVT.h"
#include "BGAL/BaseShape/KDTree.h"
#include "BGAL/Integral/Integral.h"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>

#include <Eigen/Dense>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
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
#include <omp.h> // 寮曞叆 OpenMP

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

static inline bool point_is_finite(const _Point3 &p) {
  return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

static inline bool sites_are_finite(const std::vector<_Point3> &sites) {
  return std::all_of(sites.begin(), sites.end(),
                     [](const _Point3 &p) { return point_is_finite(p); });
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

// 涓ヨ皑鐨勫姩鎬佸崐寰勮瘎浼颁綋
struct DynamicRadiusEval {
  double total_loss = 0.0;
  double qem_loss = 0.0;
  double hinge_loss = 0.0;
  double weighted_hinge_loss = 0.0;
  double hinge_lambda = 0.0;
  int active_quads = 0;
  double min_g = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd qem_grads;
  Eigen::MatrixXd hinge_grads;
  Eigen::MatrixXd grads;

  explicit DynamicRadiusEval(int n = 0, bool need_grad = false)
      : qem_grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()),
        hinge_grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()),
        grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()) {}
};

struct HingeStats {
  double hinge_loss = 0.0;
  int active_quads = 0;
  double min_g = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd grads;

  explicit HingeStats(int n = 0, bool need_grad = false)
      : grads(need_grad ? Eigen::MatrixXd::Zero(n, 3) : Eigen::MatrixXd()) {}
};

static inline void apply_eval_weights(DynamicRadiusEval &eval,
                                      double qem_weight_value,
                                      double hinge_lambda_value) {
  eval.hinge_lambda = hinge_lambda_value;
  eval.weighted_hinge_loss = eval.hinge_lambda * eval.hinge_loss;
  eval.total_loss = qem_weight_value * eval.qem_loss + eval.weighted_hinge_loss;
  if (eval.grads.size() != 0 &&
      eval.qem_grads.rows() == eval.grads.rows() &&
      eval.hinge_grads.rows() == eval.grads.rows()) {
    eval.grads = qem_weight_value * eval.qem_grads +
                 eval.hinge_lambda * eval.hinge_grads;
  }
}

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
  
  // 骞惰鍖栧鎵捐〃闈㈡硶绾?
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

static inline Eigen::MatrixXd sites_to_matrix(const std::vector<_Point3> &sites) {
  Eigen::MatrixXd M((int)sites.size(), 3);
  #pragma omp parallel for
  for (int i = 0; i < (int)sites.size(); ++i) {
    M(i, 0) = sites[i].x();
    M(i, 1) = sites[i].y();
    M(i, 2) = sites[i].z();
  }
  return M;
}

static inline torch::Tensor sites_to_tensor(const std::vector<_Point3> &sites,
                                            torch::Device device = torch::kCPU) {
  auto t = torch::empty({(long long)sites.size(), 3},
                        torch::TensorOptions().dtype(torch::kFloat64).device(device));
  auto acc = t.accessor<double, 2>();
  #pragma omp parallel for
  for (int i = 0; i < (int)sites.size(); ++i) {
    acc[i][0] = sites[i].x();
    acc[i][1] = sites[i].y();
    acc[i][2] = sites[i].z();
  }
  return t;
}

static inline torch::Tensor eigen_matrix_to_tensor(const Eigen::MatrixXd &M,
                                                   torch::Device device = torch::kCPU) {
  auto t = torch::empty({(long long)M.rows(), (long long)M.cols()},
                        torch::TensorOptions().dtype(torch::kFloat64).device(device));
  auto acc = t.accessor<double, 2>();
  #pragma omp parallel for
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      acc[i][j] = M(i, j);
    }
  }
  return t;
}

static inline Eigen::MatrixXd tensor_to_matrix(const torch::Tensor &tensor_in) {
  torch::Tensor t = tensor_in.detach().to(torch::kCPU).contiguous();
  TORCH_CHECK(t.dim() == 2 && t.size(1) == 3, "tensor_to_matrix expects shape [N,3]");
  Eigen::MatrixXd M((int)t.size(0), 3);
  auto acc = t.accessor<double, 2>();
  #pragma omp parallel for
  for (int i = 0; i < (int)t.size(0); ++i) {
    M(i, 0) = acc[i][0];
    M(i, 1) = acc[i][1];
    M(i, 2) = acc[i][2];
  }
  return M;
}

static inline double get_adam_lr(torch::optim::Adam &optimizer) {
  auto &options =
      static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options());
  return options.lr();
}

static inline void set_adam_lr(torch::optim::Adam &optimizer, double lr) {
  auto &options =
      static_cast<torch::optim::AdamOptions &>(optimizer.param_groups()[0].options());
  options.lr(lr);
}

static inline double max_row_norm(const Eigen::MatrixXd &M) {
  double ret = 0.0;
  // 瀵瑰綊绾︽搷浣滆繘琛屽绾跨▼鍔犻€?
  #pragma omp parallel for reduction(max:ret)
  for (int i = 0; i < M.rows(); ++i) {
    ret = std::max(ret, M.row(i).norm());
  }
  return ret;
}

static constexpr double kSphereRadiusPadding = 1e-6;

static inline double padded_radius_squared(const Vec3 &site,
                                           const Vec3 &pole) {
  const double r = (site - pole).norm() + kSphereRadiusPadding;
  return r * r;
}

static inline Vec3 padded_hinge_grad_factor(const Vec3 &site,
                                            const Vec3 &pole,
                                            const Vec3 &quad_center) {
  const Vec3 site_to_pole = site - pole;
  const double d = site_to_pole.norm();
  Vec3 ret = pole - quad_center;
  if (d > 1e-30) {
    ret -= kSphereRadiusPadding * (site_to_pole / d);
  }
  return ret;
}

static inline HingeStats recompute_hinge_stats_serial(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<std::array<int, 4>> &quads,
    double eps,
    bool need_grad) {
  const int n_sites = (int)sites.size();
  HingeStats stats(n_sites, need_grad);
  const double eps_safe = (std::isfinite(eps) && eps >= 0.0) ? eps : 0.0;

  for (const auto &q : quads) {
    Vec3 P_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) P_bar += to_eigen(sites[q[k]]);
    P_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 Pi = to_eigen(sites[sid]);
      const double ri2 = padded_radius_squared(Pi, frozen_poles[sid]);
      g_val += (P_bar - Pi).squaredNorm() - ri2;
    }

    stats.min_g = std::min(stats.min_g, g_val);
    const double violation = -(g_val + eps_safe);
    if (violation > 0.0) {
      stats.hinge_loss += violation * violation;
      ++stats.active_quads;
      if (need_grad) {
        for (int k = 0; k < 4; ++k) {
          const int sid = q[k];
          const Vec3 Pi = to_eigen(sites[sid]);
          const Vec3 factor =
              padded_hinge_grad_factor(Pi, frozen_poles[sid], P_bar);
          stats.grads.row(sid) += (-4.0 * violation * factor).transpose();
        }
      }
    }
  }

  if (!std::isfinite(stats.min_g)) stats.min_g = 0.0;
  return stats;
}

static inline DynamicRadiusEval evaluate_qem_hinge_objective(
    const std::vector<_Point3> &sites,
    const _Restricted_Tessellation3D &rvd,
    const std::vector<Vec3> &frozen_poles,
    const std::vector<Vec3> &normals,
    const std::vector<std::array<int, 4>> &quads,
    double eps,
    double qem_weight_value,
    double hinge_lambda_value,
    bool need_grad) {

  const int n_sites = (int)sites.size();
  DynamicRadiusEval eval(n_sites, need_grad);
  const auto &cell_tris = rvd.get_cells_();

  double qem_loss = 0.0;
  #pragma omp parallel for reduction(+:qem_loss) schedule(dynamic)
  for (int i = 0; i < (int)cell_tris.size(); ++i) {
    Eigen::RowVector3d grad_i = Eigen::RowVector3d::Zero();
    for (const auto &tri : cell_tris[i]) {
      const _Point3 p0 = rvd.vertex_(std::get<0>(tri));
      const _Point3 p1 = rvd.vertex_(std::get<1>(tri));
      const _Point3 p2 = rvd.vertex_(std::get<2>(tri));
      _Point3 tri_normal = (p1 - p0).cross_(p2 - p0);
      if (tri_normal.length_() <= 1e-20) continue;
      tri_normal.normalized_();

      Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
          [&](BGAL::_Point3 p) {
            Eigen::VectorXd r(4);
            const double h = tri_normal.dot_(p - sites[i]);
            r(0) = h * h;
            r(1) = -2.0 * tri_normal.x() * h;
            r(2) = -2.0 * tri_normal.y() * h;
            r(3) = -2.0 * tri_normal.z() * h;
            return r;
          },
          p0, p1, p2);

      qem_loss += inte(0);
      if (need_grad) {
        grad_i += Eigen::RowVector3d(inte(1), inte(2), inte(3));
      }
    }
    if (need_grad) {
      eval.qem_grads.row(i) = grad_i;
    }
  }
  eval.qem_loss = qem_loss;

  const int num_quads = (int)quads.size();
  double hinge_loss = 0.0;
  int active_quads = 0;
  double min_g = std::numeric_limits<double>::infinity();
  std::vector<double> hinge_grad_flat(need_grad ? 3 * n_sites : 0, 0.0);

  #pragma omp parallel for reduction(+:hinge_loss,active_quads) reduction(min:min_g) schedule(static)
  for (int qi = 0; qi < num_quads; ++qi) {
    const auto &q = quads[qi];
    Vec3 P_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) P_bar += to_eigen(sites[q[k]]);
    P_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 Pi = to_eigen(sites[sid]);
      const double ri2 = padded_radius_squared(Pi, frozen_poles[sid]);
      g_val += (P_bar - Pi).squaredNorm() - ri2;
    }

    min_g = std::min(min_g, g_val);
    const double violation = -(g_val + eps);
    if (violation > 0.0) {
      hinge_loss += violation * violation;
      active_quads += 1;
      if (need_grad) {
        for (int k = 0; k < 4; ++k) {
          const int sid = q[k];
          const Vec3 Pi = to_eigen(sites[sid]);
          const Vec3 factor =
              padded_hinge_grad_factor(Pi, frozen_poles[sid], P_bar);
          const Vec3 contrib = -4.0 * violation * factor;
          #pragma omp atomic update
          hinge_grad_flat[3 * sid + 0] += contrib.x();
          #pragma omp atomic update
          hinge_grad_flat[3 * sid + 1] += contrib.y();
          #pragma omp atomic update
          hinge_grad_flat[3 * sid + 2] += contrib.z();
        }
      }
    }
  }

  eval.hinge_loss = hinge_loss;
  eval.active_quads = active_quads;
  eval.min_g = min_g;
  if (need_grad) {
    #pragma omp parallel for
    for (int i = 0; i < n_sites; ++i) {
      eval.hinge_grads(i, 0) = hinge_grad_flat[3 * i + 0];
      eval.hinge_grads(i, 1) = hinge_grad_flat[3 * i + 1];
      eval.hinge_grads(i, 2) = hinge_grad_flat[3 * i + 2];
    }
  }

  const double eps_safe = (std::isfinite(eps) && eps >= 0.0) ? eps : 0.0;
  const bool hinge_inconsistent =
      (num_quads > 0) &&
      (eval.active_quads == 0) &&
      (eval.hinge_loss <= 0.0) &&
      std::isfinite(eval.min_g) &&
      (eval.min_g < -eps_safe - 1e-15);
  const bool need_hinge_recount =
      (!std::isfinite(eps) || eps < 0.0) || hinge_inconsistent;
  if (need_hinge_recount) {
    const HingeStats recounted =
        recompute_hinge_stats_serial(sites, frozen_poles, quads, eps_safe, need_grad);
    eval.hinge_loss = recounted.hinge_loss;
    eval.active_quads = recounted.active_quads;
    eval.min_g = recounted.min_g;
    if (need_grad) {
      eval.hinge_grads = recounted.grads;
    }
  }

  if (need_grad && !normals.empty()) {
    #pragma omp parallel for
    for (int i = 0; i < eval.qem_grads.rows(); ++i) {
      if (eval.qem_grads.row(i).squaredNorm() > 1e-24) {
        Vec3 gq = eval.qem_grads.row(i).transpose();
        gq -= gq.dot(normals[i]) * normals[i];
        eval.qem_grads.row(i) = gq.transpose();
      }
      if (eval.hinge_grads.row(i).squaredNorm() > 1e-24) {
        Vec3 gh = eval.hinge_grads.row(i).transpose();
        gh -= gh.dot(normals[i]) * normals[i];
        eval.hinge_grads.row(i) = gh.transpose();
      }
    }
  }

  apply_eval_weights(eval, qem_weight_value, hinge_lambda_value);

  if (!std::isfinite(eval.min_g)) eval.min_g = 0.0;
  return eval;
}

static inline std::vector<_Point3> apply_projected_step(
    const std::vector<_Point3> &sites,
    const Eigen::MatrixXd &step,
    const BGAL::_ManifoldModel &model) {
  std::vector<_Point3> trial = sites;
  
  // 骞惰璁＄畻棰勬祴姝ュ苟鎶曞奖
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
  std::vector<int> flags(sites.size(), 0);

  #pragma omp parallel for schedule(static)
  for (int qi = 0; qi < (int)quads.size(); ++qi) {
    const auto &q = quads[qi];
    Vec3 P_bar = Vec3::Zero();
    for (int k = 0; k < 4; ++k) P_bar += to_eigen(sites[q[k]]);
    P_bar /= 4.0;

    double g_val = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int sid = q[k];
      const Vec3 Pi = to_eigen(sites[sid]);
      const double ri2 = padded_radius_squared(Pi, frozen_poles[sid]);
      g_val += (P_bar - Pi).squaredNorm() - ri2;
    }

    if (-(g_val + eps) > 0.0) {
      for (int k = 0; k < 4; ++k) {
        #pragma omp atomic write
        flags[q[k]] = 1;
      }
    }
  }

  std::vector<char> mask(flags.size(), 0);
  #pragma omp parallel for
  for (int i = 0; i < (int)flags.size(); ++i) {
    mask[i] = static_cast<char>(flags[i] != 0);
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

  std::vector<std::vector<int>> adj(num_sites);
  for (const auto &f : faces) {
    const int a = f.x(), b = f.y(), c = f.z();
    if (a < 0 || a >= num_sites || b < 0 || b >= num_sites || c < 0 || c >= num_sites) {
      continue;
    }
    adj[a].push_back(b); adj[a].push_back(c);
    adj[b].push_back(a); adj[b].push_back(c);
    adj[c].push_back(a); adj[c].push_back(b);
  }

  #pragma omp parallel for
  for (int i = 0; i < num_sites; ++i) {
    auto &nbrs = adj[i];
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  for (int r = 0; r < rings; ++r) {
    std::vector<int> next(mask.begin(), mask.end());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_sites; ++i) {
      if (!mask[i]) continue;
      for (int j : adj[i]) {
        #pragma omp atomic write
        next[j] = 1;
      }
    }
    #pragma omp parallel for
    for (int i = 0; i < num_sites; ++i) {
      mask[i] = static_cast<char>(next[i] != 0);
    }
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

static inline std::vector<double> compute_local_site_scales(
    const std::vector<_Point3> &sites,
    const std::vector<Eigen::Vector3i> &rdt_faces,
    double fallback_scale) {
  const int n = (int)sites.size();
  std::vector<std::vector<int>> neighbors(n);
  for (const auto &f : rdt_faces) {
    const int a = f.x();
    const int b = f.y();
    const int c = f.z();
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) continue;
    neighbors[a].push_back(b);
    neighbors[a].push_back(c);
    neighbors[b].push_back(a);
    neighbors[b].push_back(c);
    neighbors[c].push_back(a);
    neighbors[c].push_back(b);
  }

  std::vector<double> scales(n, fallback_scale);
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    auto &nbrs = neighbors[i];
    if (nbrs.empty()) continue;
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

    double sum_dist = 0.0;
    int cnt = 0;
    const Vec3 pi = to_eigen(sites[i]);
    for (int j : nbrs) {
      if (j < 0 || j >= n || j == i) continue;
      sum_dist += (pi - to_eigen(sites[j])).norm();
      ++cnt;
    }
    if (cnt > 0) {
      scales[i] = sum_dist / double(cnt);
    }
  }
  return scales;
}

struct TangentPerturbFrame {
  int site_id = -1;
  Vec3 t1 = Vec3::UnitX();
  Vec3 t2 = Vec3::UnitY();
  double local_scale = 1.0;
};

static inline std::vector<TangentPerturbFrame> build_tangent_perturb_frames(
    const std::vector<_Point3> &base_sites,
    const std::vector<Vec3> &surface_normals,
    const std::vector<char> &perturb_mask,
    const std::vector<Eigen::Vector3i> &rdt_faces,
    double fallback_scale) {
  std::vector<TangentPerturbFrame> frames;
  if (base_sites.empty()) return frames;

  const std::vector<double> local_scales =
      compute_local_site_scales(base_sites, rdt_faces, fallback_scale);

  int active_count = 0;
  for (int i = 0; i < (int)base_sites.size(); ++i) {
    if (i < (int)perturb_mask.size() && perturb_mask[i] != 0) ++active_count;
  }
  frames.reserve(active_count);

  for (int i = 0; i < (int)base_sites.size(); ++i) {
    if (!(i < (int)perturb_mask.size() && perturb_mask[i] != 0)) continue;

    Vec3 n = (i < (int)surface_normals.size()) ? surface_normals[i]
                                               : Vec3(0.0, 0.0, 1.0);
    const double n_norm = n.norm();
    if (n_norm < 1e-12) n = Vec3(0.0, 0.0, 1.0);
    else n /= n_norm;

    Vec3 t1 = orthogonal_unit(n);
    Vec3 t2 = n.cross(t1);
    const double t2_norm = t2.norm();
    if (t2_norm < 1e-12) t2 = orthogonal_unit(t1);
    else t2 /= t2_norm;

    double hi = fallback_scale;
    if (i < (int)local_scales.size() && std::isfinite(local_scales[i]) &&
        local_scales[i] > 0.0) {
      hi = local_scales[i];
    }
    if (!(hi > 0.0) || !std::isfinite(hi)) hi = std::max(fallback_scale, 1e-8);

    TangentPerturbFrame frame;
    frame.site_id = i;
    frame.t1 = t1;
    frame.t2 = t2;
    frame.local_scale = hi;
    frames.push_back(frame);
  }
  return frames;
}

static inline std::vector<_Point3> perturb_sites_from_frames(
    const std::vector<_Point3> &base_sites,
    const std::vector<TangentPerturbFrame> &frames,
    double alpha,
    const BGAL::_ManifoldModel &model,
    std::mt19937 &rng) {
  std::vector<_Point3> perturbed = base_sites;
  if (!(alpha > 0.0) || !std::isfinite(alpha) || frames.empty()) {
    return perturbed;
  }

  std::normal_distribution<double> normal01(0.0, 1.0);
  std::vector<double> xi1(frames.size(), 0.0);
  std::vector<double> xi2(frames.size(), 0.0);
  for (int k = 0; k < (int)frames.size(); ++k) {
    xi1[k] = normal01(rng);
    xi2[k] = normal01(rng);
  }

  auto apply_one = [&](int k) {
    const auto &frame = frames[k];
    const int i = frame.site_id;
    if (i < 0 || i >= (int)base_sites.size()) return;

    const double sigma_i = alpha * frame.local_scale;
    if (!(sigma_i > 0.0) || !std::isfinite(sigma_i)) return;

    const Vec3 tangent_step =
        sigma_i * (xi1[k] * frame.t1 + xi2[k] * frame.t2);
    const Vec3 p = to_eigen(base_sites[i]) + tangent_step;
    perturbed[i] = project_to_surface(model, to_point(p));
  };

  if (omp_in_parallel()) {
    for (int k = 0; k < (int)frames.size(); ++k) apply_one(k);
  } else {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < (int)frames.size(); ++k) apply_one(k);
  }
  return perturbed;
}


struct EdgeKey {
  int a = -1;
  int b = -1;

  EdgeKey() = default;

  EdgeKey(int x, int y) {
    if (x < y) {
      a = x;
      b = y;
    } else {
      a = y;
      b = x;
    }
  }

  bool operator<(const EdgeKey &o) const {
    if (a != o.a) return a < o.a;
    return b < o.b;
  }
};

struct FaceKey {
  int a = -1;
  int b = -1;
  int c = -1;

  FaceKey() = default;

  FaceKey(int x, int y, int z) {
    std::array<int, 3> t{x, y, z};
    std::sort(t.begin(), t.end());
    a = t[0];
    b = t[1];
    c = t[2];
  }

  bool valid(int n) const {
    return a >= 0 && b >= 0 && c >= 0 &&
           a < n && b < n && c < n &&
           a != b && b != c && a != c;
  }

  Eigen::Vector3i vec() const {
    return Eigen::Vector3i(a, b, c);
  }

  bool operator<(const FaceKey &o) const {
    if (a != o.a) return a < o.a;
    if (b != o.b) return b < o.b;
    return c < o.c;
  }
};

static inline Vec3 safe_face_normal(
    const std::vector<_Point3> &sites,
    const Eigen::Vector3i &f) {
  const int a = f.x();
  const int b = f.y();
  const int c = f.z();

  const Vec3 p0 = to_eigen(sites[a]);
  const Vec3 p1 = to_eigen(sites[b]);
  const Vec3 p2 = to_eigen(sites[c]);

  Vec3 n = (p1 - p0).cross(p2 - p0);
  const double len = n.norm();
  if (!(len > 1e-30) || !std::isfinite(len)) {
    return Vec3::Zero();
  }
  return n / len;
}

static inline double face_double_area(
    const std::vector<_Point3> &sites,
    const Eigen::Vector3i &f) {
  const int a = f.x();
  const int b = f.y();
  const int c = f.z();

  const Vec3 p0 = to_eigen(sites[a]);
  const Vec3 p1 = to_eigen(sites[b]);
  const Vec3 p2 = to_eigen(sites[c]);

  const double da = (p1 - p0).cross(p2 - p0).norm();
  if (!std::isfinite(da)) return 0.0;
  return da;
}

static inline double mean_edge_length_squared(
    const std::vector<_Point3> &sites,
    const std::vector<Eigen::Vector3i> &faces) {
  double sum = 0.0;
  long long cnt = 0;

  #pragma omp parallel for reduction(+:sum,cnt)
  for (int i = 0; i < (int)faces.size(); ++i) {
    const auto &f = faces[i];
    const int a = f.x();
    const int b = f.y();
    const int c = f.z();

    if (a < 0 || b < 0 || c < 0 ||
        a >= (int)sites.size() ||
        b >= (int)sites.size() ||
        c >= (int)sites.size() ||
        a == b || b == c || a == c) {
      continue;
    }

    const Vec3 pa = to_eigen(sites[a]);
    const Vec3 pb = to_eigen(sites[b]);
    const Vec3 pc = to_eigen(sites[c]);

    sum += (pa - pb).squaredNorm();
    sum += (pb - pc).squaredNorm();
    sum += (pc - pa).squaredNorm();
    cnt += 3;
  }

  if (cnt <= 0) return 1.0;
  const double ret = sum / double(cnt);
  if (!(ret > 0.0) || !std::isfinite(ret)) return 1.0;
  return ret;
}

static inline double face_badness_score(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &site_normals,
    const Eigen::Vector3i &f,
    double mean_edge2) {
  const int a = f.x();
  const int b = f.y();
  const int c = f.z();

  if (a < 0 || b < 0 || c < 0 ||
      a >= (int)sites.size() ||
      b >= (int)sites.size() ||
      c >= (int)sites.size() ||
      a == b || b == c || a == c) {
    return std::numeric_limits<double>::infinity();
  }

  const Vec3 nf = safe_face_normal(sites, f);
  if (nf.squaredNorm() < 1e-24) {
    return std::numeric_limits<double>::infinity();
  }

  const double area2 = face_double_area(sites, f);
  const double area_score = area2 / (mean_edge2 + 1e-30);

  double normal_alignment_bad = 0.0;
  double normal_spread_bad = 0.0;

  if ((int)site_normals.size() == (int)sites.size()) {
    Vec3 n0 = site_normals[a];
    Vec3 n1 = site_normals[b];
    Vec3 n2 = site_normals[c];

    const double l0 = n0.norm();
    const double l1 = n1.norm();
    const double l2 = n2.norm();

    if (l0 > 1e-12 && l1 > 1e-12 && l2 > 1e-12) {
      n0 /= l0;
      n1 /= l1;
      n2 /= l2;

      const double align =
          (std::abs(nf.dot(n0)) +
           std::abs(nf.dot(n1)) +
           std::abs(nf.dot(n2))) / 3.0;

      normal_alignment_bad = 1.0 - std::min(1.0, std::max(0.0, align));

      const double d01 = std::abs(n0.dot(n1));
      const double d12 = std::abs(n1.dot(n2));
      const double d20 = std::abs(n2.dot(n0));
      const double min_consistency = std::min(d01, std::min(d12, d20));

      normal_spread_bad = 1.0 - std::min(1.0, std::max(0.0, min_consistency));
    }
  }

  // Prefer removing faces crossing sharp features, then unusually large faces.
  return 0.20 * area_score +
         2.50 * normal_alignment_bad +
         4.00 * normal_spread_bad;
}

static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_edges_raw(
    int num_sites,
    const std::vector<std::map<int, std::vector<std::pair<int, int>>>> &edges) {
  std::vector<std::vector<int>> adj(num_sites);

  const int edge_count = std::min(num_sites, (int)edges.size());
  for (int i = 0; i < edge_count; ++i) {
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

  #pragma omp parallel
  {
    std::vector<Eigen::Vector3i> local;

    #pragma omp for schedule(dynamic, 64)
    for (int u = 0; u < num_sites; ++u) {
      for (int v : adj[u]) {
        if (v <= u) continue;

        int i = 0;
        int j = 0;

        while (i < (int)adj[u].size() && j < (int)adj[v].size()) {
          if (adj[u][i] == adj[v][j]) {
            const int w = adj[u][i];
            if (w > v) {
              local.emplace_back(u, v, w);
            }
            ++i;
            ++j;
          } else if (adj[u][i] < adj[v][j]) {
            ++i;
          } else {
            ++j;
          }
        }
      }
    }

    #pragma omp critical
    tris.insert(tris.end(), local.begin(), local.end());
  }

  std::set<FaceKey> uniq;
  for (const auto &f : tris) {
    FaceKey key(f.x(), f.y(), f.z());
    if (key.valid(num_sites)) {
      uniq.insert(key);
    }
  }

  std::vector<Eigen::Vector3i> ret;
  ret.reserve(uniq.size());
  for (const auto &k : uniq) {
    ret.push_back(k.vec());
  }

  return ret;
}

static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_rvd_corners(
    int num_sites,
    const _Restricted_Tessellation3D &rvd) {
  const auto &cells = rvd.get_cells_();
  const int nv = rvd.number_vertices_();

  if (num_sites <= 0 || nv <= 0 || cells.empty()) {
    return {};
  }

  std::vector<std::vector<int>> vertex_to_sites(nv);

  for (int sid = 0; sid < (int)cells.size() && sid < num_sites; ++sid) {
    for (const auto &tri : cells[sid]) {
      const int vids[3] = {
          std::get<0>(tri),
          std::get<1>(tri),
          std::get<2>(tri)
      };

      for (int k = 0; k < 3; ++k) {
        const int vid = vids[k];
        if (vid >= 0 && vid < nv) {
          vertex_to_sites[vid].push_back(sid);
        }
      }
    }
  }

  std::set<FaceKey> face_keys;

  #pragma omp parallel
  {
    std::set<FaceKey> local_keys;

    #pragma omp for schedule(dynamic, 256)
    for (int vid = 0; vid < nv; ++vid) {
      auto s = vertex_to_sites[vid];

      if (s.empty()) continue;

      std::sort(s.begin(), s.end());
      s.erase(std::unique(s.begin(), s.end()), s.end());

      if (s.size() == 3) {
        FaceKey key(s[0], s[1], s[2]);
        if (key.valid(num_sites)) {
          local_keys.insert(key);
        }
      }

      // Do not emit all C(k,3) faces when s.size() > 3.
      // Those cases are usually sharp corners / degeneracies / multi-sheet junctions.
    }

    #pragma omp critical
    {
      face_keys.insert(local_keys.begin(), local_keys.end());
    }
  }

  std::vector<Eigen::Vector3i> ret;
  ret.reserve(face_keys.size());

  for (const auto &key : face_keys) {
    ret.push_back(key.vec());
  }

  return ret;
}

static inline std::vector<Eigen::Vector3i> filter_nonmanifold_faces_by_edge_valence(
    const std::vector<_Point3> &sites,
    const std::vector<Eigen::Vector3i> &faces_in,
    const std::vector<Vec3> &site_normals) {
  const int n = (int)sites.size();
  const int nf = (int)faces_in.size();

  if (n <= 0 || nf <= 0) return {};

  std::vector<char> removed(nf, 0);
  const double mean_edge2 = mean_edge_length_squared(sites, faces_in);

  for (int iter = 0; iter < nf; ++iter) {
    std::map<EdgeKey, std::vector<int>> edge_faces;

    for (int fi = 0; fi < nf; ++fi) {
      if (removed[fi]) continue;

      const auto &f = faces_in[fi];
      const int a = f.x();
      const int b = f.y();
      const int c = f.z();

      if (a < 0 || b < 0 || c < 0 ||
          a >= n || b >= n || c >= n ||
          a == b || b == c || a == c) {
        removed[fi] = 1;
        continue;
      }

      edge_faces[EdgeKey(a, b)].push_back(fi);
      edge_faces[EdgeKey(b, c)].push_back(fi);
      edge_faces[EdgeKey(c, a)].push_back(fi);
    }

    std::vector<int> face_conflict_count(nf, 0);
    int num_bad_edges = 0;

    for (const auto &kv : edge_faces) {
      const auto &inc = kv.second;
      if ((int)inc.size() > 2) {
        ++num_bad_edges;
        for (int fi : inc) {
          if (!removed[fi]) {
            face_conflict_count[fi]++;
          }
        }
      }
    }

    if (num_bad_edges == 0) {
      break;
    }

    int best_remove = -1;
    double best_score = -std::numeric_limits<double>::infinity();

    for (int fi = 0; fi < nf; ++fi) {
      if (removed[fi]) continue;
      if (face_conflict_count[fi] <= 0) continue;

      const double badness =
          face_badness_score(sites, site_normals, faces_in[fi], mean_edge2);

      const double score =
          1000.0 * double(face_conflict_count[fi]) + badness;

      if (score > best_score) {
        best_score = score;
        best_remove = fi;
      }
    }

    if (best_remove < 0) {
      break;
    }

    removed[best_remove] = 1;
  }

  std::vector<Eigen::Vector3i> faces_out;
  faces_out.reserve(faces_in.size());

  for (int fi = 0; fi < nf; ++fi) {
    if (!removed[fi]) {
      faces_out.push_back(faces_in[fi]);
    }
  }

  return faces_out;
}

static inline std::vector<Eigen::Vector3i> build_rdt_faces_robust(
    int num_sites,
    const std::vector<_Point3> &sites,
    const _Restricted_Tessellation3D &rvd,
    const BGAL::_ManifoldModel &model) {
  std::vector<Eigen::Vector3i> corner_faces =
      build_rdt_faces_from_rvd_corners(num_sites, rvd);

  std::vector<Eigen::Vector3i> clique_faces =
      build_rdt_faces_from_edges_raw(num_sites, rvd.get_edges_());

  std::vector<Eigen::Vector3i> candidates;

  if (!corner_faces.empty() &&
      corner_faces.size() >= std::max<std::size_t>(16, clique_faces.size() / 4)) {
    candidates = std::move(corner_faces);
  } else {
    candidates = std::move(clique_faces);
  }

  std::vector<Vec3> site_normals;
  update_surface_normals(model, sites, site_normals);

  return filter_nonmanifold_faces_by_edge_valence(sites, candidates, site_normals);
}

namespace {

static inline bool valid_radius(double r) {
  return std::isfinite(r) && r >= 0.0;
}

static inline bool pairwise_sphere_overlap(const _Point3 &a, double ra,
                                           const _Point3 &b, double rb) {
  const double reach = ra + rb;
  if (!(reach >= 0.0) || !std::isfinite(reach)) return false;
  const double tol = 1e-12 * (1.0 + reach * reach);
  return (a - b).sqlength_() <= reach * reach + tol;
}

static inline void build_quads_from_search_faces(
    const std::vector<Eigen::Vector3i> &faces,
    const std::vector<_Point3> &sites,
    const std::vector<Sphere::Sphere> &spheres,
    std::vector<std::array<int, 4>> &quads) {

  quads.clear();
  const int n = (int)sites.size();
  if (n <= 0 || faces.empty() || (int)spheres.size() != n) return;

  std::vector<double> radii(n, -1.0);
  double global_rmax = 0.0;
  #pragma omp parallel for reduction(max:global_rmax)
  for (int i = 0; i < n; ++i) {
    const double ri = (double)spheres[i].r;
    if (valid_radius(ri)) {
      radii[i] = ri;
      global_rmax = std::max(global_rmax, ri);
    }
  }
  if (!(global_rmax >= 0.0) || !std::isfinite(global_rmax)) return;

  BGAL::_KDTree tree(sites);
  std::vector<std::vector<int>> overlap_neighbors(n);

  #pragma omp parallel for schedule(dynamic, 32)
  for (int i = 0; i < n; ++i) {
    const double ri = radii[i];
    if (!valid_radius(ri)) continue;

    const double query_radius = ri + global_rmax;
    if (!valid_radius(query_radius)) continue;

    std::vector<int> hits = tree.rsearch_(sites[i], query_radius);
    auto &nbrs = overlap_neighbors[i];
    nbrs.reserve(hits.size());
    for (int j : hits) {
      if (j == i || j < 0 || j >= n) continue;
      const double rj = radii[j];
      if (!valid_radius(rj)) continue;
      if (pairwise_sphere_overlap(sites[i], ri, sites[j], rj)) {
        nbrs.push_back(j);
      }
    }
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  const int max_threads = std::max(1, omp_get_max_threads());
  std::vector<std::vector<std::array<int, 4>>> thread_bins(max_threads);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto &local_quads = thread_bins[tid];
    local_quads.reserve(std::min<std::size_t>(
        std::max<std::size_t>(1, faces.size() / std::max(1, max_threads)) * 4ull,
        1ull << 18));

    #pragma omp for schedule(dynamic, 64)
    for (int f_idx = 0; f_idx < (int)faces.size(); ++f_idx) {
      const auto &f = faces[f_idx];
      const int i = f.x(), j = f.y(), k = f.z();
      if (i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n) continue;
      if (!valid_radius(radii[i]) || !valid_radius(radii[j]) || !valid_radius(radii[k])) {
        continue;
      }

      const std::vector<int> *lists[3] = {
          &overlap_neighbors[i], &overlap_neighbors[j], &overlap_neighbors[k]};
      int ref = 0;
      if (lists[1]->size() < lists[ref]->size()) ref = 1;
      if (lists[2]->size() < lists[ref]->size()) ref = 2;
      const int other_a = (ref + 1) % 3;
      const int other_b = (ref + 2) % 3;

      for (int l : *lists[ref]) {
        if (l == i || l == j || l == k) continue;
        if (!std::binary_search(lists[other_a]->begin(), lists[other_a]->end(), l)) continue;
        if (!std::binary_search(lists[other_b]->begin(), lists[other_b]->end(), l)) continue;

        std::array<int, 4> q{i, j, k, l};
        std::sort(q.begin(), q.end());
        local_quads.push_back(q);
      }
    }
  }

  std::size_t total = 0;
  for (const auto &bin : thread_bins) total += bin.size();
  quads.reserve(total);
  for (auto &bin : thread_bins) {
    quads.insert(quads.end(), bin.begin(), bin.end());
  }
  std::sort(quads.begin(), quads.end());
  quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
}

} // namespace

static inline double quad_margin_value(
    const std::vector<_Point3> &sites,
    const std::vector<Vec3> &frozen_poles,
    const std::array<int, 4> &q) {
  Vec3 p_bar = Vec3::Zero();
  for (int k = 0; k < 4; ++k) p_bar += to_eigen(sites[q[k]]);
  p_bar /= 4.0;

  double g_val = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int sid = q[k];
    const Vec3 pi = to_eigen(sites[sid]);
    const double ri2 = padded_radius_squared(pi, frozen_poles[sid]);
    g_val += (p_bar - pi).squaredNorm() - ri2;
  }
  return g_val;
}

static inline int write_active_quads_obj(
    const std::vector<_Point3> &sites,
    const std::vector<std::array<int, 4>> &quads,
    const std::vector<Vec3> &frozen_poles,
    double eps,
    int num,
    const std::string &outpath,
    const std::string &modelname,
    const std::string &tag) {
  namespace fs = std::filesystem;
  fs::create_directories(outpath);

  const std::string filepath =
      outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
      "_" + tag + "ActiveQuads.obj";
  std::ofstream out(filepath, std::ios::out | std::ios::trunc);
  if (!out) return 0;

  out << std::setprecision(17);
  out << "# Active QuadCover quads. Each active quad is written as four\n";
  out << "# duplicated vertices plus all six pairwise line segments.\n";
  out << "# eps " << eps << "\n";
  out << "g " << tag << "ActiveQuads\n";

  int active_count = 0;
  int next_vertex_index = 1;
  const double eps_safe = (std::isfinite(eps) && eps >= 0.0) ? eps : 0.0;

  for (const auto &q : quads) {
    bool valid = true;
    for (int k = 0; k < 4; ++k) {
      valid = valid && q[k] >= 0 && q[k] < (int)sites.size() &&
              q[k] < (int)frozen_poles.size();
    }
    if (!valid) continue;

    const double g_val = quad_margin_value(sites, frozen_poles, q);
    const double violation = -(g_val + eps_safe);
    if (!(violation > 0.0)) continue;

    ++active_count;
    out << "\n# active_quad " << active_count
        << " ids " << q[0] << " " << q[1] << " " << q[2] << " " << q[3]
        << " g " << g_val
        << " violation " << violation << "\n";

    for (int k = 0; k < 4; ++k) {
      out << "v " << sites[q[k]] << "\n";
    }

    const int a = next_vertex_index;
    out << "l " << a << " " << a + 1 << "\n";
    out << "l " << a << " " << a + 2 << "\n";
    out << "l " << a << " " << a + 3 << "\n";
    out << "l " << a + 1 << " " << a + 2 << "\n";
    out << "l " << a + 1 << " " << a + 3 << "\n";
    out << "l " << a + 2 << " " << a + 3 << "\n";
    next_vertex_index += 4;
  }

  return active_count;
}

static inline void compute_cell_geom_and_spheres(
    const std::vector<_Point3> &sites,
    const _Restricted_Tessellation3D &rvd,
    std::vector<CellGeom> &cells,
    std::vector<Sphere::Sphere> &spheres) {

  const auto &cell_tris = rvd.get_cells_();
  const int n_sites = (int)sites.size();
  const int n_vertices = rvd.number_vertices_();

  std::vector<_Point3> vertex_cache(std::max(0, n_vertices));
  #pragma omp parallel for schedule(static)
  for (int vid = 0; vid < n_vertices; ++vid) {
    vertex_cache[vid] = rvd.vertex_(vid);
  }

  cells.assign(n_sites, CellGeom());
  spheres.assign(n_sites, Sphere::Sphere());

  const int max_threads = std::max(1, omp_get_max_threads());
  std::vector<std::vector<unsigned int>> thread_marks(
      max_threads, std::vector<unsigned int>(std::max(0, n_vertices), 0u));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto &marks = thread_marks[tid];
    unsigned int stamp = 1u;
    std::vector<int> uniq;

    #pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < (int)cell_tris.size(); ++i) {
      if (++stamp == 0u) {
        std::fill(marks.begin(), marks.end(), 0u);
        stamp = 1u;
      }

      uniq.clear();
      uniq.reserve(cell_tris[i].size() * 3);
      for (const auto &tri : cell_tris[i]) {
        const int vids[3] = {std::get<0>(tri), std::get<1>(tri), std::get<2>(tri)};
        for (int t = 0; t < 3; ++t) {
          const int vid = vids[t];
          if (vid < 0 || vid >= n_vertices) continue;
          if (marks[vid] != stamp) {
            marks[vid] = stamp;
            uniq.push_back(vid);
          }
        }
      }

      auto &cell = cells[i];
      cell.vertex_ids.assign(uniq.begin(), uniq.end());
      cell.vertex_pos.clear();

      double best_d2 = 0.0;
      _Point3 best_v = sites[i];
      for (int vid : uniq) {
        const _Point3 &pv = vertex_cache[vid];
        const double d2 = (pv - sites[i]).sqlength_();
        if (d2 > best_d2) {
          best_d2 = d2;
          best_v = pv;
        }
      }

      cell.r2 = best_d2;
      cell.far_v = best_v;

      spheres[i].c = sites[i];
      spheres[i].r = std::sqrt(best_d2) + 1e-6;
      spheres[i].max_point = best_v;
    }
  }
}

static inline void output_mesh(const std::vector<_Point3> &sites,
                               const _Restricted_Tessellation3D &rvd,
                               const BGAL::_ManifoldModel &model,
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
  out << std::setprecision(17);
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
  outP << std::setprecision(17);
  for (const auto &s : sites) outP << s << "\n";
  outP.close();

  std::string remesh_path =
      outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
      "_Remesh.obj";
  std::ofstream out_remesh(remesh_path);
  if (out_remesh) out_remesh << std::setprecision(17);
  std::ofstream out_remesh_iter;
  if (step > 0) {
    std::string remesh_iter_path =
        outpath + "/QuadCover_" + std::to_string(num) + "_" + modelname +
        "_Iter" + std::to_string(step) + "_Remesh.obj";
    out_remesh_iter.open(remesh_iter_path, std::ios::out | std::ios::trunc);
    if (out_remesh_iter) out_remesh_iter << std::setprecision(17);
  }

  if (out_remesh) {
    const auto rdt_faces =
        build_rdt_faces_robust((int)sites.size(), sites, rvd, model);

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
  
  // IO涓茶锛屼絾鎴戜滑鍙互棰勫厛骞惰璁＄畻琛ㄩ潰娉曞悜
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
  const double overall_start = omp_get_wtime();
  const double max_wall_time_seconds = _para.max_wall_time_seconds;
  bool hit_time_limit = false;
  auto time_limit_exceeded = [&]() -> bool {
    return max_wall_time_seconds > 0.0 &&
           wall_seconds_since(overall_start) >= max_wall_time_seconds;
  };
  _history.clear();
  _sites = init_sites;
  const std::vector<_Point3> original_init_sites = init_sites;
  bool warm_start_used = false;
  bool warm_start_fell_back = false;
  if (_sites.empty()) {
    throw std::runtime_error("[QuadCoverLike] empty init_sites.");
  }

  const int n = (int)_sites.size();
  const std::string output_dir = outpath.empty()
                                     ? (std::filesystem::current_path() / "data" /
                                        "QuadCover")
                                           .string()
                                     : outpath;

  if (_para.use_cwf_warm_start && _para.cwf_max_iterations > 0) {
    std::function<double(_Point3 &)> rho = [](_Point3 &p) {
      (void)p;
      return 1.0;
    };
    BGAL::_LBFGS::_Parameter cvt_para;
    cvt_para.is_show = _para.show_cwf_progress;
    cvt_para.epsilon = 1e-30;
    cvt_para.max_linearsearch = 20;
    cvt_para.max_iteration = _para.cwf_max_iterations;
    if (max_wall_time_seconds > 0.0) {
      const double remaining_seconds =
          std::max(0.0, max_wall_time_seconds - wall_seconds_since(overall_start));
      cvt_para.max_time = std::max(1.0, remaining_seconds * 1000.0);
    }

    if (_para.is_show) {
      std::cout << "[QuadCoverLike] CWF warm start begin"
                << " | iters=" << cvt_para.max_iteration
                << " | sites=" << _sites.size() << std::endl;
    }

    BGAL::_CVT3D cvt(_model, rho, cvt_para);
    cvt.set_use_feature_density_boost(false);
    cvt.set_outpath(output_dir + "/");
    cvt.calculate_(_sites, model_name, false);
    if (time_limit_exceeded()) {
      hit_time_limit = true;
    }
    warm_start_used = true;
    const std::vector<_Point3> &warm_sites = cvt.get_sites();
    if (warm_sites.size() == _sites.size() && sites_are_finite(warm_sites)) {
      _sites = warm_sites;
    } else {
      warm_start_fell_back = true;
      _sites = original_init_sites;
      if (_para.is_show) {
        std::cout << "[QuadCoverLike] CWF warm start fallback"
                  << " | reason=nonfinite_sites" << std::endl;
      }
    }

    if (_para.is_show) {
      std::cout << "[QuadCoverLike] CWF warm start end"
                << " | sites=" << _sites.size() << std::endl;
    }
	  }

  const double quadcover_start = omp_get_wtime();

  double total_rebuild_non_rvd_time = 0.0;
  double total_rvd_time = 0.0;
  double total_adam_time = 0.0;
  double total_output_time = 0.0;
  double total_refresh_eval_time = 0.0;
  double total_perturb_wall_time = 0.0;
  double total_perturb_rebuild_task_time = 0.0;
  double total_perturb_rvd_task_time = 0.0;

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

  const double hinge_lambda_floor = std::max(_para.hinge_lambda, 1e-12);
  double penalty_k = hinge_lambda_floor;
  double fixed_hinge_lambda = penalty_k;
  double qem_weight = 1.0;
  const bool use_qem_energy = _para.use_qem_energy;
  const bool use_hinge_loss_energy = _para.use_hinge_loss_energy;
  const bool use_weight_schedule = _para.use_weight_schedule;
  const bool use_tangential_perturb = _para.use_tangential_perturb;

  auto effective_qem_weight = [&]() -> double {
    return use_qem_energy ? qem_weight : 0.0;
  };

  auto effective_hinge_lambda = [&]() -> double {
    return use_hinge_loss_energy ? fixed_hinge_lambda : 0.0;
  };

  if (!use_weight_schedule) {
    penalty_k = 1.0;
    fixed_hinge_lambda = 1.0;
    qem_weight = 1.0;
  }

  auto rebuild_state = [&](const std::vector<_Point3> &sites_in,
                           bool need_grad,
                           _Restricted_Tessellation3D &rvd,
                           double &rebuild_non_rvd_acc,
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
    if (!std::isfinite(state.avg_r) || state.avg_r < 0.0) {
      state.avg_r = 0.0;
    }
    state.eps = 1e-4 * state.avg_r * state.avg_r;
    if (!std::isfinite(state.eps) || state.eps < 0.0) {
      state.eps = 0.0;
    }

    if (need_grad) {
      update_surface_normals(_model, state.sites, state.surface_normals);
    } else {
      state.surface_normals.clear();
    }
    state.rdt_faces = build_rdt_faces_robust(n, state.sites, rvd, _model);
    build_quads_from_search_faces(state.rdt_faces, state.sites, state.spheres,
                                  state.search_quads);

    state.eval = evaluate_qem_hinge_objective(
        state.sites, rvd, state.frozen_poles, state.surface_normals,
        state.search_quads, state.eps, effective_qem_weight(),
        effective_hinge_lambda(),
        need_grad);
    rebuild_non_rvd_acc +=
        std::max(0.0, wall_seconds_since(rebuild_start) - rvd_elapsed);
    return state;
  };

  auto rebuilt_energy = [&](const RebuiltState &state) -> double {
    return state.eval.total_loss;
  };

  auto refresh_state_with_current_rvd =
      [&](RebuiltState &state, const _Restricted_Tessellation3D &rvd,
          bool need_grad) {
        if (need_grad) {
          update_surface_normals(_model, state.sites, state.surface_normals);
        } else {
          state.surface_normals.clear();
        }
        state.eval = evaluate_qem_hinge_objective(
            state.sites, rvd, state.frozen_poles, state.surface_normals,
            state.search_quads, state.eps, effective_qem_weight(),
            effective_hinge_lambda(),
            need_grad);
      };

  auto update_fixed_hinge_lambda = [&]() {
    fixed_hinge_lambda = penalty_k;
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
    if (std::abs(cand.eval.hinge_loss - ref.eval.hinge_loss) > 1e-18) {
      return cand.eval.hinge_loss < ref.eval.hinge_loss;
    }
    if (std::abs(cand.eval.min_g - ref.eval.min_g) > 1e-15) {
      return cand.eval.min_g > ref.eval.min_g;
    }
    if (std::abs(cand_E - ref_E) > 1e-15) {
      return cand_E < ref_E;
    }
    return false;
  };

  auto is_better_perturb_candidate = [&](const RebuiltState &cand,
                                         const RebuiltState &ref,
                                         double cand_E,
                                         double ref_E) -> bool {
    if (std::abs(cand.eval.hinge_loss - ref.eval.hinge_loss) > 1e-18) {
      return cand.eval.hinge_loss < ref.eval.hinge_loss;
    }
    if (cand.eval.active_quads != ref.eval.active_quads) {
      return cand.eval.active_quads < ref.eval.active_quads;
    }
    if (std::abs(cand.eval.min_g - ref.eval.min_g) > 1e-15) {
      return cand.eval.min_g > ref.eval.min_g;
    }
    if (std::abs(cand_E - ref_E) > 1e-15) {
      return cand_E < ref_E;
    }
    return false;
  };

  update_fixed_hinge_lambda();
  RebuiltState current_state = rebuild_state(
      _sites, true, _RVD, total_rebuild_non_rvd_time, total_rvd_time);
  if (warm_start_used && !warm_start_fell_back &&
      (current_state.search_quads.empty() ||
       !std::isfinite(current_state.eval.total_loss) ||
       !std::isfinite(current_state.eval.qem_loss) ||
       !sites_are_finite(current_state.sites))) {
    if (_para.is_show) {
      std::cout << "[QuadCoverLike] CWF warm start fallback"
                << " | reason=invalid_rebuild_state"
                << " | quads=" << current_state.search_quads.size()
                << std::endl;
    }
    _sites = original_init_sites;
    current_state = rebuild_state(
        _sites, true, _RVD, total_rebuild_non_rvd_time, total_rvd_time);
    warm_start_fell_back = true;
  }
  if (use_weight_schedule) {
    qem_weight = penalty_k * current_state.eval.hinge_loss /
                 (current_state.eval.qem_loss + 1e-12);
    if (!std::isfinite(qem_weight)) qem_weight = 1.0;
    qem_weight = std::max(qem_weight, 0.0);
    if (!use_qem_energy) qem_weight = 0.0;
  } else {
    qem_weight = 1.0;
    penalty_k = 1.0;
    fixed_hinge_lambda = 1.0;
  }
  {
    const double refresh_start = omp_get_wtime();
    refresh_state_with_current_rvd(current_state, _RVD, true);
    total_refresh_eval_time += wall_seconds_since(refresh_start);
  }
  double current_rebuilt_E = rebuilt_energy(current_state);
  double last_penalty_update_hinge = current_state.eval.hinge_loss;

  std::vector<_Point3> best_sites = current_state.sites;
  double best_rebuilt_E = current_rebuilt_E;
  int best_active_quads = current_state.eval.active_quads;
  double best_hinge_loss = current_state.eval.hinge_loss;
  double best_min_g = current_state.eval.min_g;

  const double initial_active_quads_output_start = omp_get_wtime();
  const int initial_active_quads_written = write_active_quads_obj(
      current_state.sites, current_state.search_quads, current_state.frozen_poles,
      current_state.eps, n, output_dir, model_name, "Initial");
  total_output_time += wall_seconds_since(initial_active_quads_output_start);
  if (_para.is_show) {
    std::cout << "[QuadCoverLike] initial active quads OBJ"
              << " | active_quads=" << initial_active_quads_written
              << " | total_quads=" << current_state.search_quads.size()
              << " | path=" << output_dir << "/QuadCover_" << n << "_"
              << model_name << "_InitialActiveQuads.obj"
              << std::endl;
  }

  auto update_best_state = [&](const RebuiltState &state, double state_E) {
    const bool better =
        (state.eval.active_quads < best_active_quads) ||
        (state.eval.active_quads == best_active_quads &&
         state.eval.hinge_loss < best_hinge_loss - 1e-18) ||
        (state.eval.active_quads == best_active_quads &&
         std::abs(state.eval.hinge_loss - best_hinge_loss) <= 1e-18 &&
         state.eval.min_g > best_min_g + 1e-15) ||
        (state.eval.active_quads == best_active_quads &&
         std::abs(state.eval.hinge_loss - best_hinge_loss) <= 1e-18 &&
         std::abs(state.eval.min_g - best_min_g) <= 1e-15 &&
         state_E < best_rebuilt_E - 1e-15);
    if (better) {
      best_sites = state.sites;
      best_rebuilt_E = state_E;
      best_active_quads = state.eval.active_quads;
      best_hinge_loss = state.eval.hinge_loss;
      best_min_g = state.eval.min_g;
    }
  };

  const double adam_lr_init = _para.adam_learning_rate;
  const double adam_beta1 = 0.9;
  const double adam_beta2 = 0.999;
  const double adam_eps = 1e-8;
  const double adam_lr_min = 3e-7;
  const double adam_lr_reduce_factor = 0.6;
  const int adam_lr_plateau_patience = 2;
  const int adam_lr_plateau_cooldown = 1;
  const double adam_lr_plateau_abs_tol = 1e-18;
  const int adam_lr_recent_best_window = 5;

  //
  const int penalty_k_update_interval = 5;
  const double penalty_k_min = hinge_lambda_floor;
  const double penalty_k_max = 1e4;
  const double penalty_k_grow_fast = 3.0;
  const double penalty_k_grow_slow = 1.5;
  const double penalty_k_shrink_factor = 1.1;
  const double hinge_progress_fast_tol = 0.20;
  const double hinge_progress_slow_tol = 0.05;
  const double qem_weight_decay = 0.95;

  const double hinge_stop_tol = std::max(_para.accept_eps, 1e-12);
  const int stagnation_patience = 4;
  const int stagnation_hard_patience = 2 * stagnation_patience;
  const double perturb_hinge_rel_tol = 1e-3;
  const double perturb_hinge_abs_tol = 1e-12;
  const int perturb_window_iters = 15;
  const double perturb_window_progress_tol = 0.03;
  const double perturb_single_step_progress_tol = 0.03;

  // Conservative perturbation: small displacement, broader trial pool.
  const double stagnation_perturb_ratio = 0.010;
  const int perturb_num_trials = 16;
  const double perturb_lr_max = 5e-5;
  const double stagnation_lr_trigger = 0.25 * adam_lr_init;

  // perturb 鎺ュ彈鍑嗗垯锛氬厑璁稿彈鎺ч潪鍗曡皟锛屼絾涓嶅厑璁告槑鏄惧姡鍖?
  const double perturb_accept_relax_E = 0.01;
  const double perturb_accept_relax_hinge = 0.05;
  const double perturb_accept_min_margin_gain = 1e-8;
  const double perturb_accept_min_hinge_gain = 0.05;

  // perturb cooldown after acceptance
  const int post_perturb_lock_iters = 10;
  const int late_post_perturb_lock_iters = 2;
  const int quick_perturb_active_quads_thresh = 4;
  const double quick_perturb_hinge_thresh = 1e-8;
  const int quick_perturb_stagnation_patience = 2;
  const double post_perturb_lr_scale = 1.2;

  // 鍚庢湡绛栫暐锛氶檺鍒舵闀裤€佸叧闂?late perturb銆佸姞蹇?hinge 涓诲
  const int late_aggressive_penalty_active_thresh = 50;

  // 瀵煎嚭棰戠巼锛氶檷浣?I/O 鍜岄澶栫殑 RVD rebuild
  const int export_interval = std::max(1, _para.export_interval);

  const int log_value_precision = 6;
  const int log_time_precision = 3;

  torch::Device optim_device(torch::kCPU);
  std::mt19937 perturb_rng(123456789u);

  torch::Tensor site_param =
      sites_to_tensor(current_state.sites, optim_device).clone().detach();
  site_param.set_requires_grad(true);

  torch::optim::Adam optimizer(
      std::vector<torch::Tensor>{site_param},
      torch::optim::AdamOptions(adam_lr_init)
          .betas(std::make_tuple(adam_beta1, adam_beta2))
          .eps(adam_eps));

  std::deque<double> recent_lr_metrics;
  std::deque<double> recent_hinge_losses;
  int lr_bad_epochs = 0;
  int lr_cooldown_counter = 0;

  auto push_recent_hinge_loss = [&](double hinge_loss) {
    recent_hinge_losses.push_back(hinge_loss);
    while ((int)recent_hinge_losses.size() > perturb_window_iters + 1) {
      recent_hinge_losses.pop_front();
    }
  };

  auto reseed_recent_lr_scheduler = [&](double seed_metric) {
    recent_lr_metrics.clear();
    recent_lr_metrics.push_back(seed_metric);
  };

  auto reset_recent_lr_scheduler = [&](double seed_metric) {
    reseed_recent_lr_scheduler(seed_metric);
    lr_bad_epochs = 0;
    lr_cooldown_counter = 0;
  };

  auto step_recent_lr_scheduler = [&](double metric) -> bool {
    bool improved_for_scheduler = true;
    if (!recent_lr_metrics.empty()) {
      const double recent_best =
          *std::min_element(recent_lr_metrics.begin(), recent_lr_metrics.end());
      improved_for_scheduler = metric < recent_best - adam_lr_plateau_abs_tol;
    }

    if (lr_cooldown_counter > 0) {
      --lr_cooldown_counter;
      lr_bad_epochs = 0;
    }

    if (improved_for_scheduler) {
      lr_bad_epochs = 0;
    } else if (lr_cooldown_counter == 0) {
      ++lr_bad_epochs;
    }

    recent_lr_metrics.push_back(metric);
    while ((int)recent_lr_metrics.size() > adam_lr_recent_best_window) {
      recent_lr_metrics.pop_front();
    }

    if (lr_bad_epochs > adam_lr_plateau_patience) {
      const double old_lr = get_adam_lr(optimizer);
      const double new_lr =
          std::max(adam_lr_min, old_lr * adam_lr_reduce_factor);
      if (new_lr < old_lr - 1e-18) {
        set_adam_lr(optimizer, new_lr);
        lr_cooldown_counter = adam_lr_plateau_cooldown;
        lr_bad_epochs = 0;
        return true;
      }
      lr_bad_epochs = 0;
    }
    return false;
  };

  reset_recent_lr_scheduler(current_state.eval.hinge_loss);
  push_recent_hinge_loss(current_state.eval.hinge_loss);
  double last_single_step_hinge_progress = 1.0;
  double last_window_hinge_progress = 1.0;
  bool last_hinge_window_ready = false;

  int hinge_stagnation_streak = 0;
  int post_perturb_lock_iters_remaining = 0;
  bool stop_requested = false;
  bool max_iter_notice_printed = false;

  auto has_strict_zero_cover_energy = [&](const RebuiltState &state) -> bool {
    return state.eval.hinge_loss == 0.0 && state.eval.active_quads == 0 &&
           state.eval.min_g >= -hinge_stop_tol;
  };

  auto accept_perturb_candidate = [&](const RebuiltState &cand,
                                      double cand_E,
                                      const RebuiltState &ref,
                                      double ref_E) -> bool {
    if (is_better_perturb_candidate(cand, ref, cand_E, ref_E)) {
      return true;
    }

    const bool hinge_gain =
        cand.eval.hinge_loss <
        ref.eval.hinge_loss * (1.0 - perturb_accept_min_hinge_gain);
    const bool active_gain = cand.eval.active_quads < ref.eval.active_quads;
    const bool margin_gain =
        cand.eval.min_g > ref.eval.min_g + perturb_accept_min_margin_gain;

    if (!(hinge_gain || active_gain || margin_gain)) {
      return false;
    }

    const double ref_E_safe = std::max(ref_E, 1e-18);
    const double ref_hinge_safe = std::max(ref.eval.hinge_loss, 1e-18);

    const bool energy_not_too_bad =
        cand_E <= ref_E_safe * (1.0 + perturb_accept_relax_E);
    const bool hinge_not_too_bad =
        cand.eval.hinge_loss <= ref_hinge_safe * (1.0 + perturb_accept_relax_hinge);

    if (ref.eval.active_quads <= 64) {
      if (!(cand.eval.active_quads < ref.eval.active_quads ||
            cand.eval.hinge_loss <= 0.9 * ref.eval.hinge_loss)) {
        return false;
      }
    }

    return energy_not_too_bad && hinge_not_too_bad;
  };

  for (int outer = 0;; ++outer) {
    if (time_limit_exceeded()) {
      hit_time_limit = true;
      break;
    }
    if (use_hinge_loss_energy && has_strict_zero_cover_energy(current_state)) {
      if (_para.is_show) {
        std::cout << "[QuadCoverLike][Adam] STOP (hinge zero)"
                  << std::scientific << std::setprecision(log_value_precision)
                  << " | hingeRaw=" << current_state.eval.hinge_loss
                  << " | hingePen=" << current_state.eval.weighted_hinge_loss
                  << " | lambda=" << current_state.eval.hinge_lambda
                  << " | num_quads=" << current_state.search_quads.size()
                  << " | active_quads=" << current_state.eval.active_quads
                  << " | min_g=" << current_state.eval.min_g
                  << std::endl;
      }
      stop_requested = true;
      break;
    }
    if (outer >= _para.max_outer_iterations) {
      if (!use_hinge_loss_energy) {
        break;
      }
      if (!max_iter_notice_printed && _para.is_show) {
        std::cout << "[QuadCoverLike][Adam] max_outer_iterations reached"
                  << " | continuing until hingeRaw==0 and active_quads==0"
                  << " | max_outer_iterations=" << _para.max_outer_iterations
                  << " | hingeRaw=" << std::scientific
                  << std::setprecision(log_value_precision)
                  << current_state.eval.hinge_loss
                  << " | active_quads=" << current_state.eval.active_quads
                  << std::endl;
      }
      max_iter_notice_printed = true;
    }

    const double iter_start = omp_get_wtime();
    double iter_rebuild_non_rvd_time = 0.0;
    double iter_rvd_time = 0.0;
    double iter_adam_time = 0.0;
    double iter_output_time = 0.0;
    double iter_refresh_eval_time = 0.0;
    double iter_perturb_wall_time = 0.0;
    double iter_perturb_rebuild_task_time = 0.0;
    double iter_perturb_rvd_task_time = 0.0;

    _sites = current_state.sites;
    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    const bool do_export =
        (_para.export_initial_state && outer == 0) ||
        (_para.export_each_iteration && (outer % export_interval == 0));
    if (do_export) {
      const double output_rvd_start = omp_get_wtime();
      _RVD.calculate_(_sites);
      {
        const double elapsed = wall_seconds_since(output_rvd_start);
        total_rvd_time += elapsed;
        iter_rvd_time += elapsed;
      }
      const double output_start = omp_get_wtime();
      output_mesh(_sites, _RVD, _model, n, output_dir, model_name, outer);
      write_spheres_csv(output_dir, n, model_name, outer, current_state.spheres,
                        _model);
      {
        const double elapsed = wall_seconds_since(output_start);
        total_output_time += elapsed;
        iter_output_time += elapsed;
      }
    }

    const double lr_before_step = get_adam_lr(optimizer);
    const bool allow_perturb_by_local_stagnation =
        (lr_before_step <= perturb_lr_max) &&
        (hinge_stagnation_streak >= stagnation_patience) &&
        (lr_before_step <= stagnation_lr_trigger ||
         hinge_stagnation_streak >= stagnation_hard_patience);
    const bool allow_perturb_by_window_stagnation =
        last_hinge_window_ready &&
        (last_window_hinge_progress <= perturb_window_progress_tol) &&
        (last_single_step_hinge_progress < perturb_single_step_progress_tol);
    const bool allow_perturb_by_quick_cleanup =
        (current_state.eval.active_quads > 0) &&
        (current_state.eval.active_quads <= quick_perturb_active_quads_thresh) &&
        (current_state.eval.hinge_loss <= quick_perturb_hinge_thresh) &&
        (hinge_stagnation_streak >= quick_perturb_stagnation_patience ||
         last_single_step_hinge_progress < perturb_single_step_progress_tol);
    const bool allow_perturb =
        use_tangential_perturb &&
        (post_perturb_lock_iters_remaining == 0) &&
        (allow_perturb_by_local_stagnation ||
         allow_perturb_by_window_stagnation ||
         allow_perturb_by_quick_cleanup);
    const double perturb_window_progress_for_log = last_window_hinge_progress;
    const double perturb_step_progress_for_log = last_single_step_hinge_progress;
    const char *perturb_trigger =
        allow_perturb_by_quick_cleanup
            ? "quick-cleanup"
            : allow_perturb_by_window_stagnation
            ? (allow_perturb_by_local_stagnation ? "local+window15"
                                                 : "window15")
            : "local";

    if (allow_perturb) {
      const double perturb_wall_start = omp_get_wtime();
      const RebuiltState pre_perturb_state = current_state;
      const double pre_perturb_E = current_rebuilt_E;

      std::vector<char> perturb_mask = build_active_site_mask(
          current_state.sites, current_state.frozen_poles,
          current_state.search_quads, current_state.eps);
      std::vector<char> perturb_opt_mask = expand_active_mask_one_ring(
          perturb_mask, current_state.rdt_faces, n, 1);
      if (!has_any_active(perturb_mask)) perturb_mask = perturb_opt_mask;
      if (!has_any_active(perturb_mask)) perturb_mask = make_all_active_mask(n);

      const double perturb_alpha = stagnation_perturb_ratio;
      const double perturb_fallback_scale = std::max(current_state.avg_r, 1e-8);

      struct PerturbEval {
        RebuiltState state;
        double energy = std::numeric_limits<double>::infinity();
        double rebuild_non_rvd_time = 0.0;
        double rvd_time = 0.0;
        bool valid = false;

        explicit PerturbEval(int n_sites = 0) : state(n_sites) {}
      };

      auto evaluate_perturb_sites =
          [&](const std::vector<_Point3> &sites_in) -> PerturbEval {
        PerturbEval eval_result;
        _Restricted_Tessellation3D local_rvd(_model);
        eval_result.state =
            rebuild_state(sites_in, false, local_rvd,
                          eval_result.rebuild_non_rvd_time,
                          eval_result.rvd_time);
        eval_result.energy = rebuilt_energy(eval_result.state);
        eval_result.valid = true;
        return eval_result;
      };

      static const std::array<double, 64> sigma_mult = []() {
        std::array<double, 64> ret{};
        for (int i = 0; i < (int)ret.size(); ++i) {
          ret[i] = 0.60 + 2.00 * double(i) / double(ret.size() - 1);
        }
        return ret;
      }();

      std::vector<unsigned int> trial_seeds(perturb_num_trials);
      for (int trial = 0; trial < perturb_num_trials; ++trial) {
        trial_seeds[trial] = perturb_rng();
      }

      const std::vector<TangentPerturbFrame> perturb_frames =
          build_tangent_perturb_frames(current_state.sites,
                                       current_state.surface_normals,
                                       perturb_mask,
                                       current_state.rdt_faces,
                                       perturb_fallback_scale);

      std::vector<PerturbEval> trial_results(perturb_num_trials);
      #pragma omp parallel for schedule(dynamic)
      for (int trial = 0; trial < perturb_num_trials; ++trial) {
        std::mt19937 trial_rng(trial_seeds[trial]);
        std::vector<_Point3> perturbed_sites = perturb_sites_from_frames(
            current_state.sites,
            perturb_frames,
            perturb_alpha * sigma_mult[trial % sigma_mult.size()],
            _model,
            trial_rng);
        trial_results[trial] = evaluate_perturb_sites(perturbed_sites);
      }

      double perturb_rebuild_non_rvd_task_time = 0.0;
      double perturb_rvd_task_time = 0.0;
      RebuiltState best_perturb_state;
      double best_perturb_E = std::numeric_limits<double>::infinity();
      int best_perturb_trial = -1;
      for (int trial = 0; trial < perturb_num_trials; ++trial) {
        if (!trial_results[trial].valid) continue;
        perturb_rebuild_non_rvd_task_time +=
            trial_results[trial].rebuild_non_rvd_time;
        perturb_rvd_task_time += trial_results[trial].rvd_time;
        if (best_perturb_trial < 0 ||
            is_better_perturb_candidate(trial_results[trial].state,
                                        best_perturb_state,
                                        trial_results[trial].energy,
                                        best_perturb_E)) {
          best_perturb_state = std::move(trial_results[trial].state);
          best_perturb_E = trial_results[trial].energy;
          best_perturb_trial = trial;
        }
      }


      bool perturb_accepted = false;
      if (best_perturb_trial >= 0 &&
          accept_perturb_candidate(best_perturb_state, best_perturb_E,
                                   pre_perturb_state, pre_perturb_E)) {
        const std::vector<_Point3> accepted_sites = best_perturb_state.sites;
        double accepted_rebuild_non_rvd_time = 0.0;
        double accepted_rvd_time = 0.0;
        current_state = rebuild_state(accepted_sites, true, _RVD,
                                      accepted_rebuild_non_rvd_time,
                                      accepted_rvd_time);
        perturb_rebuild_non_rvd_task_time += accepted_rebuild_non_rvd_time;
        perturb_rvd_task_time += accepted_rvd_time;
        current_rebuilt_E = rebuilt_energy(current_state);
        _sites = current_state.sites;
        update_best_state(current_state, current_rebuilt_E);

        {
          torch::NoGradGuard no_grad;
          site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
        }
        optimizer.state().clear();

        const double perturb_lr_reset =
            std::max(adam_lr_min, post_perturb_lr_scale * lr_before_step);
        set_adam_lr(optimizer, perturb_lr_reset);
        reset_recent_lr_scheduler(current_state.eval.hinge_loss);
        recent_hinge_losses.clear();
        push_recent_hinge_loss(current_state.eval.hinge_loss);
        last_single_step_hinge_progress = 1.0;
        last_window_hinge_progress = 1.0;
        last_hinge_window_ready = false;

        hinge_stagnation_streak = 0;
        last_penalty_update_hinge = current_state.eval.hinge_loss;
        post_perturb_lock_iters_remaining =
            current_state.eval.active_quads <= quick_perturb_active_quads_thresh
                ? late_post_perturb_lock_iters
                : post_perturb_lock_iters;
        perturb_accepted = true;

        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] perturb ACCEPT"
                    << " | trigger=" << perturb_trigger
                    << " | alpha=" << std::scientific
                    << std::setprecision(log_value_precision)
                    << perturb_alpha
                    << " | window15Progress=" << perturb_window_progress_for_log
                    << " | stepProgress=" << perturb_step_progress_for_log
                    << " | trials=" << perturb_num_trials
                    << " | bestTrial=" << best_perturb_trial
                    << " | lrReset=" << get_adam_lr(optimizer)
                    << " | E(before/after)=" << pre_perturb_E << "/"
                    << current_rebuilt_E
                    << " | hinge(before/after)="
                    << pre_perturb_state.eval.hinge_loss << "/"
                    << current_state.eval.hinge_loss
                    << " | Act(before/after)="
                    << pre_perturb_state.eval.active_quads << "/"
                    << current_state.eval.active_quads
                    << " | min_g(before/after)="
                    << pre_perturb_state.eval.min_g << "/"
                    << current_state.eval.min_g
                    << std::endl;
        }

        if (has_strict_zero_cover_energy(current_state)) {
          stop_requested = true;
          if (_para.is_show) {
            std::cout << "[QuadCoverLike][Adam] STOP (perturb feasible)"
                      << std::scientific
                      << std::setprecision(log_value_precision)
                      << " | hingeRaw=" << current_state.eval.hinge_loss
                      << " | hingePen=" << current_state.eval.weighted_hinge_loss
                      << " | lambda=" << current_state.eval.hinge_lambda
                      << " | num_quads=" << current_state.search_quads.size()
                      << " | active_quads=" << current_state.eval.active_quads
                      << " | min_g=" << current_state.eval.min_g
                      << " | bestTrial=" << best_perturb_trial
                      << std::endl;
          }
        }
      }

      if (!perturb_accepted) {
        {
          torch::NoGradGuard no_grad;
          site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
        }
        optimizer.state().clear();
        set_adam_lr(optimizer, std::max(adam_lr_min, 0.8 * lr_before_step));
        reset_recent_lr_scheduler(current_state.eval.hinge_loss);

        hinge_stagnation_streak = std::max(0, hinge_stagnation_streak - 1);

        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] perturb REJECT"
                    << " | trigger=" << perturb_trigger
                    << " | alpha=" << std::scientific
                    << std::setprecision(log_value_precision)
                    << perturb_alpha
                    << " | window15Progress=" << perturb_window_progress_for_log
                    << " | stepProgress=" << perturb_step_progress_for_log
                    << " | trials=" << perturb_num_trials
                    << " | bestTrial=" << best_perturb_trial
                    << " | E(curr/bestPert)=" << pre_perturb_E << "/"
                    << best_perturb_E
                    << " | hinge(curr/bestPert)="
                    << pre_perturb_state.eval.hinge_loss << "/"
                    << (best_perturb_trial >= 0 ? best_perturb_state.eval.hinge_loss
                                                : pre_perturb_state.eval.hinge_loss)
                    << " | Act(curr/bestPert)="
                    << pre_perturb_state.eval.active_quads << "/"
                    << (best_perturb_trial >= 0 ? best_perturb_state.eval.active_quads
                                                : pre_perturb_state.eval.active_quads)
                    << std::endl;
        }
      }

      const double perturb_wall_elapsed = wall_seconds_since(perturb_wall_start);
      total_perturb_wall_time += perturb_wall_elapsed;
      iter_perturb_wall_time += perturb_wall_elapsed;
      total_perturb_rebuild_task_time += perturb_rebuild_non_rvd_task_time;
      total_perturb_rvd_task_time += perturb_rvd_task_time;
      iter_perturb_rebuild_task_time += perturb_rebuild_non_rvd_task_time;
      iter_perturb_rvd_task_time += perturb_rvd_task_time;
    }

    if (stop_requested) {
      break;
    }

    const int prev_num_quads = (int)current_state.search_quads.size();
    const int prev_active_quads = current_state.eval.active_quads;
    const double prev_min_g = current_state.eval.min_g;
    const double prev_E = current_rebuilt_E;
    const double prev_qem = current_state.eval.qem_loss;
    const double prev_hinge = current_state.eval.hinge_loss;
    const double prev_weighted_hinge = current_state.eval.weighted_hinge_loss;


    std::vector<char> active_mask = build_active_site_mask(
        current_state.sites, current_state.frozen_poles,
        current_state.search_quads, current_state.eps);
    std::vector<char> opt_mask = expand_active_mask_one_ring(
        active_mask, current_state.rdt_faces, n, 1);
    if (!use_hinge_loss_energy) opt_mask = make_all_active_mask(n);
    if (!has_any_active(opt_mask)) opt_mask = make_all_active_mask(n);
    const int active_site_count = std::max(1, count_active_mask(opt_mask));

    Eigen::MatrixXd grad_mat = current_state.eval.grads;
    zero_out_inactive_rows(grad_mat, opt_mask);
    Eigen::VectorXd grad_vec = flatten_matrix(grad_mat);
    const double grad_norm = grad_vec.norm();

    const double lr = get_adam_lr(optimizer);

    const double adam_start = omp_get_wtime();
    {
      torch::NoGradGuard no_grad;
      site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
    }

    optimizer.zero_grad();
    site_param.mutable_grad() = eigen_matrix_to_tensor(grad_mat, optim_device).clone();
    optimizer.step();

    Eigen::MatrixXd step_mat =
        tensor_to_matrix(site_param) - sites_to_matrix(current_state.sites);
    zero_out_inactive_rows(step_mat, opt_mask);

    {
      const double elapsed = wall_seconds_since(adam_start);
      total_adam_time += elapsed;
      iter_adam_time += elapsed;
    }

    std::vector<_Point3> trial_sites =
        apply_projected_step(current_state.sites, step_mat, _model);
    double cand_rebuild_non_rvd_time = 0.0;
    double cand_rvd_time = 0.0;
    RebuiltState candidate_state = rebuild_state(
        trial_sites, false, _RVD, cand_rebuild_non_rvd_time, cand_rvd_time);
    total_rebuild_non_rvd_time += cand_rebuild_non_rvd_time;
    total_rvd_time += cand_rvd_time;
    iter_rebuild_non_rvd_time += cand_rebuild_non_rvd_time;
    iter_rvd_time += cand_rvd_time;
    const double cand_E = rebuilt_energy(candidate_state);
    const Eigen::MatrixXd realized =
        realized_step_matrix(current_state.sites, candidate_state.sites);
    const double attempted_step_for_log = max_row_norm(realized);

    {
      const double refresh_start = omp_get_wtime();
      refresh_state_with_current_rvd(candidate_state, _RVD, true);
      const double elapsed = wall_seconds_since(refresh_start);
      total_refresh_eval_time += elapsed;
      iter_refresh_eval_time += elapsed;
    }
    current_state = std::move(candidate_state);
    current_rebuilt_E = rebuilt_energy(current_state);
    _sites = current_state.sites;
    update_best_state(current_state, current_rebuilt_E);

    {
      torch::NoGradGuard no_grad;
      site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
    }

    bool lr_reduced = step_recent_lr_scheduler(current_state.eval.hinge_loss);
    double lr_after_sched = get_adam_lr(optimizer);

    const bool hinge_improved =
        current_state.eval.hinge_loss <
        prev_hinge * (1.0 - perturb_hinge_rel_tol) - perturb_hinge_abs_tol;
    if (post_perturb_lock_iters_remaining > 0) {
      --post_perturb_lock_iters_remaining;
    }

    const double step_for_log = attempted_step_for_log;
    const char *step_status = "ACCEPT";

    if (hinge_improved) {
      hinge_stagnation_streak = 0;
    } else {
      ++hinge_stagnation_streak;
    }
    push_recent_hinge_loss(current_state.eval.hinge_loss);

    double single_step_hinge_progress = 1.0;
    const double prev_hinge_safe = std::max(prev_hinge, 1e-18);
    single_step_hinge_progress =
        (prev_hinge - current_state.eval.hinge_loss) / prev_hinge_safe;

    double window_hinge_progress = 1.0;
    const bool hinge_window_ready =
        (int)recent_hinge_losses.size() >= perturb_window_iters + 1;
    if (hinge_window_ready) {
      const double window_hinge_ref =
          std::max(recent_hinge_losses.front(), 1e-18);
      window_hinge_progress =
          (window_hinge_ref - current_state.eval.hinge_loss) / window_hinge_ref;
    }
    last_single_step_hinge_progress = single_step_hinge_progress;
    last_window_hinge_progress = window_hinge_progress;
    last_hinge_window_ready = hinge_window_ready;

    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    _IterationInfo info;
    info.iteration = outer + 1;
    info.num_quads = (int)current_state.search_quads.size();
    info.active_quads = current_state.eval.active_quads;
    info.min_margin = current_state.eval.min_g;
    info.accepted_step = step_for_log;
    _history.push_back(info);


    if (use_weight_schedule && (outer + 1) % penalty_k_update_interval == 0) {
      const double old_penalty_k = penalty_k;
      const double old_fixed_hinge_lambda = fixed_hinge_lambda;
      const double old_qem_weight = qem_weight;
      const double hinge_ref = std::max(last_penalty_update_hinge, 1e-12);
      const double hinge_progress =
          (hinge_ref - current_state.eval.hinge_loss) / hinge_ref;

      if (current_state.eval.active_quads > 0 &&
          current_state.eval.hinge_loss > hinge_stop_tol) {
        if (hinge_progress < hinge_progress_slow_tol) {
          penalty_k *= penalty_k_grow_fast;
        } else if (hinge_progress < hinge_progress_fast_tol) {
          penalty_k *= penalty_k_grow_slow;
        } else {
          penalty_k /= penalty_k_shrink_factor;
        }
      }

      if (current_state.eval.active_quads <= late_aggressive_penalty_active_thresh &&
          hinge_stagnation_streak >= 3 &&
          current_state.eval.active_quads > 0) {
        penalty_k *= 3.0;
        qem_weight *= 0.85;
      }

      penalty_k = std::clamp(penalty_k, penalty_k_min, penalty_k_max);
      update_fixed_hinge_lambda();
      qem_weight *= qem_weight_decay;
      last_penalty_update_hinge = current_state.eval.hinge_loss;

      if (std::abs(penalty_k - old_penalty_k) > 1e-18 ||
          std::abs(fixed_hinge_lambda - old_fixed_hinge_lambda) > 1e-18 ||
          std::abs(qem_weight - old_qem_weight) > 1e-18) {
        apply_eval_weights(current_state.eval, effective_qem_weight(),
                           effective_hinge_lambda());
        current_rebuilt_E = rebuilt_energy(current_state);
        _sites = current_state.sites;
        update_best_state(current_state, current_rebuilt_E);
        reseed_recent_lr_scheduler(current_state.eval.hinge_loss);

        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] k update"
                    << " | old=" << std::scientific
                    << std::setprecision(log_value_precision) << old_penalty_k
                    << " | new=" << penalty_k
                    << " | qemW=" << qem_weight
                    << " | hingeRaw=" << current_state.eval.hinge_loss
                    << " | QEM=" << current_state.eval.qem_loss
                    << " | hingeProgress=" << hinge_progress
                    << " | lambda=" << fixed_hinge_lambda
                    << std::endl;
        }
      }
    }

    {
      if (use_hinge_loss_energy && has_strict_zero_cover_energy(current_state)) {
        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] STOP (hinge zero)"
                    << std::scientific << std::setprecision(log_value_precision)
                    << " | hingeRaw=" << current_state.eval.hinge_loss
                    << " | hingePen=" << current_state.eval.weighted_hinge_loss
                    << " | lambda=" << current_state.eval.hinge_lambda
                    << " | num_quads=" << current_state.search_quads.size()
                    << " | active_quads=" << current_state.eval.active_quads
                    << " | min_g=" << current_state.eval.min_g
                    << " | step=" << step_for_log
                    << std::endl;
        }
        stop_requested = true;
        break;
      }
    }

    if (_para.is_show) {
      const double iter_time = wall_seconds_since(iter_start);
      const double iter_accounted_time =
          iter_rebuild_non_rvd_time + iter_rvd_time + iter_refresh_eval_time +
          iter_adam_time + iter_output_time + iter_perturb_wall_time;
      const double iter_other_time =
          std::max(0.0, iter_time - iter_accounted_time);

      std::cout << "[QuadCoverLike][Adam] iter=" << info.iteration
                << " | Quads(prev/curr)=" << prev_num_quads << "/"
                << info.num_quads
                << " | Act(prev/curr)=" << prev_active_quads << "/"
                << info.active_quads
                << " | E(curr)=" << std::scientific
                << std::setprecision(log_value_precision) << prev_E
                << " -> " << current_rebuilt_E
                << " | QEM(prev/curr)=" << prev_qem
                << "/" << current_state.eval.qem_loss
                << " | hingeRaw(prev/curr)=" << prev_hinge
                << "/" << current_state.eval.hinge_loss
                << " | hingePen(prev/curr)=" << prev_weighted_hinge
                << "/" << current_state.eval.weighted_hinge_loss
                << " | qemW(curr)=" << qem_weight
                << " | k(curr)=" << penalty_k
                << " | lambda(curr)=" << current_state.eval.hinge_lambda
                << " | min_g(prev/curr)=" << prev_min_g << "/"
                << info.min_margin
                << " | lr=" << lr
                << " | ||g||=" << grad_norm
                << " | trialStep=" << attempted_step_for_log
                << " | step=" << step_for_log
                << " | lrDrop=" << (lr_reduced ? 1 : 0)
                << " | lrNext=" << lr_after_sched
                << " | hingeStag=" << hinge_stagnation_streak
                << " | iterT=" << std::fixed << std::setprecision(log_time_precision)
                << iter_time << "s"
                << " | rebuildT=" << iter_rebuild_non_rvd_time << "s"
                << " | rvdT=" << iter_rvd_time << "s"
                << " | refreshT=" << iter_refresh_eval_time << "s"
                << " | adamT=" << iter_adam_time << "s"
                << " | outputT=" << iter_output_time << "s"
                << " | perturbWallT=" << iter_perturb_wall_time << "s"
                << " | otherT=" << iter_other_time << "s"
                << " | " << step_status
                << std::endl;
    }

    if (time_limit_exceeded()) {
      hit_time_limit = true;
      break;
    }
  }
  _sites = best_sites;
  const double final_rvd_start = omp_get_wtime();
  _RVD.calculate_(_sites);
  total_rvd_time += wall_seconds_since(final_rvd_start);
  const double final_rebuild_start = omp_get_wtime();
  std::vector<CellGeom> final_cells;
  compute_cell_geom_and_spheres(_sites, _RVD, final_cells, _spheres);
  total_rebuild_non_rvd_time += wall_seconds_since(final_rebuild_start);
  const double final_output_start = omp_get_wtime();
  output_mesh(_sites, _RVD, _model, n, output_dir, model_name,
              (int)_history.size() + 1);
  write_spheres_csv(output_dir, n, model_name, (int)_history.size() + 1,
                    _spheres, _model);
  total_output_time += wall_seconds_since(final_output_start);

  if (hit_time_limit || time_limit_exceeded()) {
    std::cout << "tle" << std::endl;
  }

  if (_para.is_show) {
    const double total_time = wall_seconds_since(quadcover_start);
    const double accounted_wall_time =
        total_rebuild_non_rvd_time + total_rvd_time + total_refresh_eval_time +
        total_adam_time + total_output_time + total_perturb_wall_time;
    const double other_wall_time =
        std::max(0.0, total_time - accounted_wall_time);

    std::cout << "\n--- QuadCover Finished ---\n"
              << "Total Wall Time: " << std::fixed
              << std::setprecision(log_time_precision)
              << total_time << " s\n"
              << "Rebuild Wall Time (non-RVD): "
              << total_rebuild_non_rvd_time << " s\n"
              << "RVD Wall Time: " << total_rvd_time << " s\n"
              << "Refresh/Eval Wall Time: " << total_refresh_eval_time
              << " s\n"
              << "Adam Wall Time: " << total_adam_time << " s\n"
              << "Perturb Wall Time: " << total_perturb_wall_time << " s\n"
              << "Output Wall Time: " << total_output_time << " s\n"
              << "Other Wall Time: " << other_wall_time << " s\n"
              << std::endl;
  }
}

} // namespace BGAL
