#include "BGAL/QuadCoverLike/QuadCover.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/CVTLike/CVT.h"
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

  const int n_sites = sites.size();
  DynamicRadiusEval eval(n_sites, need_grad);
  const auto &cell_tris = rvd.get_cells_();
  Eigen::MatrixXd hinge_grads =
      need_grad ? Eigen::MatrixXd::Zero(n_sites, 3) : Eigen::MatrixXd();

  #pragma omp parallel
  {
    double local_qem_loss = 0.0;
    Eigen::MatrixXd local_grads =
        need_grad ? Eigen::MatrixXd::Zero(n_sites, 3) : Eigen::MatrixXd();

    #pragma omp for schedule(dynamic) nowait
    for (int i = 0; i < (int)cell_tris.size(); ++i) {
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

        local_qem_loss += inte(0);
        if (need_grad) {
          local_grads.row(i) += Vec3(inte(1), inte(2), inte(3)).transpose();
        }
      }
    }

    #pragma omp critical
    {
      eval.qem_loss += local_qem_loss;
      if (need_grad) eval.qem_grads += local_grads;
    }
  }

  const int num_quads = quads.size();
  #pragma omp parallel
  {
    double local_hinge_loss = 0.0;
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
        local_hinge_loss += violation * violation;
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

    // 绾跨▼鏁版嵁姹囪仛
    #pragma omp critical
    {
      eval.hinge_loss += local_hinge_loss;
      eval.active_quads += local_active_quads;
      eval.min_g = std::min(eval.min_g, local_min_g);
      if (need_grad) {
        hinge_grads += local_grads;
      }
    }
  }

  if (need_grad) {
    eval.hinge_grads = hinge_grads;
  }

  // 骞惰鍒囬潰鎶曞奖锛氬垎鍒姇褰?QEM / hinge 涓ら儴鍒嗘搴︼紝渚夸簬鍚庣画鏃犻噸寤鸿皟鏉?
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
  std::vector<char> mask(sites.size(), 0);

  #pragma omp parallel
  {
    std::vector<char> local_mask(sites.size(), 0);

    #pragma omp for nowait
    for (int qi = 0; qi < (int)quads.size(); ++qi) {
      const auto &q = quads[qi];
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
        for (int k = 0; k < 4; ++k) local_mask[q[k]] = 1;
      }
    }

    #pragma omp critical
    {
      for (int i = 0; i < (int)mask.size(); ++i) {
        mask[i] = static_cast<char>(mask[i] || local_mask[i]);
      }
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
    #pragma omp parallel
    {
      std::vector<char> local_next(num_sites, 0);

      #pragma omp for nowait
      for (int fi = 0; fi < (int)faces.size(); ++fi) {
        const auto &f = faces[fi];
        const int a = f.x(), b = f.y(), c = f.z();
        if (a < 0 || a >= num_sites || b < 0 || b >= num_sites || c < 0 || c >= num_sites) {
          continue;
        }
        if (mask[a] || mask[b] || mask[c]) {
          local_next[a] = local_next[b] = local_next[c] = 1;
        }
      }

      #pragma omp critical
      {
        for (int i = 0; i < num_sites; ++i) {
          next[i] = static_cast<char>(next[i] || local_next[i]);
        }
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

static inline std::vector<_Point3> perturb_sites_structured(
    const std::vector<_Point3> &base_sites,
    const std::vector<Vec3> &surface_normals,
    const std::vector<char> &perturb_mask,
    const Eigen::MatrixXd &hinge_grads,
    double sigma,
    int mode,
    const BGAL::_ManifoldModel &model,
    std::mt19937 &rng) {
  std::vector<_Point3> perturbed = base_sites;
  if (!(sigma > 0.0) || !std::isfinite(sigma)) return perturbed;

  std::normal_distribution<double> normal01(0.0, 1.0);
  std::vector<char> use_site_flags(base_sites.size(), 0);
  std::vector<double> rand_a(base_sites.size(), 0.0);
  std::vector<double> rand_b(base_sites.size(), 0.0);

  for (int i = 0; i < (int)base_sites.size(); ++i) {
    const bool use_site =
        (i < (int)perturb_mask.size() && perturb_mask[i] != 0);
    use_site_flags[i] = use_site ? 1 : 0;
    if (!use_site) continue;
    rand_a[i] = normal01(rng);
    rand_b[i] = normal01(rng);
  }

  #pragma omp parallel for
  for (int i = 0; i < (int)base_sites.size(); ++i) {
    if (!use_site_flags[i]) continue;

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

    Vec3 guide = Vec3::Zero();
    if (hinge_grads.rows() == (int)base_sites.size()) {
      guide = -hinge_grads.row(i).transpose();
    }
    guide -= guide.dot(n) * n;
    if (guide.squaredNorm() < 1e-20) {
      guide = t1;
    } else {
      guide.normalize();
    }

    Vec3 ortho = n.cross(guide);
    const double ortho_norm = ortho.norm();
    if (ortho_norm < 1e-12) ortho = t2;
    else ortho /= ortho_norm;

    Vec3 rand_tan = rand_a[i] * t1 + rand_b[i] * t2;
    const double rand_norm = rand_tan.norm();
    if (rand_norm > 1e-12) rand_tan /= rand_norm;
    else rand_tan = t1;

    double guide_scale = 1.0;
    double ortho_scale = 0.0;
    double rand_scale = 0.0;

    switch (mode) {
      case 0: guide_scale = 1.80; ortho_scale = 0.00; rand_scale = 0.30; break;
      case 1: guide_scale = 1.40; ortho_scale = 0.60; rand_scale = 0.35; break;
      case 2: guide_scale = 1.40; ortho_scale = -0.60; rand_scale = 0.35; break;
      case 3: guide_scale = 1.10; ortho_scale = 0.00; rand_scale = 0.70; break;
      case 4: guide_scale = 0.90; ortho_scale = 0.90; rand_scale = 0.60; break;
      case 5: guide_scale = 0.90; ortho_scale = -0.90; rand_scale = 0.60; break;
      case 6: guide_scale = 0.55; ortho_scale = 0.00; rand_scale = 1.10; break;
      default: guide_scale = 0.35; ortho_scale = 1.20; rand_scale = 0.95; break;
    }

    Vec3 tangent_step =
        sigma * (guide_scale * guide +
                 ortho_scale * ortho +
                 rand_scale * rand_tan);

    const double max_len = 4.0 * sigma;
    const double len = tangent_step.norm();
    if (len > max_len && len > 1e-16) tangent_step *= (max_len / len);

    const Vec3 p = to_eigen(base_sites[i]) + tangent_step;
    perturbed[i] = project_to_surface(model, to_point(p));
  }
  return perturbed;
}


// 銆愰€熷害鏍稿脊绾ф彁鍗囥€戜娇鐢ㄩ偦鎺ヨ〃浜ら泦娉曚唬鏇垮師鏉ョ殑 std::set锛屾彁鍙?Delaunay 琛ㄩ潰缁撴瀯
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
  // 閬垮厤杩囧害閿佷簤鐢紝閲囩敤灞€閮ㄦ敹闆?
  #pragma omp parallel
  {
      std::vector<Eigen::Vector3i> local_tris;
      #pragma omp for schedule(dynamic)
      for (int u = 0; u < num_sites; ++u) {
        for (int v : adj[u]) {
          if (u >= v) continue;
          
          // 姹備袱涓湁搴忔暟缁勭殑浜ら泦鏉ユ壘鍒板叡鍚岄偦灞?w
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
// 鏍稿績閫昏緫锛欳GAL kd-tree 鍗婂緞鏌ヨ + 鍊欓€夊洓鍏冪粍鏋氫妇
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
  #pragma omp parallel for reduction(max:global_rmax)
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
  tree.build();

  #pragma omp parallel
  {
    std::vector<CGALPoint3> hits;
    hits.reserve(256);
    std::vector<std::array<int, 4>> local_quads;
    local_quads.reserve(std::min<std::size_t>(
        std::max<std::size_t>(1, faces.size() / std::max(1, omp_get_max_threads())) * 8ull,
        1ull << 18));

    #pragma omp for schedule(dynamic)
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

    #pragma omp critical
    {
      quads.insert(quads.end(), local_quads.begin(), local_quads.end());
    }
  }

  std::sort(quads.begin(), quads.end());
  quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
}

} // namespace 闂悎
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
    // 閲囩敤灞€閮ㄥ畾闀挎暟缁勬垨鑰呭皬 vector 鏇夸唬 std::set锛屽姞閫熸瀬鍏舵槑鏄?
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
  const double total_start = omp_get_wtime();
  _history.clear();
  _sites = init_sites;
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

    if (_para.is_show) {
      std::cout << "[QuadCoverLike] CWF warm start begin"
                << " | iters=" << cvt_para.max_iteration
                << " | sites=" << _sites.size() << std::endl;
    }

    BGAL::_CVT3D cvt(_model, rho, cvt_para);
    cvt.set_outpath(output_dir + "/");
    cvt.calculate_(_sites, model_name, false);
    _sites = cvt.get_sites();

    if (_para.is_show) {
      std::cout << "[QuadCoverLike] CWF warm start end"
                << " | sites=" << _sites.size() << std::endl;
    }
  }

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
    state.eps = 1e-4 * state.avg_r * state.avg_r;

    if (need_grad) {
      update_surface_normals(_model, state.sites, state.surface_normals);
    } else {
      state.surface_normals.clear();
    }
    state.rdt_faces = build_rdt_faces_from_edges(n, rvd.get_edges_());
    build_quads_from_search_faces(state.rdt_faces, state.sites, state.spheres,
                                  state.search_quads);

    state.eval = evaluate_qem_hinge_objective(
        state.sites, rvd, state.frozen_poles, state.surface_normals,
        state.search_quads, state.eps, qem_weight, fixed_hinge_lambda,
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
            state.search_quads, state.eps, qem_weight, fixed_hinge_lambda,
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
  qem_weight = penalty_k * current_state.eval.hinge_loss /
               (current_state.eval.qem_loss + 1e-12);
  if (!std::isfinite(qem_weight)) qem_weight = 1.0;
  qem_weight = std::max(qem_weight, 0.0);
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

  const double adam_lr_init = 1e-4;
  const double adam_beta1 = 0.9;
  const double adam_beta2 = 0.999;
  const double adam_eps = 1e-8;
  const double adam_lr_min = 3e-7;
  const double adam_lr_reduce_factor = 0.6;
  const int adam_lr_plateau_patience = 2;
  const int adam_lr_plateau_cooldown = 1;
  const double adam_lr_plateau_rel_tol = 1e-2;
  const int adam_lr_recent_best_window = 5;

  //
  const int penalty_k_update_interval = 6;
  const double penalty_k_min = hinge_lambda_floor;
  const double penalty_k_max = 1e4;
  const double penalty_k_grow_fast = 3.0;
  const double penalty_k_grow_slow = 1.5;
  const double penalty_k_shrink_factor = 1.0;
  const double hinge_progress_fast_tol = 0.20;
  const double hinge_progress_slow_tol = 0.05;
  const double qem_weight_decay = 0.95;

  const double hinge_stop_tol = std::max(_para.accept_eps, 1e-12);
  const int hinge_feasible_patience = 3;
  const int stagnation_patience = 4;
  const int stagnation_hard_patience = 2 * stagnation_patience;
  const double perturb_hinge_rel_tol = 1e-3;
  const double perturb_hinge_abs_tol = 1e-12;
  const int perturb_window_iters = 15;
  const double perturb_window_progress_tol = 0.03;
  const double perturb_single_step_progress_tol = 0.03;

  // perturb 鏇翠繚瀹堬細鍑忓皯 trial 鏁帮紝鍑忓皬灏哄害
  const double stagnation_perturb_ratio = 0.010;
  const int perturb_num_trials = 32;
  const double perturb_lr_max = 5e-5;
  const double stagnation_lr_trigger = 0.25 * adam_lr_init;
  const double stagnation_prop_trigger = 1.0;

  // perturb 鎺ュ彈鍑嗗垯锛氬厑璁稿彈鎺ч潪鍗曡皟锛屼絾涓嶅厑璁告槑鏄惧姡鍖?
  const double perturb_accept_relax_E = 0.01;
  const double perturb_accept_relax_hinge = 0.05;
  const double perturb_accept_min_margin_gain = 1e-8;
  const double perturb_accept_min_hinge_gain = 0.05;

  // perturb 鍚庡喎鍗?
  const int post_perturb_cooldown_iters = 10;
  const double post_perturb_prop_fixed = 1.0;
  const double post_perturb_lr_scale = 1.2;
  const double post_perturb_lr_floor = 2.5e-5;
  const double post_perturb_lr_cap = 5.0e-5;
  const double late_post_perturb_lr_cap = 4.0e-5;
  const double very_late_post_perturb_lr_cap = 3.0e-5;

  // 鍚庢湡绛栫暐锛氶檺鍒舵闀裤€佸叧闂?late perturb銆佸姞蹇?hinge 涓诲
  const int late_no_perturb_active_thresh = 20;
  const int late_aggressive_penalty_active_thresh = 50;

  // 瀵煎嚭棰戠巼锛氶檷浣?I/O 鍜岄澶栫殑 RVD rebuild
  const int export_interval = 50;

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
      improved_for_scheduler =
          metric < recent_best * (1.0 - adam_lr_plateau_rel_tol);
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

  double proposal_scale = 1.5;
  const double proposal_scale_min = 1.0;
  const double proposal_scale_max = 4.0;
  const double proposal_grow = 1.10;

  int hinge_stagnation_streak = 0;
  int hinge_feasible_streak = 0;
  int post_perturb_cooldown = 0;

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

  for (int outer = 0; outer < _para.max_outer_iterations; ++outer) {
    const double iter_start = omp_get_wtime();
    double iter_rebuild_non_rvd_time = 0.0;
    double iter_rvd_time = 0.0;
    double iter_adam_time = 0.0;
    double iter_output_time = 0.0;
    double iter_refresh_eval_time = 0.0;
    double iter_perturb_wall_time = 0.0;
    double iter_perturb_rebuild_task_time = 0.0;
    double iter_perturb_rvd_task_time = 0.0;

    if (post_perturb_cooldown > 0) {
      proposal_scale = post_perturb_prop_fixed;
      --post_perturb_cooldown;
    }

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
      output_mesh(_sites, _RVD, n, output_dir, model_name, outer);
      write_spheres_csv(output_dir, n, model_name, outer, current_state.spheres,
                        _model);
      {
        const double elapsed = wall_seconds_since(output_start);
        total_output_time += elapsed;
        iter_output_time += elapsed;
      }
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
    step_mat *= proposal_scale;

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
    const bool hinge_worsened =
        current_state.eval.hinge_loss >
        prev_hinge * (1.0 + perturb_hinge_rel_tol) + perturb_hinge_abs_tol;

    if (hinge_worsened) {
      proposal_scale = std::max(proposal_scale_min, 0.95 * proposal_scale);
    } else {
      proposal_scale = std::min(proposal_scale_max,
                                proposal_grow * proposal_scale);
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

    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    _IterationInfo info;
    info.iteration = outer + 1;
    info.num_quads = (int)current_state.search_quads.size();
    info.active_quads = current_state.eval.active_quads;
    info.min_margin = current_state.eval.min_g;
    info.accepted_step = step_for_log;
    _history.push_back(info);


    if ((outer + 1) % penalty_k_update_interval == 0) {
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
        apply_eval_weights(current_state.eval, qem_weight, fixed_hinge_lambda);
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
      std::vector<char> stop_mask = build_active_site_mask(
          current_state.sites, current_state.frozen_poles,
          current_state.search_quads, current_state.eps);
      stop_mask = expand_active_mask_one_ring(
          stop_mask, current_state.rdt_faces, n, 1);
      if (!has_any_active(stop_mask)) stop_mask = make_all_active_mask(n);

      const bool hinge_exact_zero = current_state.eval.hinge_loss <= 0.0;
      if (hinge_exact_zero) {
        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] STOP (hinge zero)"
                    << std::scientific << std::setprecision(log_value_precision)
                    << " | hingeRaw=" << current_state.eval.hinge_loss
                    << " | hingePen=" << current_state.eval.weighted_hinge_loss
                    << " | lambda=" << current_state.eval.hinge_lambda
                    << " | step=" << step_for_log
                    << std::endl;
        }
        break;
      }

      const bool hinge_feasible =
          (current_state.eval.active_quads == 0) ||
          (current_state.eval.hinge_loss <= hinge_stop_tol);
      if (hinge_feasible) {
        ++hinge_feasible_streak;
      } else {
        hinge_feasible_streak = 0;
      }

      if (hinge_feasible && hinge_feasible_streak >= hinge_feasible_patience) {
        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] STOP (hinge feasible)"
                    << std::scientific << std::setprecision(log_value_precision)
                    << " | hingeRaw=" << current_state.eval.hinge_loss
                    << " | hingePen=" << current_state.eval.weighted_hinge_loss
                    << " | lambda=" << current_state.eval.hinge_lambda
                    << " | step=" << step_for_log
                    << " | streak=" << hinge_feasible_streak
                    << std::endl;
        }
        break;
      }
    }

      const bool allow_perturb_by_local_stagnation =
          (lr_after_sched <= perturb_lr_max) &&
          (hinge_stagnation_streak >= stagnation_patience) &&
          (proposal_scale <= stagnation_prop_trigger ||
           lr_after_sched <= stagnation_lr_trigger ||
           hinge_stagnation_streak >= stagnation_hard_patience);
      const bool allow_perturb_by_window_stagnation =
          hinge_window_ready &&
          (window_hinge_progress <= perturb_window_progress_tol) &&
          (single_step_hinge_progress < perturb_single_step_progress_tol);
      const bool allow_perturb =
          (post_perturb_cooldown == 0) &&
          (allow_perturb_by_local_stagnation ||
           allow_perturb_by_window_stagnation);
      const char *perturb_trigger =
          allow_perturb_by_window_stagnation
              ? (allow_perturb_by_local_stagnation ? "local+window15"
                                                   : "window15")
              : "local";

    if (allow_perturb) {
      const double perturb_wall_start = omp_get_wtime();
      const RebuiltState pre_perturb_state = current_state;
      const double pre_perturb_E = current_rebuilt_E;

      std::vector<char> perturb_mask = active_mask;
      if (!has_any_active(perturb_mask)) perturb_mask = opt_mask;
      if (!has_any_active(perturb_mask)) perturb_mask = make_all_active_mask(n);

      Eigen::MatrixXd perturb_guide = current_state.eval.hinge_grads;
      zero_out_inactive_rows(perturb_guide, perturb_mask);

      const double perturb_sigma = std::max(
          1e-6,
          stagnation_perturb_ratio * std::max(current_state.avg_r, 1e-8));

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
            rebuild_state(sites_in, true, local_rvd,
                          eval_result.rebuild_non_rvd_time,
                          eval_result.rvd_time);
        eval_result.energy = rebuilt_energy(eval_result.state);
        eval_result.valid = true;
        return eval_result;
      };

      static const std::array<double, 16> sigma_mult = {
          0.60, 0.72, 0.84, 0.96, 1.08, 1.20, 1.32, 1.44,
          1.56, 1.68, 1.82, 1.96, 2.12, 2.28, 2.44, 2.60};

      std::vector<unsigned int> trial_seeds(perturb_num_trials);
      for (int trial = 0; trial < perturb_num_trials; ++trial) {
        trial_seeds[trial] = perturb_rng();
      }

      std::vector<PerturbEval> trial_results(perturb_num_trials);
      #pragma omp parallel for schedule(dynamic)
      for (int trial = 0; trial < perturb_num_trials; ++trial) {
        std::mt19937 trial_rng(trial_seeds[trial]);
        std::vector<_Point3> perturbed_sites = perturb_sites_structured(
            current_state.sites,
            current_state.surface_normals,
            perturb_mask,
            perturb_guide,
            sigma_mult[trial % sigma_mult.size()] * perturb_sigma,
            trial % 8,
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

      const double perturb_wall_elapsed = wall_seconds_since(perturb_wall_start);
      total_perturb_wall_time += perturb_wall_elapsed;
      iter_perturb_wall_time += perturb_wall_elapsed;
      total_perturb_rebuild_task_time += perturb_rebuild_non_rvd_task_time;
      total_perturb_rvd_task_time += perturb_rvd_task_time;
      iter_perturb_rebuild_task_time += perturb_rebuild_non_rvd_task_time;
      iter_perturb_rvd_task_time += perturb_rvd_task_time;

      bool perturb_accepted = false;
      if (best_perturb_trial >= 0 &&
          accept_perturb_candidate(best_perturb_state, best_perturb_E,
                                   pre_perturb_state, pre_perturb_E)) {
        current_state = std::move(best_perturb_state);
        current_rebuilt_E = best_perturb_E;
        _sites = current_state.sites;
        update_best_state(current_state, current_rebuilt_E);

        {
          torch::NoGradGuard no_grad;
          site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
        }
        optimizer.state().clear();

        const double post_perturb_lr_cap_now =
            (current_state.eval.active_quads <= late_no_perturb_active_thresh)
                ? very_late_post_perturb_lr_cap
                : (current_state.eval.active_quads <=
                           late_aggressive_penalty_active_thresh
                       ? late_post_perturb_lr_cap
                       : post_perturb_lr_cap);
        const double perturb_lr_reset =
            std::min(post_perturb_lr_cap_now,
                     std::max(post_perturb_lr_floor,
                              post_perturb_lr_scale * lr_after_sched));
        set_adam_lr(optimizer, perturb_lr_reset);
        reset_recent_lr_scheduler(current_state.eval.hinge_loss);
        recent_hinge_losses.clear();
        push_recent_hinge_loss(current_state.eval.hinge_loss);

        proposal_scale = post_perturb_prop_fixed;
        hinge_stagnation_streak = 0;
        hinge_feasible_streak = 0;
        last_penalty_update_hinge = current_state.eval.hinge_loss;
        post_perturb_cooldown = post_perturb_cooldown_iters;
        perturb_accepted = true;

        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] perturb ACCEPT"
                    << " | trigger=" << perturb_trigger
                    << " | sigma=" << std::scientific
                    << std::setprecision(log_value_precision)
                    << perturb_sigma
                    << " | window15Progress=" << window_hinge_progress
                    << " | stepProgress=" << single_step_hinge_progress
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
      }

      if (!perturb_accepted) {
        {
          torch::NoGradGuard no_grad;
          site_param.copy_(sites_to_tensor(current_state.sites, optim_device));
        }
        optimizer.state().clear();
        set_adam_lr(optimizer, std::max(adam_lr_min, 0.8 * lr_after_sched));
        reset_recent_lr_scheduler(current_state.eval.hinge_loss);

        proposal_scale = std::max(proposal_scale_min, 0.5 * proposal_scale);
        hinge_stagnation_streak = std::max(0, hinge_stagnation_streak - 1);

        if (_para.is_show) {
          std::cout << "[QuadCoverLike][Adam] perturb REJECT"
                    << " | trigger=" << perturb_trigger
                    << " | sigma=" << std::scientific
                    << std::setprecision(log_value_precision)
                    << perturb_sigma
                    << " | window15Progress=" << window_hinge_progress
                    << " | stepProgress=" << single_step_hinge_progress
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
                << " | propScale=" << proposal_scale
                << " | hingeStag=" << hinge_stagnation_streak
                << " | iterT=" << std::fixed << std::setprecision(log_time_precision)
                << iter_time << "s"
                << " | rebuildT=" << iter_rebuild_non_rvd_time << "s"
                << " | rvdT=" << iter_rvd_time << "s"
                << " | refreshT=" << iter_refresh_eval_time << "s"
                << " | adamT=" << iter_adam_time << "s"
                << " | outputT=" << iter_output_time << "s"
                << " | perturbWallT=" << iter_perturb_wall_time << "s"
                << " | perturbTask(rebuild/rvd)="
                << iter_perturb_rebuild_task_time << "/"
                << iter_perturb_rvd_task_time
                << " | otherT=" << iter_other_time << "s"
                << " | " << step_status
                << std::endl;
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
  output_mesh(_sites, _RVD, n, output_dir, model_name,
              (int)_history.size() + 1);
  write_spheres_csv(output_dir, n, model_name, (int)_history.size() + 1,
                    _spheres, _model);
  total_output_time += wall_seconds_since(final_output_start);

  if (_para.is_show) {
    const double total_time = wall_seconds_since(total_start);
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
              << "Accumulated Perturb Rebuild Task Time: "
              << total_perturb_rebuild_task_time << " s\n"
              << "Accumulated Perturb RVD Task Time: "
              << total_perturb_rvd_task_time << " s\n"
              << std::endl;
  }
}

} // namespace BGAL
