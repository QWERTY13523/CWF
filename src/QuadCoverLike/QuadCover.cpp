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
  _history.clear();
  _sites = init_sites;
  if (_sites.empty()) {
    throw std::runtime_error("[QuadCoverLike] empty init_sites.");
  }

  const int n = (int)_sites.size();
  std::string outpath = (std::filesystem::current_path() / "data" / "QuadCover").string();

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
                           bool need_grad) -> RebuiltState {
    RebuiltState state(need_grad ? n : 0);
    state.sites = sites_in;

    _RVD.calculate_(state.sites);

    std::vector<CellGeom> cells;
    compute_cell_geom_and_spheres(state.sites, _RVD, cells, state.spheres);

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
    state.rdt_faces = build_rdt_faces_from_edges(n, _RVD.get_edges_());
    build_quads_from_search_faces(state.rdt_faces, state.sites, state.spheres,
                                  state.search_quads);

    state.eval = evaluate_dynamic_radius_surrogate(
        state.sites, state.frozen_poles, state.surface_normals,
        state.search_quads, state.eps, need_grad);
    return state;
  };

  auto refresh_quads_only_state = [&](const RebuiltState &base_state,
                                      const std::vector<_Point3> &sites_in,
                                      bool need_grad) -> RebuiltState {
    RebuiltState state(need_grad ? n : 0);
    state.sites = sites_in;
    state.spheres = base_state.spheres;
    state.frozen_poles = base_state.frozen_poles;
    state.eps = base_state.eps;
    state.avg_r = base_state.avg_r;
    state.rdt_faces = base_state.rdt_faces;

    update_surface_normals(_model, state.sites, state.surface_normals);
    build_quads_from_search_faces(state.rdt_faces, state.sites, state.spheres,
                                  state.search_quads);

    state.eval = evaluate_dynamic_radius_surrogate(
        state.sites, state.frozen_poles, state.surface_normals,
        state.search_quads, state.eps, need_grad);
    return state;
  };

  auto rebuilt_energy = [&](const RebuiltState &state) -> double {
    return state.eval.total_loss;
  };

  auto count_active_sites = [&](const RebuiltState &state) -> int {
    std::vector<char> active_mask = build_active_site_mask(
        state.sites, state.frozen_poles, state.search_quads, state.eps);
    return (int)std::count(active_mask.begin(), active_mask.end(), (char)1);
  };


  auto model_value = [&](const RebuiltState &base_state,
                              const RebuiltState &state,
                              double *step_norm_out) -> double {
    Eigen::MatrixXd step_from_base =
        realized_step_matrix(base_state.sites, state.sites);
    const double step_norm = flatten_matrix(step_from_base).norm();
    if (step_norm_out) *step_norm_out = step_norm;
    return state.eval.total_loss;
  };

  auto model_gradient = [&](const RebuiltState &base_state,
                                 const RebuiltState &state,
                                 Eigen::MatrixXd *step_from_base_out,
                                 Eigen::VectorXd *grad_vec_out) -> Eigen::MatrixXd {
    Eigen::MatrixXd step_from_base =
        realized_step_matrix(base_state.sites, state.sites);
    Eigen::MatrixXd grad_model = state.eval.grads;
    if (step_from_base_out) *step_from_base_out = std::move(step_from_base);
    if (grad_vec_out) *grad_vec_out = flatten_matrix(grad_model);
    return grad_model;
  };

  struct InnerSolveResult {
    RebuiltState state;
    Eigen::MatrixXd step_from_base;
    Eigen::VectorXd step_vec_from_base;
    double step_norm = 0.0;
    double surrogate_start_E = 0.0;
    double surrogate_final_E = 0.0;
    int start_active = 0;
    int final_active = 0;
    int num_quads = 0;
    double last_accepted_step = 0.0;
    bool hit_boundary = false;
    bool late_stage = false;
    int active_site_count = 0;
  };

  double initial_rebuilt_E = 0.0;
  std::mt19937 rng(123456789u);
  const int late_active_quad_threshold = 128;
  const int late_active_site_threshold = 384;
  const double late_energy_ratio = 1e-3;

  auto solve_tr_subproblem = [&](const RebuiltState &base_state,
                                  double trust_radius,
                                  bool force_late_stage) -> InnerSolveResult {
    InnerSolveResult result;
    result.state = refresh_quads_only_state(base_state, base_state.sites, true);
    result.surrogate_start_E = result.state.eval.total_loss;
    result.start_active = result.state.eval.active_quads;

    std::vector<char> active_mask = build_active_site_mask(
        base_state.sites, base_state.frozen_poles, base_state.search_quads,
        base_state.eps);
    result.active_site_count = std::max(1, count_active_mask(active_mask));
    result.late_stage =
        force_late_stage ||
        (base_state.eval.active_quads <= late_active_quad_threshold) ||
        (result.active_site_count <= late_active_site_threshold) ||
        (base_state.eval.total_loss <=
         late_energy_ratio * std::max(initial_rebuilt_E, 1e-30));

    const int inner_iters = result.late_stage ? 3 : 8;
    const int max_linesearch = result.late_stage ? 12 : 10;
    const int lbfgs_memory = result.late_stage ? 3 : 6;
    const double c1 = 1e-4;
    std::vector<LBFGSPair> history;

    for (int inner = 0; inner < inner_iters; ++inner) {
      Eigen::MatrixXd current_step_from_base;
      Eigen::VectorXd model_grad_vec;
      Eigen::MatrixXd model_grad_mat = model_gradient(
          base_state, result.state, &current_step_from_base, &model_grad_vec);
      if (result.late_stage) {
        zero_out_inactive_rows(model_grad_mat, active_mask);
        model_grad_vec = flatten_matrix(model_grad_mat);
      }
      const double grad_norm = model_grad_vec.norm();
      if (grad_norm < 1e-14) break;

      Eigen::VectorXd dir = lbfgs_two_loop_direction(model_grad_vec, history);
      Eigen::MatrixXd dir_mat = unflatten_vector(dir, n);
      if (result.late_stage) {
        zero_out_inactive_rows(dir_mat, active_mask);
        dir = flatten_matrix(dir_mat);
      }
      const double dir_norm = dir.norm();
      if (dir_norm < 1e-16) break;

      double alpha = std::min(1.0, trust_radius / std::max(dir_norm, 1e-16));
      if (result.late_stage) alpha = std::min(alpha, 0.75);
      if (!(alpha > 0.0) || !std::isfinite(alpha)) break;

      bool accepted = false;
      RebuiltState accepted_state;
      Eigen::MatrixXd accepted_realized_step;

      double current_step_norm = flatten_matrix(current_step_from_base).norm();
      const double current_model = result.state.eval.total_loss;
      if (current_step_norm >= 0.999 * trust_radius) {
        result.hit_boundary = true;
        break;
      }

      for (int ls = 0; ls < max_linesearch; ++ls) {
        const Eigen::MatrixXd proposed_step = alpha * dir_mat;
        std::vector<_Point3> trial_sites =
            apply_projected_step(result.state.sites, proposed_step, _model);
        Eigen::MatrixXd realized_step_current =
            realized_step_matrix(result.state.sites, trial_sites);
        if (result.late_stage) {
          zero_out_inactive_rows(realized_step_current, active_mask);
        }
        const Eigen::VectorXd realized_step_current_vec =
            flatten_matrix(realized_step_current);
        if (realized_step_current_vec.norm() < 1e-16) {
          alpha *= 0.5;
          continue;
        }

        const double directional_derivative =
            model_grad_vec.dot(realized_step_current_vec);
        if (!(directional_derivative < -1e-18)) {
          alpha *= 0.5;
          continue;
        }

        RebuiltState trial_state =
            refresh_quads_only_state(base_state, trial_sites, false);
        double trial_step_norm = 0.0;
        const double trial_model =
            model_value(base_state, trial_state, &trial_step_norm);

        if (trial_step_norm > trust_radius * (1.0 + 1e-8)) {
          alpha *= 0.5;
          continue;
        }

        const double armijo_rhs = current_model + c1 * alpha * directional_derivative;
        if (trial_model <= armijo_rhs + 1e-14) {
          accepted = true;
          accepted_realized_step = std::move(realized_step_current);
          accepted_state =
              refresh_quads_only_state(base_state, trial_sites, true);
          result.last_accepted_step =
              std::max(result.last_accepted_step, max_row_norm(accepted_realized_step));
          if (trial_step_norm >= 0.98 * trust_radius) {
            result.hit_boundary = true;
          }
          break;
        }

        alpha *= 0.5;
      }

      if (!accepted) {
        history.clear();
        break;
      }

      Eigen::MatrixXd new_step_from_base;
      Eigen::VectorXd new_model_grad_vec;
      Eigen::MatrixXd new_model_grad_mat = model_gradient(
          base_state, accepted_state, &new_step_from_base, &new_model_grad_vec);
      if (result.late_stage) {
        zero_out_inactive_rows(new_model_grad_mat, active_mask);
        new_model_grad_vec = flatten_matrix(new_model_grad_mat);
      }

      const bool same_topology =
          (result.state.search_quads == accepted_state.search_quads);
      const Eigen::VectorXd s = flatten_matrix(accepted_realized_step);
      const Eigen::VectorXd y = new_model_grad_vec - model_grad_vec;
      const double sy = s.dot(y);
      const double s2 = s.squaredNorm();

      if (!same_topology || sy <= 1e-10 * std::max(1.0, s2) || alpha < 0.25) {
        history.clear();
      } else {
        push_lbfgs_pair(history, s, y, lbfgs_memory);
      }

      result.state = std::move(accepted_state);
    }

    result.step_from_base = realized_step_matrix(base_state.sites, result.state.sites);
    if (result.late_stage) {
      zero_out_inactive_rows(result.step_from_base, active_mask);
    }
    result.step_vec_from_base = flatten_matrix(result.step_from_base);
    result.step_norm = result.step_vec_from_base.norm();
    result.surrogate_final_E = result.state.eval.total_loss;
    result.final_active = result.state.eval.active_quads;
    result.num_quads = (int)result.state.search_quads.size();
    return result;
  };

  RebuiltState current_state = rebuild_state(_sites, true);
  double current_rebuilt_E = rebuilt_energy(current_state);
  initial_rebuilt_E = std::max(current_rebuilt_E, 1e-30);

  std::vector<_Point3> best_sites = current_state.sites;
  double best_rebuilt_E = current_rebuilt_E;

  int active_site_count = std::max(1, count_active_sites(current_state));
  double trust_radius =
      0.03 * current_state.avg_r * std::sqrt((double)active_site_count);

  int stagnation_streak = 0;
  int topology_freeze_streak = 0;
  int rebuilt_close_streak = 0;

  const double eta_accept = 0.02;
  const double eta_good = 0.60;
  const double eta_ok = 0.15;
  const double outer_stop_tol = 1e-12;

  auto tiny_actual_decrease = [&](double E) -> double {
    return std::max(1e-8, 1e-4 * std::max(E, 1e-12));
  };

  auto margin_improve_tol = [&](double mg) -> double {
    return std::max(1e-6, 5e-3 * std::max(std::abs(mg), 1e-3));
  };

  struct EscapeAttemptInfo {
    bool attempted = false;
    bool accepted = false;
    int trial_count = 0;
    int accepted_trial = -1;
    double sigma_base_max = 0.0;
    double accepted_sigma = 0.0;
    double best_sigma = 0.0;
    int best_active = std::numeric_limits<int>::max();
    double best_min_g = -std::numeric_limits<double>::infinity();
    double best_E = std::numeric_limits<double>::infinity();
    bool best_better_active = false;
    bool best_better_margin = false;
    bool best_better_energy = false;
  };

  auto try_escape_perturbation = [&](RebuiltState &state,
                                     double &state_E,
                                     double &delta,
                                     EscapeAttemptInfo *info = nullptr) -> bool {
    std::vector<char> active_mask = build_active_site_mask(
        state.sites, state.frozen_poles, state.search_quads, state.eps);
    const int n_active_sites = std::max(1, count_active_mask(active_mask));
    const int plateau_level = std::max(stagnation_streak,
                                       std::max(topology_freeze_streak, rebuilt_close_streak));
    const double avg_r = std::max(state.avg_r, 1e-8);
    const double sigma_base_max =
        std::max(1e-6,
                 (0.024 + 0.012 * std::min(plateau_level, 10)) * avg_r);
    const std::array<double, 6> sigma_scales = {1.0, 0.72, 0.50, 0.33, 0.20, 0.10};

    if (info) {
      *info = EscapeAttemptInfo{};
      info->attempted = true;
      info->sigma_base_max = sigma_base_max;
    }

    auto better_trial = [](int act_a, double mg_a, double E_a,
                           int act_b, double mg_b, double E_b) -> bool {
      if (act_a != act_b) return act_a < act_b;
      if (std::abs(mg_a - mg_b) > 1e-15) return mg_a > mg_b;
      return E_a < E_b;
    };

    for (int trial_id = 0; trial_id < (int)sigma_scales.size(); ++trial_id) {
      const double sigma = std::max(1e-6, sigma_base_max * sigma_scales[trial_id]);
      std::vector<_Point3> escaped_sites = perturb_sites_for_next_proposal(
          state.sites, state.surface_normals, active_mask, sigma, _model, rng);
      RebuiltState escaped_state = rebuild_state(escaped_sites, true);
      const double escaped_E = rebuilt_energy(escaped_state);

      const bool better_active =
          escaped_state.eval.active_quads + 1 <= state.eval.active_quads;
      const bool better_margin =
          escaped_state.eval.min_g >
          state.eval.min_g + margin_improve_tol(state.eval.min_g);
      const bool better_energy =
          escaped_E < state_E - tiny_actual_decrease(state_E);

      if (info) {
        ++info->trial_count;
        if (trial_id == 0 ||
            better_trial(escaped_state.eval.active_quads, escaped_state.eval.min_g, escaped_E,
                         info->best_active, info->best_min_g, info->best_E)) {
          info->best_sigma = sigma;
          info->best_active = escaped_state.eval.active_quads;
          info->best_min_g = escaped_state.eval.min_g;
          info->best_E = escaped_E;
          info->best_better_active = better_active;
          info->best_better_margin = better_margin;
          info->best_better_energy = better_energy;
        }
      }

      if (!(better_active || better_margin || better_energy)) continue;

      state = std::move(escaped_state);
      state_E = escaped_E;
      delta = std::max(delta, 0.02 * std::max(state.avg_r, 1e-8) *
                                 std::sqrt((double)n_active_sites));
      if (info) {
        info->accepted = true;
        info->accepted_trial = trial_id;
        info->accepted_sigma = sigma;
      }
      return true;
    }
    return false;
  };

  for (int outer = 0; outer < _para.max_outer_iterations; ++outer) {
    const double local_delta_min = std::max(1e-12, 1e-4 * std::max(current_state.avg_r, 1e-12));
    const double local_delta_max = std::max(local_delta_min * 10.0,
        0.10 * std::max(current_state.avg_r, 1e-12) * std::sqrt((double)std::max(1, n)));
    trust_radius = std::min(local_delta_max, std::max(local_delta_min, trust_radius));

    // if (stagnation_streak >= 3 || topology_freeze_streak >= 3) {
    //   const double rescue_delta =
    //       0.01 * std::max(current_state.avg_r, 1e-8) *
    //       std::sqrt((double)std::max(1, active_site_count));
    //   trust_radius = std::max(trust_radius, rescue_delta);
    // }

    if (stagnation_streak >= 6 || topology_freeze_streak >= 6 || rebuilt_close_streak >= 4) {
      EscapeAttemptInfo escape_info;
      if (try_escape_perturbation(current_state, current_rebuilt_E,
                                  trust_radius, &escape_info)) {
        active_site_count = std::max(1, count_active_sites(current_state));
        _sites = current_state.sites;
        _quads.clear();
        for (const auto &q : current_state.search_quads) _quads.push_back({q});
        if (_para.is_show) {
          std::cout << "[QuadCoverLike][TR-LBFGS] iter=" << (outer + 1)
                    << " | escapeTry=1"
                    << " | escapeTrial=" << (escape_info.accepted_trial + 1)
                    << "/" << escape_info.trial_count
                    << " | sigmaBaseMax=" << std::scientific << std::setprecision(3)
                    << escape_info.sigma_base_max
                    << " | sigma=" << escape_info.accepted_sigma
                    << " | Quads=" << current_state.search_quads.size()
                    << " | Act=" << current_state.eval.active_quads
                    << " | E(rebuilt)=" << current_rebuilt_E
                    << " | min_g=" << current_state.eval.min_g
                    << " | Delta=" << trust_radius
                    << " | rebuiltClose=" << rebuilt_close_streak
                    << " | ESCAPE-ACCEPT"
                    << std::endl;
        }
        stagnation_streak = 0;
        topology_freeze_streak = 0;
        rebuilt_close_streak = 0;
      } else if (_para.is_show && escape_info.attempted) {
        std::cout << "[QuadCoverLike][TR-LBFGS] iter=" << (outer + 1)
                  << " | escapeTry=1"
                  << " | escapeTrial=0/" << escape_info.trial_count
                  << " | sigmaBaseMax=" << std::scientific << std::setprecision(3)
                  << escape_info.sigma_base_max
                  << " | bestSigma=" << escape_info.best_sigma
                  << " | bestAct=" << escape_info.best_active
                  << " | bestE(rebuilt)=" << escape_info.best_E
                  << " | bestMinG=" << escape_info.best_min_g
                  << " | bestBetter(a/m/e)="
                  << (escape_info.best_better_active ? 1 : 0) << "/"
                  << (escape_info.best_better_margin ? 1 : 0) << "/"
                  << (escape_info.best_better_energy ? 1 : 0)
                  << " | rebuiltClose=" << rebuilt_close_streak
                  << " | ESCAPE-REJECT"
                  << std::endl;
      }
    }

    _sites = current_state.sites;
    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    if (_para.export_each_iteration || outer == 0) {
      _RVD.calculate_(_sites);
      output_mesh(_sites, _RVD, n, outpath, model_name, outer);
      write_spheres_csv(outpath, n, model_name, outer, current_state.spheres, _model);
    }

    if (current_state.eval.active_quads == 0 && current_rebuilt_E < outer_stop_tol) {
      _IterationInfo info;
      info.iteration = outer + 1;
      info.num_quads = (int)current_state.search_quads.size();
      info.active_quads = current_state.eval.active_quads;
      info.min_margin = current_state.eval.min_g;
      info.accepted_step = 0.0;
      _history.push_back(info);

      if (_para.is_show) {
        std::cout << "[QuadCoverLike][TR-LBFGS] iter=" << info.iteration
                  << " | Quads=" << info.num_quads
                  << " | Act=" << info.active_quads
                  << " | E(rebuilt)=" << std::scientific << std::setprecision(3)
                  << current_rebuilt_E
                  << " | Delta=" << trust_radius
                    << " | STOP (Constraint Satisfied)"
                  << std::endl;
      }
      break;
    }

    const int prev_rebuilt_num_quads = (int)current_state.search_quads.size();
    const int prev_rebuilt_active = current_state.eval.active_quads;
    const double prev_rebuilt_min_g = current_state.eval.min_g;
    const double prev_rebuilt_E = current_rebuilt_E;

    bool accept_outer = false;
    double rho = -std::numeric_limits<double>::infinity();
    double predicted_decrease = 0.0;
    double actual_decrease = 0.0;
    double surrogate_decrease = 0.0;
    double cand_rebuilt_E = prev_rebuilt_E;
    int cand_num_quads = prev_rebuilt_num_quads;
    int cand_active_quads = prev_rebuilt_active;
    double cand_min_margin = prev_rebuilt_min_g;
    int inner_start_active = prev_rebuilt_active;
    int inner_final_active = prev_rebuilt_active;
    int inner_num_quads = prev_rebuilt_num_quads;
    double inner_start_E = prev_rebuilt_E;
    double inner_final_E = prev_rebuilt_E;
    double trial_step_norm = 0.0;
    double accepted_step = 0.0;
    int used_attempts = 0;
    bool used_late_stage = false;
    int late_active_sites = active_site_count;
    bool has_rebuilt_candidate = false;
    double cand_rebuilt_rel_diff = std::numeric_limits<double>::infinity();

    const bool force_late_stage =
        (stagnation_streak >= 3) || (topology_freeze_streak >= 3) || (rebuilt_close_streak >= 4);
    const bool outer_late_stage =
        force_late_stage ||
        (current_state.eval.active_quads <= late_active_quad_threshold) ||
        (active_site_count <= late_active_site_threshold) ||
        (current_rebuilt_E <= late_energy_ratio * std::max(initial_rebuilt_E, 1e-30));
    const int max_outer_attempts = outer_late_stage ? 5 : 3;

    const double outer_trust_radius = trust_radius;
    double attempt_trust_radius = trust_radius;

    for (int attempt = 0; attempt < max_outer_attempts; ++attempt) {
      used_attempts = attempt + 1;
      InnerSolveResult trial =
          solve_tr_subproblem(current_state, attempt_trust_radius,
                               force_late_stage);
      used_late_stage = trial.late_stage;
      late_active_sites = trial.active_site_count;

      inner_start_active = trial.start_active;
      inner_final_active = trial.final_active;
      inner_num_quads = trial.num_quads;
      inner_start_E = trial.surrogate_start_E;
      inner_final_E = trial.surrogate_final_E;
      trial_step_norm = trial.step_norm;
      accepted_step = trial.last_accepted_step;

      surrogate_decrease = trial.surrogate_start_E - trial.surrogate_final_E;
      predicted_decrease = surrogate_decrease;
      if (!(predicted_decrease > 1e-16) || !(trial.step_norm > 1e-14)) {
        attempt_trust_radius = std::max(local_delta_min, 0.65 * attempt_trust_radius);
        if (attempt_trust_radius <= 1.01 * local_delta_min) break;
        continue;
      }

      RebuiltState candidate_state = rebuild_state(trial.state.sites, true);
      cand_num_quads = (int)candidate_state.search_quads.size();
      cand_active_quads = candidate_state.eval.active_quads;
      cand_min_margin = candidate_state.eval.min_g;
      cand_rebuilt_E = rebuilt_energy(candidate_state);
      has_rebuilt_candidate = true;
      cand_rebuilt_rel_diff = std::abs(cand_rebuilt_E - prev_rebuilt_E) /
                              std::max(std::abs(prev_rebuilt_E), 1e-30);
      actual_decrease = prev_rebuilt_E - cand_rebuilt_E;
      rho = actual_decrease / predicted_decrease;

      if (std::isfinite(rho) && rho > eta_accept && actual_decrease > 0.0) {
        accept_outer = true;
        current_state = std::move(candidate_state);
        current_rebuilt_E = cand_rebuilt_E;
        _sites = current_state.sites;

        if (current_rebuilt_E < best_rebuilt_E) {
          best_rebuilt_E = current_rebuilt_E;
          best_sites = current_state.sites;
        }

        if (rho > eta_good && trial.step_norm >= 0.8 * attempt_trust_radius) {
          trust_radius = std::min(local_delta_max,
                                  std::max(1.50 * attempt_trust_radius,
                                           1.25 * trial.step_norm));
        } else if (rho > eta_ok) {
          trust_radius = std::min(local_delta_max,
                                  std::max(1.10 * attempt_trust_radius,
                                           1.50 * trial.step_norm));
        } else {
          trust_radius = std::min(local_delta_max,
                                  std::max(0.90 * outer_trust_radius,
                                           1.50 * trial.step_norm));
        }
        break;
      }

      if (rho < 0.0 || !std::isfinite(rho)) {
        attempt_trust_radius = std::max(local_delta_min, 0.35 * attempt_trust_radius);
      } else {
        attempt_trust_radius = std::max(local_delta_min, 0.60 * attempt_trust_radius);
      }

      if (attempt_trust_radius <= 1.01 * local_delta_min) break;
    }

    bool stagnating_now = false;
    const bool topology_same_now =
        (cand_num_quads == prev_rebuilt_num_quads) &&
        (cand_active_quads == prev_rebuilt_active);

    if (accept_outer) {
      const bool tiny_gain_now =
          actual_decrease <= tiny_actual_decrease(prev_rebuilt_E);
      const bool no_margin_gain_now =
          cand_min_margin <= prev_rebuilt_min_g + margin_improve_tol(prev_rebuilt_min_g);
      stagnating_now = topology_same_now && tiny_gain_now && no_margin_gain_now;
    } else {
      const bool tiny_pred_now =
          predicted_decrease <= tiny_actual_decrease(prev_rebuilt_E);
      stagnating_now = topology_same_now || tiny_pred_now || (!std::isfinite(rho)) ||
                       (rho < eta_accept);
    }

    if (topology_same_now) ++topology_freeze_streak;
    else topology_freeze_streak = 0;

    const bool rebuilt_close_now =
        has_rebuilt_candidate && std::isfinite(cand_rebuilt_rel_diff) &&
        (cand_rebuilt_rel_diff < 1e-3);
    if (rebuilt_close_now) ++rebuilt_close_streak;
    else rebuilt_close_streak = 0;

    if (stagnating_now) ++stagnation_streak;
    else stagnation_streak = 0;

    if (!accept_outer) {
      trust_radius = attempt_trust_radius;
      _sites = current_state.sites;
    }

    _quads.clear();
    for (const auto &q : current_state.search_quads) _quads.push_back({q});

    _IterationInfo info;
    info.iteration = outer + 1;
    info.num_quads = (int)current_state.search_quads.size();
    info.active_quads = current_state.eval.active_quads;
    info.min_margin = current_state.eval.min_g;
    info.accepted_step = accept_outer ? accepted_step : 0.0;
    _history.push_back(info);

    if (_para.is_show) {
      std::cout << "[QuadCoverLike][TR-LBFGS] iter=" << info.iteration
                << " | tries=" << used_attempts
                << " | Quads(inner/rebuilt/cand)=" << inner_num_quads << "/"
                << prev_rebuilt_num_quads << "/" << cand_num_quads
                << " | Act(inner/rebuilt/cand)=" << inner_start_active << "->"
                << inner_final_active << "/" << prev_rebuilt_active
                << "/" << cand_active_quads
                << " | E(inner)=" << std::scientific << std::setprecision(3)
                << inner_start_E << " -> " << inner_final_E
                << " | E(rebuilt)=" << prev_rebuilt_E << " -> " << cand_rebuilt_E
                << " | relDiff(rebuilt)=" << cand_rebuilt_rel_diff
                << " | pred=" << predicted_decrease
                << " | surDec=" << surrogate_decrease
                << " | ared=" << actual_decrease
                << " | rho=" << rho
                << " | ||d||=" << trial_step_norm
                << " | min_g(rebuilt/cand)=" << prev_rebuilt_min_g << "/" << cand_min_margin
                << " | late=" << (used_late_stage ? 1 : 0)
                << " | actSites=" << late_active_sites
                << " | stag=" << stagnation_streak
                << " | freeze=" << topology_freeze_streak
                << " | rebuiltClose=" << rebuilt_close_streak
                << " | Delta=" << trust_radius
                << " | " << (accept_outer ? "ACCEPT" : "REJECT")
                << std::endl;
    }
  }

  _sites = best_sites;
  _RVD.calculate_(_sites);
  std::vector<CellGeom> final_cells;
  compute_cell_geom_and_spheres(_sites, _RVD, final_cells, _spheres);
  output_mesh(_sites, _RVD, n, outpath, model_name, (int)_history.size() + 1);
  write_spheres_csv(outpath, n, model_name, (int)_history.size() + 1, _spheres, _model);
}

} // namespace BGAL
