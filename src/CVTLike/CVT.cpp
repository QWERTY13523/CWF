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
#include <vector>
#include <cmath>
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
#include <memory>
#include <igl/gaussian_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/principal_curvature.h>
#include <igl/adjacency_list.h>
#include <igl/avg_edge_length.h>

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
struct FeatureSegment {
  BGAL::_Point3 a;
  BGAL::_Point3 b;
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

// ============================
// 数值稳定 softplus/sigmoid
// ============================
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
  // log(1+exp(x))
  if (x > 50.0) return x;
  if (x < -50.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

// ============================
// 三球交（trilateration）两解
// 返回 false 表示无实交点或退化
// ============================
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

  // 坐标解
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

// 用 64-bit key 去重三角形（假设点数 < 2^21 ~ 2M）
static inline uint64_t tri_key_sorted(int a, int b, int c) {
  int x=a,y=b,z=c;
  if (x>y) std::swap(x,y);
  if (y>z) std::swap(y,z);
  if (x>y) std::swap(x,y);
  return ( (uint64_t)x << 42 ) | ( (uint64_t)y << 21 ) | (uint64_t)z;
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

    auto Vs = sites;
    auto Edges = RVD.get_edges_();
    std::set<std::pair<int, int>> RDT_Edges;
    std::vector<std::set<int>> neibors;
    neibors.resize(Vs.size());

    for (int i = 0; i < (int)Edges.size(); i++) {
      for (const auto ee : Edges[i]) {
        RDT_Edges.insert(
            std::make_pair(std::min(i, ee.first), std::max(i, ee.first)));
        neibors[i].insert(ee.first);
        neibors[ee.first].insert(i);
      }
    }

    for (auto v : Vs) {
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
                f1 != std::min(f1, std::min(f2, f3))) {
              mid = f1;
            }
            if (f2 != std::max(f1, std::max(f2, f3)) &&
                f2 != std::min(f1, std::min(f2, f3))) {
              mid = f2;
            }
            if (f3 != std::max(f1, std::max(f2, f3)) &&
                f3 != std::min(f1, std::min(f2, f3))) {
              mid = f3;
            }
            rdtFaces.insert(MyFace(std::max(f1, std::max(f2, f3)), mid,
                                   std::min(f1, std::min(f2, f3))));
          }
        }
      }
    }

    for (auto f : rdtFaces) {
      outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " "
             << f.p.z() + 1 << std::endl;
      outRDT1 << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " "
              << f.p.z() + 1 << std::endl;
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

  _rho = [feature_segments, inv_two_sigma2, kBaseDensity,
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
    inPointsName = (base / ("n" + std::to_string(num_sites) + "_" + modelname +
                           "_inputPoints.xyz")).string();
  } else {
    inPointsName = pointsName;
  }

  std::ifstream inPoints(inPointsName.c_str());
  std::vector<Eigen::Vector3d> Pts, Nors;

  double x, y, z, nx, ny, nz;
  while (inPoints >> x >> y >> z >> nx >> ny >> nz) {
    Pts.push_back(Eigen::Vector3d(x, y, z));
    Nors.push_back(Eigen::Vector3d(nx, ny, nz));
  }
  inPoints.close();
  std::cout << "Pts.size(): " << Pts.size() << std::endl;

  if (pointsName != nullptr) {
    num_sites = static_cast<int>(Pts.size());
  }

  int num = (int)Pts.size();
  std::cout << "\nBegin CWF.\n" << std::endl;

  // ======= 参数（你可以调）=======
  int Fnum = 4;
  double alpha = 1.0, eplison = 1.0, lambda = 1.0;
  double decay = 0.965;

  // 覆盖惩罚项参数（核心新增）
  const auto bbox = _model.bounding_box_();
  const double bbox_diag = (bbox.second - bbox.first).length_();
  const double cover_weight = 0.15;          // μ：覆盖惩罚强度（建议 0.05~1 试）
  const double cover_margin = 0.0;           // margin：留点余量避免擦边抖动
  const double cover_beta   = std::max(1e-12, 1e-3 * bbox_diag); // softplus 平滑尺度
  const double cover_damp   = 1e-10;         // 解 J^T y = dE/ds 的阻尼
  const int cover_neighbor_ring = 2;         // 第四球候选范围：从 1-ring 略微放宽到 2-ring
  const bool cover_global_fallback = true;   // 局部候选没命中时，退化到全局第四球检查
  const int cover_start_iter = 55;

  std::vector<int> FaceIDs(num, -1);
  int eval_count = 0;

  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm2 =
      [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
        ++eval_count;
        const bool enable_cover = (eval_count > cover_start_iter);

        eplison = eplison * decay;
        double lossCVT = 0, loss = 0;
        double lossCover = 0;

        startRVD = clock();
        std::vector<Sphere::Sphere> spheres(num);

        // 1) 把 X 投影回基准面，更新 _sites 和 Nors
        for (int i = 0; i < num; ++i) {
          Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2));
          Point_T closest = tree.closest_point(query);
          auto tri = tree.closest_point_and_primitive(query);

          Polyhedron::Face_handle f = tri.second;
          int fid = -1;
          auto it = face_id_map.find(static_cast<const void *>(&*f));
          if (it != face_id_map.end()) fid = it->second;
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

          this->_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());
        }

        // 2) RVD
        this->_RVD.calculate_(this->_sites);

        Fnum++;
        if (Fnum % 1 == 0) {
          OutputMesh(this->_sites, this->_RVD, num_sites,
                     (std::filesystem::current_path() / "data" / "Block").string(),
                     modelname, Fnum);
        }

        endRVD = clock();
        RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

        const auto &cells = this->_RVD.get_cells_();
        const auto &edges = this->_RVD.get_edges_();

        double energy = 0.0;
        g.setZero();

        std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());

        // 3) CVT + QE 主项（原逻辑）
        //    注意：你这里把 rho 固定成 rho(site)，不是 rho(p)，我保持你的写法不动
        omp_set_num_threads(128);
#pragma omp parallel for reduction(+ : lossCVT, loss)
        for (int i = 0; i < num; ++i) {
          BGAL::_Point3 site = this->_sites[i];
          for (int j = 0; j < (int)cells[i].size(); ++j) {
            auto [a, b, c] = cells[i][j];
            BGAL::_Point3 pa = this->_RVD.vertex_(a);
            BGAL::_Point3 pb = this->_RVD.vertex_(b);
            BGAL::_Point3 pc = this->_RVD.vertex_(c);

            Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
                [&, site](BGAL::_Point3 p) {
                  const double rho_val = this->_rho(p);
                  Eigen::VectorXd r(5);
                  BGAL::_Point3 NorTriM = (pb - pa).cross_(pc - pa);
                  NorTriM.normalized_();

                  r(0) = (eplison * rho_val * ((site - p).sqlength_()));
                  r(1) = lambda * (NorTriM.dot_(p - site)) * (NorTriM.dot_(p - site)) +
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

          // 4) 半径：站点到 cell 边界最远点距离（你原来的逻辑）
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
          boundary_pts.reserve(bnd.size());

          for (int vid : bnd) {
            boundary_pts.emplace_back(this->_RVD.vertex_(vid));
          }

          BGAL::_Point3 farp = site;
          double best_dist = 0.0;
          for (const auto &p : boundary_pts) {
            double dist = (p - site).length_(); // 注意：这是距离，不是 squared
            if (dist > best_dist) {
              best_dist = dist;
              farp = p;
            }
          }

          spheres[i].c = decltype(spheres[i].c)(site.x(), site.y(), site.z());
          spheres[i].r = best_dist;
          spheres[i].max_point =
              decltype(spheres[i].max_point)(farp.x(), farp.y(), farp.z());
        } // end omp

        // ============================================================
        // 5) 新增：覆盖约束 E_cover（严格：对三球交点做隐式微分传梯度）
        // ============================================================
        if (enable_cover && cover_weight > 0.0) {
          // 5.1 建邻接（RDT 图）
          std::vector<std::vector<int>> nbr(num);
          std::vector<std::unordered_set<int>> nbr_set(num);
          for (int i = 0; i < (int)edges.size(); ++i) {
            nbr[i].reserve(edges[i].size());
            nbr_set[i].reserve(edges[i].size() * 2 + 8);
            for (const auto &kv : edges[i]) {
              int j = kv.first;
              if (j < 0 || j >= num || j == i) continue;
              nbr[i].push_back(j);
              nbr_set[i].insert(j);
            }
          }

          // 5.2 枚举三角形（i<j<k）并去重
          std::vector<Eigen::Vector3i> rdt_tris;
          rdt_tris.reserve(num * 4);

          std::unordered_set<uint64_t> tri_seen;
          tri_seen.reserve(num * 8);

          for (int i = 0; i < num; ++i) {
            auto &ni = nbr[i];
            // 为了更好控制唯一性，做一个排序（可选）
            std::sort(ni.begin(), ni.end());
            for (int idxj = 0; idxj < (int)ni.size(); ++idxj) {
              int j = ni[idxj];
              if (j <= i) continue;
              for (int idxk = idxj + 1; idxk < (int)ni.size(); ++idxk) {
                int k = ni[idxk];
                if (k <= j) continue;
                if (nbr_set[j].find(k) == nbr_set[j].end()) continue;

                uint64_t key = tri_key_sorted(i, j, k);
                if (tri_seen.insert(key).second) {
                  rdt_tris.emplace_back(i, j, k);
                }
              }
            }
          }

          // 5.3 预取中心和半径（半径在本次评估里当常量）
          std::vector<Eigen::Vector3d> C(num);
          std::vector<double> R(num);
          for (int i = 0; i < num; ++i) {
            C[i] = Eigen::Vector3d(this->_sites[i].x(), this->_sites[i].y(), this->_sites[i].z());
            R[i] = std::max(0.0, (double)spheres[i].r);
          }

          // 5.4 对每个三角形：三球交两点 -> 找覆盖球 -> 计算 dE/ds -> 解 (J^T)y=dE/ds -> 回传到 ci,cj,ck
          for (const auto &t : rdt_tris) {
            int i = t.x(), j = t.y(), k = t.z();
            if (R[i] <= 0 || R[j] <= 0 || R[k] <= 0) continue;

            Eigen::Vector3d s1, s2;
            if (!intersect_three_spheres(C[i], R[i], C[j], R[j], C[k], R[k], s1, s2)) {
              // 这里你若想把“必须形成种子点(约束1)”也软化，可以在这里加惩罚
              continue;
            }

            // 两个解都检查
            Eigen::Vector3d seeds[2] = {s1, s2};

            for (int si = 0; si < 2; ++si) {
              const Eigen::Vector3d s = seeds[si];

              // 候选第四球：默认 1-ring，这里额外并入一层 2-ring 以减少漏检
              std::unordered_set<int> cand;
              cand.reserve((nbr[i].size() + nbr[j].size() + nbr[k].size()) *
                           cover_neighbor_ring * 2 + 8);
              auto add_candidate_ring = [&](int center) {
                for (int u : nbr[center]) {
                  if (u < 0 || u >= num) continue;
                  cand.insert(u);
                  if (cover_neighbor_ring >= 2) {
                    for (int uu : nbr[u]) {
                      if (uu < 0 || uu >= num) continue;
                      cand.insert(uu);
                    }
                  }
                }
              };
              add_candidate_ring(i);
              add_candidate_ring(j);
              add_candidate_ring(k);
              cand.erase(i); cand.erase(j); cand.erase(k);

              int best_l = -1;
              double best_v = 0.0; // v = (r_l - margin) - dist
              auto try_update_best = [&](int l) {
                if (l < 0 || l >= num) return;
                if (l == i || l == j || l == k) return;
                if (R[l] <= 0) return;
                double dist = (s - C[l]).norm();
                if (dist < 1e-12) return;

                double v = (R[l] - cover_margin) - dist;
                if (v > best_v) {
                  best_v = v;
                  best_l = l;
                }
              };
              for (int l : cand) {
                try_update_best(l);
              }
              if (cover_global_fallback && best_v <= 0.0) {
                for (int l = 0; l < num; ++l) {
                  try_update_best(l);
                }
              }

              if (best_l < 0 || best_v <= 0.0) continue;

              // ---- softplus barrier: E = μ * (β log(1+exp(v/β)))^2 ----
              double tval = best_v / cover_beta;
              double sp = cover_beta * stable_softplus(tval);
              double sig = stable_sigmoid(tval); // d/dv softplus(v/β) = sigmoid(v/β)/β; 乘回 cover_beta 后就是 sigmoid(v/β)
              double dE_dv = 2.0 * cover_weight * sp * sig; // dE/dv
              lossCover += cover_weight * sp * sp;

              // v = (r_l - margin) - ||s-c_l||
              // dv/ds = -(s-c_l)/dist
              Eigen::Vector3d d = (s - C[best_l]);
              double dist = d.norm();
              if (dist < 1e-12) continue;
              Eigen::Vector3d dir = d / dist;

              // dE/ds
              Eigen::Vector3d grad_s = -dE_dv * dir;

              // 直接项：对第四球中心 c_l 的梯度
              // dv/dc_l = +(s-c_l)/dist = dir
              gi[best_l] += (+dE_dv) * dir;

              // --------- 隐式微分回传到 (c_i,c_j,c_k) ----------
              // F1 = ||s-ci||^2 - ri^2, ...
              // J = dF/ds, row1 = 2(s-ci)^T, row2 = 2(s-cj)^T, row3 = 2(s-ck)^T
              // Solve J^T y = dE/ds
              Eigen::Vector3d di = (s - C[i]);
              Eigen::Vector3d dj = (s - C[j]);
              Eigen::Vector3d dk = (s - C[k]);

              Eigen::Matrix3d JT;
              JT.col(0) = 2.0 * di;
              JT.col(1) = 2.0 * dj;
              JT.col(2) = 2.0 * dk;

              // 阻尼避免奇异
              JT(0,0) += cover_damp;
              JT(1,1) += cover_damp;
              JT(2,2) += cover_damp;

              Eigen::Vector3d yvec = JT.colPivHouseholderQr().solve(grad_s);

              // dE/dci = 2(s-ci)*y1, dE/dcj = 2(s-cj)*y2, dE/dck = 2(s-ck)*y3
              gi[i] += 2.0 * di * yvec(0);
              gi[j] += 2.0 * dj * yvec(1);
              gi[k] += 2.0 * dk * yvec(2);
            }
          }
        }

        // 6) 投影到切平面（保持你原来的曲面约束）
        for (int i = 0; i < num; ++i) {
          gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
          g(i * 3 + 0) += gi[i].x();
          g(i * 3 + 1) += gi[i].y();
          g(i * 3 + 2) += gi[i].z();
        }

        energy += loss + lossCover;

        std::cout << std::setprecision(7)
                  << "energy: " << energy
                  << " LossCVT: " << lossCVT / eplison
                  << " LossQE: "  << (loss - lossCVT) / lambda
                  << " LossCover: " << lossCover
                  << " Lambda_CVT: " << eplison
                  << std::endl;

        // dump sphere info（你原来就有）
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

  // ======= LBFGS 初始化（原逻辑）=======
  std::vector<Eigen::Vector3d> Pts2 = Pts;
  num = (int)Pts2.size();
  std::cout << Pts2.size() << "  " << num << std::endl;

  _sites.resize(num);
  _para.max_linearsearch = 20;

  BGAL::_LBFGS lbfgs2(_para);
  Eigen::VectorXd iterX2(num * 3);

  for (int i = 0; i < num; ++i) {
    iterX2(i * 3 + 0) = Pts2[i].x();
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

  // 最终投影回面 + 输出
  for (int i = 0; i < num; ++i) {
    Point_T query(iterX2(i * 3 + 0), iterX2(i * 3 + 1), iterX2(i * 3 + 2));
    Point_T closest = tree.closest_point(query);
    auto tri = tree.closest_point_and_primitive(query);

    Polyhedron::Face_handle f = tri.second;
    int fid = -1;
    auto it = face_id_map.find(static_cast<const void *>(&*tri.second));
    if (it != face_id_map.end()) fid = it->second;
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
             (std::filesystem::current_path() / "data" / "Block").string(),
             modelname, 2);
}

} // namespace BGAL
