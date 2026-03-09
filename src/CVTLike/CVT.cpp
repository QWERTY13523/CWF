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
// 数值稳定 softplus/sigmoid (保留以备其他地方使用)
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
  double decay = 0.95;

  // 覆盖惩罚项参数（核心新增）
  const auto bbox = _model.bounding_box_();
  const double bbox_diag = (bbox.second - bbox.first).length_();
  const double cover_weight = 0.15;          // μ：覆盖惩罚强度（若 v^3 惩罚不够，可适当调大至 1.0~100.0）
  const double cover_margin = 0.0;           // margin：留点余量避免擦边抖动
  const double cover_beta   = std::max(1e-12, 1e-3 * bbox_diag); 
  const double cover_damp   = 1e-10;         // 解 J^T y = dE/ds 的阻尼
  const double cover_violation_tolerance = 1e-6;
  const int cover_neighbor_ring = 1;         // 第四球候选范围：从 1-ring 略微放宽到 2-ring
  const bool cover_global_fallback = true;   // 局部候选没命中时，退化到全局第四球检查
  const int cvt_qem_warmup_iterations = 50;

  std::vector<int> FaceIDs(num, -1);
  bool suppress_intermediate_output = false;

  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm2 =
      [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
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

        if (!suppress_intermediate_output) {
          Fnum++;
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
        // 注意：如果你想在主优化中开启它，请把这里的 false && 删掉
        if (false && cover_weight > 0.0) {
          // 5.1 建邻接（RDT 图）
          std::vector<std::vector<int>> nbr(num);
          double maxCoverViolation = 0.0;
          int invalidSeedCount = 0;
          for (int i = 0; i < (int)edges.size(); ++i) {
            nbr[i].reserve(edges[i].size());
            for (const auto &kv : edges[i]) {
              int j = kv.first;
              if (j < 0 || j >= num || j == i) continue;
              nbr[i].push_back(j);
            }
          }

          const std::vector<Eigen::Vector3i> rdt_tris =
              build_rdt_faces_from_edges(num, edges);

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
              continue;
            }

            // 两个解都检查
            Eigen::Vector3d seeds[2] = {s1, s2};

           for (int si = 0; si < 2; ++si) {
              const Eigen::Vector3d s = seeds[si];

              // 候选第四球：默认 1-ring，额外并入一层 2-ring
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

              double best_v_for_tracking = 0.0; 
              bool has_violation = false;
              Eigen::Vector3d seed_grad_s = Eigen::Vector3d::Zero();

              // 定义一个 Lambda 来评估并累加所有候选球的能量（使用 v^3 保证平滑截断）
              auto eval_and_accumulate = [&](int l) {
                if (l < 0 || l >= num || l == i || l == j || l == k || R[l] <= 0) return;
                Eigen::Vector3d d = (s - C[l]);
                double dist = d.norm();
                if (dist < 1e-12) return;

                double v = (R[l] - cover_margin) - dist;
                best_v_for_tracking = std::max(best_v_for_tracking, v);

                // 严格判定：只有真正进入第四个球内部才产生能量和梯度！
                if (v > 0.0) {
                  has_violation = true;
                  
                  // 使用 v^3 保证 v=0 处能量为0且二阶导连续
                  double v2 = v * v;
                  double v3 = v2 * v;
                  double dE_dv = 3.0 * cover_weight * v2; 
                  
                  lossCover += cover_weight * v3;

                  Eigen::Vector3d dir = d / dist;
                  // 累加到交点 s 的总梯度上
                  seed_grad_s += -dE_dv * dir;
                  // 直接项：对第四球中心 c_l 的梯度
                  gi[l] += (+dE_dv) * dir;
                }
              };

              // 遍历所有候选并累加
              for (int l : cand) {
                eval_and_accumulate(l);
              }

              // 全局 Fallback
              if (cover_global_fallback && !has_violation) {
                for (int l = 0; l < num; ++l) {
                  eval_and_accumulate(l);
                }
              }

              // 如果没有任何球产生惩罚，直接跳过隐式微分
              if (!has_violation) continue;

              ++invalidSeedCount;
              maxCoverViolation = std::max(maxCoverViolation, best_v_for_tracking);

              // --------- 修正后的隐式微分回传到 (c_i,c_j,c_k) ----------
              Eigen::Vector3d di = (s - C[i]);
              Eigen::Vector3d dj = (s - C[j]);
              Eigen::Vector3d dk = (s - C[k]);

              Eigen::Matrix3d JT;
              JT.col(0) = 2.0 * di;
              JT.col(1) = 2.0 * dj;
              JT.col(2) = 2.0 * dk;

              // 【关键修复】：正确的 Tikhonov 正则化求解 (J * J^T + damp * I) y = J * g_s
              Eigen::Matrix3d J = JT.transpose();
              Eigen::Matrix3d JJT = J * JT; 
              
              // 阻尼加在 J*J^T 的对角线上，这保证了梯度的方向不会被扭曲！
              JJT.diagonal().array() += cover_damp; 

              // 求解 yvec
              Eigen::Vector3d yvec = JJT.ldlt().solve(J * seed_grad_s);

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
        if (!suppress_intermediate_output) {
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
          }
        }

        return energy;
      };

  bool cover_satisfied = false;
  int cover_invalid_seed_count = -1;
  int current_cover_previous_invalid_seed_count = -1;
  int current_cover_same_invalid_seed_count_rounds = 0;
  double last_cover_loss = 0.0;
  int cover_nonzero_loss_rounds = 0;
  bool skip_cover_after_six_nonzero_rounds = false;
  std::function<double(const Eigen::VectorXd &X, Eigen::VectorXd &g)> fgm_cover =
      [&](const Eigen::VectorXd &X, Eigen::VectorXd &g) {
        double lossCover = 0.0;

        startRVD = clock();
        std::vector<Sphere::Sphere> spheres(num);
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

        this->_RVD.calculate_(this->_sites);

        endRVD = clock();
        RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

        const auto &edges = this->_RVD.get_edges_();
        build_spheres_from_rvd(this->_sites, this->_RVD, spheres);

        std::vector<std::vector<int>> nbr(num);
        for (int i = 0; i < (int)edges.size(); ++i) {
          nbr[i].reserve(edges[i].size());
          for (const auto &kv : edges[i]) {
            int j = kv.first;
            if (j < 0 || j >= num || j == i) continue;
            nbr[i].push_back(j);
          }
        }

        const std::vector<Eigen::Vector3i> rdt_tris =
            build_rdt_faces_from_edges(num, edges);

        std::vector<Eigen::Vector3d> C(num);
        std::vector<double> R(num);
        for (int i = 0; i < num; ++i) {
          C[i] = Eigen::Vector3d(this->_sites[i].x(), this->_sites[i].y(), this->_sites[i].z());
          R[i] = std::max(0.0, (double)spheres[i].r);
        }

        g.setZero();
        double maxCoverViolation = 0.0;
        int invalidSeedCount = 0;
        std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());
        #pragma omp parallel
        {
          double localLossCover = 0.0;
          double localMaxCoverViolation = 0.0;
          int localInvalidSeedCount = 0;
          std::vector<Eigen::Vector3d> localGi(num, Eigen::Vector3d::Zero());

          #pragma omp for nowait
          for (int tidx = 0; tidx < (int)rdt_tris.size(); ++tidx) {
            const auto &t = rdt_tris[tidx];
            const int i = t.x();
            const int j = t.y();
            const int k = t.z();
            if (R[i] <= 0 || R[j] <= 0 || R[k] <= 0) continue;

            Eigen::Vector3d s1, s2;
            if (!intersect_three_spheres(C[i], R[i], C[j], R[j], C[k], R[k],
                                         s1, s2)) {
              continue;
            }

            Eigen::Vector3d seeds[2] = {s1, s2};
            for (int si = 0; si < 2; ++si) {
              const Eigen::Vector3d s = seeds[si];
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
              cand.erase(i);
              cand.erase(j);
              cand.erase(k);

              double best_v_for_tracking = 0.0;
              bool has_violation = false;
              Eigen::Vector3d seed_grad_s = Eigen::Vector3d::Zero();

              auto eval_and_accumulate = [&](int l) {
                if (l < 0 || l >= num || l == i || l == j || l == k ||
                    R[l] <= 0) {
                  return;
                }
                Eigen::Vector3d d = (s - C[l]);
                double dist = d.norm();
                if (dist < 1e-12) return;

                double v = (R[l] - cover_margin) - dist;
                best_v_for_tracking = std::max(best_v_for_tracking, v);

                if (v > 0.0) {
                  has_violation = true;

                  double v2 = v * v;
                  double v3 = v2 * v;
                  double dE_dv = 3.0 * cover_weight * v2;

                  localLossCover += cover_weight * v3;

                  Eigen::Vector3d dir = d / dist;
                  seed_grad_s += -dE_dv * dir;
                  localGi[l] += (+dE_dv) * dir;
                }
              };

              for (int l : cand) {
                eval_and_accumulate(l);
              }
              if (cover_global_fallback && !has_violation) {
                for (int l = 0; l < num; ++l) {
                  eval_and_accumulate(l);
                }
              }
              if (!has_violation) continue;

              ++localInvalidSeedCount;
              localMaxCoverViolation =
                  std::max(localMaxCoverViolation, best_v_for_tracking);

              Eigen::Vector3d di = (s - C[i]);
              Eigen::Vector3d dj = (s - C[j]);
              Eigen::Vector3d dk = (s - C[k]);
              Eigen::Matrix3d JT;
              JT.col(0) = 2.0 * di;
              JT.col(1) = 2.0 * dj;
              JT.col(2) = 2.0 * dk;

              Eigen::Matrix3d J = JT.transpose();
              Eigen::Matrix3d JJT = J * JT;
              JJT.diagonal().array() += cover_damp;
              Eigen::Vector3d yvec = JJT.ldlt().solve(J * seed_grad_s);
              localGi[i] += 2.0 * di * yvec(0);
              localGi[j] += 2.0 * dj * yvec(1);
              localGi[k] += 2.0 * dk * yvec(2);
            }
          }

          #pragma omp critical
          {
            lossCover += localLossCover;
            invalidSeedCount += localInvalidSeedCount;
            maxCoverViolation =
                std::max(maxCoverViolation, localMaxCoverViolation);
            for (int i = 0; i < num; ++i) {
              gi[i] += localGi[i];
            }
          }
        }

        for (int i = 0; i < num; ++i) {
          gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
          g(i * 3 + 0) += gi[i].x();
          g(i * 3 + 1) += gi[i].y();
          g(i * 3 + 2) += gi[i].z();
        }

        if (invalidSeedCount == 0 ||
            maxCoverViolation < cover_violation_tolerance) {
          cover_satisfied = true;
          g.setZero();
        } else {
          cover_satisfied = false;
        }
        cover_invalid_seed_count = invalidSeedCount;
        last_cover_loss = lossCover;
        if (cover_invalid_seed_count ==
            current_cover_previous_invalid_seed_count) {
          ++current_cover_same_invalid_seed_count_rounds;
        } else {
          current_cover_previous_invalid_seed_count = cover_invalid_seed_count;
          current_cover_same_invalid_seed_count_rounds = 1;
        }

        std::cout << std::setprecision(7)
                  << "cover-only energy: " << lossCover
                  << " LossCover: " << lossCover
                  << " maxViolation: " << maxCoverViolation
                  << " invalidSeeds: " << invalidSeedCount << std::endl;
        if (current_cover_same_invalid_seed_count_rounds >= 3 &&
            !cover_satisfied) {
          std::cout << "stop current cover-only optimization: invalidSeeds unchanged for 3 evaluations"
                    << " (" << invalidSeedCount << ")" << std::endl;
          g.setZero();
        }
        return lossCover;
      };

  // ======= LBFGS 初始化（原逻辑）=======
  std::vector<Eigen::Vector3d> Pts2 = Pts;
  num = (int)Pts2.size();
  std::cout << Pts2.size() << "  " << num << std::endl;

  _sites.resize(num);
  _para.max_linearsearch = 20;
  Eigen::VectorXd iterX2(num * 3);

  for (int i = 0; i < num; ++i) {
    iterX2(i * 3 + 0) = Pts2[i].x();
    iterX2(i * 3 + 1) = Pts2[i].y();
    iterX2(i * 3 + 2) = Pts2[i].z();
    _sites[i] = BGAL::_Point3(Pts2[i](0), Pts2[i](1), Pts2[i](2));
  }

  _RVD.calculate_(_sites);
  const std::string final_outdir =
      (std::filesystem::current_path() / "data" / "Block").string();
  auto flush_final_outputs = [&]() {
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
    std::vector<Sphere::Sphere> final_spheres;
    build_spheres_from_rvd(_sites, _RVD, final_spheres);

    std::ofstream out(final_outdir + "/Sphere_" + modelname + ".csv",
                      std::ios::out | std::ios::trunc);
    if (!out) {
      std::cerr << "[io] open failed, errno=" << errno << " ("
                << std::strerror(errno) << ")\n";
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

    OutputMesh(_sites, _RVD, num_sites, final_outdir, modelname, 2);
  };

  start = clock();
  const bool unlimited_qem_iterations = (_para.max_iteration == 0);
  const int total_qem_iterations =
      unlimited_qem_iterations ? std::numeric_limits<int>::max()
                               : _para.max_iteration;
  int qem_iterations_done = 0;

  BGAL::_LBFGS::_Parameter warmup_para(_para);
  warmup_para.max_iteration =
      std::min(cvt_qem_warmup_iterations, total_qem_iterations);
  BGAL::_LBFGS warmup_lbfgs(warmup_para);
  qem_iterations_done += warmup_lbfgs.minimize(fgm2, iterX2);

  BGAL::_LBFGS::_Parameter one_step_para(_para);
  one_step_para.max_iteration = 1;
  BGAL::_LBFGS one_step_lbfgs(one_step_para);

  BGAL::_LBFGS::_Parameter cover_para(_para);
  cover_para.max_iteration = 10;
  BGAL::_LBFGS cover_lbfgs(cover_para);

  suppress_intermediate_output = true;
  while (qem_iterations_done < total_qem_iterations) {
    const int qem_step = one_step_lbfgs.minimize(fgm2, iterX2);
    qem_iterations_done += qem_step;

    int cover_step = 0;
    if (skip_cover_after_six_nonzero_rounds) {
      std::cout << "skip cover-only optimization: lossCover stayed nonzero for 6 runs"
                << std::endl;
    } else {
      current_cover_previous_invalid_seed_count = -1;
      current_cover_same_invalid_seed_count_rounds = 0;
      cover_step = cover_lbfgs.minimize(fgm_cover, iterX2);
      if (last_cover_loss > 0.0) {
        ++cover_nonzero_loss_rounds;
        if (cover_nonzero_loss_rounds >= 6) {
          skip_cover_after_six_nonzero_rounds = true;
        }
      } else {
        cover_nonzero_loss_rounds = 0;
      }
    }
    flush_final_outputs();

    if (cover_satisfied) {
      std::cout << "stop alternating optimization: maxViolation < "
                << cover_violation_tolerance << std::endl;
      break;
    }

    if (qem_step == 0 && cover_step == 0) {
      break;
    }

    if (qem_iterations_done >= total_qem_iterations) {
      break;
    }
  }
  end = clock();

  allTime += (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << "allTime: " << allTime << " RVDtime: " << RVDtime
            << " L-BFGS time: " << allTime - RVDtime << std::endl;

  flush_final_outputs();
}

} // namespace BGAL
