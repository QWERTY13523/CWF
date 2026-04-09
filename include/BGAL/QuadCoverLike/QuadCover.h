#pragma once
#ifdef _HAS_STD_BYTE
#undef _HAS_STD_BYTE
#endif
#define _HAS_STD_BYTE 0

#include "BGAL/BaseShape/Point.h"
#include "BGAL/Model/ManifoldModel.h"
#include "BGAL/Tessellation3D/Tessellation3D.h"
#include "BGAL/Sphere/Sphere.h"

#include <array>
#include <string>
#include <vector>

namespace BGAL {

class _QuadCover3D {
public:
  struct _Parameter {
    bool is_show = true;
    bool export_initial_state = true;
    bool export_each_iteration = false;
    int export_interval = 50;
    bool use_cwf_warm_start = true;
    bool show_cwf_progress = true;
    int cwf_max_iterations = 50;
    int max_outer_iterations = 30;
    int max_line_search = 10;
    double active_eps = 1e-8;
    double accept_eps = 1e-12;
    double hinge_lambda = 1.0;
    double step_cap_scale = 0.02;
    double fallback_step_cap_scale = 1e-3;
    double min_step_scale = 1e-6;
    int max_incident_cells_per_rvd_vertex = 8;
  };

  struct _Quad {
    std::array<int, 4> ids{};
  };

  struct _IterationInfo {
    int iteration = 0;
    int num_quads = 0;
    int active_quads = 0;
    double min_margin = 0.0;
    double accepted_step = 0.0;
  };

public:
  explicit _QuadCover3D(const _ManifoldModel &model);
  _QuadCover3D(const _ManifoldModel &model, const _Parameter &para);

  void set_outpath(const std::string &path) { outpath = path; }

  void calculate_(const std::vector<_Point3> &init_sites,
                  const std::string &model_name = "model");
  void calculate_(int site_num, char *modelNamee, char *pointsName = nullptr);

  const std::vector<_Point3> &get_sites() const { return _sites; }
  const _Restricted_Tessellation3D &get_RVD() const { return _RVD; }
  const std::vector<Sphere::Sphere> &get_spheres() const { return _spheres; }
  const std::vector<_Quad> &get_quads() const { return _quads; }
  const std::vector<_IterationInfo> &get_history() const { return _history; }

private:
  std::string outpath{};

private:
  const _ManifoldModel &_model;
  _Restricted_Tessellation3D _RVD;
  std::vector<_Point3> _sites{};
  std::vector<Sphere::Sphere> _spheres{};
  std::vector<_Quad> _quads{};
  std::vector<_IterationInfo> _history{};
  _Parameter _para;
};

} // namespace BGAL
