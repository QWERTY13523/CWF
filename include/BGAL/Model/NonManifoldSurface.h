#pragma once

#include "BGAL/Model/ManifoldModel.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Linear_cell_complex_for_combinatorial_map.h>
#include <CGAL/Linear_cell_complex_incremental_builder_3.h>
#include <CGAL/Linear_cell_complex_traits.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace BGAL {

namespace NonManifoldSurface {

namespace PMP = CGAL::Polygon_mesh_processing;

using CgalKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using CgalPoint3 = CgalKernel::Point_3;
using CgalLccTraits = CGAL::Linear_cell_complex_traits<3, CgalKernel>;
using CgalLcc = CGAL::Linear_cell_complex_for_combinatorial_map<2, 3, CgalLccTraits>;
using CgalLccBuilder = CGAL::Linear_cell_complex_incremental_builder_3<CgalLcc>;
using SoupTriangle = std::array<std::size_t, 3>;

struct UndirectedEdgeKey {
  std::size_t a = 0;
  std::size_t b = 0;

  bool operator==(const UndirectedEdgeKey& other) const {
    return a == other.a && b == other.b;
  }
};

struct UndirectedEdgeKeyHash {
  std::size_t operator()(const UndirectedEdgeKey& key) const {
    return std::hash<std::size_t>{}(key.a) ^
           (std::hash<std::size_t>{}(key.b) << 1);
  }
};

struct EdgeIncident {
  int face_id = -1;
  std::size_t src = 0;
  std::size_t dst = 0;
};

struct PreparedTriangleMesh {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::size_t input_vertex_count = 0;
  std::size_t input_face_count = 0;
  std::size_t output_vertex_count = 0;
  std::size_t output_face_count = 0;
  bool non_manifold_edges_duplicated_without_growth = true;
  bool orientation_fixed_without_growth = true;
  bool polygon_mesh_check_passed = true;
};

static inline bool triangle_has_duplicate_indices(const SoupTriangle& tri) {
  return tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2];
}

static inline bool is_degenerate_triangle(const CgalPoint3& a,
                                          const CgalPoint3& b,
                                          const CgalPoint3& c) {
  const auto ab = b - a;
  const auto ac = c - a;
  const auto area2 = CGAL::cross_product(ab, ac).squared_length();
  return !(area2 > 0.0);
}

static inline void filter_valid_triangles(const std::vector<CgalPoint3>& points,
                                          std::vector<SoupTriangle>& triangles) {
  std::vector<SoupTriangle> filtered;
  filtered.reserve(triangles.size());
  for (const auto& tri : triangles) {
    if (triangle_has_duplicate_indices(tri)) {
      continue;
    }
    if (tri[0] >= points.size() || tri[1] >= points.size() || tri[2] >= points.size()) {
      continue;
    }
    if (is_degenerate_triangle(points[tri[0]], points[tri[1]], points[tri[2]])) {
      continue;
    }
    filtered.push_back(tri);
  }
  triangles.swap(filtered);
}

static inline void sanitize_triangle_soup(std::vector<CgalPoint3>& points,
                                          std::vector<SoupTriangle>& triangles) {
  PMP::repair_polygon_soup(points, triangles);
  filter_valid_triangles(points, triangles);
}

static inline void split_face_along_edge(std::vector<CgalPoint3>& points,
                                         std::vector<SoupTriangle>& triangles,
                                         int face_id,
                                         std::size_t edge_src,
                                         std::size_t edge_dst) {
  if (face_id < 0 || face_id >= static_cast<int>(triangles.size())) {
    throw std::runtime_error(
        "invalid face id while splitting a residual non-manifold edge.");
  }
  SoupTriangle& tri = triangles[static_cast<std::size_t>(face_id)];

  bool replaced_src = false;
  bool replaced_dst = false;
  for (std::size_t corner = 0; corner < 3; ++corner) {
    if (tri[corner] == edge_src) {
      tri[corner] = points.size();
      points.push_back(points[edge_src]);
      replaced_src = true;
    } else if (tri[corner] == edge_dst) {
      tri[corner] = points.size();
      points.push_back(points[edge_dst]);
      replaced_dst = true;
    }
  }

  if (!replaced_src || !replaced_dst) {
    throw std::runtime_error(
        "failed to find both edge endpoints while splitting a residual non-manifold edge.");
  }
}

static inline bool split_residual_nonmanifold_edges(std::vector<CgalPoint3>& points,
                                                    std::vector<SoupTriangle>& triangles) {
  bool modified = false;

  while (true) {
    std::unordered_map<UndirectedEdgeKey, std::vector<EdgeIncident>, UndirectedEdgeKeyHash>
        incidences;

    for (int face_id = 0; face_id < static_cast<int>(triangles.size()); ++face_id) {
      const SoupTriangle& tri = triangles[static_cast<std::size_t>(face_id)];
      for (int e = 0; e < 3; ++e) {
        const std::size_t src = tri[static_cast<std::size_t>(e)];
        const std::size_t dst = tri[static_cast<std::size_t>((e + 1) % 3)];
        const UndirectedEdgeKey key{std::min(src, dst), std::max(src, dst)};
        incidences[key].push_back(EdgeIncident{face_id, src, dst});
      }
    }

    bool split_one = false;
    for (const auto& entry : incidences) {
      const auto& group = entry.second;
      const bool too_many_faces = group.size() > 2;
      const bool inconsistent_pair =
          group.size() == 2 &&
          !(group[0].src == group[1].dst && group[0].dst == group[1].src);

      if (!(too_many_faces || inconsistent_pair)) {
        continue;
      }

      split_face_along_edge(points, triangles, group.back().face_id,
                            group.back().src, group.back().dst);
      modified = true;
      split_one = true;
      break;
    }

    if (!split_one) {
      break;
    }
  }

  return modified;
}

static inline _ManifoldModel build_manifold_model(const Eigen::MatrixXd& V,
                                                  const Eigen::MatrixXi& F) {
  std::vector<_Point3> vertices;
  vertices.reserve(V.rows());
  for (int i = 0; i < V.rows(); ++i) {
    vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  }

  std::vector<_Model::_MFace> faces;
  faces.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i) {
    if (F.cols() != 3) {
      throw std::runtime_error("surface mesh is not triangulated.");
    }
    faces.emplace_back(F(i, 0), F(i, 1), F(i, 2));
  }
  return _ManifoldModel(vertices, faces);
}

static inline PreparedTriangleMesh prepare_surface_with_cgal(const Eigen::MatrixXd& V,
                                                             const Eigen::MatrixXi& F) {
  if (F.cols() != 3) {
    throw std::runtime_error("surface mesh is not triangulated.");
  }

  PreparedTriangleMesh prepared;
  prepared.input_vertex_count = static_cast<std::size_t>(V.rows());
  prepared.input_face_count = static_cast<std::size_t>(F.rows());

  std::vector<CgalPoint3> points;
  points.reserve(static_cast<std::size_t>(V.rows()));
  for (int i = 0; i < V.rows(); ++i) {
    points.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  }

  std::vector<SoupTriangle> triangles;
  triangles.reserve(static_cast<std::size_t>(F.rows()));
  for (int i = 0; i < F.rows(); ++i) {
    triangles.push_back({
        static_cast<std::size_t>(F(i, 0)),
        static_cast<std::size_t>(F(i, 1)),
        static_cast<std::size_t>(F(i, 2)),
    });
  }

  sanitize_triangle_soup(points, triangles);

  const std::size_t before_duplicate_points = points.size();
  prepared.non_manifold_edges_duplicated_without_growth =
      PMP::duplicate_non_manifold_edges_in_polygon_soup(points, triangles);
  if (points.size() < before_duplicate_points) {
    throw std::runtime_error(
        "CGAL duplicated non-manifold edges but reduced the point count unexpectedly.");
  }

  const std::size_t before_orient_points = points.size();
  prepared.orientation_fixed_without_growth = PMP::orient_polygon_soup(points, triangles);
  if (points.size() < before_orient_points) {
    throw std::runtime_error(
        "CGAL oriented the polygon soup but reduced the point count unexpectedly.");
  }

  split_residual_nonmanifold_edges(points, triangles);
  filter_valid_triangles(points, triangles);

  if (triangles.empty()) {
    throw std::runtime_error(
        "surface mesh became empty after CGAL non-manifold preprocessing.");
  }
  prepared.polygon_mesh_check_passed = PMP::is_polygon_soup_a_polygon_mesh(triangles);

  CgalLcc lcc;
  CgalLccBuilder builder(lcc);
  builder.begin_surface();
  for (const auto& p : points) {
    builder.add_vertex(p);
  }
  for (const auto& tri : triangles) {
    builder.begin_facet();
    builder.add_vertex_to_facet(tri[0]);
    builder.add_vertex_to_facet(tri[1]);
    builder.add_vertex_to_facet(tri[2]);
    builder.end_facet();
  }
  builder.end_surface();

  prepared.output_vertex_count = points.size();
  prepared.output_face_count = triangles.size();
  prepared.V.resize(static_cast<int>(points.size()), 3);
  prepared.F.resize(static_cast<int>(triangles.size()), 3);

  for (int i = 0; i < prepared.V.rows(); ++i) {
    prepared.V(i, 0) = points[static_cast<std::size_t>(i)].x();
    prepared.V(i, 1) = points[static_cast<std::size_t>(i)].y();
    prepared.V(i, 2) = points[static_cast<std::size_t>(i)].z();
  }
  for (int i = 0; i < prepared.F.rows(); ++i) {
    prepared.F(i, 0) = static_cast<int>(triangles[static_cast<std::size_t>(i)][0]);
    prepared.F(i, 1) = static_cast<int>(triangles[static_cast<std::size_t>(i)][1]);
    prepared.F(i, 2) = static_cast<int>(triangles[static_cast<std::size_t>(i)][2]);
  }

  return prepared;
}

static inline _ManifoldModel build_manifold_model_allow_non_manifold(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    PreparedTriangleMesh* prepared = nullptr,
    bool* used_cgal_fallback = nullptr) {
  _ManifoldModel model(V.rows() == 0 ? std::vector<_Point3>{} : [&]() {
      std::vector<_Point3> vertices;
      vertices.reserve(V.rows());
      for (int i = 0; i < V.rows(); ++i) {
        vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
      }
      return vertices;
    }(), [&]() {
      std::vector<_Model::_MFace> faces;
      faces.reserve(F.rows());
      for (int i = 0; i < F.rows(); ++i) {
        faces.emplace_back(F(i, 0), F(i, 1), F(i, 2));
      }
      return faces;
    }());
  if (used_cgal_fallback != nullptr) {
    *used_cgal_fallback = model.used_nonmanifold_fallback_();
  }
  if (prepared != nullptr) {
    prepared->input_vertex_count = static_cast<std::size_t>(V.rows());
    prepared->input_face_count = static_cast<std::size_t>(F.rows());
    prepared->output_vertex_count = static_cast<std::size_t>(model.number_vertices_());
    prepared->output_face_count = static_cast<std::size_t>(model.number_faces_());
    prepared->V.resize(model.number_vertices_(), 3);
    prepared->F.resize(model.number_faces_(), 3);
    for (int i = 0; i < model.number_vertices_(); ++i) {
      prepared->V(i, 0) = model.vertex_(i).x();
      prepared->V(i, 1) = model.vertex_(i).y();
      prepared->V(i, 2) = model.vertex_(i).z();
    }
    for (int i = 0; i < model.number_faces_(); ++i) {
      prepared->F(i, 0) = model.face_(i)[0];
      prepared->F(i, 1) = model.face_(i)[1];
      prepared->F(i, 2) = model.face_(i)[2];
    }
  }
  return model;
}

static inline void save_prepared_triangle_mesh_as_obj(const PreparedTriangleMesh& prepared,
                                                      const std::string& out_file_name) {
  std::ofstream out(out_file_name);
  if (!out) {
    throw std::runtime_error("failed to open output OBJ file: " + out_file_name);
  }
  for (int i = 0; i < prepared.V.rows(); ++i) {
    out << "v " << prepared.V(i, 0) << " " << prepared.V(i, 1) << " " << prepared.V(i, 2) << "\n";
  }
  for (int i = 0; i < prepared.F.rows(); ++i) {
    out << "f " << prepared.F(i, 0) + 1 << " " << prepared.F(i, 1) + 1 << " " << prepared.F(i, 2) + 1 << "\n";
  }
}

static inline std::string format_preprocess_summary(const PreparedTriangleMesh& prepared,
                                                    const std::string& prefix) {
  std::ostringstream out;
  out << prefix << " CGAL non-manifold preprocessing"
      << " | V: " << prepared.input_vertex_count
      << " -> " << prepared.output_vertex_count
      << " | F: " << prepared.input_face_count
      << " -> " << prepared.output_face_count
      << " | duplicate_nm_edges="
      << (prepared.non_manifold_edges_duplicated_without_growth ? "no_new_points"
                                                                : "duplicated_points")
      << " | orient_soup="
      << (prepared.orientation_fixed_without_growth ? "no_new_points"
                                                    : "duplicated_points")
      << " | polygon_mesh_check="
      << (prepared.polygon_mesh_check_passed ? "pass" : "soft_fail");
  return out.str();
}

}  // namespace NonManifoldSurface

}  // namespace BGAL
