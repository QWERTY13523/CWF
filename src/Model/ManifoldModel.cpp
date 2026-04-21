#pragma once
#include "BGAL/Model/ManifoldModel.h"
#include "BGAL/Model/Model_Iterator.h"
#include "BGAL/Model/NonManifoldSurface.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace BGAL
{
  namespace
  {
    static inline std::pair<int, int> undirected_key(const int a, const int b)
    {
      return std::make_pair(std::min(a, b), std::max(a, b));
    }

    static inline void mesh_to_eigen(const std::vector<_Point3> &vertices,
                                     const std::vector<_Model::_MFace> &faces,
                                     Eigen::MatrixXd &V,
                                     Eigen::MatrixXi &F)
    {
      V.resize(static_cast<int>(vertices.size()), 3);
      for (int i = 0; i < V.rows(); ++i)
      {
        V(i, 0) = vertices[static_cast<std::size_t>(i)].x();
        V(i, 1) = vertices[static_cast<std::size_t>(i)].y();
        V(i, 2) = vertices[static_cast<std::size_t>(i)].z();
      }

      F.resize(static_cast<int>(faces.size()), 3);
      for (int i = 0; i < F.rows(); ++i)
      {
        F(i, 0) = faces[static_cast<std::size_t>(i)][0];
        F(i, 1) = faces[static_cast<std::size_t>(i)][1];
        F(i, 2) = faces[static_cast<std::size_t>(i)][2];
      }
    }

    static inline void eigen_to_mesh(const Eigen::MatrixXd &V,
                                     const Eigen::MatrixXi &F,
                                     std::vector<_Point3> &vertices,
                                     std::vector<_Model::_MFace> &faces)
    {
      vertices.clear();
      faces.clear();
      vertices.reserve(static_cast<std::size_t>(V.rows()));
      faces.reserve(static_cast<std::size_t>(F.rows()));

      for (int i = 0; i < V.rows(); ++i)
      {
        vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
      }

      for (int i = 0; i < F.rows(); ++i)
      {
        _Model::_MFace face(F(i, 0), F(i, 1), F(i, 2));
        face.id = i;
        faces.push_back(face);
      }
    }
  } // namespace

  _ManifoldModel::_MMEdge::_MMEdge()
      : _Segment3(),
        _id_left_vertex(-1),
        _id_right_vertex(-1),
        _id_opposite_vertex(-1),
        _id_left_edge(-1),
        _id_right_edge(-1),
        _id_reverse_edge(-1),
        _id_face(-1),
        _is_boundary_placeholder(false)
  {
  }
  _ManifoldModel::_MMEdge::_MMEdge(const _Point3 &in_s, const _Point3 &in_t)
      : _Segment3(in_s, in_t),
        _id_left_vertex(-1),
        _id_right_vertex(-1),
        _id_opposite_vertex(-1),
        _id_left_edge(-1),
        _id_right_edge(-1),
        _id_reverse_edge(-1),
        _id_face(-1),
        _is_boundary_placeholder(false)
  {
  }
  _ManifoldModel::_ManifoldModel()
  {
  }
  _ManifoldModel::_ManifoldModel(const std::string &in_file_name) : _Model()
  {
    int folder_loc = in_file_name.rfind("\\") > in_file_name.rfind("/") ? in_file_name.rfind("\\") : in_file_name.rfind("/");
    int dot_loc = in_file_name.rfind('.');
    _name = in_file_name.substr(folder_loc + 1, dot_loc - folder_loc - 1);
    read_file_(in_file_name);
    const std::vector<_Point3> raw_vertices = _vertices;
    const std::vector<_Model::_MFace> raw_faces = _faces;
    load_with_nonmanifold_support_(raw_vertices, raw_faces);
  }
  _ManifoldModel::_ManifoldModel(const std::vector<_Point3> &in_vertices, const std::vector<_Model::_MFace> &in_faces)
      : _Model()
  {
    load_with_nonmanifold_support_(in_vertices, in_faces);
  }
  _ManifoldModel::_ManifoldModel(const _ManifoldModel &in_mmodel)
  {
    _name = in_mmodel._name;
    _used_nonmanifold_fallback = in_mmodel._used_nonmanifold_fallback;
    assign_raw_mesh_(in_mmodel._vertices, in_mmodel._faces);
  }
  void _ManifoldModel::assign_raw_mesh_(const std::vector<_Point3> &in_vertices,
                                        const std::vector<_Model::_MFace> &in_faces)
  {
    _vertices = in_vertices;
    _faces = in_faces;
    _faces_useless.clear();
    for (int i = 0; i < static_cast<int>(_faces.size()); ++i)
    {
      _faces[static_cast<std::size_t>(i)].id = i;
    }
    compute_normal_boundingbox_();
    preprocess_model_();
  }
  void _ManifoldModel::load_with_nonmanifold_support_(const std::vector<_Point3> &in_vertices,
                                                      const std::vector<_Model::_MFace> &in_faces)
  {
    _used_nonmanifold_fallback = false;
    assign_raw_mesh_(in_vertices, in_faces);

    if (!_has_nonmanifold_topology)
    {
      return;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    mesh_to_eigen(in_vertices, in_faces, V, F);

    const auto prepared = NonManifoldSurface::prepare_surface_with_cgal(V, F);

    std::vector<_Point3> prepared_vertices;
    std::vector<_Model::_MFace> prepared_faces;
    eigen_to_mesh(prepared.V, prepared.F, prepared_vertices, prepared_faces);

    _used_nonmanifold_fallback = true;
    assign_raw_mesh_(prepared_vertices, prepared_faces);

    if (_has_nonmanifold_topology)
    {
      throw std::runtime_error(
          "non-manifold preprocessing finished, but the resulting mesh is still non-manifold in _ManifoldModel.");
    }
  }
  void _ManifoldModel::preprocess_model_()
  {
    creat_edges_from_vertices_faces_();
    arrange_neighs_of_vertex_face_();
  }
  void _ManifoldModel::save_processed_obj_file_(const std::string &out_file_name) const
  {
    save_obj_file_(out_file_name);
  }
  void _ManifoldModel::export_processed_obj_(const std::string &in_file_name,
                                             const std::string &out_file_name)
  {
    _ManifoldModel model(in_file_name);
    model.save_processed_obj_file_(out_file_name);
  }
  _Edge_Iterator _ManifoldModel::edge_begin() const
  {
    return _Edge_Iterator(this, 0);
  }
  _FE_Iterator _ManifoldModel::fe_begin(const int &fid) const
  {
    if (fid < 0 || fid >= number_faces_())
    {
      throw std::runtime_error("Beyond the index!");
    }
    return _FE_Iterator(this, fid, 0);
  }
  _FF_Iterator _ManifoldModel::ff_begin(const int &fid) const
  {
    if (fid < 0 || fid >= number_faces_())
    {
      throw std::runtime_error("Beyond the index!");
    }
    return _FF_Iterator(this, fid, 0);
  }
  _VV_Iterator _ManifoldModel::vv_begin(const int &vid) const
  {
    if (vid < 0 || vid >= number_vertices_())
    {
      throw std::runtime_error("Beyond the index!");
    }
    return _VV_Iterator(this, vid, 0);
  }
  _VE_Iterator _ManifoldModel::ve_begin(const int &vid) const
  {
    if (vid < 0 || vid >= number_vertices_())
    {
      throw std::runtime_error("Beyond the index!");
    }
    return _VE_Iterator(this, vid, 0);
  }
  _VF_Iterator _ManifoldModel::vf_begin(const int &vid) const
  {
    if (vid < 0 || vid >= number_vertices_())
    {
      throw std::runtime_error("Beyond the index!");
    }
    return _VF_Iterator(this, vid, 0);
  }
  void _ManifoldModel::creat_edges_from_vertices_faces_()
  {
    _edges.clear();
    _face_edges.clear();
    _face_edges.resize(_faces.size(), std::array<int, 3>{-1, -1, -1});
    _face_adjacent_faces.clear();
    _face_adjacent_faces.resize(_faces.size(), std::array<int, 3>{-1, -1, -1});
    _boundary_edge_flags.clear();
    _has_nonmanifold_topology = false;

    std::map<std::pair<int, int>, std::vector<int>> directed_edges;
    std::map<std::pair<int, int>, std::vector<int>> undirected_edges;

    for (int i = 0; i < static_cast<int>(_faces.size()); ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        int post = (j + 1) % 3;
        int pre = (j + 2) % 3;

        const int left_vertex = _faces[i][pre];
        const int right_vertex = _faces[i][j];

        _MMEdge e(_vertices[left_vertex], _vertices[right_vertex]);
        e._id_left_vertex = left_vertex;
        e._id_right_vertex = right_vertex;
        e._id_face = i;
        e._id_opposite_vertex = _faces[i][post];
        const int eid = static_cast<int>(_edges.size());
        _edges.push_back(e);
        _face_edges[i][j] = eid;
        directed_edges[std::make_pair(left_vertex, right_vertex)].push_back(eid);
        undirected_edges[undirected_key(left_vertex, right_vertex)].push_back(eid);
      }
    }

    for (const auto &entry : directed_edges)
    {
      if (entry.second.size() > 1)
      {
        _has_nonmanifold_topology = true;
        break;
      }
    }

    for (const auto &entry : undirected_edges)
    {
      if (entry.second.size() > 2)
      {
        _has_nonmanifold_topology = true;
        break;
      }
    }

    _boundary_edge_flags.resize(_edges.size(), false);

    auto choose_best_opposite = [&](const int eid,
                                    const std::vector<int> &candidates) -> int
    {
      int best_eid = -1;
      double best_score = -std::numeric_limits<double>::infinity();
      for (const int candidate_eid : candidates)
      {
        if (candidate_eid == eid)
        {
          continue;
        }
        double score = 0.0;
        if (_edges[candidate_eid]._id_left_vertex == _edges[eid]._id_right_vertex &&
            _edges[candidate_eid]._id_right_vertex == _edges[eid]._id_left_vertex)
        {
          score += 10.0;
        }
        const int face_a = _edges[eid]._id_face;
        const int face_b = _edges[candidate_eid]._id_face;
        if (face_a >= 0 && face_a < static_cast<int>(_normals_face.size()) &&
            face_b >= 0 && face_b < static_cast<int>(_normals_face.size()))
        {
          score += std::abs(_normals_face[face_a].dot_(_normals_face[face_b]));
        }
        if (score > best_score)
        {
          best_score = score;
          best_eid = candidate_eid;
        }
      }
      return best_eid;
    };

    for (int fid = 0; fid < static_cast<int>(_faces.size()); ++fid)
    {
      for (int local_edge = 0; local_edge < 3; ++local_edge)
      {
        const int eid = _face_edges[fid][local_edge];
        const auto edge_key =
            undirected_key(_edges[eid]._id_left_vertex, _edges[eid]._id_right_vertex);
        const std::vector<int> &incident_edges = undirected_edges[edge_key];

        std::vector<int> opposite_edges;
        opposite_edges.reserve(incident_edges.size());
        for (const int other_eid : incident_edges)
        {
          if (other_eid != eid)
          {
            opposite_edges.push_back(other_eid);
          }
        }
        _edges[eid]._id_opposite_edges = opposite_edges;

        if (opposite_edges.empty())
        {
          _MMEdge boundary_edge(_vertices[_edges[eid]._id_right_vertex],
                                _vertices[_edges[eid]._id_left_vertex]);
          boundary_edge._id_left_vertex = _edges[eid]._id_right_vertex;
          boundary_edge._id_right_vertex = _edges[eid]._id_left_vertex;
          boundary_edge._id_face = -1;
          boundary_edge._id_opposite_vertex = -1;
          boundary_edge._id_reverse_edge = eid;
          boundary_edge._is_boundary_placeholder = true;
          const int boundary_eid = static_cast<int>(_edges.size());
          _edges.push_back(boundary_edge);
          _boundary_edge_flags.push_back(true);
          _edges[eid]._id_reverse_edge = boundary_eid;
          _boundary_edge_flags[eid] = true;
          _face_adjacent_faces[fid][local_edge] = -1;
        }
        else
        {
          if (opposite_edges.size() > 1)
          {
            _has_nonmanifold_topology = true;
          }
          const int reverse_eid = choose_best_opposite(eid, opposite_edges);
          _edges[eid]._id_reverse_edge = reverse_eid;
          _face_adjacent_faces[fid][local_edge] =
              reverse_eid >= 0 ? _edges[reverse_eid]._id_face : -1;
        }
      }
    }

    for (int fid = 0; fid < static_cast<int>(_faces.size()); ++fid)
    {
      for (int local_edge = 0; local_edge < 3; ++local_edge)
      {
        const int eid = _face_edges[fid][local_edge];
        _edges[eid]._id_left_edge = _face_edges[fid][(local_edge + 2) % 3];
        _edges[eid]._id_right_edge = _face_edges[fid][(local_edge + 1) % 3];
      }
    }
  }
  void _ManifoldModel::arrange_neighs_of_vertex_face_()
  {
    _neight_edge_of_vertices.clear();
    _neight_edge_of_vertices.resize(_vertices.size(), -1);
    _neigh_edge_of_faces.clear();
    _neigh_edge_of_faces.resize(_faces.size(), -1);
    _degree_of_vertices.clear();
    _degree_of_vertices.resize(_vertices.size(), 0);
    _incident_edges_of_vertices.clear();
    _incident_edges_of_vertices.resize(_vertices.size());
    _incident_faces_of_vertices.clear();
    _incident_faces_of_vertices.resize(_vertices.size());
    _adjacent_vertices_of_vertices.clear();
    _adjacent_vertices_of_vertices.resize(_vertices.size());
    _boundary_vertex_flags.clear();
    _boundary_vertex_flags.resize(_vertices.size(), false);

    std::vector<std::set<int>> incident_faces(_vertices.size());
    std::vector<std::set<int>> adjacent_vertices(_vertices.size());

    for (int fid = 0; fid < static_cast<int>(_faces.size()); ++fid)
    {
      _neigh_edge_of_faces[fid] = _face_edges[fid][0];
      for (int local_edge = 0; local_edge < 3; ++local_edge)
      {
        const int eid = _face_edges[fid][local_edge];
        const int vid = _edges[eid]._id_left_vertex;
        const int nbr = _edges[eid]._id_right_vertex;
        _incident_edges_of_vertices[vid].push_back(eid);
        incident_faces[vid].insert(fid);
        adjacent_vertices[vid].insert(nbr);
        adjacent_vertices[nbr].insert(vid);
        if (_neight_edge_of_vertices[vid] == -1 || _boundary_edge_flags[eid])
        {
          _neight_edge_of_vertices[vid] = eid;
        }
        if (_boundary_edge_flags[eid])
        {
          _boundary_vertex_flags[vid] = true;
          _boundary_vertex_flags[nbr] = true;
        }
      }
    }

    _isolated_vertices.resize(0);
    for (int i = 0; i < static_cast<int>(_vertices.size()); ++i)
    {
      _incident_faces_of_vertices[i].assign(incident_faces[i].begin(), incident_faces[i].end());
      _adjacent_vertices_of_vertices[i].assign(adjacent_vertices[i].begin(),
                                               adjacent_vertices[i].end());
      _degree_of_vertices[i] = static_cast<int>(_incident_faces_of_vertices[i].size());
      if (_neight_edge_of_vertices[i] == -1)
      {
        _isolated_vertices.push_back(i);
      }
      if (_incident_faces_of_vertices[i].size() > _adjacent_vertices_of_vertices[i].size() + 1)
      {
        _has_nonmanifold_topology = true;
      }
    }
  }
} // namespace BGAL
