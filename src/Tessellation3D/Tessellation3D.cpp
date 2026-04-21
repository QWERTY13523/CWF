#include "BGAL/Tessellation3D/Side3D.h"
#include "BGAL/Tessellation3D/Tessellation3D.h"

#include <sstream>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cmath>
namespace BGAL
{
  namespace
  {
    inline _Point3 face_plane_normal_from_vertices(const _ManifoldModel& model,
                                                   const int v0_id,
                                                   const int v1_id,
                                                   const int v2_id,
                                                   const int fallback_face_id = -1)
    {
      const _Point3& p0 = model.vertex_(v0_id);
      const _Point3& p1 = model.vertex_(v1_id);
      const _Point3& p2 = model.vertex_(v2_id);
      _Point3 n = (p1 - p0).cross_(p2 - p0);
      if (n.length_() > 1e-20)
      {
        n.normalized_();
        return n;
      }

      if (fallback_face_id >= 0 && fallback_face_id < model.number_faces_())
      {
        _Point3 fallback = model.normal_face_(fallback_face_id);
        fallback.normalized_();
        return fallback;
      }

      throw std::runtime_error("degenerate face plane in _Restricted_Tessellation3D");
    }


    struct FaceSeedData
    {
      std::array<int, 3> vids{};
      std::array<int, 3> adj_faces{};
      _Point3 center;
    };

    struct SymbolKey
    {
      std::array<int, 6> data{};
      std::uint8_t size = 0;

      bool operator==(const SymbolKey& other) const
      {
        if (size != other.size)
        {
          return false;
        }
        for (std::uint8_t i = 0; i < size; ++i)
        {
          if (data[i] != other.data[i])
          {
            return false;
          }
        }
        return true;
      }
    };

    struct SymbolKeyHash
    {
      std::size_t operator()(const SymbolKey& key) const noexcept
      {
        std::size_t h = static_cast<std::size_t>(key.size);
        for (std::uint8_t i = 0; i < key.size; ++i)
        {
          h ^= std::hash<int>{}(key.data[i]) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        }
        return h;
      }
    };
  }

  _Tessellation3D_Skeleton::_Tessellation3D_Skeleton()
  {
    _neights.resize(0);
  }
  _Tessellation3D_Skeleton::_Tessellation3D_Skeleton(Rt &rt, const int &num_vertices)
  {
    _neights.resize(num_vertices);
    _neight_vecs.resize(num_vertices);
    for (auto c_it = rt.finite_cells_begin(); c_it != rt.finite_cells_end(); c_it++)
    {
      int i0, i1, i2, i3;
      i0 = c_it->vertex(0)->info();
      i1 = c_it->vertex(1)->info();
      i2 = c_it->vertex(2)->info();
      i3 = c_it->vertex(3)->info();
      if (i0 != -1)
      {
        if (i1 != -1)
        {
          _neights[i0].insert(i1);
          _neights[i1].insert(i0);
        }
        if (i2 != -1)
        {
          _neights[i0].insert(i2);
          _neights[i2].insert(i0);
        }
        if (i3 != -1)
        {
          _neights[i0].insert(i3);
          _neights[i3].insert(i0);
        }
      }
      if (i1 != -1)
      {
        if (i2 != -1)
        {
          _neights[i1].insert(i2);
          _neights[i2].insert(i1);
        }
        if (i3 != -1)
        {
          _neights[i3].insert(i1);
          _neights[i1].insert(i3);
        }
      }
      if (i2 != -1 && i3 != -1)
      {
        _neights[i3].insert(i2);
        _neights[i2].insert(i3);
      }
    }
    for (int i = 0; i < num_vertices; ++i)
    {
      _neight_vecs[i].assign(_neights[i].begin(), _neights[i].end());
    }
  }

  _BOC::_Sign _Restricted_Tessellation3D::side_(const int &ip1, const int &ip2, _Symbolic_Point &v)
  {
    const _Point3 &p1 = _sites[ip1];
    const _Point3 &p2 = _sites[ip2];
    const auto format_symbolic_point = [&v]() -> std::string
    {
      std::ostringstream out;
      out << "flag=" << v.flag << " p=[";
      for (std::size_t i = 0; i < v.p.size(); ++i)
      {
        if (i > 0)
        {
          out << ",";
        }
        out << v.p[i];
      }
      out << "] sym={";
      bool first = true;
      for (const int s : v._sym)
      {
        if (!first)
        {
          out << ",";
        }
        out << s;
        first = false;
      }
      out << "}";
      return out.str();
    };
    _BOC::_Sign res;
    switch (v.flag)
    {
    case 1:
    {
      if (v.p.size() < 1 || v.p[0] < 0 || v.p[0] >= _model.number_vertices_())
      {
        throw std::runtime_error("invalid symbolic point (flag=1): " +
                                 format_symbolic_point() +
                                 " vertices=" + std::to_string(_model.number_vertices_()) +
                                 " sites=" + std::to_string(_sites.size()));
      }
      _Point3 q = _model.vertex_(v.p[0]);
      res = _Side3D::side1_(p1.x(),
                            p1.y(),
                            p1.z(),
                            p2.x(),
                            p2.y(),
                            p2.z(),
                            _weights[ip1],
                            _weights[ip2],
                            q.x(),
                            q.y(),
                            q.z());
      break;
    }
    case 2:
    {
      if (v.p.size() < 3 || v.p[0] < 0 || v.p[0] >= _model.number_vertices_() ||
          v.p[1] < 0 || v.p[1] >= _model.number_vertices_() ||
          v.p[2] < 0 || v.p[2] >= static_cast<int>(_sites.size()))
      {
        throw std::runtime_error("invalid symbolic point (flag=2): " +
                                 format_symbolic_point() +
                                 " vertices=" + std::to_string(_model.number_vertices_()) +
                                 " sites=" + std::to_string(_sites.size()));
      }
      _Point3 q1 = _model.vertex_(v.p[0]);
      _Point3 q2 = _model.vertex_(v.p[1]);
      const _Point3 &p3 = _sites[v.p[2]];
      res = _Side3D::side2_(p1.x(),
                            p1.y(),
                            p1.z(),
                            p2.x(),
                            p2.y(),
                            p2.z(),
                            p3.x(),
                            p3.y(),
                            p3.z(),
                            _weights[ip1],
                            _weights[ip2],
                            _weights[v.p[2]],
                            q1.x(),
                            q1.y(),
                            q1.z(),
                            q2.x(),
                            q2.y(),
                            q2.z());
      break;
    }
    case 3:
    {
      if (v.p.size() < 3 || v.p[0] < 0 || v.p[0] >= _model.number_vertices_() ||
          v.p[1] < 0 || v.p[1] >= _model.number_vertices_() ||
          v.p[2] < 0 || v.p[2] >= _model.number_vertices_())
      {
        throw std::runtime_error("invalid symbolic point (flag=3): " +
                                 format_symbolic_point() +
                                 " vertices=" + std::to_string(_model.number_vertices_()) +
                                 " sites=" + std::to_string(_sites.size()));
      }
      _Point3 q1 = _model.vertex_(v.p[0]);
      _Point3 q2 = _model.vertex_(v.p[1]);
      _Point3 q3 = _model.vertex_(v.p[2]);
      std::vector<int> ip34;
      for (auto it = v._sym.begin(); it != v._sym.end(); ++it)
      {
        if (*it > 0)
        {
          ip34.push_back(*it);
        }
      }
      const int &ip3 = ip34[0] - 1;
      const int &ip4 = ip34[1] - 1;
      const _Point3 &p3 = _sites[ip3];
      const _Point3 &p4 = _sites[ip4];
      res = _Side3D::side3_(p1.x(),
                            p1.y(),
                            p1.z(),
                            p2.x(),
                            p2.y(),
                            p2.z(),
                            p3.x(),
                            p3.y(),
                            p3.z(),
                            p4.x(),
                            p4.y(),
                            p4.z(),
                            _weights[ip1],
                            _weights[ip2],
                            _weights[ip3],
                            _weights[ip4],
                            q1.x(),
                            q1.y(),
                            q1.z(),
                            q2.x(),
                            q2.y(),
                            q2.z(),
                            q3.x(),
                            q3.y(),
                            q3.z());
      break;
    }
    default:
      throw std::runtime_error("flag error");
      break;
    }
    return res;
  }

  _Restricted_Tessellation3D::_Symbolic_Point _Restricted_Tessellation3D::insec_bisector_(const _Symbolic_Point &p1,
                                                                                          const _Symbolic_Point &p2,
                                                                                          const int &neigh,
                                                                                          const int &center,
                                                                                          const std::array<int, 3> &current_face_vid) const
  {
    std::set<int> insec = p1.insec_(p2);
    insec.insert(neigh + 1);
    _Symbolic_Point insec_p(insec, center);
    const auto is_valid_face_vertex = [&](const int vid) -> bool
    {
      return vid >= 0 &&
             vid < _model.number_vertices_() &&
             (vid == current_face_vid[0] || vid == current_face_vid[1] || vid == current_face_vid[2]);
    };
    const auto extract_pair_from_point = [&](const _Symbolic_Point &point,
                                             std::array<int, 2> &pair) -> bool
    {
      std::vector<int> valid_vertices;
      for (const int vid : point.p)
      {
        if (!is_valid_face_vertex(vid))
        {
          continue;
        }
        if (std::find(valid_vertices.begin(), valid_vertices.end(), vid) == valid_vertices.end())
        {
          valid_vertices.push_back(vid);
        }
      }
      if (valid_vertices.size() < 2)
      {
        return false;
      }
      pair[0] = valid_vertices[0];
      pair[1] = valid_vertices[1];
      return true;
    };
    if (insec_p.num_sites_() == 1)
    {
      std::array<int, 2> support_edge = {-1, -1};
      bool found_support_edge = false;

      if (p1.num_sites_() == 0 && p2.num_sites_() == 0 &&
          !p1.p.empty() && !p2.p.empty() &&
          is_valid_face_vertex(p1.p[0]) && is_valid_face_vertex(p2.p[0]) &&
          p1.p[0] != p2.p[0])
      {
        support_edge[0] = p1.p[0];
        support_edge[1] = p2.p[0];
        found_support_edge = true;
      }
      if (!found_support_edge && p2.num_sites_() == 1)
      {
        found_support_edge = extract_pair_from_point(p2, support_edge);
      }
      if (!found_support_edge && p1.num_sites_() == 1)
      {
        found_support_edge = extract_pair_from_point(p1, support_edge);
      }
      if (!found_support_edge)
      {
        std::vector<int> merged_vertices;
        for (const int vid : p1.p)
        {
          if (is_valid_face_vertex(vid) &&
              std::find(merged_vertices.begin(), merged_vertices.end(), vid) == merged_vertices.end())
          {
            merged_vertices.push_back(vid);
          }
        }
        for (const int vid : p2.p)
        {
          if (is_valid_face_vertex(vid) &&
              std::find(merged_vertices.begin(), merged_vertices.end(), vid) == merged_vertices.end())
          {
            merged_vertices.push_back(vid);
          }
        }
        if (merged_vertices.size() >= 2)
        {
          support_edge[0] = merged_vertices[0];
          support_edge[1] = merged_vertices[1];
          found_support_edge = true;
        }
      }
      if (!found_support_edge)
      {
        throw std::runtime_error("failed to derive a valid face edge for symbolic point");
      }

      insec_p.flag = 2;
      insec_p.p.clear();
      insec_p.p.push_back(support_edge[0]);
      insec_p.p.push_back(support_edge[1]);
      insec_p.p.push_back(neigh);
    }
    else
    {
      insec_p.flag = 3;
      insec_p.p.clear();
      insec_p.p.push_back(current_face_vid[0]);
      insec_p.p.push_back(current_face_vid[1]);
      insec_p.p.push_back(current_face_vid[2]);
    }
    //std::cout << "insec:  ";
    //for (auto tit = insec_p._sym.begin(); tit != insec_p._sym.end(); ++tit)
    //{
    //std::cout << *tit << " ";
    //}
    //std::cout << std::endl;
    return insec_p;
  }

  void _Restricted_Tessellation3D::calculate_()
  {
    std::vector<std::pair<Weighted_point, int>> wps(_num_sites);
    const double min_weight = *(std::min_element(_weights.begin(), _weights.end()));
    for (int i = 0; i < _num_sites; ++i)
    {
      wps[i] = std::make_pair(Weighted_point(Point(_sites[i].x(), _sites[i].y(), _sites[i].z()), _weights[i]), i);
    }

    const std::pair<_Point3, _Point3> bbox = _model.bounding_box_();
    const _Point3 &min_p = bbox.first;
    const _Point3 &max_p = bbox.second;
    wps.push_back(std::make_pair(Weighted_point(Point(3 * min_p.x() - 2 * max_p.x(),
                                                      3 * min_p.y() - 2 * max_p.y(),
                                                      3 * min_p.z() - 2 * max_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * max_p.x() - 2 * min_p.x(),
                                                      3 * min_p.y() - 2 * max_p.y(),
                                                      3 * min_p.z() - 2 * max_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * max_p.x() - 2 * min_p.x(),
                                                      3 * max_p.y() - 2 * min_p.y(),
                                                      3 * min_p.z() - 2 * max_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * min_p.x() - 2 * max_p.x(),
                                                      3 * max_p.y() - 2 * min_p.y(),
                                                      3 * min_p.z() - 2 * max_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * min_p.x() - 2 * max_p.x(),
                                                      3 * min_p.y() - 2 * max_p.y(),
                                                      3 * max_p.z() - 2 * min_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * max_p.x() - 2 * min_p.x(),
                                                      3 * min_p.y() - 2 * max_p.y(),
                                                      3 * max_p.z() - 2 * min_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * max_p.x() - 2 * min_p.x(),
                                                      3 * max_p.y() - 2 * min_p.y(),
                                                      3 * max_p.z() - 2 * min_p.z()),
                                                min_weight),
                                 -1));
    wps.push_back(std::make_pair(Weighted_point(Point(3 * min_p.x() - 2 * max_p.x(),
                                                      3 * max_p.y() - 2 * min_p.y(),
                                                      3 * max_p.z() - 2 * min_p.z()),
                                                min_weight),
                                 -1));

    Rt rt(wps.begin(), wps.end());
    _skeleton = _Tessellation3D_Skeleton(rt, _num_sites);

    const int num_faces = _model.number_faces_();
    std::vector<char> face_is_visited(num_faces, 0);
    std::vector<FaceSeedData> face_cache(num_faces);
    for (_Face_Iterator f_it = _model.face_begin(); f_it != _model.face_end(); ++f_it)
    {
      const int fid = f_it.id();
      FaceSeedData data;
      data.center = _Point3(0, 0, 0);
      int edge_slot = 0;
      for (auto fe_it = _model.fe_begin(fid); fe_it != _model.fe_end(fid); ++fe_it, ++edge_slot)
      {
        const int cv = (*fe_it)._id_left_vertex;
        data.vids[edge_slot] = cv;
        const int reverse_eid = (*fe_it)._id_reverse_edge;
        if (reverse_eid < 0 || reverse_eid >= _model.number_edges_())
        {
          std::ostringstream out;
          out << "invalid reverse edge while caching face data"
              << " | fid=" << fid
              << " | eid=" << fe_it.id()
              << " | reverse_eid=" << reverse_eid
              << " | lv=" << (*fe_it)._id_left_vertex
              << " | rv=" << (*fe_it)._id_right_vertex
              << " | opposite_count=" << (*fe_it)._id_opposite_edges.size();
          throw std::runtime_error(out.str());
        }
        data.adj_faces[edge_slot] = _model.edge_(reverse_eid)._id_face;
        data.center = data.center + _model.vertex_(cv);
      }
      data.center = data.center / 3.0;
      face_cache[fid] = data;
    }

    std::vector<std::unordered_map<int, int>> from_idx_to_locations(_num_sites);
    for (int i = 0; i < _num_sites; ++i)
    {
      from_idx_to_locations[i].reserve(16);
    }
    std::vector<int> from_locations_to_idx;
    from_locations_to_idx.reserve(std::max(32, num_faces * 2));
    std::vector<std::vector<_Symbolic_Point>> cliped_faces;
    cliped_faces.reserve(std::max(32, num_faces * 2));
    std::queue<std::pair<int, int>> Qfs;

    const auto make_seed_polygon = [&](const int face_id, const int site_id) -> std::vector<_Symbolic_Point>
    {
      const FaceSeedData &fd = face_cache[face_id];
      std::vector<_Symbolic_Point> fps;
      fps.reserve(3);

      _Symbolic_Point fp1(-face_id - 1, -fd.adj_faces[0] - 1, -fd.adj_faces[2] - 1, site_id);
      fp1.flag = 1;
      fp1.p.push_back(fd.vids[0]);

      _Symbolic_Point fp2(-face_id - 1, -fd.adj_faces[0] - 1, -fd.adj_faces[1] - 1, site_id);
      fp2.flag = 1;
      fp2.p.push_back(fd.vids[1]);

      _Symbolic_Point fp3(-face_id - 1, -fd.adj_faces[1] - 1, -fd.adj_faces[2] - 1, site_id);
      fp3.flag = 1;
      fp3.p.push_back(fd.vids[2]);

      fps.push_back(std::move(fp1));
      fps.push_back(std::move(fp2));
      fps.push_back(std::move(fp3));
      return fps;
    };

    const auto enqueue_face_for_site = [&](const int site_id, const int face_id)
    {
      auto &site_map = from_idx_to_locations[site_id];
      if (site_map.find(face_id) != site_map.end())
      {
        return;
      }
      const int loc = static_cast<int>(cliped_faces.size());
      site_map.emplace(face_id, loc);
      from_locations_to_idx.push_back(face_id);
      cliped_faces.push_back(make_seed_polygon(face_id, site_id));
      Qfs.emplace(site_id, loc);
    };

    for (_Face_Iterator f_it = _model.face_begin(); f_it != _model.face_end(); ++f_it)
    {
      const int seed_face_id = f_it.id();
      if (face_is_visited[seed_face_id])
      {
        continue;
      }

      const FaceSeedData &seed_face = face_cache[seed_face_id];
      const int near_site = rt.nearest_power_vertex(K::Point_3(seed_face.center.x(), seed_face.center.y(), seed_face.center.z()))->info();
      enqueue_face_for_site(near_site, seed_face_id);

      while (!Qfs.empty())
      {
        const int current_site = Qfs.front().first;
        const int current_cliped = Qfs.front().second;
        Qfs.pop();

        const int current_face = from_locations_to_idx[current_cliped];
        std::vector<_Symbolic_Point> old_cliped = cliped_faces[current_cliped];
        const std::vector<_Symbolic_Point> original_cliped = cliped_faces[current_cliped];
        const std::array<int, 3> &current_face_vid = face_cache[current_face].vids;
        std::vector<_Symbolic_Point> update_cliped;
        const std::vector<int> &planes = _skeleton.neights_(current_site);

        for (const int plane_site : planes)
        {
          if (old_cliped.empty())
          {
            break;
          }

          update_cliped.clear();
          update_cliped.reserve(old_cliped.size() + 2);

          _BOC::_Sign pre_state = side_(current_site, plane_site, old_cliped[0]);
          _BOC::_Sign cur_state = side_(current_site, plane_site, old_cliped[1]);
          _BOC::_Sign nex_state;

          if (pre_state == _BOC::_Sign::PositivE)
          {
            update_cliped.push_back(old_cliped[0]);
            if (cur_state == _BOC::_Sign::PositivE)
            {
              update_cliped.push_back(old_cliped[1]);
            }
            else if (cur_state == _BOC::_Sign::NegativE)
            {
              update_cliped.push_back(insec_bisector_(old_cliped[0], old_cliped[1], plane_site, current_site, current_face_vid));
            }
            else if (cur_state == _BOC::_Sign::ZerO)
            {
              _Symbolic_Point updateSym = old_cliped[1];
              updateSym.update_(plane_site, old_cliped[0]);
              update_cliped.push_back(std::move(updateSym));
            }
          }
          else if (pre_state == _BOC::_Sign::NegativE)
          {
            if (cur_state == _BOC::_Sign::PositivE)
            {
              update_cliped.push_back(insec_bisector_(old_cliped[0], old_cliped[1], plane_site, current_site, current_face_vid));
              update_cliped.push_back(old_cliped[1]);
            }
          }
          else if (pre_state == _BOC::_Sign::ZerO)
          {
            if (cur_state == _BOC::_Sign::PositivE)
            {
              _Symbolic_Point updateSym = old_cliped[0];
              updateSym.update_(plane_site, old_cliped[1]);
              update_cliped.push_back(std::move(updateSym));
              update_cliped.push_back(old_cliped[1]);
            }
            else if (cur_state == _BOC::_Sign::ZerO)
            {
              _BOC::_Sign sp2 = side_(current_site, plane_site, old_cliped[2]);
              if (sp2 == _BOC::_Sign::PositivE)
              {
                update_cliped.clear();
                continue;
              }
              else if (sp2 == _BOC::_Sign::NegativE)
              {
                old_cliped.clear();
                break;
              }
            }
          }

          _BOC::_Sign sp0 = pre_state;
          _BOC::_Sign sp1 = cur_state;
          bool ifbreak = false, ifcontinue = false;
          for (int i = 2; i < static_cast<int>(old_cliped.size()); ++i)
          {
            nex_state = side_(current_site, plane_site, old_cliped[i]);
            if (cur_state == _BOC::_Sign::PositivE)
            {
              if (nex_state == _BOC::_Sign::PositivE)
              {
                update_cliped.push_back(old_cliped[i]);
              }
              else if (nex_state == _BOC::_Sign::NegativE)
              {
                update_cliped.push_back(insec_bisector_(old_cliped[i - 1], old_cliped[i], plane_site, current_site, current_face_vid));
              }
              else
              {
                _Symbolic_Point updatesym = old_cliped[i];
                updatesym.update_(plane_site, old_cliped[i - 1]);
                update_cliped.push_back(std::move(updatesym));
              }
            }
            else if (cur_state == _BOC::_Sign::NegativE)
            {
              if (nex_state == _BOC::_Sign::PositivE)
              {
                update_cliped.push_back(insec_bisector_(old_cliped[i - 1], old_cliped[i], plane_site, current_site, current_face_vid));
                update_cliped.push_back(old_cliped[i]);
              }
            }
            else if (cur_state == _BOC::_Sign::ZerO)
            {
              if (nex_state == _BOC::_Sign::PositivE)
              {
                if (pre_state == _BOC::_Sign::NegativE)
                {
                  _Symbolic_Point updateSym = old_cliped[i - 1];
                  updateSym.update_(plane_site, old_cliped[i]);
                  update_cliped.push_back(std::move(updateSym));
                  update_cliped.push_back(old_cliped[i]);
                }
                else if (pre_state == _BOC::_Sign::PositivE)
                {
                  update_cliped.clear();
                  ifcontinue = true;
                  break;
                }
                else
                {
                  throw std::runtime_error("Two consecutive 0s appear in front.");
                }
              }
              else if (nex_state == _BOC::_Sign::ZerO)
              {
                if (pre_state == _BOC::_Sign::PositivE)
                {
                  update_cliped.clear();
                  ifcontinue = true;
                  break;
                }
                else if (pre_state == _BOC::_Sign::NegativE)
                {
                  old_cliped.clear();
                  ifbreak = true;
                  break;
                }
              }
              else if (nex_state == _BOC::_Sign::NegativE)
              {
                if (pre_state == _BOC::_Sign::NegativE)
                {
                  old_cliped.clear();
                  ifbreak = true;
                  break;
                }
              }
            }
            pre_state = cur_state;
            cur_state = nex_state;
          }
          if (ifbreak)
          {
            break;
          }
          if (ifcontinue)
          {
            continue;
          }
          if (sp0 == _BOC::_Sign::PositivE)
          {
            if (cur_state == _BOC::_Sign::NegativE)
            {
              update_cliped.push_back(insec_bisector_(old_cliped.back(), old_cliped[0], plane_site, current_site, current_face_vid));
            }
            else if (cur_state == _BOC::_Sign::ZerO)
            {
              if (pre_state == _BOC::_Sign::NegativE)
              {
                _Symbolic_Point updateSym = old_cliped.back();
                updateSym.update_(plane_site, old_cliped[0]);
                update_cliped.push_back(std::move(updateSym));
              }
              else if (pre_state == _BOC::_Sign::PositivE)
              {
                update_cliped.clear();
                continue;
              }
            }
          }
          else if (sp0 == _BOC::_Sign::NegativE)
          {
            if (cur_state == _BOC::_Sign::PositivE)
            {
              update_cliped.push_back(insec_bisector_(old_cliped.back(), old_cliped[0], plane_site, current_site, current_face_vid));
            }
            else if (cur_state == _BOC::_Sign::ZerO)
            {
              if (pre_state == _BOC::_Sign::NegativE)
              {
                old_cliped.clear();
                break;
              }
            }
          }
          else if (sp0 == _BOC::_Sign::ZerO)
          {
            if (sp1 == _BOC::_Sign::NegativE)
            {
              if (cur_state == _BOC::_Sign::PositivE)
              {
                _Symbolic_Point updateSym = old_cliped[0];
                updateSym.update_(plane_site, old_cliped.back());
                update_cliped.push_back(std::move(updateSym));
              }
              else
              {
                old_cliped.clear();
                break;
              }
            }
            else if (sp1 == _BOC::_Sign::PositivE)
            {
              if (cur_state == _BOC::_Sign::PositivE || cur_state == _BOC::_Sign::ZerO)
              {
                update_cliped.clear();
                continue;
              }
            }
          }
          old_cliped.swap(update_cliped);
        }

        for (const auto &sp : old_cliped)
        {
          for (const int sym_id : sp._sym)
          {
            if (sym_id > 0)
            {
              const int next_site = sym_id - 1;
              auto &next_map = from_idx_to_locations[next_site];
              if (next_map.find(current_face) == next_map.end())
              {
                const int loc = static_cast<int>(cliped_faces.size());
                next_map.emplace(current_face, loc);
                from_locations_to_idx.push_back(current_face);
                std::vector<_Symbolic_Point> copy_cliped = original_cliped;
                copy_cliped[0]._site = next_site;
                copy_cliped[1]._site = next_site;
                copy_cliped[2]._site = next_site;
                cliped_faces.push_back(std::move(copy_cliped));
                Qfs.emplace(next_site, loc);
              }
            }
            else if (sym_id < 0)
            {
              const int next_face = -sym_id - 1;
              auto &site_map = from_idx_to_locations[current_site];
              if (site_map.find(next_face) == site_map.end())
              {
                const int loc = static_cast<int>(cliped_faces.size());
                site_map.emplace(next_face, loc);
                from_locations_to_idx.push_back(next_face);
                cliped_faces.push_back(make_seed_polygon(next_face, current_site));
                Qfs.emplace(current_site, loc);
              }
            }
          }
        }

        cliped_faces[current_cliped] = std::move(old_cliped);
        face_is_visited[current_face] = 1;
      }
    }

    _vertices.clear();
    _cells.clear();
    _cells.resize(_num_sites);
    _edges.clear();
    _edges.resize(_num_sites);

    std::unordered_map<SymbolKey, int, SymbolKeyHash> from_sym_to_vertex;
    from_sym_to_vertex.reserve(std::max(64, static_cast<int>(cliped_faces.size()) * 3));

    const auto make_symbol_key = [&](const int site_i, const _Symbolic_Point &sp) -> SymbolKey
    {
      SymbolKey key;
      auto push_raw = [&](const int value)
      {
        if (key.size >= key.data.size())
        {
          throw std::runtime_error("symbol key overflow");
        }
        key.data[key.size++] = value;
      };
      if (sp.flag == 1)
      {
        push_raw(sp.p[0]);
      }
      else if (sp.flag == 2)
      {
        push_raw(-sp.p[0] - 1);
        push_raw(-sp.p[1] - 1);
        push_raw(sp.p[2] + 1);
        push_raw(site_i + 1);
      }
      else if (sp.flag == 3)
      {
        push_raw(-sp.p[0] - 1);
        push_raw(-sp.p[1] - 1);
        push_raw(-sp.p[2] - 1);
        push_raw(site_i + 1);
        for (const int sym_id : sp._sym)
        {
          if (sym_id > 0)
          {
            push_raw(sym_id);
          }
        }
      }
      else
      {
        throw std::runtime_error("flag error");
      }
      std::sort(key.data.begin(), key.data.begin() + key.size);
      auto unique_end = std::unique(key.data.begin(), key.data.begin() + key.size);
      key.size = static_cast<std::uint8_t>(unique_end - key.data.begin());
      if ((sp.flag == 2 && key.size != 4) || (sp.flag == 3 && key.size != 6))
      {
        throw std::runtime_error("unexpected symbolic key size");
      }
      return key;
    };

    const auto resolve_symbolic_vertex = [&](const int site_i, const _Symbolic_Point &sp) -> int
    {
      const SymbolKey key = make_symbol_key(site_i, sp);
      const auto found = from_sym_to_vertex.find(key);
      if (found != from_sym_to_vertex.end())
      {
        return found->second;
      }

      const int new_idx = static_cast<int>(_vertices.size());
      from_sym_to_vertex.emplace(key, new_idx);

      if (sp.flag == 1)
      {
        _vertices.push_back(_model.vertex_(sp.p[0]));
      }
      else if (sp.flag == 2)
      {
        const int q1 = site_i;
        const int q2 = sp.p[2];
        const _Point3 p1 = _model.vertex_(sp.p[0]);
        const _Point3 p2 = _model.vertex_(sp.p[1]);
        const _Point3 v = (_sites[q2] - _sites[q1]) * 2;
        const double d = _weights[q2] - _weights[q1] - 0.5 * v.dot_(_sites[q2] + _sites[q1]);
        const double d1 = std::fabs(p1.dot_(v) + d);
        const double d2 = std::fabs(p2.dot_(v) + d);
        _vertices.push_back(p1 + (p2 - p1) * d1 / (d1 + d2));
      }
      else if (sp.flag == 3)
      {
        std::vector<int> p12;
        p12.reserve(2);
        int q = -1;
        for (const int sym_id : sp._sym)
        {
          if (sym_id > 0)
          {
            p12.push_back(sym_id - 1);
          }
          else
          {
            q = -sym_id - 1;
          }
        }
        if (p12.size() != 2)
        {
          throw std::runtime_error("psym.size() != 6");
        }
        const _Point3 v1 = (_sites[p12[0]] - _sites[site_i]) * 2;
        const double d1 = _weights[p12[0]] - _weights[site_i] - 0.5 * v1.dot_(_sites[p12[0]] + _sites[site_i]);
        const _Point3 v2 = (_sites[p12[1]] - _sites[site_i]) * 2;
        const double d2 = _weights[p12[1]] - _weights[site_i] - 0.5 * v2.dot_(_sites[p12[1]] + _sites[site_i]);
        const _Point3 v0 = face_plane_normal_from_vertices(_model, sp.p[0], sp.p[1], sp.p[2], q);
        const double d0 = -v0.dot_(_model.vertex_(sp.p[0]));
        _vertices.push_back(_Point3::intersection_three_plane(v0, d0, v1, d1, v2, d2));
      }
      else
      {
        throw std::runtime_error("flag error");
      }
      return new_idx;
    };

    for (int i = 0; i < _num_sites; ++i)
    {
      for (auto it = from_idx_to_locations[i].begin(); it != from_idx_to_locations[i].end(); ++it)
      {
        const int clipped_idx = it->second;
        if (cliped_faces[clipped_idx].empty())
        {
          continue;
        }
        if (cliped_faces[clipped_idx].size() < 3)
        {
          throw std::runtime_error("size error");
        }

        int first_p = resolve_symbolic_vertex(i, cliped_faces[clipped_idx][0]);
        int second_p = resolve_symbolic_vertex(i, cliped_faces[clipped_idx][1]);

        {
          const std::set<int> insec_sym = cliped_faces[clipped_idx][0].insec_(cliped_faces[clipped_idx][1]);
          int adj_sites = -1;
          for (const int sym_id : insec_sym)
          {
            if (sym_id > 0)
            {
              adj_sites = sym_id - 1;
            }
          }
          if (adj_sites != -1)
          {
            _edges[i][adj_sites].push_back(std::make_pair(first_p, second_p));
          }
        }

        for (int j = 2; j < static_cast<int>(cliped_faces[clipped_idx].size()); ++j)
        {
          const int third_p = resolve_symbolic_vertex(i, cliped_faces[clipped_idx][j]);
          _cells[i].push_back(std::make_tuple(first_p, second_p, third_p));

          {
            const std::set<int> insec_sym = cliped_faces[clipped_idx][j - 1].insec_(cliped_faces[clipped_idx][j]);
            int adj_sites = -1;
            for (const int sym_id : insec_sym)
            {
              if (sym_id > 0)
              {
                adj_sites = sym_id - 1;
              }
            }
            if (adj_sites != -1)
            {
              _edges[i][adj_sites].push_back(std::make_pair(second_p, third_p));
            }
          }
          second_p = third_p;
        }

        {
          const std::set<int> insec_sym = cliped_faces[clipped_idx][0].insec_(cliped_faces[clipped_idx].back());
          int adj_sites = -1;
          for (const int sym_id : insec_sym)
          {
            if (sym_id > 0)
            {
              adj_sites = sym_id - 1;
            }
          }
          if (adj_sites != -1)
          {
            _edges[i][adj_sites].push_back(std::make_pair(second_p, first_p));
          }
        }
      }
    }
  }
  _Restricted_Tessellation3D::_Restricted_Tessellation3D(const _ManifoldModel& in_model)
      :_model(in_model)
  {
      _num_sites = 0;
      _sites.clear();
      _weights.clear();
      _vertices.clear();
      _cells.clear();
  }
  _Restricted_Tessellation3D::_Restricted_Tessellation3D(const _ManifoldModel &in_model,
                                                         const std::vector<_Point3> &in_sites,
                                                         const std::vector<double> &in_weights)
      : _num_sites(in_sites.size()),
        _sites(in_sites),
        _weights(in_weights),
        _model(in_model)
  {
    _vertices.clear();
    _cells.clear();
    _cells.resize(_num_sites);
    calculate_();
  }
  _Restricted_Tessellation3D::_Restricted_Tessellation3D(const _ManifoldModel &in_model,
                                                         const std::vector<_Point3> &in_sites)
      : _num_sites(in_sites.size()),
        _sites(in_sites),
        _model(in_model)
  {
    _vertices.clear();
    _cells.clear();
    _cells.resize(_num_sites);
    _weights.clear();
    _weights.resize(_num_sites, 0);
    calculate_();
  }
  void _Restricted_Tessellation3D::calculate_(const std::vector<_Point3> &in_sites)
  {
    _sites = in_sites;
    _num_sites = _sites.size();
    _weights.clear();
    _weights.resize(_num_sites, 0);
    calculate_();
  }
  void _Restricted_Tessellation3D::calculate_(const std::vector<_Point3>& in_sites, const std::vector<double>& in_weights)
  {
      _sites = in_sites;
      _num_sites = _sites.size();
      _weights = in_weights;
      calculate_();
  }
} // namespace BGAL
