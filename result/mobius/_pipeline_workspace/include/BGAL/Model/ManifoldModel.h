#pragma once
#include "Model.h"
#include "BGAL/BaseShape/Line.h"
#include <array>
#include <map>
namespace BGAL {
	class _Edge_Iterator;
	class _FE_Iterator;
	class _FF_Iterator;
	class _VV_Iterator;
	class _VE_Iterator;
	class _VF_Iterator;
	class _ManifoldModel : public _Model 
	{
		friend class _Edge_Iterator;
		friend class _FE_Iterator;
		friend class _FF_Iterator;
		friend class _VV_Iterator;
		friend class _VE_Iterator;
		friend class _VF_Iterator;
	public:
		class _MMEdge : public _Segment3 
		{
		public:
			int _id_left_vertex;
			int _id_right_vertex;
			int _id_opposite_vertex;
			int _id_left_edge;
			int _id_right_edge;
			int _id_reverse_edge;
			int _id_face;
			bool _is_boundary_placeholder;
			std::vector<int> _id_opposite_edges;
			_MMEdge();
			_MMEdge(const _Point3& in_s, const _Point3& in_t);
		};
		_ManifoldModel();
		_ManifoldModel(const std::string& in_file_name);
		_ManifoldModel(const std::vector<_Point3>& in_vertices, const std::vector<_Model::_MFace>& in_faces);
		_ManifoldModel(const _ManifoldModel& in_mmodel);
		void preprocess_model_();
		void save_processed_obj_file_(const std::string& out_file_name) const;
		static void export_processed_obj_(const std::string& in_file_name,
			const std::string& out_file_name);
		inline bool used_nonmanifold_fallback_() const
		{
			return _used_nonmanifold_fallback;
		}
		inline int number_edges_() const 
		{
			return _edges.size();
		}

		inline const std::array<int, 3>& face_edges_(const int& fid) const
		{
			if (fid < 0 || fid >= number_faces_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return _face_edges[fid];
		}
		inline const std::array<int, 3>& face_adjacent_faces_(const int& fid) const
		{
			if (fid < 0 || fid >= number_faces_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return _face_adjacent_faces[fid];
		}
		inline const std::vector<int>& incident_edges_of_vertex_(const int& vid) const
		{
			if (vid < 0 || vid >= number_vertices_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return _incident_edges_of_vertices[vid];
		}
		inline const std::vector<int>& incident_faces_of_vertex_(const int& vid) const
		{
			if (vid < 0 || vid >= number_vertices_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return _incident_faces_of_vertices[vid];
		}
		inline const std::vector<int>& adjacent_vertices_of_vertex_(const int& vid) const
		{
			if (vid < 0 || vid >= number_vertices_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return _adjacent_vertices_of_vertices[vid];
		}
		inline const _MMEdge& edge_(const int& id) const 
		{
			if (id < 0 || id >= _edges.size())
				throw std::runtime_error("Beyond the index!");
			return _edges[id];
		}
		_Edge_Iterator edge_begin() const;
		inline int edge_end() const 
		{
			return number_edges_();
		}
		_FE_Iterator fe_begin(const int& fid) const;
		inline int fe_end(const int& fid) const 
		{
			if (fid < 0 || fid >= number_faces_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return 3;
		}
		_FF_Iterator ff_begin(const int& fid) const;
		inline int ff_end(const int& fid) const 
		{
			if (fid < 0 || fid >= number_faces_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return 3;
		}
		_VV_Iterator vv_begin(const int& vid) const;
		inline int vv_end(const int& vid) const 
		{
			if (vid < 0 || vid >= number_vertices_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return static_cast<int>(_adjacent_vertices_of_vertices[vid].size());
		}
		_VE_Iterator ve_begin(const int& vid) const;
		inline int ve_end(const int& vid) const 
		{
			if (vid < 0 || vid >= number_vertices_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return static_cast<int>(_incident_edges_of_vertices[vid].size());
		}
		_VF_Iterator vf_begin(const int& vid) const;
		inline int vf_end(const int& vid) const 
		{
			if (vid < 0 || vid >= number_vertices_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return static_cast<int>(_incident_faces_of_vertices[vid].size());
		}
		inline int degree_of_vertex_(const int& vid) const 
		{
			if (vid < 0 || vid >= number_vertices_()) 
			{
				throw std::runtime_error("Beyond the index!");
			}
			return static_cast<int>(_incident_faces_of_vertices[vid].size());
		}
		inline bool is_boundary_vertex_(const int& vid) const
		{
			if (vid < 0 || vid >= number_vertices_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return vid < static_cast<int>(_boundary_vertex_flags.size()) && _boundary_vertex_flags[vid];
		}
		inline bool is_boundary_edge_(const int& eid) const
		{
			if (eid < 0 || eid >= number_edges_())
			{
				throw std::runtime_error("Beyond the index!");
			}
			return eid < static_cast<int>(_boundary_edge_flags.size()) && _boundary_edge_flags[eid];
		}
		inline bool has_nonmanifold_topology_() const
		{
			return _has_nonmanifold_topology;
		}
	protected:
		void creat_edges_from_vertices_faces_();
		void arrange_neighs_of_vertex_face_();
		void assign_raw_mesh_(const std::vector<_Point3>& in_vertices,
			const std::vector<_Model::_MFace>& in_faces);
		void load_with_nonmanifold_support_(const std::vector<_Point3>& in_vertices,
			const std::vector<_Model::_MFace>& in_faces);
	protected:
		std::vector<_MMEdge> _edges;
		std::vector<std::array<int, 3>> _face_edges;
		std::vector<std::array<int, 3>> _face_adjacent_faces;
		std::vector<int> _neight_edge_of_vertices;
		std::vector<int> _neigh_edge_of_faces;
		std::vector<int> _isolated_vertices;
		std::vector<int> _degree_of_vertices;
		std::vector<std::vector<int>> _incident_edges_of_vertices;
		std::vector<std::vector<int>> _incident_faces_of_vertices;
		std::vector<std::vector<int>> _adjacent_vertices_of_vertices;
		std::vector<bool> _boundary_edge_flags;
		std::vector<bool> _boundary_vertex_flags;
		bool _has_nonmanifold_topology = false;
		bool _used_nonmanifold_fallback = false;
	};
}
