#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <time.h>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_segment_primitive_3.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <omp.h>

#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/BaseShape/KDTree.h"
#include "BGAL/Integral/Integral.h"

struct MyFace
{
	MyFace(Eigen::Vector3i a)
	{
		p = a;
	}
	MyFace(int a, int b, int c)
	{
		p.x() = a;
		p.y() = b;
		p.z() = c;
	}
	Eigen::Vector3i p;
	bool operator<(const MyFace& a) const
	{
		if (p.x() == a.p.x())
		{
			if (p.y() == a.p.y())
			{
				return p.z() > a.p.z();
			}
			return p.y() > a.p.y();
		}
		return p.x() > a.p.x();
	}
};

namespace
{
	using CGALKernel = CGAL::Simple_cartesian<double>;
	using CGALPoint3 = CGALKernel::Point_3;
	using CGALSegment3 = CGALKernel::Segment_3;
	using CGALPolyhedron = CGAL::Polyhedron_3<CGALKernel>;
	using CGALFacePrimitive =
		CGAL::AABB_face_graph_triangle_primitive<CGALPolyhedron>;
	using CGALFaceTraits = CGAL::AABB_traits_3<CGALKernel, CGALFacePrimitive>;
	using CGALFaceTree = CGAL::AABB_tree<CGALFaceTraits>;
	using CGALSegmentIterator = std::list<CGALSegment3>::const_iterator;
	using CGALSegmentPrimitive =
		CGAL::AABB_segment_primitive_3<CGALKernel, CGALSegmentIterator>;
	using CGALSegmentTraits =
		CGAL::AABB_traits_3<CGALKernel, CGALSegmentPrimitive>;
	using CGALSegmentTree = CGAL::AABB_tree<CGALSegmentTraits>;
	using CGALClosestFace = CGALFaceTree::Point_and_primitive_id;

	constexpr double kFeatureAngleDegrees = 35.0;
	constexpr double kFeatureDensityBoost = 12.0;
	constexpr double kFeatureSigmaScale = 2.0;
	constexpr double kMinFeatureSigma = 1e-8;

	struct OutputSphere
	{
		BGAL::_Point3 c;
		double r = 0.0;
		BGAL::_Point3 max_point;
	};

	static inline void build_spheres_from_rvd(
		const std::vector<BGAL::_Point3>& sites,
		const BGAL::_Restricted_Tessellation3D& rvd,
		std::vector<OutputSphere>& spheres)
	{
		const auto& edges = rvd.get_edges_();
		spheres.assign(sites.size(), OutputSphere());

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < (int)sites.size(); ++i)
		{
			const BGAL::_Point3 site = sites[i];
			std::vector<int> boundary_vertices;
			if (i < (int)edges.size())
			{
				boundary_vertices.reserve(edges[i].size() * 4);
				for (const auto& kv : edges[i])
				{
					for (const auto& e : kv.second)
					{
						boundary_vertices.push_back(e.first);
						boundary_vertices.push_back(e.second);
					}
				}
			}

			std::sort(boundary_vertices.begin(), boundary_vertices.end());
			boundary_vertices.erase(
				std::unique(boundary_vertices.begin(), boundary_vertices.end()),
				boundary_vertices.end());

			BGAL::_Point3 far_point = site;
			double max_dist = 0.0;
			for (int vid : boundary_vertices)
			{
				const BGAL::_Point3 pv = rvd.vertex_(vid);
				const double dist = (pv - site).length_();
				if (dist > max_dist)
				{
					max_dist = dist;
					far_point = pv;
				}
			}

			spheres[i].c = site;
			spheres[i].r = max_dist;
			spheres[i].max_point = far_point;
		}
	}

	static inline void write_spheres_csv(
		const std::string& filepath,
		const std::vector<OutputSphere>& spheres,
		const BGAL::_ManifoldModel& model)
	{
		std::ofstream out(filepath);
		if (!out)
		{
			return;
		}

		out << std::setprecision(17);
		for (size_t i = 0; i < spheres.size(); ++i)
		{
			auto nearest = const_cast<BGAL::_ManifoldModel&>(model).nearest_point_(spheres[i].c);
			int face_id = std::get<2>(nearest);
			BGAL::_Point3 normal(0.0, 0.0, 0.0);
			if (face_id >= 0 && face_id < model.number_faces_())
			{
				if (model.has_nonmanifold_topology_())
				{
					normal = model.normal_face_(face_id);
				}
				else
				{
					const BGAL::_Point3& nearest_point = std::get<0>(nearest);
					const auto& face = model.face_(face_id);
					const double dis1 = (nearest_point - model.vertex_(face[0])).length_();
					const double dis2 = (nearest_point - model.vertex_(face[1])).length_();
					const double dis3 = (nearest_point - model.vertex_(face[2])).length_();
					normal += model.normal_vertex_(face[0]) * (dis2 + dis3);
					normal += model.normal_vertex_(face[1]) * (dis1 + dis3);
					normal += model.normal_vertex_(face[2]) * (dis1 + dis2);
				}
				if (normal.sqlength_() > 1e-30)
				{
					normal.normalized_();
				}
			}

			out << spheres[i].c.x() << "," << spheres[i].c.y() << "," << spheres[i].c.z() << ","
				<< spheres[i].r << "," << face_id << ","
				<< normal.x() << "," << normal.y() << "," << normal.z() << std::endl;
		}
	}

	static inline Eigen::Vector3d to_eigen(const BGAL::_Point3& p)
	{
		return Eigen::Vector3d(p.x(), p.y(), p.z());
	}

	static inline CGALPoint3 to_cgal_point(const BGAL::_Point3& p)
	{
		return CGALPoint3(p.x(), p.y(), p.z());
	}

	static inline BGAL::_Point3 to_bgal_point(const CGALPoint3& p)
	{
		return BGAL::_Point3(p.x(), p.y(), p.z());
	}

	static inline bool is_finite_point(const BGAL::_Point3& p)
	{
		return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
	}

	static inline bool sites_are_finite(const std::vector<BGAL::_Point3>& sites)
	{
		for (const auto& p : sites)
		{
			if (!is_finite_point(p))
			{
				return false;
			}
		}
		return true;
	}

	static inline bool project_with_aabb_tree(
		const CGALFaceTree& tree,
		const BGAL::_Point3& query,
		BGAL::_Point3& projected,
		Eigen::Vector3d& normal);

	static inline void project_points_with_aabb_tree_parallel(
		const CGALFaceTree& tree,
		const Eigen::VectorXd& X,
		std::vector<BGAL::_Point3>& sites,
		std::vector<Eigen::Vector3d>& normals,
		const std::vector<BGAL::_Point3>* fallback_sites = nullptr)
	{
		const int num = static_cast<int>(sites.size());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num; ++i)
		{
			const double x = X(i * 3);
			const double y = X(i * 3 + 1);
			const double z = X(i * 3 + 2);

			if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
			{
				if (fallback_sites != nullptr && i < (int)fallback_sites->size())
				{
					sites[i] = (*fallback_sites)[i];
				}
				normals[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
				continue;
			}

			const BGAL::_Point3 query(x, y, z);
			BGAL::_Point3 projected;
			Eigen::Vector3d projected_normal(0.0, 0.0, 1.0);
			if (!project_with_aabb_tree(tree, query, projected, projected_normal))
			{
				if (fallback_sites != nullptr && i < (int)fallback_sites->size())
				{
					sites[i] = (*fallback_sites)[i];
				}
				normals[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
				continue;
			}

			sites[i] = projected;
			normals[i] = projected_normal;
			if (!normals[i].allFinite() || normals[i].squaredNorm() <= 1e-30)
			{
				normals[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
			}
		}
	}

	struct CwfEdgeKey
	{
		int a = -1;
		int b = -1;

		CwfEdgeKey() = default;
		CwfEdgeKey(int x, int y)
		{
			if (x < y)
			{
				a = x;
				b = y;
			}
			else
			{
				a = y;
				b = x;
			}
		}

		bool operator<(const CwfEdgeKey& o) const
		{
			if (a != o.a)
			{
				return a < o.a;
			}
			return b < o.b;
		}
	};

	struct CwfFaceKey
	{
		int a = -1;
		int b = -1;
		int c = -1;

		CwfFaceKey() = default;
		CwfFaceKey(int x, int y, int z)
		{
			std::array<int, 3> ids{x, y, z};
			std::sort(ids.begin(), ids.end());
			a = ids[0];
			b = ids[1];
			c = ids[2];
		}

		bool valid(int n) const
		{
			return a >= 0 && b >= 0 && c >= 0 &&
				a < n && b < n && c < n &&
				a != b && b != c && a != c;
		}

		Eigen::Vector3i vec() const
		{
			return Eigen::Vector3i(a, b, c);
		}

		bool operator<(const CwfFaceKey& o) const
		{
			if (a != o.a)
			{
				return a < o.a;
			}
			if (b != o.b)
			{
				return b < o.b;
			}
			return c < o.c;
		}
	};

	static inline Eigen::Vector3d safe_triangle_normal(
		const std::vector<BGAL::_Point3>& sites,
		const Eigen::Vector3i& f)
	{
		const int a = f.x();
		const int b = f.y();
		const int c = f.z();
		if (a < 0 || b < 0 || c < 0 ||
			a >= static_cast<int>(sites.size()) ||
			b >= static_cast<int>(sites.size()) ||
			c >= static_cast<int>(sites.size()))
		{
			return Eigen::Vector3d::Zero();
		}

		const Eigen::Vector3d p0 = to_eigen(sites[a]);
		const Eigen::Vector3d p1 = to_eigen(sites[b]);
		const Eigen::Vector3d p2 = to_eigen(sites[c]);
		Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
		const double len = n.norm();
		if (!(len > 1e-30) || !std::isfinite(len))
		{
			return Eigen::Vector3d::Zero();
		}
		return n / len;
	}

	static inline double triangle_double_area(
		const std::vector<BGAL::_Point3>& sites,
		const Eigen::Vector3i& f)
	{
		const int a = f.x();
		const int b = f.y();
		const int c = f.z();
		if (a < 0 || b < 0 || c < 0 ||
			a >= static_cast<int>(sites.size()) ||
			b >= static_cast<int>(sites.size()) ||
			c >= static_cast<int>(sites.size()))
		{
			return 0.0;
		}
		const Eigen::Vector3d p0 = to_eigen(sites[a]);
		const Eigen::Vector3d p1 = to_eigen(sites[b]);
		const Eigen::Vector3d p2 = to_eigen(sites[c]);
		const double area2 = (p1 - p0).cross(p2 - p0).norm();
		return std::isfinite(area2) ? area2 : 0.0;
	}

	static inline double mean_edge_length_squared(
		const std::vector<BGAL::_Point3>& sites,
		const std::vector<Eigen::Vector3i>& faces)
	{
		double sum = 0.0;
		long long cnt = 0;
#pragma omp parallel for reduction(+ : sum, cnt)
		for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi)
		{
			const auto& f = faces[fi];
			const int a = f.x();
			const int b = f.y();
			const int c = f.z();
			if (a < 0 || b < 0 || c < 0 ||
				a >= static_cast<int>(sites.size()) ||
				b >= static_cast<int>(sites.size()) ||
				c >= static_cast<int>(sites.size()) ||
				a == b || b == c || a == c)
			{
				continue;
			}
			const Eigen::Vector3d pa = to_eigen(sites[a]);
			const Eigen::Vector3d pb = to_eigen(sites[b]);
			const Eigen::Vector3d pc = to_eigen(sites[c]);
			sum += (pa - pb).squaredNorm();
			sum += (pb - pc).squaredNorm();
			sum += (pc - pa).squaredNorm();
			cnt += 3;
		}
		if (cnt <= 0)
		{
			return 1.0;
		}
		const double ret = sum / static_cast<double>(cnt);
		return (ret > 0.0 && std::isfinite(ret)) ? ret : 1.0;
	}

	static inline std::vector<Eigen::Vector3d> compute_site_face_normals(
		const std::vector<BGAL::_Point3>& sites,
		const BGAL::_ManifoldModel& model)
	{
		std::vector<Eigen::Vector3d> normals(sites.size(), Eigen::Vector3d(0.0, 0.0, 1.0));
		for (int i = 0; i < static_cast<int>(sites.size()); ++i)
		{
			auto nearest = const_cast<BGAL::_ManifoldModel&>(model).nearest_point_(sites[i]);
			const int face_id = std::get<2>(nearest);
			if (face_id >= 0 && face_id < model.number_faces_())
			{
				BGAL::_Point3 n = model.normal_face_(face_id);
				if (n.sqlength_() > 1e-30)
				{
					n.normalized_();
					normals[i] = to_eigen(n);
				}
			}
		}
		return normals;
	}

	static inline double face_badness_score(
		const std::vector<BGAL::_Point3>& sites,
		const std::vector<Eigen::Vector3d>& site_normals,
		const Eigen::Vector3i& f,
		double mean_edge2)
	{
		const int a = f.x();
		const int b = f.y();
		const int c = f.z();
		if (a < 0 || b < 0 || c < 0 ||
			a >= static_cast<int>(sites.size()) ||
			b >= static_cast<int>(sites.size()) ||
			c >= static_cast<int>(sites.size()) ||
			a == b || b == c || a == c)
		{
			return std::numeric_limits<double>::infinity();
		}

		const Eigen::Vector3d nf = safe_triangle_normal(sites, f);
		if (nf.squaredNorm() <= 1e-24)
		{
			return std::numeric_limits<double>::infinity();
		}

		const double area_score = triangle_double_area(sites, f) / (mean_edge2 + 1e-30);
		double normal_alignment_bad = 0.0;
		double normal_spread_bad = 0.0;

		if (site_normals.size() == sites.size())
		{
			Eigen::Vector3d n0 = site_normals[a];
			Eigen::Vector3d n1 = site_normals[b];
			Eigen::Vector3d n2 = site_normals[c];
			const double l0 = n0.norm();
			const double l1 = n1.norm();
			const double l2 = n2.norm();
			if (l0 > 1e-12 && l1 > 1e-12 && l2 > 1e-12)
			{
				n0 /= l0;
				n1 /= l1;
				n2 /= l2;
				const double align =
					(std::abs(nf.dot(n0)) + std::abs(nf.dot(n1)) + std::abs(nf.dot(n2))) / 3.0;
				normal_alignment_bad = 1.0 - std::min(1.0, std::max(0.0, align));

				const double d01 = std::abs(n0.dot(n1));
				const double d12 = std::abs(n1.dot(n2));
				const double d20 = std::abs(n2.dot(n0));
				const double min_consistency = std::min(d01, std::min(d12, d20));
				normal_spread_bad = 1.0 - std::min(1.0, std::max(0.0, min_consistency));
			}
		}

		return 0.20 * area_score +
			2.50 * normal_alignment_bad +
			4.00 * normal_spread_bad;
	}

	static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_edges_fast_raw(
		int num_sites,
		const std::vector<std::map<int, std::vector<std::pair<int, int>>>>& edges)
	{
		std::vector<std::vector<int>> adj(num_sites);
		const int edge_count = std::min(num_sites, static_cast<int>(edges.size()));
		for (int i = 0; i < edge_count; ++i)
		{
			for (const auto& ee : edges[i])
			{
				if (ee.first >= 0 && ee.first < num_sites && ee.first != i)
				{
					adj[i].push_back(ee.first);
				}
			}
		}

#pragma omp parallel for
		for (int i = 0; i < num_sites; ++i)
		{
			std::sort(adj[i].begin(), adj[i].end());
			adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
		}

		std::vector<Eigen::Vector3i> tris;
#pragma omp parallel
		{
			std::vector<Eigen::Vector3i> local;
#pragma omp for schedule(dynamic, 64)
			for (int u = 0; u < num_sites; ++u)
			{
				for (int v : adj[u])
				{
					if (v <= u)
					{
						continue;
					}
					int i = 0;
					int j = 0;
					while (i < static_cast<int>(adj[u].size()) && j < static_cast<int>(adj[v].size()))
					{
						if (adj[u][i] == adj[v][j])
						{
							const int w = adj[u][i];
							if (w > v)
							{
								local.emplace_back(u, v, w);
							}
							++i;
							++j;
						}
						else if (adj[u][i] < adj[v][j])
						{
							++i;
						}
						else
						{
							++j;
						}
					}
				}
			}
#pragma omp critical
			tris.insert(tris.end(), local.begin(), local.end());
		}

		std::set<CwfFaceKey> uniq;
		for (const auto& f : tris)
		{
			CwfFaceKey key(f.x(), f.y(), f.z());
			if (key.valid(num_sites))
			{
				uniq.insert(key);
			}
		}

		std::vector<Eigen::Vector3i> ret;
		ret.reserve(uniq.size());
		for (const auto& key : uniq)
		{
			ret.push_back(key.vec());
		}
		return ret;
	}

	static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_rvd_corners(
		int num_sites,
		const BGAL::_Restricted_Tessellation3D& rvd)
	{
		const auto& cells = rvd.get_cells_();
		const int nv = rvd.number_vertices_();
		if (num_sites <= 0 || nv <= 0 || cells.empty())
		{
			return {};
		}

		std::vector<std::vector<int>> vertex_to_sites(nv);
		const int cell_count = std::min(num_sites, static_cast<int>(cells.size()));
		for (int sid = 0; sid < cell_count; ++sid)
		{
			for (const auto& tri : cells[sid])
			{
				const int vids[3] = {
					std::get<0>(tri),
					std::get<1>(tri),
					std::get<2>(tri)};
				for (int k = 0; k < 3; ++k)
				{
					const int vid = vids[k];
					if (vid >= 0 && vid < nv)
					{
						vertex_to_sites[vid].push_back(sid);
					}
				}
			}
		}

		std::set<CwfFaceKey> face_keys;
#pragma omp parallel
		{
			std::set<CwfFaceKey> local_keys;
#pragma omp for schedule(dynamic, 256)
			for (int vid = 0; vid < nv; ++vid)
			{
				auto s = vertex_to_sites[vid];
				if (s.empty())
				{
					continue;
				}
				std::sort(s.begin(), s.end());
				s.erase(std::unique(s.begin(), s.end()), s.end());

				// Only accept strict 3-cell corners.  Corners with more than
				// three incident cells are usually sharp-corner degeneracies;
				// emitting all C(k,3) combinations recreates false triangles.
				if (s.size() == 3)
				{
					CwfFaceKey key(s[0], s[1], s[2]);
					if (key.valid(num_sites))
					{
						local_keys.insert(key);
					}
				}
			}
#pragma omp critical
			face_keys.insert(local_keys.begin(), local_keys.end());
		}

		std::vector<Eigen::Vector3i> ret;
		ret.reserve(face_keys.size());
		for (const auto& key : face_keys)
		{
			ret.push_back(key.vec());
		}
		return ret;
	}

	static inline std::vector<Eigen::Vector3i> filter_nonmanifold_faces_by_edge_valence(
		const std::vector<BGAL::_Point3>& sites,
		const std::vector<Eigen::Vector3i>& faces_in,
		const std::vector<Eigen::Vector3d>& site_normals)
	{
		const int n = static_cast<int>(sites.size());
		const int nf = static_cast<int>(faces_in.size());
		if (n <= 0 || nf <= 0)
		{
			return {};
		}

		std::vector<char> removed(nf, 0);
		const double mean_edge2 = mean_edge_length_squared(sites, faces_in);

		for (int iter = 0; iter < nf; ++iter)
		{
			std::map<CwfEdgeKey, std::vector<int>> edge_faces;
			for (int fi = 0; fi < nf; ++fi)
			{
				if (removed[fi])
				{
					continue;
				}
				const auto& f = faces_in[fi];
				const int a = f.x();
				const int b = f.y();
				const int c = f.z();
				if (a < 0 || b < 0 || c < 0 ||
					a >= n || b >= n || c >= n ||
					a == b || b == c || a == c ||
					triangle_double_area(sites, f) <= 1e-30)
				{
					removed[fi] = 1;
					continue;
				}
				edge_faces[CwfEdgeKey(a, b)].push_back(fi);
				edge_faces[CwfEdgeKey(b, c)].push_back(fi);
				edge_faces[CwfEdgeKey(c, a)].push_back(fi);
			}

			std::vector<int> face_conflict_count(nf, 0);
			int bad_edges = 0;
			for (const auto& kv : edge_faces)
			{
				const auto& inc = kv.second;
				if (static_cast<int>(inc.size()) > 2)
				{
					++bad_edges;
					for (int fi : inc)
					{
						if (!removed[fi])
						{
							face_conflict_count[fi]++;
						}
					}
				}
			}

			if (bad_edges == 0)
			{
				break;
			}

			int best_remove = -1;
			double best_score = -std::numeric_limits<double>::infinity();
			for (int fi = 0; fi < nf; ++fi)
			{
				if (removed[fi] || face_conflict_count[fi] <= 0)
				{
					continue;
				}
				const double badness = face_badness_score(sites, site_normals, faces_in[fi], mean_edge2);
				const double score = 1000.0 * static_cast<double>(face_conflict_count[fi]) + badness;
				if (score > best_score)
				{
					best_score = score;
					best_remove = fi;
				}
			}

			if (best_remove < 0)
			{
				break;
			}
			removed[best_remove] = 1;
		}

		std::vector<Eigen::Vector3i> faces_out;
		faces_out.reserve(faces_in.size());
		for (int fi = 0; fi < nf; ++fi)
		{
			if (!removed[fi])
			{
				faces_out.push_back(faces_in[fi]);
			}
		}
		return faces_out;
	}

	static inline std::vector<Eigen::Vector3i> build_rdt_faces_robust(
		int num_sites,
		const std::vector<BGAL::_Point3>& sites,
		const BGAL::_Restricted_Tessellation3D& rvd,
		const BGAL::_ManifoldModel* model = nullptr,
		bool use_model_normals = false)
	{
		std::vector<Eigen::Vector3i> corner_faces =
			build_rdt_faces_from_rvd_corners(num_sites, rvd);
		std::vector<Eigen::Vector3i> clique_faces =
			build_rdt_faces_from_edges_fast_raw(num_sites, rvd.get_edges_());

		std::vector<Eigen::Vector3i> candidates;
		if (!corner_faces.empty() &&
			corner_faces.size() >= std::max<std::size_t>(16, clique_faces.size() / 4))
		{
			candidates = std::move(corner_faces);
		}
		else
		{
			candidates = std::move(clique_faces);
		}

		std::vector<Eigen::Vector3d> site_normals;
		if (use_model_normals && model != nullptr)
		{
			site_normals = compute_site_face_normals(sites, *model);
		}

		return filter_nonmanifold_faces_by_edge_valence(sites, candidates, site_normals);
	}

	struct CwfEnaStats
	{
		int num_quads = 0;
		int active_quads = 0;
		double ena = 0.0;
		double min_margin = 0.0;
	};

	static inline bool valid_radius(double r)
	{
		return std::isfinite(r) && r >= 0.0;
	}

	static inline bool pairwise_sphere_overlap(
		const BGAL::_Point3& a,
		double ra,
		const BGAL::_Point3& b,
		double rb)
	{
		const double reach = ra + rb;
		if (!valid_radius(reach))
		{
			return false;
		}
		const double tol = 1e-12 * (1.0 + reach * reach);
		return (a - b).sqlength_() <= reach * reach + tol;
	}

	static inline void build_ena_quads(
		const std::vector<Eigen::Vector3i>& faces,
		const std::vector<BGAL::_Point3>& sites,
		const std::vector<OutputSphere>& spheres,
		std::vector<std::array<int, 4>>& quads)
	{
		quads.clear();
		const int n = static_cast<int>(sites.size());
		if (n <= 0 || faces.empty() || static_cast<int>(spheres.size()) != n)
		{
			return;
		}

		std::vector<double> radii(n, -1.0);
		double global_rmax = 0.0;
#pragma omp parallel for reduction(max : global_rmax)
		for (int i = 0; i < n; ++i)
		{
			const double ri = spheres[i].r;
			if (valid_radius(ri))
			{
				radii[i] = ri;
				global_rmax = std::max(global_rmax, ri);
			}
		}
		if (!valid_radius(global_rmax))
		{
			return;
		}

		BGAL::_KDTree tree(sites);
		std::vector<std::vector<int>> overlap_neighbors(n);
#pragma omp parallel for schedule(dynamic, 32)
		for (int i = 0; i < n; ++i)
		{
			const double ri = radii[i];
			if (!valid_radius(ri))
			{
				continue;
			}
			const double query_radius = ri + global_rmax;
			if (!valid_radius(query_radius))
			{
				continue;
			}

			std::vector<int> hits = tree.rsearch_(sites[i], query_radius);
			auto& nbrs = overlap_neighbors[i];
			nbrs.reserve(hits.size());
			for (int j : hits)
			{
				if (j == i || j < 0 || j >= n || !valid_radius(radii[j]))
				{
					continue;
				}
				if (pairwise_sphere_overlap(sites[i], ri, sites[j], radii[j]))
				{
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
			auto& local_quads = thread_bins[tid];
#pragma omp for schedule(dynamic, 64)
			for (int f_idx = 0; f_idx < static_cast<int>(faces.size()); ++f_idx)
			{
				const auto& f = faces[f_idx];
				const int i = f.x();
				const int j = f.y();
				const int k = f.z();
				if (i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n)
				{
					continue;
				}
				const std::vector<int>* lists[3] = {
					&overlap_neighbors[i], &overlap_neighbors[j], &overlap_neighbors[k]};
				int ref = 0;
				if (lists[1]->size() < lists[ref]->size())
				{
					ref = 1;
				}
				if (lists[2]->size() < lists[ref]->size())
				{
					ref = 2;
				}
				const int other_a = (ref + 1) % 3;
				const int other_b = (ref + 2) % 3;
				for (int l : *lists[ref])
				{
					if (l == i || l == j || l == k)
					{
						continue;
					}
					if (!std::binary_search(lists[other_a]->begin(), lists[other_a]->end(), l) ||
						!std::binary_search(lists[other_b]->begin(), lists[other_b]->end(), l))
					{
						continue;
					}
					std::array<int, 4> q{i, j, k, l};
					std::sort(q.begin(), q.end());
					local_quads.push_back(q);
				}
			}
		}

		std::size_t total = 0;
		for (const auto& bin : thread_bins)
		{
			total += bin.size();
		}
		quads.reserve(total);
		for (auto& bin : thread_bins)
		{
			quads.insert(quads.end(), bin.begin(), bin.end());
		}
		std::sort(quads.begin(), quads.end());
		quads.erase(std::unique(quads.begin(), quads.end()), quads.end());
	}

	static inline CwfEnaStats compute_cwf_ena_stats(
		const std::vector<BGAL::_Point3>& sites,
		const BGAL::_Restricted_Tessellation3D& rvd,
		const BGAL::_ManifoldModel& model)
	{
		CwfEnaStats stats;
		if (sites.empty())
		{
			return stats;
		}

		std::vector<OutputSphere> spheres;
		build_spheres_from_rvd(sites, rvd, spheres);
		const auto rdt_faces =
			build_rdt_faces_robust(static_cast<int>(sites.size()), sites, rvd, &model, false);
		std::vector<std::array<int, 4>> quads;
		build_ena_quads(rdt_faces, sites, spheres, quads);
		stats.num_quads = static_cast<int>(quads.size());
		stats.min_margin = std::numeric_limits<double>::infinity();

		double ena = 0.0;
		int active_quads = 0;
		double min_margin = std::numeric_limits<double>::infinity();
#pragma omp parallel for reduction(+ : ena, active_quads) reduction(min : min_margin)
		for (int qi = 0; qi < static_cast<int>(quads.size()); ++qi)
		{
			const auto& q = quads[qi];
			Eigen::Vector3d p_bar = Eigen::Vector3d::Zero();
			for (int k = 0; k < 4; ++k)
			{
				p_bar += to_eigen(sites[q[k]]);
			}
			p_bar /= 4.0;

			double g_val = 0.0;
			for (int k = 0; k < 4; ++k)
			{
				const int sid = q[k];
				const Eigen::Vector3d pi = to_eigen(sites[sid]);
				const Eigen::Vector3d vi = to_eigen(spheres[sid].max_point);
				g_val += (p_bar - pi).squaredNorm() - (pi - vi).squaredNorm();
			}

			min_margin = std::min(min_margin, g_val);
			const double violation = -g_val;
			if (violation > 0.0)
			{
				ena += violation * violation;
				active_quads += 1;
			}
		}
		stats.ena = ena;
		stats.active_quads = active_quads;
		stats.min_margin = min_margin;
		if (!std::isfinite(stats.min_margin))
		{
			stats.min_margin = 0.0;
		}
		return stats;
	}

	static inline void write_cwf_metrics_svg(
		const std::filesystem::path& svg_path,
		const std::vector<int>& iterations,
		const std::vector<double>& ena_values,
		const std::vector<int>& active_values)
	{
		if (iterations.empty())
		{
			return;
		}
		std::ofstream out(svg_path);
		if (!out)
		{
			return;
		}

		const double width = 960.0;
		const double height = 520.0;
		const double left = 76.0;
		const double right = 34.0;
		const double top = 34.0;
		const double bottom = 68.0;
		const double plot_w = width - left - right;
		const double plot_h = height - top - bottom;
		const int x_min = iterations.front();
		const int x_max = std::max(x_min + 1, iterations.back());
		const double ena_max = std::max(1e-30, *std::max_element(ena_values.begin(), ena_values.end()));
		const int active_max_i = std::max(1, *std::max_element(active_values.begin(), active_values.end()));

		auto x_of = [&](int iter) {
			return left + plot_w * double(iter - x_min) / double(x_max - x_min);
		};
		auto y_ena = [&](double value) {
			return top + plot_h * (1.0 - std::log10(std::max(value, 1e-30)) / std::log10(ena_max + 10.0));
		};
		auto y_active = [&](int value) {
			return top + plot_h * (1.0 - double(value) / double(active_max_i));
		};

		out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
			<< "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
		out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
		out << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"18\">CWF E_na and active quads</text>\n";
		out << "<line x1=\"" << left << "\" y1=\"" << (top + plot_h) << "\" x2=\"" << (left + plot_w)
			<< "\" y2=\"" << (top + plot_h) << "\" stroke=\"#222\"/>\n";
		out << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
			<< "\" y2=\"" << (top + plot_h) << "\" stroke=\"#222\"/>\n";
		out << "<text x=\"" << (left + plot_w / 2.0 - 28.0) << "\" y=\"" << (height - 22.0)
			<< "\" font-family=\"sans-serif\" font-size=\"13\">iteration</text>\n";
		out << "<text x=\"18\" y=\"" << (top + 18.0)
			<< "\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#1565c0\">E_na</text>\n";
		out << "<text x=\"" << (width - 138.0) << "\" y=\"" << (top + 18.0)
			<< "\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#c62828\">active quads</text>\n";

		auto write_polyline = [&](const char* color, auto y_func, const auto& values) {
			out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2.2\" points=\"";
			for (std::size_t i = 0; i < iterations.size(); ++i)
			{
				out << x_of(iterations[i]) << "," << y_func(values[i]) << " ";
			}
			out << "\"/>\n";
		};
		write_polyline("#1565c0", y_ena, ena_values);
		write_polyline("#c62828", y_active, active_values);
		out << "</svg>\n";
	}

	static inline Eigen::Vector3d surface_normal_at_point(
		const BGAL::_ManifoldModel& model,
		const BGAL::_Point3& nearest_point,
		int face_id)
	{
		if (face_id >= 0 && face_id < model.number_faces_())
		{
			BGAL::_Point3 normal(0.0, 0.0, 0.0);
			if (model.has_nonmanifold_topology_())
			{
				normal = model.normal_face_(face_id);
			}
			else
			{
				const auto& face = model.face_(face_id);
				const double dis1 = (nearest_point - model.vertex_(face[0])).length_();
				const double dis2 = (nearest_point - model.vertex_(face[1])).length_();
				const double dis3 = (nearest_point - model.vertex_(face[2])).length_();
				normal += model.normal_vertex_(face[0]) * (dis2 + dis3);
				normal += model.normal_vertex_(face[1]) * (dis1 + dis3);
				normal += model.normal_vertex_(face[2]) * (dis1 + dis2);
			}
			if (normal.sqlength_() > 1e-30)
			{
				normal.normalized_();
				return to_eigen(normal);
			}

			normal = model.normal_face_(face_id);
			if (normal.sqlength_() > 1e-30)
			{
				normal.normalized_();
				return to_eigen(normal);
			}
		}
		return Eigen::Vector3d(0.0, 0.0, 1.0);
	}

	static inline bool write_model_to_off(
		const BGAL::_ManifoldModel& model,
		const std::filesystem::path& path)
	{
		std::ofstream out(path);
		if (!out)
		{
			return false;
		}

		out << std::setprecision(17);
		out << "OFF\n";
		out << model.number_vertices_() << " " << model.number_faces_() << " 0\n";
		for (int vid = 0; vid < model.number_vertices_(); ++vid)
		{
			const auto& v = model.vertex_(vid);
			out << v.x() << " " << v.y() << " " << v.z() << "\n";
		}
		for (int fid = 0; fid < model.number_faces_(); ++fid)
		{
			const auto& f = model.face_(fid);
			out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
		}
		return true;
	}

	static inline bool project_with_aabb_tree(
		const CGALFaceTree& tree,
		const BGAL::_Point3& query,
		BGAL::_Point3& projected,
		Eigen::Vector3d& normal)
	{
		if (!is_finite_point(query))
		{
			return false;
		}

		const CGALClosestFace hit = tree.closest_point_and_primitive(to_cgal_point(query));
		projected = to_bgal_point(hit.first);
		if (!is_finite_point(projected))
		{
			return false;
		}

		auto face = hit.second;
		const auto p1 = face->halfedge()->vertex()->point();
		const auto p2 = face->halfedge()->next()->vertex()->point();
		const auto p3 = face->halfedge()->next()->next()->vertex()->point();
		const Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
		const Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
		const Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
		normal = (v2 - v1).cross(v3 - v1);
		if (!normal.allFinite() || normal.squaredNorm() <= 1e-30)
		{
			normal = Eigen::Vector3d(0.0, 0.0, 1.0);
			return true;
		}
		normal.normalize();
		return true;
	}

	static inline std::list<CGALSegment3> collect_feature_segments(
		const BGAL::_ManifoldModel& model,
		double angle_threshold_degrees,
		double& mean_edge_length)
	{
		std::list<CGALSegment3> segments;
		std::set<std::pair<int, int>> seen_edges;
		double total_edge_length = 0.0;
		int unique_edge_count = 0;
		const double cos_threshold =
			std::cos(angle_threshold_degrees * 3.14159265358979323846 / 180.0);

		for (int eid = 0; eid < model.number_edges_(); ++eid)
		{
			const auto edge = model.edge_(eid);
			if (edge._is_boundary_placeholder)
			{
				continue;
			}

			const int a = edge._id_left_vertex;
			const int b = edge._id_right_vertex;
			if (a < 0 || b < 0)
			{
				continue;
			}

			const std::pair<int, int> key(std::min(a, b), std::max(a, b));
			if (!seen_edges.insert(key).second)
			{
				continue;
			}

			const double length = (model.vertex_(a) - model.vertex_(b)).length_();
			if (std::isfinite(length) && length > 1e-12)
			{
				total_edge_length += length;
				++unique_edge_count;
			}

			bool is_feature = model.is_boundary_edge_(eid);
			if (!is_feature)
			{
				const int reverse_eid = edge._id_reverse_edge;
				if (reverse_eid >= 0)
				{
					const auto reverse_edge = model.edge_(reverse_eid);
					if (reverse_edge._id_face >= 0 && edge._id_face >= 0)
					{
						BGAL::_Point3 n0 = model.normal_face_(edge._id_face);
						BGAL::_Point3 n1 = model.normal_face_(reverse_edge._id_face);
						if (n0.sqlength_() > 1e-30 && n1.sqlength_() > 1e-30)
						{
							n0.normalized_();
							n1.normalized_();
							is_feature = n0.dot_(n1) < cos_threshold;
						}
					}
				}
			}

			if (is_feature)
			{
				segments.emplace_back(to_cgal_point(model.vertex_(a)),
									  to_cgal_point(model.vertex_(b)));
			}
		}

		mean_edge_length =
			(unique_edge_count > 0) ? (total_edge_length / unique_edge_count) : 0.0;
		return segments;
	}

	static inline double feature_density_weight(
		const std::unique_ptr<CGALSegmentTree>& feature_tree,
		double sigma,
		const BGAL::_Point3& p)
	{
		if (!feature_tree || !(sigma > 0.0) || !is_finite_point(p))
		{
			return 1.0;
		}

		const CGALPoint3 closest = feature_tree->closest_point(to_cgal_point(p));
		const double dx = closest.x() - p.x();
		const double dy = closest.y() - p.y();
		const double dz = closest.z() - p.z();
		const double dist2 = dx * dx + dy * dy + dz * dz;
		const double gaussian = std::exp(-0.5 * dist2 / (sigma * sigma));
		return 1.0 + kFeatureDensityBoost * gaussian;
	}

	static inline void load_points_from_xyz(
		const std::string& filepath,
		std::vector<BGAL::_Point3>& sites)
	{
		sites.clear();
		std::ifstream in(filepath);
		if (!in)
		{
			throw std::runtime_error("failed to open init points file: " + filepath);
		}

		std::string line;
		while (std::getline(in, line))
		{
			if (line.empty())
			{
				continue;
			}
			std::istringstream iss(line);
			double x = 0.0, y = 0.0, z = 0.0;
			if (!(iss >> x >> y >> z))
			{
				continue;
			}
			sites.emplace_back(x, y, z);
		}
	}
}



namespace BGAL
{
	_CVT3D::_CVT3D(const _ManifoldModel& model) : _model(model), _RVD(model), _RVD2(model), _para()
	{
		_rho = [](BGAL::_Point3& p)
		{
			return 1;
		};
		_para.is_show = true;
		_para.epsilon = 1e-30;
		_para.max_linearsearch = 20;
		const_cast<_ManifoldModel&>(_model).initialization_PQP_();
	}
	_CVT3D::_CVT3D(const _ManifoldModel& model, std::function<double(_Point3& p)>& rho, _LBFGS::_Parameter para) : _model(model), _RVD(model), _RVD2(model), _rho(rho), _para(para)
	{
		const_cast<_ManifoldModel&>(_model).initialization_PQP_();
	}
	void OutputMesh(const std::vector<_Point3>& sites, const _Restricted_Tessellation3D& RVD, int num, std::string outpath, std::string modelname, int step, const _ManifoldModel& model)
	{
		const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RVD.get_cells_();
		std::string filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		if (step == 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		}

		if (step > 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_RVD.obj";
			}
			std::cout << "filepath = " << filepath << std::endl;
			std::ofstream out(filepath);
			out << std::setprecision(17);
			out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
		for (int i = 0; i < RVD.number_vertices_(); ++i)
		{
			out << "v " << RVD.vertex_(i) << std::endl;
		}
		double totarea = 0, parea = 0;
		for (int i = 0; i < cells.size(); ++i)
		{
			double area = 0;
			for (int j = 0; j < cells[i].size(); ++j)
			{
				BGAL::_Point3 p1 = RVD.vertex_(std::get<0>(cells[i][j]));
				BGAL::_Point3 p2 = RVD.vertex_(std::get<1>(cells[i][j]));
				BGAL::_Point3 p3 = RVD.vertex_(std::get<2>(cells[i][j]));
				area += (p2 - p1).cross_(p3 - p1).length_() / 2;
			}
			totarea += area;

			double color = (double)BGAL::_BOC::rand_();
			if (i > cells.size() / 3)
			{
				if (step == 1)
				{
					color = 0;
				}
				//
			}
			else
			{
				parea += area;
			}

			out << "vt " << color << " 0" << std::endl;


			for (int j = 0; j < cells[i].size(); ++j)
			{
				out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
					<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
					<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
			}
		}
		out.close();


		filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		if (step == 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		}

		if (step > 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_Points.xyz";
		}

			std::ofstream outP(filepath);
			outP << std::setprecision(17);

			int outnum = sites.size();
		if (step == 1)
			outnum = sites.size() / 3;

		for (int i = 0; i < outnum; ++i)
		{
			outP << sites[i] << std::endl;
		}
		outP.close();

		filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Spheres.csv";
		if (step == 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Spheres.csv";
		}

		if (step > 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_Spheres.csv";
		}
		std::vector<OutputSphere> spheres;
		build_spheres_from_rvd(sites, RVD, spheres);
		write_spheres_csv(filepath, spheres, model);


		if (step >= 2)
		{
			std::string filepath = outpath + "\\Ours_" + std::to_string(num) + "_" + modelname + "_Remesh.obj";


				std::string	filepath1 = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "Iter" + std::to_string(step - 3) + "_Remesh.obj";
				std::ofstream outRDT(filepath);
				std::ofstream outRDT1(filepath1);
				outRDT << std::setprecision(17);
				outRDT1 << std::setprecision(17);

				const auto& Vs = sites;
			for (const auto& v : Vs)
			{
				if (step >= 2)
					outRDT << "v " << v << std::endl;
				outRDT1 << "v " << v << std::endl;
			}

			const auto rdtFaces = build_rdt_faces_robust(
				static_cast<int>(Vs.size()), Vs, RVD, &model, true);

			for (const auto& f : rdtFaces)
			{
				if (step >= 2)
					outRDT << "f " << f.x() + 1 << " " << f.y() + 1 << " " << f.z() + 1 << std::endl;
				outRDT1 << "f " << f.x() + 1 << " " << f.y() + 1 << " " << f.z() + 1 << std::endl;
			}

			outRDT.close();
			outRDT1.close();

		}



	}


	void _CVT3D::calculate_(int num_sites, char* modelNamee, char* pointsName)
	{
		std::string modelname = modelNamee == nullptr ? std::string("model") : std::string(modelNamee);
		std::string inPointsName;
		if (pointsName == nullptr)
		{
			inPointsName = std::string("..\\..\\data\\n") + std::to_string(num_sites) + "_" + modelname + "_inputPoints.xyz";
		}
		else
		{
			inPointsName = pointsName;
		}

		std::vector<_Point3> init_sites;
		load_points_from_xyz(inPointsName, init_sites);
		if (pointsName != nullptr)
		{
			num_sites = static_cast<int>(init_sites.size());
		}
		calculate_(init_sites, modelname, false);
	}

	void _CVT3D::calculate_(const std::vector<_Point3>& init_sites,
		const std::string& modelname,
		bool export_process)
	{

		double allTime = 0, RVDtime = 0;
		const double wall_start_time = omp_get_wtime();
		const double max_wall_time_seconds =
			(_para.max_time > 0.0) ? (_para.max_time / 1000.0) : -1.0;
		bool hit_time_limit = false;
		const std::filesystem::path temp_off_path =
			std::filesystem::current_path() / "Temp.off";
		if (!write_model_to_off(_model, temp_off_path))
		{
			throw std::runtime_error("failed to write Temp.off for CWF projection.");
		}

		CGALPolyhedron projection_mesh;
		{
			std::ifstream temp_off_input(temp_off_path);
			if (!temp_off_input || !(temp_off_input >> projection_mesh) ||
				projection_mesh.empty())
			{
				throw std::runtime_error(
					"failed to load Temp.off for CWF AABB projection.");
			}
		}
		CGALFaceTree projection_tree(
			faces(projection_mesh).first,
			faces(projection_mesh).second,
			projection_mesh);
		projection_tree.accelerate_distance_queries();

		double mean_edge_length = 0.0;
		std::list<CGALSegment3> feature_segments;
		if (_use_feature_density_boost)
		{
			feature_segments =
				collect_feature_segments(_model, kFeatureAngleDegrees, mean_edge_length);
		}
		std::unique_ptr<CGALSegmentTree> feature_tree;
		double feature_sigma = 0.0;
		if (_use_feature_density_boost)
		{
			feature_sigma = std::max(kMinFeatureSigma,
									 kFeatureSigmaScale * mean_edge_length);
		}
		if (!feature_segments.empty())
		{
			feature_tree = std::make_unique<CGALSegmentTree>(
				feature_segments.begin(), feature_segments.end());
			feature_tree->accelerate_distance_queries();
		}

		std::vector<Eigen::Vector3d> Pts, Nors;
		Pts.reserve(init_sites.size());
		Nors.reserve(init_sites.size());
		for (const auto& p : init_sites)
		{
			Pts.push_back(to_eigen(p));
			Nors.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
		}
		std::cout << "Pts.size(): " << Pts.size() << std::endl;

		const int num_sites = static_cast<int>(Pts.size());
		int num = static_cast<int>(Pts.size());
		std::cout<< "\nBegin CWF.\n" << std::endl;
		if (_para.is_show)
		{
		std::cout << "[CWF] projection: Temp.off + CGAL AABB tree"
				  << " | feature_edges: " << feature_segments.size()
				  << " | feature_sigma: " << feature_sigma
				  << " | feature_boost: "
				  << (_use_feature_density_boost ? kFeatureDensityBoost : 0.0)
				  << std::endl;
		}


		int Fnum = 4;
		double alpha = 1.0, eplison = 1, lambda = 1; // eplison is CVT weight,  lambda is qe weight.
		double decay = 0.95;
			int accepted_iterations = 0;
		double last_finite_energy = std::numeric_limits<double>::infinity();
		Eigen::VectorXd last_finite_grad = Eigen::VectorXd::Zero(num * 3);
		std::vector<_Point3> last_valid_sites = _sites;
		bool has_last_finite_eval = false;
		CwfEnaStats current_ena_stats;
		double current_loss_qe = 0.0;
		std::vector<int> metric_iterations;
		std::vector<double> metric_ena_values;
		std::vector<int> metric_active_values;
		const std::filesystem::path metrics_csv_path =
			std::filesystem::path(outpath) /
			("CWF_" + std::to_string(num_sites) + "_" + modelname + "_metrics.csv");
		const std::filesystem::path metrics_svg_path =
			std::filesystem::path(outpath) /
			("CWF_" + std::to_string(num_sites) + "_" + modelname + "_metrics.svg");
		std::filesystem::create_directories(metrics_csv_path.parent_path());
		std::ofstream metrics_csv(metrics_csv_path, std::ios::out | std::ios::trunc);
		if (metrics_csv)
		{
			metrics_csv << "iteration,lbfgs_linear_searches,time_seconds,grad_norm,energy,num_quads,active_quads,E_na,hinge_loss,min_margin\n";
			metrics_csv << std::setprecision(17);
		}

		std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgm2
			= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
			{
				if (max_wall_time_seconds > 0.0 &&
					omp_get_wtime() - wall_start_time >= max_wall_time_seconds)
				{
					hit_time_limit = true;
					g.setZero();
					if (has_last_finite_eval)
					{
						_sites = last_valid_sites;
						return last_finite_energy;
					}
					return 0.0;
				}
				eplison = eplison * decay;
				const double cvt_weight = eplison;
				double lossCVT = 0, lossQE = 0, loss = 0;

				const double startRVD = omp_get_wtime();
				project_points_with_aabb_tree_parallel(
					projection_tree, X, _sites, Nors, &last_valid_sites);
				_RVD.calculate_(_sites);
				Fnum++;
				if (export_process && Fnum % 1 == 0)
				{
					OutputMesh(_sites, _RVD, num_sites, outpath, modelname, Fnum, _model); //output process
				}
				RVDtime += omp_get_wtime() - startRVD;

				const std::vector<std::vector<std::tuple<int, int, int>>>& cells = _RVD.get_cells_();
				double energy = 0.0;
				g.setZero();
				std::vector<Eigen::Vector3d> gi;
				gi.resize(num);
				for (int i = 0; i < num; ++i)
				{
					gi[i] = Eigen::Vector3d(0, 0, 0);
				}
				int degenerate_triangles = 0;
				int nonfinite_integrals = 0;

#pragma omp parallel for reduction(+ : lossCVT, loss, degenerate_triangles, nonfinite_integrals)
				for (int i = 0; i < num; ++i)
				{

					for (int j = 0; j < cells[i].size(); ++j)
					{
						BGAL::_Point3  NorTriM = (_RVD.vertex_(std::get<1>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j]))).cross_(_RVD.vertex_(std::get<2>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j])));
						const double tri_norm2 = NorTriM.sqlength_();
						if (!(tri_norm2 > 1e-30) || !std::isfinite(tri_norm2))
						{
							++degenerate_triangles;
							continue;
						}
						NorTriM /= std::sqrt(tri_norm2);

							Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
								[&, NorTriM](BGAL::_Point3 p)
								{
									Eigen::VectorXd r(5);
									BGAL::_Point3 rho_query = p;
									const double rho_p =
										_rho(rho_query) *
										feature_density_weight(feature_tree, feature_sigma, rho_query);

									r(0) = cvt_weight * rho_p * ((_sites[i] - p).sqlength_()); //CVT

									r(1) = lambda*(NorTriM.dot_(p - _sites[i]))* (NorTriM.dot_(p - _sites[i])) + cvt_weight * rho_p * ((p - _sites[i]).sqlength_()); // qe+CVT

								r(2) = lambda* -2 * NorTriM.x() * (NorTriM.dot_(p - _sites[i])) + cvt_weight * rho_p * -2 * (p - _sites[i]).x();  	 //g
								r(3) = lambda* -2 * NorTriM.y() * (NorTriM.dot_(p - _sites[i])) + cvt_weight * rho_p * -2 * (p - _sites[i]).y();	 //g
								r(4) = lambda* -2 * NorTriM.z() * (NorTriM.dot_(p - _sites[i])) + cvt_weight * rho_p * -2 * (p - _sites[i]).z();	 //g


								return r;

								}, _RVD.vertex_(std::get<0>(cells[i][j])), _RVD.vertex_(std::get<1>(cells[i][j])), _RVD.vertex_(std::get<2>(cells[i][j]))
									);
						if (!inte.allFinite())
						{
							++nonfinite_integrals;
							continue;
						}
						lossCVT += alpha * inte(0);
						loss += alpha * inte(1);
						gi[i].x()+= alpha * inte(2);
						gi[i].y()+= alpha * inte(3);
						gi[i].z()+= alpha * inte(4);
					}


				}

				for (int i = 0; i < num; i++)
				{
					if (!gi[i].allFinite())
					{
						gi[i].setZero();
					}
					const double nor_sq = Nors[i].squaredNorm();
					if (Nors[i].allFinite() && nor_sq > 1e-30)
					{
						gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / nor_sq);
					}
					g(i * 3) += gi[i].x();
					g(i * 3 + 1) += gi[i].y();
					g(i * 3 + 2) += gi[i].z();
				}
				energy += loss;

				if (!std::isfinite(energy) || !std::isfinite(lossCVT) || !std::isfinite(loss) || !g.allFinite())
				{
					if (has_last_finite_eval)
					{
						g = last_finite_grad;
						_sites = last_valid_sites;
						energy = last_finite_energy;
					}
					else
					{
						g.setZero();
						energy = 0.0;
					}
				}
				else
				{
					last_finite_energy = energy;
					last_finite_grad = g;
					last_valid_sites = _sites;
					has_last_finite_eval = true;
				}

				current_ena_stats = compute_cwf_ena_stats(_sites, _RVD, _model);
				current_loss_qe = loss - lossCVT;
				const double normalized_loss_cvt =
					(cvt_weight > 0.0) ? (lossCVT / cvt_weight) : 0.0;
				std::cout << std::setprecision(7) << "energy: " << energy << " LossCVT: " << normalized_loss_cvt << " LossQE: " << loss - lossCVT << " Lambda_CVT: " << cvt_weight << std::endl;
				std::cout << "[CWF][E_na] quads=" << current_ena_stats.num_quads
						  << " active_quads=" << current_ena_stats.active_quads
						  << " E_na=" << current_loss_qe
						  << " hinge_loss=" << current_ena_stats.ena
						  << " min_margin=" << current_ena_stats.min_margin << std::endl;

				return energy;
			};


		std::cout << Pts.size()<<"  "<<num << std::endl;
		_sites.resize(num);
		_para.max_linearsearch = 20;
		_para.iteration_callback =
			[&](int iteration, int linear_searches, double elapsed, double grad_norm, double energy)
			{
				if (metrics_csv)
				{
					metrics_csv << iteration << ","
								<< linear_searches << ","
								<< elapsed << ","
								<< grad_norm << ","
								<< energy << ","
								<< current_ena_stats.num_quads << ","
								<< current_ena_stats.active_quads << ","
								<< current_loss_qe << ","
								<< current_ena_stats.ena << ","
								<< current_ena_stats.min_margin << "\n";
					metrics_csv.flush();
				}
				metric_iterations.push_back(iteration);
				metric_ena_values.push_back(current_loss_qe);
				metric_active_values.push_back(current_ena_stats.active_quads);
				accepted_iterations = iteration;
			};
		BGAL::_LBFGS lbfgs2(_para);
		Eigen::VectorXd iterX2(num * 3);
		for (int i = 0; i < num; ++i)
		{
			iterX2(i * 3) =     Pts[i].x();
			iterX2(i * 3 + 1) = Pts[i].y();
			iterX2(i * 3 + 2) = Pts[i].z();
			_sites[i] = BGAL::_Point3(Pts[i](0), Pts[i](1), Pts[i](2));
		}
		{
			const double startRVD = omp_get_wtime();
			_RVD.calculate_(_sites);
			RVDtime += omp_get_wtime() - startRVD;
		}
		const double lbfgs_wall_start = omp_get_wtime();
		lbfgs2.minimize(fgm2, iterX2);
		const double lbfgs_wall_time = omp_get_wtime() - lbfgs_wall_start;
		if (metrics_csv)
		{
			metrics_csv.close();
		}
		write_cwf_metrics_svg(metrics_svg_path, metric_iterations, metric_ena_values, metric_active_values);
		std::cout << "[CWF] metrics csv: " << metrics_csv_path << std::endl;
		std::cout << "[CWF] metrics plot: " << metrics_svg_path << std::endl;
		if (max_wall_time_seconds > 0.0 &&
			omp_get_wtime() - wall_start_time >= max_wall_time_seconds)
		{
			hit_time_limit = true;
		}
		project_points_with_aabb_tree_parallel(
			projection_tree, iterX2, _sites, Nors, &last_valid_sites);
		if (!sites_are_finite(_sites) && sites_are_finite(last_valid_sites))
		{
			_sites = last_valid_sites;
		}
		{
			const double startRVD = omp_get_wtime();
			_RVD.calculate_(_sites);
			RVDtime += omp_get_wtime() - startRVD;
		}

		OutputMesh(_sites, _RVD, num_sites, outpath, modelname, 2, _model);
		allTime = omp_get_wtime() - wall_start_time;
		std::cout << "CWF Wall Time: " << allTime
				  << " s | CWF RVD Wall Time: " << RVDtime
				  << " s | CWF LBFGS Wall Time: " << lbfgs_wall_time
				  << " s | CWF Non-RVD Wall Time: "
				  << std::max(0.0, allTime - RVDtime) << " s"
				  << std::endl;
		if (hit_time_limit)
		{
			std::cout << "tle" << std::endl;
		}


	}
} // namespace BGAL
