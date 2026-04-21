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

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_set>
#include <omp.h>

#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Algorithm/BOC/BOC.h"
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

	static inline std::vector<Eigen::Vector3i> build_rdt_faces_from_edges_fast(
		int num_sites,
		const std::vector<std::map<int, std::vector<std::pair<int, int>>>>& edges)
	{
		std::vector<std::vector<int>> adj(num_sites);
		for (int i = 0; i < (int)edges.size(); ++i)
		{
			for (const auto& ee : edges[i])
			{
				const int j = ee.first;
				if (j >= 0 && j < num_sites && j != i)
				{
					adj[i].push_back(j);
				}
			}
		}

#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_sites; ++i)
		{
			auto& nbrs = adj[i];
			std::sort(nbrs.begin(), nbrs.end());
			nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
		}

		std::vector<Eigen::Vector3i> tris;
#pragma omp parallel
		{
			std::vector<Eigen::Vector3i> local_tris;
#pragma omp for schedule(dynamic)
			for (int u = 0; u < num_sites; ++u)
			{
				for (int v : adj[u])
				{
					if (u >= v)
					{
						continue;
					}

					int pu = 0, pv = 0;
					while (pu < (int)adj[u].size() && pv < (int)adj[v].size())
					{
						if (adj[u][pu] == adj[v][pv])
						{
							const int w = adj[u][pu];
							if (v < w)
							{
								local_tris.emplace_back(u, v, w);
							}
							++pu;
							++pv;
						}
						else if (adj[u][pu] < adj[v][pv])
						{
							++pu;
						}
						else
						{
							++pv;
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

			const auto& Vs = sites;
			for (const auto& v : Vs)
			{
				if (step >= 2)
					outRDT << "v " << v << std::endl;
				outRDT1 << "v " << v << std::endl;
			}

			const auto rdtFaces = build_rdt_faces_from_edges_fast(
				static_cast<int>(Vs.size()), RVD.get_edges_());

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
		calculate_(init_sites, modelname, true);
	}

	void _CVT3D::calculate_(const std::vector<_Point3>& init_sites,
		const std::string& modelname,
		bool export_process)
	{

		double allTime = 0, RVDtime = 0;
		clock_t start, end;
		clock_t startRVD, endRVD;
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
		double last_finite_energy = std::numeric_limits<double>::infinity();
		Eigen::VectorXd last_finite_grad = Eigen::VectorXd::Zero(num * 3);
		std::vector<_Point3> last_valid_sites = _sites;
		bool has_last_finite_eval = false;

		std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgm2
			= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
			{
				eplison = eplison * decay;
				double lossCVT = 0, lossQE = 0, loss = 0;

				startRVD = clock();
				project_points_with_aabb_tree_parallel(
					projection_tree, X, _sites, Nors, &last_valid_sites);
				_RVD.calculate_(_sites);
				Fnum++;
				if (export_process && Fnum % 1 == 0)
				{
					OutputMesh(_sites, _RVD, num_sites, outpath, modelname, Fnum, _model); //output process
				}
				endRVD = clock();
				RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

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

									r(0) = eplison * rho_p * ((_sites[i] - p).sqlength_()); //CVT

									r(1) = lambda*(NorTriM.dot_(p - _sites[i]))* (NorTriM.dot_(p - _sites[i])) + eplison * rho_p * ((p - _sites[i]).sqlength_()); // qe+CVT

								r(2) = lambda* -2 * NorTriM.x() * (NorTriM.dot_(p - _sites[i])) + eplison * rho_p * -2 * (p - _sites[i]).x();  	 //g
								r(3) = lambda* -2 * NorTriM.y() * (NorTriM.dot_(p - _sites[i])) + eplison * rho_p * -2 * (p - _sites[i]).y();	 //g
								r(4) = lambda* -2 * NorTriM.z() * (NorTriM.dot_(p - _sites[i])) + eplison * rho_p * -2 * (p - _sites[i]).z();	 //g


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

				std::cout << std::setprecision(7) << "energy: " << energy << " LossCVT: " << lossCVT/eplison << " LossQE: " << loss - lossCVT << " Lambda_CVT: " << eplison << std::endl;

				return energy;
			};


		std::cout << Pts.size()<<"  "<<num << std::endl;
		_sites.resize(num);
		_para.max_linearsearch = 20;
		BGAL::_LBFGS lbfgs2(_para);
		Eigen::VectorXd iterX2(num * 3);
		for (int i = 0; i < num; ++i)
		{
			iterX2(i * 3) =     Pts[i].x();
			iterX2(i * 3 + 1) = Pts[i].y();
			iterX2(i * 3 + 2) = Pts[i].z();
			_sites[i] = BGAL::_Point3(Pts[i](0), Pts[i](1), Pts[i](2));
		}
		_RVD.calculate_(_sites);
		start = clock();
		lbfgs2.minimize(fgm2, iterX2);
		end = clock();
		allTime += (double)(end - start) / CLOCKS_PER_SEC;
		std::cout<<"allTime: "<<allTime<<" RVDtime: "<<RVDtime<< " L-BFGS time: "<< allTime - RVDtime << std::endl;
		project_points_with_aabb_tree_parallel(
			projection_tree, iterX2, _sites, Nors, &last_valid_sites);
		if (!sites_are_finite(_sites) && sites_are_finite(last_valid_sites))
		{
			_sites = last_valid_sites;
		}
		_RVD.calculate_(_sites);

		if (export_process)
		{
			OutputMesh(_sites, _RVD, num_sites, outpath, modelname, 2, _model);
		}


	}
} // namespace BGAL
