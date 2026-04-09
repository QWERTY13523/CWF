#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <time.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <unordered_set>

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
		for (int i = 0; i < (int)sites.size(); ++i)
		{
			const BGAL::_Point3 site = sites[i];
			std::unordered_set<int> boundary_vertices;
			if (i < (int)edges.size())
			{
				for (const auto& kv : edges[i])
				{
					for (const auto& e : kv.second)
					{
						boundary_vertices.insert(e.first);
						boundary_vertices.insert(e.second);
					}
				}
			}

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
				const BGAL::_Point3& nearest_point = std::get<0>(nearest);
				const auto& face = model.face_(face_id);
				const double dis1 = (nearest_point - model.vertex_(face[0])).length_();
				const double dis2 = (nearest_point - model.vertex_(face[1])).length_();
				const double dis3 = (nearest_point - model.vertex_(face[2])).length_();
				normal += model.normal_vertex_(face[0]) * (dis2 + dis3);
				normal += model.normal_vertex_(face[1]) * (dis1 + dis3);
				normal += model.normal_vertex_(face[2]) * (dis1 + dis2);
				normal.normalized_();
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

	static inline Eigen::Vector3d surface_normal_at_point(
		const BGAL::_ManifoldModel& model,
		const BGAL::_Point3& nearest_point,
		int face_id)
	{
		if (face_id >= 0 && face_id < model.number_faces_())
		{
			const auto& face = model.face_(face_id);
			const double dis1 = (nearest_point - model.vertex_(face[0])).length_();
			const double dis2 = (nearest_point - model.vertex_(face[1])).length_();
			const double dis3 = (nearest_point - model.vertex_(face[2])).length_();
			BGAL::_Point3 normal(0.0, 0.0, 0.0);
			normal += model.normal_vertex_(face[0]) * (dis2 + dis3);
			normal += model.normal_vertex_(face[1]) * (dis1 + dis3);
			normal += model.normal_vertex_(face[2]) * (dis1 + dis2);
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
			auto Edges = RVD.get_edges_();
			std::set<std::pair<int, int>> RDT_Edges;
			std::vector<std::set<int>> neibors;
			neibors.resize(Vs.size());
			for (int i = 0; i < Edges.size(); i++)
			{
				for (auto ee : Edges[i])
				{
					RDT_Edges.insert(std::make_pair(std::min(i, ee.first), std::max(i, ee.first)));
					neibors[i].insert(ee.first);
					neibors[ee.first].insert(i);
					//std::cout << ee.first << std::endl;

				}
			}

			for (auto v : Vs)
			{
				if (step >= 2)
					outRDT << "v " << v << std::endl;
				outRDT1 << "v " << v << std::endl;
			}

			std::set<MyFace> rdtFaces;

			for (auto e : RDT_Edges)
			{
				for (int pid : neibors[e.first])
				{
					if (RDT_Edges.find(std::make_pair(std::min(pid, e.first), std::max(pid, e.first))) != RDT_Edges.end())
					{
						if (RDT_Edges.find(std::make_pair(std::min(pid, e.second), std::max(pid, e.second))) != RDT_Edges.end())
						{
							int f1 = pid, f2 = e.first, f3 = e.second;

							int mid;
							if (f1 != std::max(f1, std::max(f2, f3)) && f1 != std::min(f1, min(f2, f3)))
							{
								mid = f1;
							}
							if (f2 != std::max(f1, std::max(f2, f3)) && f2 != std::min(f1, std::min(f2, f3)))
							{
								mid = f2;
							}
							if (f3 != std::max(f1, max(f2, f3)) && f3 != std::min(f1, min(f2, f3)))
							{
								mid = f3;
							}
							rdtFaces.insert(MyFace(std::max(f1, std::max(f2, f3)), mid, std::min(f1, std::min(f2, f3))));
						}
					}
				}
			}
			for (auto f : rdtFaces)
			{
				if (step >= 2)
					outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << std::endl;
				outRDT1 << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << std::endl;
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
				for (int i = 0; i < num; ++i)
				{
					const double x = X(i * 3);
					const double y = X(i * 3 + 1);
					const double z = X(i * 3 + 2);
					if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
					{
						if (i < (int)last_valid_sites.size())
						{
							_sites[i] = last_valid_sites[i];
						}
						Nors[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
						continue;
					}
					const BGAL::_Point3 query(x, y, z);
					auto nearest = const_cast<_ManifoldModel&>(_model).nearest_point_(query);
					const BGAL::_Point3 projected = std::get<0>(nearest);
					if (!is_finite_point(projected))
					{
						if (i < (int)last_valid_sites.size())
						{
							_sites[i] = last_valid_sites[i];
						}
						Nors[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
						continue;
					}
					_sites[i] = projected;
					Nors[i] = surface_normal_at_point(_model, _sites[i], std::get<2>(nearest));
					if (!Nors[i].allFinite() || Nors[i].squaredNorm() <= 1e-30)
					{
						Nors[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
					}
				}
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
								const double rho_p = _rho(p);

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
		for (int i = 0; i < num; ++i)
		{
			const double x = iterX2(i * 3);
			const double y = iterX2(i * 3 + 1);
			const double z = iterX2(i * 3 + 2);
			if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
			{
				if (i < (int)last_valid_sites.size())
				{
					_sites[i] = last_valid_sites[i];
				}
				continue;
			}
			const BGAL::_Point3 query(x, y, z);
			auto nearest = const_cast<_ManifoldModel&>(_model).nearest_point_(query);
			const BGAL::_Point3 projected = std::get<0>(nearest);
			if (is_finite_point(projected))
			{
				_sites[i] = projected;
				Nors[i] = surface_normal_at_point(_model, _sites[i], std::get<2>(nearest));
			}
			else if (i < (int)last_valid_sites.size())
			{
				_sites[i] = last_valid_sites[i];
			}

		}
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
