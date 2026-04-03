#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <time.h>

#include <fstream>
#include <iomanip>
#include <unordered_set>

#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/Integral/Integral.h"
#include "BGAL/Optimization/LinearSystem/LinearSystem.h"


#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/OBJ.h>

#include <igl/gaussian_curvature.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>



typedef CGAL::Simple_cartesian<double> K_T;
typedef K_T::FT FT;
typedef K_T::Point_3 Point_T;

typedef K_T::Segment_3 Segment;
typedef CGAL::Polyhedron_3<K_T> Polyhedron;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits_3<K_T, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;
constexpr double kPointTolerance = 1e-14;
struct MyPoint
{
	MyPoint(Eigen::Vector3d a)
	{
		p = a;

	}

	MyPoint(double a, double b, double c)
	{
		p.x() = a;
		p.y() = b;
		p.z() = c;
	}
	Eigen::Vector3d p;

	bool operator<(const MyPoint& a) const
	{



		double dis = (p - a.p).norm();
		if (dis < kPointTolerance)
		{
			return false;
		}

		if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001)
		{
			if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001)
			{
				return (p.z() < a.p.z());
			}
			return (p.y() < a.p.y());
		}
		return (p.x() < a.p.x());



	}
	bool operator==(const MyPoint& a) const
	{
		if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001)
		{
			if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001)
			{
				if ((p.z() - a.p.z()) < 0.00000000001 && (p.z() - a.p.z()) > -0.00000000001)
				{
					return 1;
				}
			}

		}
		return 0;
	}
};

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
	}
	_CVT3D::_CVT3D(const _ManifoldModel& model, std::function<double(_Point3& p)>& rho, _LBFGS::_Parameter para) : _model(model), _RVD(model), _RVD2(model), _rho(rho), _para(para)
	{

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

		double allTime = 0, RVDtime = 0;
		clock_t start, end;
		clock_t startRVD, endRVD;

		std::string filepath = "../../data/";
		double PI = 3.14159265358;
		std::string modelname = modelNamee;
		Polyhedron polyhedron;
		std::ifstream input("Temp.off");
		input >> polyhedron;
		Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);



		//if anisotropic
		/*
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		igl::readOFF("Temp.off", V, F);
		Eigen::VectorXd K;

		Eigen::MatrixXd HN;
		Eigen::SparseMatrix<double> L, M, Minv;
		igl::cotmatrix(V, F, L);
		igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
		igl::invert_diag(M, Minv);
		// Laplace-Beltrami of position
		HN = -Minv * (L * V);
		// Extract magnitude as mean curvature
		Eigen::VectorXd H = HN.rowwise().norm();

		// Compute curvature directions via quadric fitting
		Eigen::MatrixXd PD1, PD2;
		Eigen::VectorXd PV1, PV2;
		igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
		// mean curvature
		H = 0.5 * (PV1 + PV2);
		map<MyPoint, int> Point2ID;
		for (int i = 0; i < V.rows(); ++i)
		{
			MyPoint p(V(i, 0), V(i, 1), V(i, 2));
			Point2ID[p] = i;
			PD1.row(i) = PD1.row(i).normalized();
			PD2.row(i) = PD2.row(i).normalized();
			double ep = 1;
			PV1(i) += ep;
			PV2(i) += ep;
			double bata = (PV1(i)* PV1(i))*(PV2(i)* PV2(i));
			PV1(i) = sqrt((PV1(i) * PV1(i) ) / bata);
			PV2(i) = sqrt((PV2(i) * PV2(i) ) / bata);
		}
		*/


		double Movement = 0.01;
		std::string inPointsName;
		if(pointsName == nullptr){
			inPointsName = std::string("..\\..\\data\\n") + std::to_string(num_sites)+"_" + modelname + "_inputPoints.xyz";
		}else{
			inPointsName = pointsName;
		}
		std::ifstream inPoints(inPointsName.c_str());

		std::vector<Eigen::Vector3d> Pts,Nors;

		int count = 0;
		double x, y, z, nx, ny, nz; // if xyz file has normal
		while (inPoints >>x >> y >> z >>nx>>ny>>nz)
		{
			Pts.push_back(Eigen::Vector3d(x, y, z));
			Nors.push_back(Eigen::Vector3d(nx,ny,nz)); // Nors here is useless, if do not have normal, just set it to (1,0,0)
			++count;
		}
		inPoints.close();
		std::cout<<"Pts.size(): "<<Pts.size()<< std::endl;

         if(pointsName != nullptr){
			num_sites = count;
		 }
		// begin step 1.
		int num = Pts.size();

		std::vector<Eigen::Vector3d> Pts3;
		std::cout<< "\nBegin CWF.\n" << std::endl;


		int Fnum = 4;
		double alpha = 1.0, eplison = 1, lambda = 1; // eplison is CVT weight,  lambda is qe weight.
		double decay = 0.95;

		std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgm2
			= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
			{
				eplison = eplison * decay;
				double lossCVT = 0, lossQE = 0, loss = 0;

				startRVD = clock();
				for (int i = 0; i < num; ++i)
				{
					Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2)); //project to base surface
					Point_T closest = tree.closest_point(query);
					auto tri = tree.closest_point_and_primitive(query);

					Polyhedron::Face_handle f = tri.second;
					auto p1 = f->halfedge()->vertex()->point();
					auto p2 = f->halfedge()->next()->vertex()->point();
					auto p3 = f->halfedge()->next()->next()->vertex()->point();
					Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
					Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
					Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
					Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
					N.normalize();
					Nors[i] = N;
					BGAL::_Point3 p(closest.x(), closest.y(), closest.z());
					_sites[i] = p;
				}
				_RVD.calculate_(_sites);
				Fnum++;
				if (Fnum % 1 == 0)
				{
					OutputMesh(_sites, _RVD, num_sites, outpath, modelname, Fnum, _model); //output process
				}
				endRVD = clock();
				RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;


				const std::vector<std::vector<std::tuple<int, int, int>>>& cells = _RVD.get_cells_();
				const std::vector<std::map<int, std::vector<std::pair<int, int>>>>& edges = _RVD.get_edges_();
				double energy = 0.0;
				g.setZero();
				std::vector<Eigen::Vector3d> gi;
				gi.resize(num);
				for (int i = 0; i < num; ++i)
				{
					gi[i] = Eigen::Vector3d(0, 0, 0);
				}

#pragma omp parallel for reduction(+ : lossCVT, loss)
				for (int i = 0; i < num; ++i)
				{

					for (int j = 0; j < cells[i].size(); ++j)
					{
						BGAL::_Point3  NorTriM = (_RVD.vertex_(std::get<1>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j]))).cross_(_RVD.vertex_(std::get<2>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j])));
						NorTriM.normalized_();

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
						//energy += alpha * inte(1);
						lossCVT += alpha * inte(0);
						loss += alpha * inte(1);
						gi[i].x()+= alpha * inte(2);
						gi[i].y()+= alpha * inte(3);
						gi[i].z()+= alpha * inte(4);
					}


					// if use exact gradient, then use this

					//for (auto e : edges[i])
					//{
					//
					//	auto p = (0.5 * (_sites[e.first] + _sites[i]));
					//	auto nx = BGAL::_Point3((0.5 * (Nors[e.first] + Nors[i])).x(), (0.5 * (Nors[e.first] + Nors[i])).y(), (0.5 * (Nors[e.first] + Nors[i])).z());
					//	auto addgi = (0.5 * (_sites[e.first] - _sites[i]) / (_sites[e.first] - _sites[i]).length_()) *(  pow((p - _sites[i]).dot_(nx),2) - pow((p - _sites[e.first]).dot_(nx), 2)) * (_sites[e.first] - _sites[i]).length_();
					//	//cout << addgi.length_() << endl;
					//	gi[i] += lambda*Eigen::Vector3d(addgi.x(), addgi.y(), addgi.z());
					//}
					//cout << gi[i].norm() << endl;

				}

				for (int i = 0; i < num; i++)
				{
					gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
					g(i * 3) += gi[i].x();
					g(i * 3 + 1) += gi[i].y();
					g(i * 3 + 2) += gi[i].z();
				}
				energy += loss;

				std::cout << std::setprecision(7) << "energy: " << energy << " LossCVT: " << lossCVT/eplison << " LossQE: " << loss - lossCVT << " Lambda_CVT: " << eplison << std::endl;

				return energy;
			};


			std::vector<Eigen::Vector3d> Pts2;

		Pts2 = Pts;
		num = Pts2.size();
		std::cout << Pts2.size()<<"  "<<num << std::endl;
		_sites.resize(num);
		_para.max_linearsearch = 20;
		BGAL::_LBFGS lbfgs2(_para);
		Eigen::VectorXd iterX2(num * 3);
		for (int i = 0; i < num; ++i)
		{
			iterX2(i * 3) =     Pts2[i].x();
			iterX2(i * 3 + 1) = Pts2[i].y();
			iterX2(i * 3 + 2) = Pts2[i].z();
			_sites[i] = BGAL::_Point3(Pts2[i](0), Pts2[i](1), Pts2[i](2));
		}
		_RVD.calculate_(_sites);
		start = clock();
		lbfgs2.minimize(fgm2, iterX2);
		end = clock();
		allTime += (double)(end - start) / CLOCKS_PER_SEC;
		std::cout<<"allTime: "<<allTime<<" RVDtime: "<<RVDtime<< " L-BFGS time: "<< allTime - RVDtime << std::endl;
		for (int i = 0; i < num; ++i)
		{
			//Point_T query(x0[i * 3], x0[i * 3+1], x0[i * 3+2]);
			Point_T query(iterX2(i * 3), iterX2(i * 3+1), iterX2(i * 3+2));
			Point_T closest = tree.closest_point(query);
			auto tri = tree.closest_point_and_primitive(query);

			Polyhedron::Face_handle f = tri.second;
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

		OutputMesh(_sites, _RVD, num_sites, outpath, modelname, 2, _model);


	}
} // namespace BGAL
