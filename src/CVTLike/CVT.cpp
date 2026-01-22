#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <system_error>
#include <cstring>


#include "BGAL/CVTLike/CVT.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/Integral/Integral.h"
#include "BGAL/Sphere/Sphere.h"


#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/OBJ.h>

#include <igl/gaussian_curvature.h>
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
double kgammaTol = 0.00000000000001;
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
		if (dis < kgammaTol)
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
					return true;
				}
			}

		}
		return false;
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
	void OutputMesh(std::vector<_Point3>& sites, _Restricted_Tessellation3D RVD, int num, std::string outpath, std::string modelname, int step)
	{
		const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RVD.get_cells_();
		std::string filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		if (step == 2)
		{
			filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		}

		if (step > 2)
		{
			filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_RVD.obj";
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

			auto color = (double)BGAL::_BOC::rand_();
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


		filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		if (step == 2)
		{
			filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		}

		if (step > 2)
		{
			filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_Points.xyz";
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


		if (step >= 2)
		{
			std::string filepath = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Remesh.obj";


			std::string	filepath1 = outpath + "/Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_Remesh.obj";
			std::ofstream outRDT(filepath);
			std::ofstream outRDT1(filepath1);

			auto Vs = sites;
			auto Edges = RVD.get_edges_();
			std::set<std::pair<int, int>> RDT_Edges;
			std::vector<std::set<int>> neibors;
			neibors.resize(Vs.size());
			for (int i = 0; i < Edges.size(); i++)
			{
				for (const auto ee : Edges[i])
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
		std::cout<<std::filesystem::current_path()<<std::endl;

		double PI = 3.14159265359;
		std::string modelname = modelNamee;
		Polyhedron polyhedron;
		std::ifstream input("Temp.off");
		input >> polyhedron;
		Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);
		std::unordered_map<const void*, int> face_id_map;
		{
			int idx = 0;
			for (auto fit = polyhedron.facets_begin(); fit != polyhedron.facets_end(); ++fit, ++idx) {
				face_id_map[ static_cast<const void*>(&*fit) ] = idx; // 0-based
			}
		}
		double Movement = 0.01;
		std::string inPointsName;
		namespace fs = std::filesystem;
		fs::path obj("/home/yiming/research/CWF/data/block.obj");
		fs::path base = obj.parent_path();
		if(pointsName == nullptr){
			inPointsName =  base / ("n" + std::to_string(num_sites) + "_" + modelname + "_inputPoints.xyz");
		}else{
			inPointsName = pointsName;
		}
		std::ifstream inPoints(inPointsName.c_str());	
	 	std::vector<Eigen::Vector3d> Pts,Nors;

		int count = 0;
		double x, y, z, nx, ny, nz; // if xyz file has normal
		while (inPoints >> x >> y >> z >> nx >> ny >> nz)
		{
			Pts.push_back(Eigen::Vector3d(x, y, z));
			Nors.push_back(Eigen::Vector3d(nx,ny,nz)); // Nors here is useless, if do not have normal, just set it to (1,0,0)
			++count;
		}
		inPoints.close();
		std::cout<<"Pts.size(): "<<Pts.size()<< std::endl;

         if(pointsName != nullptr){
			num_sites = static_cast<int>(Pts.size());;
		 }
		// begin step 1.
		int num = Pts.size();

		std::vector<Eigen::Vector3d> Pts3;
		std::cout<< "\nBegin CWF.\n" << std::endl;


		int Fnum = 4;
		double alpha = 1.0, eplison = 1, lambda = 1; // eplison is CVT weight,  lambda is qe weight.
		double decay = 0.95;
		std::vector<int> FaceIDs;
		FaceIDs.assign(num, -1);
		std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgm2
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
		{
		    eplison = eplison * decay;
		    double lossCVT = 0, lossQE = 0, loss = 0;
			double lossCenter = 0.0;

		    startRVD = clock();
		    std::vector<Sphere::Sphere> spheres(num);
		    for (int i = 0; i < num; ++i)
		    {
		        Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2)); // project to base surface
		        Point_T closest = tree.closest_point(query);
		        auto tri = tree.closest_point_and_primitive(query);
		        Polyhedron::Face_handle f = tri.second;
		    	int fid = -1;
		    	auto it = face_id_map.find(static_cast<const void*>(&*f));
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
		        BGAL::_Point3 p(closest.x(), closest.y(), closest.z());
		        _sites[i] = p;
		    }
		    _RVD.calculate_(_sites);
		    Fnum++;
		    if (Fnum % 1 == 0)
		    {
		        OutputMesh(_sites, _RVD, num_sites, std::filesystem::current_path()/"data"/"Block", modelname, Fnum);
		    }
		    endRVD = clock();
		    RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

		    const std::vector<std::vector<std::tuple<int, int, int>>>& cells  = _RVD.get_cells_();
		    const std::vector<std::map<int, std::vector<std::pair<int, int>>>>& edges = _RVD.get_edges_();

		    double energy = 0.0;
		    g.setZero();
		    std::vector<Eigen::Vector3d> gi(num, Eigen::Vector3d::Zero());

		    omp_set_num_threads(30);  // change to your CPU core numbers
		#pragma omp parallel for reduction(+:lossCVT,loss,lossCenter)
		    for (int i = 0; i < num; ++i)
		    {
		        _Point3 site = _sites[i];
		    	Eigen::Vector3d xi(site.x(), site.y(), site.z());

		        // ----------------------
		        // 积分与梯度（原逻辑不变）
		        // ----------------------
		        for (int j = 0; j < (int)cells[i].size(); ++j)
		        {
		            auto [a,b,c] = cells[i][j];
		            _Point3 pa = _RVD.vertex_(a), pb = _RVD.vertex_(b), pc = _RVD.vertex_(c);

		            Eigen::VectorXd inte = BGAL::_Integral::integral_triangle3D(
		                [&](BGAL::_Point3 p)
		                {
		                    Eigen::VectorXd r(5);

		                    BGAL::_Point3 NorTriM =
		                        (_RVD.vertex_(std::get<1>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j]))).cross_(
		                         _RVD.vertex_(std::get<2>(cells[i][j])) - _RVD.vertex_(std::get<0>(cells[i][j])));

		                    // 确保真正归一化（若 normalized_ 为返回新向量的版本，可改为：NorTriM = NorTriM.normalized_();）
		                    NorTriM.normalized_();

		                    r(0) = (eplison * _rho(p) * ((_sites[i] - p).sqlength_())); // CVT
		                    r(1) = lambda * (NorTriM.dot_(p - _sites[i])) * (NorTriM.dot_(p - _sites[i])) +
		                           eplison * ((p - _sites[i]).sqlength_());             // QE + CVT

		                    r(2) = lambda * -2 * NorTriM.x() * (NorTriM.dot_(p - _sites[i])) + eplison * -2 * (p - _sites[i]).x(); // gx
		                    r(3) = lambda * -2 * NorTriM.y() * (NorTriM.dot_(p - _sites[i])) + eplison * -2 * (p - _sites[i]).y(); // gy
		                    r(4) = lambda * -2 * NorTriM.z() * (NorTriM.dot_(p - _sites[i])) + eplison * -2 * (p - _sites[i]).z(); // gz

		                    return r;
		                }, pa, pb, pc
		            );

		            lossCVT += alpha * inte(0);
		            loss    += alpha * inte(1);
		            gi[i].x() += alpha * inte(2);
		            gi[i].y() += alpha * inte(3);
		            gi[i].z() += alpha * inte(4);
		        }

		        // -----------------------------------------
		        // 求“定心外接球”的最远点：仅在 cell 边界顶点集合上搜索
		        // -----------------------------------------
		        int    best_vid = -1;
		        double x = site.x(), y = site.y(), z = site.z(); // 默认回退为 site 本身

		        std::unordered_set<int> bnd;
		        if (i < (int)edges.size()) {
		            for (const auto& kv : edges[i]) {
		                for (const auto& e : kv.second) {
		                    bnd.insert(e.first);
		                    bnd.insert(e.second);
		                }
		            }
		        }

		    	std::vector<BGAL::_Point3> boundary_pts;
		    	boundary_pts.reserve(bnd.size() + cells[i].size()*3);

		    	if (!bnd.empty()) {
		    		for (int vid : bnd) {
		    			const auto& pv = _RVD.vertex_(vid);
		    			boundary_pts.emplace_back(pv.x(), pv.y(), pv.z());
		    		}
		    	}
		        // auto consider_vid = [&](int vid) {
		        //     const auto& pv = _RVD.vertex_(vid);
		        //     const double dx = site.x() - pv.x();
		        //     const double dy = site.y() - pv.y();
		        //     const double dz = site.z() - pv.z();
		        //     const double d2 = dx*dx + dy*dy + dz*dz; // 用距离平方比较稳定
		        //     if (d2 > best_d2 || (d2 == best_d2 && vid < best_vid)) {
		        //         best_d2 = d2; best_vid = vid;
		        //         x = pv.x(); y = pv.y(); z = pv.z();
		        //     }
		        // };
		        //
		        // if (!bnd.empty())
		        // {
			       //  for (int vid : bnd) consider_vid(vid);
		        // }
		        // } else {
		        //     // 罕见兜底：边界集合缺失时，退回到 cell 的顶点集合（去重）
		        //     std::unordered_set<int> used;
		        //     used.reserve(cells[i].size()*3);
		        //     for (int j = 0; j < (int)cells[i].size(); ++j) {
		        //         int a,b,c; std::tie(a,b,c) = cells[i][j];
		        //         if (used.insert(a).second) consider_vid(a);
		        //         if (used.insert(b).second) consider_vid(b);
		        //         if (used.insert(c).second) consider_vid(c);
		        //     }
		        // }
		    	// if (Fnum >= 37 && !boundary_pts.empty())
		    	// {
		    	// 	// 1. 计算当前 cell 边界点的最小外接球
		    	// 	BGAL::MEBall B = BGAL::minimum_enclosing_ball(boundary_pts);
			    //
		    	// 	// 2. 把最小外接球球心投影到原始三角网表面
		    	// 	//    Point_T 就是你前面用来投影 site 的 CGAL 点类型
		    	// 	Point_T bc_query(B.c.x(), B.c.y(), B.c.z());
		    	// 	Point_T bc_proj  = tree.closest_point(bc_query);
			    //
		    	// 	// 3. 能量：site 到这个"投影点"的距离平方
		    	// 	double dx = site.x() - bc_proj.x();
		    	// 	double dy = site.y() - bc_proj.y();
		    	// 	double dz = site.z() - bc_proj.z();
		    	// 	double Ei_center = dx*dx + dy*dy + dz*dz;
			    //
		    	// 	// 4. 累加能量和梯度（把投影点当常量，和你之前对 B.c 的处理一致）
		    	// 	lossCenter += eplison * Ei_center;
		    	// 	gi[i].x()  += 2.0 * eplison * dx;
		    	// 	gi[i].y()  += 2.0 * eplison * dy;
		    	// 	gi[i].z()  += 2.0 * eplison * dz;
		    	// }
				BGAL::_Point3 farp = site;
		    	double best_d2 = -1.0;
		    	for (const auto& p : boundary_pts) {
		    		double d2 = (p - site).length_();
		    		if (d2 > best_d2) {
		    			best_d2 = d2;
		    			farp = p;
		    		}
		    	}

		        spheres[i].c = decltype(spheres[i].c)(site.x(), site.y(), site.z());
		        spheres[i].r = best_d2;
		    	spheres[i].max_point =
			decltype(spheres[i].max_point)(farp.x(), farp.y(), farp.z());

		    } // end omp for

		    for (int i = 0; i < num; i++)
		    {
		        gi[i] = gi[i] - Nors[i] * (gi[i].dot(Nors[i]) / Nors[i].dot(Nors[i]));
		        g(i * 3)     += gi[i].x();
		        g(i * 3 + 1) += gi[i].y();
		        g(i * 3 + 2) += gi[i].z();
		    }
		    energy += loss + lossCenter;

		    std::cout << std::setprecision(7)
		              << "energy: "   << energy
		              << " LossCVT: " << lossCVT/eplison
		              << " LossQE: "  << loss - lossCVT
			          <<" LossCenter: " << lossCenter/eplison
		              << " Lambda_CVT: " << eplison << std::endl;

		    namespace fs = std::filesystem;
		    fs::path file  = fs::absolute(fs::current_path() / "data"/ "Block" / ("Sphere_8000_" + std::to_string(Fnum) + ".csv"));
		    fs::path file1 = fs::absolute(fs::current_path() / "data"/ "Block" / "Max_point" / ("MaxPoint_8000_" + std::to_string(Fnum) + ".xyz"));
		    std::cerr << "[io] cwd   = " << fs::current_path().string() << "\n";
		    std::cerr << "[io] write = " << file.string() << "\n";

		    std::ofstream out(file, std::ios::out | std::ios::trunc);
		    if (!out) {
		        std::cerr << "[io] open failed, errno=" << errno << " (" << std::strerror(errno) << ")\n";
		    } else {
		        out << std::setprecision(17);
		        for (int i = 0; i < (int)spheres.size(); ++i) {
		            out << spheres[i].c.x() << "," << spheres[i].c.y() << "," << spheres[i].c.z() << ","
		                << spheres[i].r << "," << FaceIDs[i] << "\n";
		        }
		        out.close();
		        if (!out.good()) std::cerr << "[io] write failed (badbit/failbit set)\n";
		        else std::cerr << "[io] ok\n";
		    }

		    fs::path edge_dir = fs::current_path() / "data" / "Block" / "Edges";
		    fs::create_directories(edge_dir);
		    fs::path edge_obj = fs::absolute(edge_dir / ("Center2Max_8000_" + std::to_string(Fnum) + ".obj"));
		    std::cerr << "[io] write = " << edge_obj.string() << "\n";

		    {
		        std::ofstream eout(edge_obj, std::ios::out | std::ios::trunc);
		        if (!eout) {
		            std::cerr << "[io] open failed (OBJ), errno=" << errno << " (" << std::strerror(errno) << ")\n";
		        } else {
		            eout << std::setprecision(17);
		            for (int i = 0; i < (int)spheres.size(); ++i) {
		                eout << "v " << spheres[i].c.x() << " " << spheres[i].c.y() << " " << spheres[i].c.z() << "\n";
		                eout << "v " << spheres[i].max_point.x() << " " << spheres[i].max_point.y() << " " << spheres[i].max_point.z() << "\n";
		            }
		            for (int i = 0; i < (int)spheres.size(); ++i) {
		                eout << "l " << (2*i + 1) << " " << (2*i + 2) << "\n";
		            }
		            eout.close();
		            if (!eout.good()) std::cerr << "[io] write failed (OBJ)\n"; else std::cerr << "[io] ok (OBJ)\n";
		        }
		    }

		    // ……(后续 coverage 采样代码保持不变，如需可继续保留)
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
			int fid = -1;
			auto it = face_id_map.find(static_cast<const void*>(&*tri.second));
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

		OutputMesh(_sites, _RVD, num_sites, std::filesystem::current_path()/"data"/"Block", modelname, 2);


	}
} // namespace BGAL
