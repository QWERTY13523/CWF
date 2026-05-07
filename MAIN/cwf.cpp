#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include <igl/readOBJ.h>

#include <BGAL/CVTLike/CVT.h>
#include <BGAL/Model/NonManifoldSurface.h>

namespace {

int count_valid_xyz_points(const std::string& filepath)
{
	std::ifstream in(filepath);
	if (!in)
	{
		return 0;
	}

	int count = 0;
	std::string line;
	while (std::getline(in, line))
	{
		if (line.empty())
		{
			continue;
		}
		std::istringstream iss(line);
		double x = 0.0, y = 0.0, z = 0.0;
		if (iss >> x >> y >> z)
		{
			++count;
		}
	}
	return count;
}

std::string shell_quote(const std::string& value)
{
	std::string quoted = "'";
	for (char c : value)
	{
		if (c == '\'')
		{
			quoted += "'\"'\"'";
		}
		else
		{
			quoted.push_back(c);
		}
	}
	quoted.push_back('\'');
	return quoted;
}

std::filesystem::path find_vcg_poisson_sampler()
{
	namespace fs = std::filesystem;
	if (const char* env = std::getenv("CWF_NONMANIFOLD_SAMPLER_EXE"))
	{
		const fs::path configured(env);
		if (!configured.empty() && fs::exists(configured))
		{
			return configured;
		}
	}

	const fs::path candidates[] = {
		fs::path("/home/yiming/research/CWFSampling/build/vcg_poisson_sampling"),
		fs::path("/home/yiming/research/CWFSampling/build-vcpkg/vcg_poisson_sampling"),
		fs::path("/home/yiming/research/CWFSampling/cmake-build-debug/vcg_poisson_sampling"),
	};
	for (const auto& candidate : candidates)
	{
		if (fs::exists(candidate))
		{
			return candidate;
		}
	}
	return {};
}

bool write_obj_mesh(const std::filesystem::path& path,
					const Eigen::MatrixXd& V,
					const Eigen::MatrixXi& F)
{
	std::ofstream out(path);
	if (!out)
	{
		return false;
	}

	out << std::setprecision(17);
	for (int i = 0; i < V.rows(); ++i)
	{
		out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
	}
	for (int i = 0; i < F.rows(); ++i)
	{
		out << "f " << (F(i, 0) + 1) << " " << (F(i, 1) + 1) << " " << (F(i, 2) + 1) << "\n";
	}
	return true;
}

std::filesystem::path make_preprocessed_surface_path(const std::string& points_file,
													 const std::string& model_name)
{
	namespace fs = std::filesystem;
	if (!points_file.empty())
	{
		const fs::path original(points_file);
		const fs::path parent = original.has_parent_path() ? original.parent_path() : fs::current_path();
		return parent / (model_name + "_nonmanifold_surface.obj");
	}
	return fs::current_path() / (model_name + "_nonmanifold_surface.obj");
}

void sample_surface_with_vcg_poisson(const std::filesystem::path& sampler_exe,
									 const std::filesystem::path& surface_obj,
									 const std::filesystem::path& output_xyz,
									 int sample_num,
									 unsigned int seed = 0)
{
	if (sample_num <= 0)
	{
		throw std::runtime_error("surface sample count must be positive.");
	}

	const std::string command =
		shell_quote(sampler_exe.string()) + " " +
		shell_quote(surface_obj.string()) + " " +
		shell_quote(output_xyz.string()) + " " +
		std::to_string(sample_num) + " " +
		std::to_string(seed);

	const int exit_code = std::system(command.c_str());
	if (exit_code != 0)
	{
		throw std::runtime_error(
			"vcg_poisson_sampling failed with exit code " + std::to_string(exit_code));
	}
	if (!std::filesystem::exists(output_xyz))
	{
		throw std::runtime_error(
			"vcg_poisson_sampling did not produce output file: " + output_xyz.string());
	}
}

std::filesystem::path make_resampled_points_path(const std::string& points_file,
												 const std::string& model_name)
{
	namespace fs = std::filesystem;
	if (!points_file.empty())
	{
		const fs::path original(points_file);
		const fs::path parent = original.has_parent_path() ? original.parent_path() : fs::current_path();
		const std::string stem = original.stem().empty() ? model_name : original.stem().string();
		return parent / (stem + "_nonmanifold_resampled.xyz");
	}
	return fs::current_path() / (model_name + "_nonmanifold_resampled.xyz");
}

int resolve_nonmanifold_sample_count(const BGAL::NonManifoldSurface::PreparedTriangleMesh& prepared_surface,
									 const std::string& points_file)
{
	const int file_count = count_valid_xyz_points(points_file);
	if (file_count > 0)
	{
		return file_count;
	}

	if (prepared_surface.output_vertex_count > 0)
	{
		return static_cast<int>(prepared_surface.output_vertex_count);
	}
	if (prepared_surface.input_vertex_count > 0)
	{
		return static_cast<int>(prepared_surface.input_vertex_count);
	}
	return 1024;
}

} // namespace


void CWF3D(std::string file, std::string pointsFile, int max_iteration)
{
	std::string modelname = file;

	// .obj to .off
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOBJ(modelname, V, F))
	{
		throw std::runtime_error("failed to read OBJ surface: " + modelname);
	}

	BGAL::NonManifoldSurface::PreparedTriangleMesh prepared_surface;
	bool used_cgal_fallback = false;
	BGAL::_ManifoldModel model =
		BGAL::NonManifoldSurface::build_manifold_model_allow_non_manifold(
			V, F, &prepared_surface, &used_cgal_fallback);
	if (used_cgal_fallback)
	{
		std::cout
			<< BGAL::NonManifoldSurface::format_preprocess_summary(
				   prepared_surface, "[cwf]")
			<< std::endl;
	}
	else if (model.has_nonmanifold_topology_())
	{
		std::cout << "[cwf] native non-manifold topology enabled | V: "
				  << model.number_vertices_() << " | F: " << model.number_faces_()
				  << std::endl;
	}

	std::function<double(BGAL::_Point3& p)> rho = [](BGAL::_Point3& p)
		{
			return 1;
		};

	BGAL::_LBFGS::_Parameter para;
	para.is_show = true;
	para.epsilon = 1e-30;
	para.max_iteration = max_iteration;
	para.max_time = 0.0;
	BGAL::_CVT3D cvt(model, rho, para);
	cvt.set_use_feature_density_boost(false);
	cvt.set_outpath("./");
	std::string filename = std::filesystem::path(modelname).filename().string();

	if (used_cgal_fallback || model.has_nonmanifold_topology_())
	{
		const auto& sample_surface = used_cgal_fallback ? prepared_surface.V : V;
		const auto& sample_faces = used_cgal_fallback ? prepared_surface.F : F;
		const int sample_num = resolve_nonmanifold_sample_count(prepared_surface, pointsFile);
		const std::filesystem::path sampler_exe = find_vcg_poisson_sampler();
		const std::filesystem::path sampled_points_path =
			make_resampled_points_path(pointsFile, std::filesystem::path(filename).stem().string());
		const std::filesystem::path preprocessed_surface_path =
			make_preprocessed_surface_path(pointsFile, std::filesystem::path(filename).stem().string());

		if (sampler_exe.empty())
		{
			throw std::runtime_error(
				"cannot find vcg_poisson_sampling. Set CWF_NONMANIFOLD_SAMPLER_EXE or build "
				"/home/yiming/research/CWFSampling.");
		}
		if (!write_obj_mesh(preprocessed_surface_path, sample_surface, sample_faces))
		{
			throw std::runtime_error(
				"failed to save preprocessed non-manifold surface: " +
				preprocessed_surface_path.string());
		}

		std::cout << "[cwf] sampling preprocessed surface with "
				  << sampler_exe << std::endl;
		sample_surface_with_vcg_poisson(
			sampler_exe, preprocessed_surface_path, sampled_points_path, sample_num, 0u);
		const int sampled_count = count_valid_xyz_points(sampled_points_path.string());
		std::cout << "[cwf] resampled " << sampled_count
				  << " points on the preprocessed surface -> "
				  << sampled_points_path << std::endl;

		int num = 0;
		cvt.calculate_(num, (char*)filename.c_str(), (char*)sampled_points_path.string().c_str());
		return;
	}

	int num = 0;
	cvt.calculate_(num, (char*)filename.c_str(), (char*)pointsFile.c_str());
}


int main(int argc, char* argv[])
{
	std::cout << "argc = " << argc << std::endl;
	if (argc < 3)
	{
		std::cerr << "Usage: cwf <surface.obj> <inputPoints.xyz> [max_iteration]" << std::endl;
		return 1;
	}
	int max_iteration = (argc>3)?std::stoi(argv[3]):80;
	CWF3D(argv[1], argv[2], max_iteration);

	return 0;
}
