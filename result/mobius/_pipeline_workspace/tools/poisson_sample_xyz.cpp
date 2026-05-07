#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/bounding.h>

#include <wrap/io_trimesh/import_obj.h>

class MyVertex;
class MyFace;
class MyEdge;

struct MyUsedTypes : public vcg::UsedTypes<
    vcg::Use<MyVertex>::AsVertexType,
    vcg::Use<MyEdge>::AsEdgeType,
    vcg::Use<MyFace>::AsFaceType> {};

class MyVertex : public vcg::Vertex<
    MyUsedTypes,
    vcg::vertex::Coord3f,
    vcg::vertex::Normal3f,
    vcg::vertex::BitFlags> {};

class MyFace : public vcg::Face<
    MyUsedTypes,
    vcg::face::VertexRef,
    vcg::face::Normal3f,
    vcg::face::BitFlags> {};

class MyEdge : public vcg::Edge<
    MyUsedTypes> {};

class MyMesh : public vcg::tri::TriMesh<
    std::vector<MyVertex>,
    std::vector<MyFace>,
    std::vector<MyEdge>> {};

namespace {

size_t CountLiveVertices(const MyMesh& mesh) {
    size_t count = 0;
    for (const auto& v : mesh.vert) {
        if (!v.IsD()) {
            ++count;
        }
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " input.obj output.xyz sample_num [seed]\n\n"
                  << "Example:\n"
                  << "  " << argv[0]
                  << " bunny.obj samples.xyz 10000 12345\n";
        return 1;
    }

    const std::string input_obj  = argv[1];
    const std::string output_xyz = argv[2];
    const int sample_num         = std::atoi(argv[3]);
    const unsigned int seed      = (argc >= 5) ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 0u;

    if (sample_num <= 0) {
        std::cerr << "sample_num must be > 0\n";
        return 1;
    }

    MyMesh mesh;

    int load_mask = 0;
    int err = vcg::tri::io::ImporterOBJ<MyMesh>::Open(mesh, input_obj.c_str(), load_mask);
    if (err != 0) {
        std::cerr << "Failed to load OBJ: " << input_obj << "\n";
        std::cerr << "Error code: " << err << "\n";
        std::cerr << "Error msg : " << vcg::tri::io::ImporterOBJ<MyMesh>::ErrorMsg(err) << "\n";
        return 1;
    }

    if (mesh.fn == 0 || mesh.vn == 0) {
        std::cerr << "Mesh is empty or has no valid triangles.\n";
        return 1;
    }

    vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalizedPerFaceNormalized(mesh);
    vcg::tri::UpdateBounding<MyMesh>::Box(mesh);

    MyMesh montecarlo_mesh;
    MyMesh sample_mesh;

    using MeshSampler = vcg::tri::MeshSampler<MyMesh>;
    using SurfaceSampler = vcg::tri::SurfaceSampling<MyMesh, MeshSampler>;

    MeshSampler mc_sampler(montecarlo_mesh);
    MeshSampler output_sampler(sample_mesh);

    const int fast_candidate_num = std::max(10000, sample_num * 12);
    const int fallback_candidate_num = std::max(10000, sample_num * 40);
    const float pruning_tolerance = 0.04f;
    const size_t min_acceptable_samples = static_cast<size_t>(
        std::ceil(sample_num * (1.0f - pruning_tolerance))
    );

    MyMesh::ScalarType radius = vcg::tri::ComputePoissonDiskRadius(mesh, sample_num);
    SurfaceSampler::PoissonDiskParam pp;
    pp.pds.sampleNum = sample_num;
    pp.randomSeed = seed;

    if (seed) {
        SurfaceSampler::SamplingRandomGenerator().initialize(seed);
    }

    SurfaceSampler::Montecarlo(mesh, mc_sampler, fast_candidate_num);
    vcg::tri::UpdateBounding<MyMesh>::Box(montecarlo_mesh);

    bool fast_path_succeeded = false;
    size_t produced_samples = 0;

    for (int iter = 0; iter < 3; ++iter) {
        output_sampler.reset();
        SurfaceSampler::PoissonDiskPruning(output_sampler, montecarlo_mesh, radius, pp);
        produced_samples = CountLiveVertices(sample_mesh);

        if (produced_samples >= min_acceptable_samples) {
            fast_path_succeeded = true;
            break;
        }

        if (produced_samples == 0) {
            radius *= 0.5f;
            continue;
        }

        const float ratio = std::sqrt(
            static_cast<float>(produced_samples) / static_cast<float>(sample_num)
        );
        radius *= std::max(0.5f, std::min(0.95f, ratio * 0.98f));
    }

    if (!fast_path_succeeded) {
        montecarlo_mesh.Clear();
        sample_mesh.Clear();

        SurfaceSampler::Montecarlo(mesh, mc_sampler, fallback_candidate_num);
        vcg::tri::UpdateBounding<MyMesh>::Box(montecarlo_mesh);

        radius = 0.0f;
        SurfaceSampler::PoissonDiskPruningByNumber(
            output_sampler,
            montecarlo_mesh,
            sample_num,
            radius,
            pp,
            pruning_tolerance
        );
    }

    std::ofstream ofs(output_xyz);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_xyz << "\n";
        return 1;
    }

    ofs.setf(std::ios::fixed);
    ofs.precision(8);

    size_t written_samples = 0;
    for (const auto& v : sample_mesh.vert) {
        if (v.IsD()) {
            continue;
        }

        if (written_samples >= static_cast<size_t>(sample_num)) {
            break;
        }

        auto n = v.cN();
        if (n.SquaredNorm() > 0) {
            n.Normalize();
        }

        ofs << v.cP()[0] << ' ' << v.cP()[1] << ' ' << v.cP()[2] << ' '
            << n[0] << ' ' << n[1] << ' ' << n[2] << '\n';
        ++written_samples;
    }

    ofs.close();

    std::cout << "Input mesh     : " << input_obj << "\n";
    std::cout << "Vertices       : " << mesh.vn << "\n";
    std::cout << "Faces          : " << mesh.fn << "\n";
    std::cout << "Sample number  : " << written_samples << "\n";
    std::cout << "Estimated r    : " << radius << "\n";
    std::cout << "Saved to       : " << output_xyz << "\n";

    return 0;
}
