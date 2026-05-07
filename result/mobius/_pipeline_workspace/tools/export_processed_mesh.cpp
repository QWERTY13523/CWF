#include <filesystem>
#include <iostream>
#include <string>

#include "BGAL/Model/ManifoldModel.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.obj output.obj\n";
        return 1;
    }

    const std::string input_obj = argv[1];
    const std::string output_obj = argv[2];

    try {
        std::filesystem::create_directories(std::filesystem::path(output_obj).parent_path());
        BGAL::_ManifoldModel model(input_obj);
        model.save_obj_file_(output_obj);
        std::cout << "Saved processed mesh to: " << output_obj << "\n";
        std::cout << "Vertices: " << model.number_vertices_() << "\n";
        std::cout << "Faces   : " << model.number_faces_() << "\n";
        std::cout << "Has nonmanifold topology flag: " << model.has_nonmanifold_topology_() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[export_processed_mesh] Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
