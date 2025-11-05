//
// Created by yiming on 2025/9/25.
//
#include "BGAL/Sphere/Sphere.h"
#include "BGAL/Tessellation3D/Tessellation3D.h"

#include <fstream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <stdexcept>

namespace Sphere
{
    int ExternalBallComputer::writeCSV(
    const std::string& filename,
    const std::vector<Sphere>& spheres,
    bool withHeader)
    {
        std::ofstream out(filename);
        if (!out.is_open()) return -1;

        if (withHeader) out << "x,y,z,r\n";
        out << std::setprecision(17);

        for (const auto& s : spheres) {
            out << s.c.x() << "," << s.c.y() << "," << s.c.z() << "," << s.r << "\n";
        }
        out.close();
        return 0;
    }
};