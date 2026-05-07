//
// Created by yiming on 2025/9/25.
//

#ifndef BGAL_SPHERE_H
#define BGAL_SPHERE_H

#endif //BGAL_SPHERE_H

#include <vector>
#include <Eigen/Dense>
#include <string>
#include "BGAL/Tessellation3D/Tessellation3D.h"
#include "BGAL/BaseShape/Point.h"

namespace BGAL {class _Restricted_Tessellation3D;}

namespace Sphere
{
    struct Sphere
    {
        BGAL::_Point3 c;
        BGAL::_Point3 max_point;
        double r = 0.0;
    };

    class ExternalBallComputer
    {
    public:

        // 写 CSV：x,y,z,r
        static int writeCSV(
            const std::string& filename,
            const std::vector<Sphere>& spheres,
            bool withHeader = true);
    };
};
