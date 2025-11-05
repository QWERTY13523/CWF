//
// Created by yiming on 2025/11/3.
//

#ifndef BGAL_MINENCLOSINGBALL_H
#define BGAL_MINENCLOSINGBALL_H

#pragma once

#include <vector>
#include <Eigen/Core>
#include <BGAL/BaseShape/Point.h>

namespace BGAL {

struct MEBall {
    BGAL::_Point3 c;
    double  r;

    MEBall() : c(BGAL::_Point3(0.0,0.0,0.0)), r(0.0) {}
    MEBall(const BGAL::_Point3& center, double radius) : c(center), r(radius) {}
};

/**
 * @brief 对 3D 点集（_Point3）计算最小外接球（Minimum Enclosing Ball）
 *
 * @param points  输入点集
 * @return MEBall 球心为 _Point3 类型，半径 double
 */
MEBall minimum_enclosing_ball(const std::vector<BGAL::_Point3>& points);

}// namespace geom

#endif //BGAL_MINENCLOSINGBALL_H