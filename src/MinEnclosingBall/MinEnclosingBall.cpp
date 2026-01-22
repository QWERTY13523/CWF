#include "BGAL/MinEnclosingBall/MinEnclosingBall.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

namespace BGAL {

namespace detail {

static const double EPS = 1e-14;

// 0~4 个点构成的支撑球 -----------------------------------

inline MEBall ball_from_1(const _Point3& p) {
    return MEBall(p, 0.0);
}

inline MEBall ball_from_2(const _Point3& a, const _Point3& b) {
    _Point3 center = (a + b) / 2.0;
    double r = std::sqrt((a - b).sqlength_()) * 0.5;
    return MEBall(center, r);
}

// 三点外接圆（在三点所在平面上），再当作球 -----------------
inline MEBall ball_from_3(const _Point3& a,
                          const _Point3& b,
                          const _Point3& c) {
    _Point3 e1 = b - a;
    double e1n2 = e1.sqlength_();
    if (e1n2 < EPS) {
        return ball_from_2(a, c);
    }
    double e1n = std::sqrt(e1n2);
    e1 /= e1n;

    _Point3 n = (b - a).cross_(c - a);
    double nn2 = n.sqlength_();
    if (nn2 < EPS) {
        // 三点近乎共线：退化成最长线段
        double ab = (a - b).sqlength_();
        double ac = (a - c).sqlength_();
        double bc = (b - c).sqlength_();
        if (ab >= ac && ab >= bc) return ball_from_2(a, b);
        if (ac >= ab && ac >= bc) return ball_from_2(a, c);
        return ball_from_2(b, c);
    }
    double nn = std::sqrt(nn2);
    n /= nn;
    _Point3 e2 = n.cross_(e1);  // e1, e2 构成平面基

    auto to2D = [&](const _Point3& p) {
        _Point3 d = p - a;
        double u = d.dot_(e1);
        double v = d.dot_(e2);
        return std::array<double,2>{u, v};
    };

    auto A2 = to2D(a);
    auto B2 = to2D(b);
    auto C2 = to2D(c);

    double ax = A2[0], ay = A2[1];
    double bx = B2[0], by = B2[1];
    double cx = C2[0], cy = C2[1];

    double det = 2.0 * ( ax*(by-cy) + bx*(cy-ay) + cx*(ay-by) );
    if (std::abs(det) < EPS) {
        // 再退化一次
        double ab = (a - b).sqlength_();
        double ac = (a - c).sqlength_();
        double bc = (b - c).sqlength_();
        if (ab >= ac && ab >= bc) return ball_from_2(a, b);
        if (ac >= ab && ac >= bc) return ball_from_2(a, c);
        return ball_from_2(b, c);
    }

    double ax2ay2 = ax*ax + ay*ay;
    double bx2by2 = bx*bx + by*by;
    double cx2cy2 = cx*cx + cy*cy;

    double ux = ( ax2ay2*(by-cy) + bx2by2*(cy-ay) + cx2cy2*(ay-by) ) / det;
    double uy = ( ax2ay2*(cx-bx) + bx2by2*(ax-cx) + cx2cy2*(bx-ax) ) / det;

    _Point3 center = a + e1 * ux + e2 * uy;
    double r = std::sqrt((center - a).sqlength_());
    return MEBall(center, r);
}

// 四点外接球 ---------------------------------------------

inline double det3(double a11, double a12, double a13,
                   double a21, double a22, double a23,
                   double a31, double a32, double a33) {
    return
        a11 * (a22*a33 - a23*a32)
      - a12 * (a21*a33 - a23*a31)
      + a13 * (a21*a32 - a22*a31);
}

inline MEBall ball_from_4(const _Point3& a,
                          const _Point3& b,
                          const _Point3& c,
                          const _Point3& d) {
    double ax = a.x(), ay = a.y(), az = a.z();
    double bx = b.x(), by = b.y(), bz = b.z();
    double cx = c.x(), cy = c.y(), cz = c.z();
    double dx = d.x(), dy = d.y(), dz = d.z();

    // 构造 A * center = rhs
    double a11 = bx - ax;
    double a12 = by - ay;
    double a13 = bz - az;

    double a21 = cx - ax;
    double a22 = cy - ay;
    double a23 = cz - az;

    double a31 = dx - ax;
    double a32 = dy - ay;
    double a33 = dz - az;

    double rhs1 = 0.5 * ( (bx*bx+by*by+bz*bz) - (ax*ax+ay*ay+az*az) );
    double rhs2 = 0.5 * ( (cx*cx+cy*cy+cz*cz) - (ax*ax+ay*ay+az*az) );
    double rhs3 = 0.5 * ( (dx*dx+dy*dy+dz*dz) - (ax*ax+ay*ay+az*az) );

    double detA = det3(a11,a12,a13,
                       a21,a22,a23,
                       a31,a32,a33);

    // 退化：用 ≤3 点支撑的球兜底
    if (std::abs(detA) < EPS) {
        MEBall best;
        best.r = std::numeric_limits<double>::infinity();
        std::array<_Point3,4> Q{a,b,c,d};

        auto try_ball = [&](const MEBall& B) {
            if (!std::isfinite(best.r) || B.r < best.r) {
                bool ok = true;
                for (int i=0;i<4;++i) {
                    if ((Q[i] - B.c).sqlength_() > B.r*B.r + 1e-12) {
                        ok = false; break;
                    }
                }
                if (ok) best = B;
            }
        };

        // 2 点
        for (int i=0;i<4;i++)
            for (int j=i+1;j<4;j++)
                try_ball(ball_from_2(Q[i], Q[j]));
        // 3 点
        for (int i=0;i<4;i++)
            for (int j=i+1;j<4;j++)
                for (int k=j+1;k<4;k++)
                    try_ball(ball_from_3(Q[i], Q[j], Q[k]));

        if (!std::isfinite(best.r)) {
            return ball_from_3(a,b,c);
        }
        return best;
    }

    // Cramer's rule
    double detX = det3(rhs1,a12,a13,
                       rhs2,a22,a23,
                       rhs3,a32,a33);

    double detY = det3(a11,rhs1,a13,
                       a21,rhs2,a23,
                       a31,rhs3,a33);

    double detZ = det3(a11,a12,rhs1,
                       a21,a22,rhs2,
                       a31,a32,rhs3);

    double cx_c = detX / detA;
    double cy_c = detY / detA;
    double cz_c = detZ / detA;

    _Point3 center(cx_c, cy_c, cz_c);
    double r = std::sqrt((center - a).sqlength_());
    return MEBall(center, r);
}

inline MEBall ball_from_support(const std::vector<_Point3>& R) {
    if (R.empty())   return MEBall(_Point3(0.0,0.0,0.0), 0.0);
    if (R.size()==1) return ball_from_1(R[0]);
    if (R.size()==2) return ball_from_2(R[0],R[1]);
    if (R.size()==3) return ball_from_3(R[0],R[1],R[2]);
    return ball_from_4(R[0],R[1],R[2],R[3]);
}

inline bool in_ball(const _Point3& p, const MEBall& B) {
    return (p - B.c).sqlength_() <= B.r*B.r + 1e-12;
}

// Welzl 递归 ----------------------------------------------

inline MEBall welzl_recursive(std::vector<_Point3>& P,
                              std::vector<_Point3> R,
                              int n) {
    if (n == 0 || (int)R.size() == 4) {
        return ball_from_support(R);
    }
    MEBall B = welzl_recursive(P, R, n-1);
    if (in_ball(P[n-1], B)) return B;
    R.push_back(P[n-1]);
    return welzl_recursive(P, R, n-1);
}

} // namespace detail

// ====== 对外接口：吃 _Point3，吐 MEBall（球心也是 _Point3） ======

MEBall minimum_enclosing_ball(const std::vector<_Point3>& points)
{
    if (points.empty()) {
        return MEBall(_Point3(0.0,0.0,0.0), 0.0);
    }

    // Welzl 要打乱顺序，这里拷一份
    std::vector<_Point3> P = points;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(P.begin(), P.end(), g);

    std::vector<_Point3> R;
    R.reserve(4);

    return detail::welzl_recursive(P, R, (int)P.size());
}

} // namespace BGAL
