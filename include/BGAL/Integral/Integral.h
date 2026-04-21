#pragma once
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "BGAL/BaseShape/Polygon.h"
#include "Tetrahedron_arbq_rule.h"

namespace BGAL
{
	class _Integral
	{
	public:
		template <class F>
		static Eigen::VectorXd integral_triangle(F f, const _Polygon &poly)
		{
			if (poly.num_() != 3)
				throw "poly is not a triangle!";
			return integral_triangle(f, poly[0], poly[1], poly[2]);
		}

		template <class F>
		static Eigen::VectorXd integral_triangle(F f, const _Point2 &p1, const _Point2 &p2, const _Point2 &p3)
		{
			const _Point2 m23 = (p2 + p3) * 0.5;
			const _Point2 m12 = (p1 + p2) * 0.5;
			const _Point2 m13 = (p1 + p3) * 0.5;
			const _Point2 c1 = p1 / 6.0 + p2 / 6.0 + p3 * (2.0 / 3.0);
			const _Point2 c2 = p3 / 6.0 + p1 / 6.0 + p2 * (2.0 / 3.0);
			const _Point2 c3 = p2 / 6.0 + p3 / 6.0 + p1 * (2.0 / 3.0);
			Eigen::VectorXd r1 = (1.0 / 30.0) * f(m23);
			Eigen::VectorXd r2 = (1.0 / 30.0) * f(m12);
			Eigen::VectorXd r3 = (1.0 / 30.0) * f(m13);
			Eigen::VectorXd r4 = (9.0 / 30.0) * f(c1);
			Eigen::VectorXd r5 = (9.0 / 30.0) * f(c2);
			Eigen::VectorXd r6 = (9.0 / 30.0) * f(c3);
			const _Point2 v1 = p2 - p1;
			const _Point2 v2 = p3 - p1;
			const double area = 0.5 * std::fabs(v1.cross_(v2).z());
			return area * (r1 + r2 + r3 + r4 + r5 + r6);
		}

		template <class F>
		static Eigen::VectorXd integral_triangle3D(F f, const _Point3 &p1, const _Point3 &p2, const _Point3 &p3)
		{
			const _Point3 m23 = (p2 + p3) * 0.5;
			const _Point3 m12 = (p1 + p2) * 0.5;
			const _Point3 m13 = (p1 + p3) * 0.5;
			const _Point3 c1 = p1 / 6.0 + p2 / 6.0 + p3 * (2.0 / 3.0);
			const _Point3 c2 = p3 / 6.0 + p1 / 6.0 + p2 * (2.0 / 3.0);
			const _Point3 c3 = p2 / 6.0 + p3 / 6.0 + p1 * (2.0 / 3.0);
			Eigen::VectorXd r1 = (1.0 / 30.0) * f(m23);
			Eigen::VectorXd r2 = (1.0 / 30.0) * f(m12);
			Eigen::VectorXd r3 = (1.0 / 30.0) * f(m13);
			Eigen::VectorXd r4 = (9.0 / 30.0) * f(c1);
			Eigen::VectorXd r5 = (9.0 / 30.0) * f(c2);
			Eigen::VectorXd r6 = (9.0 / 30.0) * f(c3);
			const _Point3 v1 = p2 - p1;
			const _Point3 v2 = p3 - p1;
			const double area = 0.5 * std::sqrt(v1.cross_(v2).sqlength_());
			return area * (r1 + r2 + r3 + r4 + r5 + r6);
		}

		template <class F>
		static Eigen::VectorXd integral_polygon(F f, const _Polygon &poly)
		{
			std::vector<_Polygon> tris = poly.constrained_delaunay_triangulation_();
			Eigen::VectorXd r = integral_triangle(f, tris[0]);
			for (int i = 1; i < static_cast<int>(tris.size()); ++i)
			{
				r += integral_triangle(f, tris[static_cast<std::size_t>(i)]);
			}
			return r;
		}

		template <class F>
		static Eigen::VectorXd integral_polygon_fast(F f, const _Polygon &poly)
		{
			Eigen::VectorXd r = integral_triangle(f, poly[0], poly[1], poly[2]);
			for (int i = 2; i < poly.num_() - 1; ++i)
			{
				r += integral_triangle(f, poly[0], poly[i], poly[i + 1]);
			}
			return r;
		}

		template <class F>
		static Eigen::VectorXd integral_tetrahedron(F f, const _Point3 &p1, const _Point3 &p2, const _Point3 &p3, const _Point3 &p4)
		{
			struct _KeastCache
			{
				int order_num = 0;
				std::vector<double> xyz;
				std::vector<double> w;
				_KeastCache()
				{
					order_num = keast_order_num(4);
					xyz.resize(static_cast<std::size_t>(3 * order_num));
					w.resize(static_cast<std::size_t>(order_num));
					keast_rule(4, order_num, xyz.data(), w.data());
				}
			};
			static const _KeastCache cache;

			const _Point3 e1 = p2 - p1;
			const _Point3 e2 = p3 - p1;
			const _Point3 e3 = p4 - p1;
			const double volume = std::fabs(e1.dot_(e2.cross_(e3))) / 6.0;

			_Point3 sample0 = p1 + e1 * cache.xyz[0] + e2 * cache.xyz[1] + e3 * cache.xyz[2];
			Eigen::VectorXd result = volume * cache.w[0] * f(sample0);
			for (int i = 1; i < cache.order_num; ++i)
			{
				const int base = 3 * i;
				_Point3 sample = p1 + e1 * cache.xyz[static_cast<std::size_t>(base + 0)] +
				                 e2 * cache.xyz[static_cast<std::size_t>(base + 1)] +
				                 e3 * cache.xyz[static_cast<std::size_t>(base + 2)];
				result += volume * cache.w[static_cast<std::size_t>(i)] * f(sample);
			}
			return result;
		}
	};
}
