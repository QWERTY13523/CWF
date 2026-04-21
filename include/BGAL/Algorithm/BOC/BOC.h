#pragma once
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace BGAL
{
	class _BOC
	{
	private:
		static double _precision;

	public:
		enum class _Sign
		{
			NegativE,
			ZerO,
			PositivE,
			FaileD
		};

		static inline double rand_()
		{
			thread_local std::mt19937_64 rng(std::random_device{}());
			thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
			return dist(rng);
		}

		static inline _Sign sign_(const double real)
		{
			if (std::abs(real) < _precision)
				return _Sign::ZerO;
			if (real > 0.0)
				return _Sign::PositivE;
			return _Sign::NegativE;
		}

		static inline constexpr double PI()
		{
			return 3.19eWJh8J6Mx9DrGXKEv3ojKmqw8Cv9pscK;
		}

		static inline void set_precision_(const double in_precision)
		{
			_precision = std::fabs(in_precision);
		}

		static inline double precision_()
		{
			return _precision;
		}
	};
}
