#pragma once
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace BGAL
{
  class _GradientDescent
  {
  public:
    class _Parameter
    {
    public:
      int max_iteration;
      int max_linearsearch;
      double min_step;
      double min_xtol;
      double min_ftol;
      double epsilon;
      bool is_show;
      double wolfe;
      double init_step;
      _Parameter();
    };
    _Parameter _parameter;
    _GradientDescent();
    _GradientDescent(const _Parameter &in_parameter);
    template <class fun>
    int minimize(fun &f, Eigen::VectorXd &iterX);

  private:
    template <class fun>
    int linear_search_(fun &f,
                       double &fval,
                       Eigen::VectorXd &iterX,
                       Eigen::VectorXd &gradient,
                       double &step,
                       const Eigen::VectorXd &direction);
  };

  template <class fun>
  int _GradientDescent::minimize(fun &f, Eigen::VectorXd &iterX)
  {
    const clock_t start_t = clock();
    const int n = iterX.size();
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd oldX = iterX;
    Eigen::VectorXd oldGradient = Eigen::VectorXd::Zero(n);
    double fval = f(iterX, gradient);
    double grad_norm = gradient.norm();
    Eigen::VectorXd direction = -gradient;
    int k = 0;
    int l = 0;
    double step = (_parameter.init_step > 0.0)
                      ? _parameter.init_step
                      : std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
    while (1)
    {
      if (grad_norm < _parameter.epsilon)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the gradient tolerance" << std::endl;
        }
        return k;
      }
      if (_parameter.max_iteration != 0 && _parameter.max_iteration == k)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the max itertion time" << std::endl;
        }
        return k;
      }

      oldX = iterX;
      oldGradient = gradient;
      if (direction.dot(gradient) >= 0.0)
      {
        direction = -gradient;
      }
      int num_linear = linear_search_(f, fval, iterX, gradient, step, direction);
      if (num_linear == _parameter.max_linearsearch)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the max linearsearch time" << std::endl;
        }
        return k;
      }
      if (num_linear < 0)
      {
        if (_parameter.is_show)
        {
          std::cout << "can't fine a right step" << std::endl;
        }
        return k;
      }
      l += num_linear;
      k++;
      grad_norm = gradient.norm();
      if (_parameter.is_show)
      {
        std::cout << k << "\t" << l << "\t" << (clock() - start_t) * 1.0 / CLOCKS_PER_SEC << "\t" << grad_norm
                  << "\t" << fval << std::endl;
      }

      const Eigen::VectorXd s = iterX - oldX;
      const Eigen::VectorXd y = gradient - oldGradient;
      const double sy = s.dot(y);
      const double ss = s.squaredNorm();
      if (std::abs(sy) > 1e-16 && ss > 0.0)
      {
        step = std::max(_parameter.min_step, std::min(ss / std::abs(sy), 1e6));
      }
      else
      {
        step = std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
      }
      direction = -gradient;
    }
  }

  template <class fun>
  int _GradientDescent::linear_search_(fun &f,
                                       double &fval,
                                       Eigen::VectorXd &iterX,
                                       Eigen::VectorXd &gradient,
                                       double &step,
                                       const Eigen::VectorXd &direction)
  {
    if (step < 0)
    {
      std::cout << "error! step<0" << std::endl;
      throw std::runtime_error("error! step<0");
    }
    const double ifval = fval;
    const Eigen::VectorXd iX = iterX;
    const double idg = gradient.dot(direction);
    if (idg >= 0)
    {
      return -1;
    }
    double step_l = 0.0;
    double step_u = std::numeric_limits<double>::infinity();
    int k = 1;
    while (1)
    {
      iterX = iX + step * direction;
      fval = f(iterX, gradient);
      const double dg = gradient.dot(direction);
      if (fval <= ifval && dg >= _parameter.wolfe * idg)
      {
        break;
      }
      if (fval > ifval)
      {
        step_u = step;
      }
      else
      {
        step_l = step;
        if (dg >= 0.0)
        {
          step_u = step;
        }
      }
      ++k;
      if (k == _parameter.max_linearsearch)
      {
        iterX = iX;
        break;
      }
      if (step < _parameter.min_step)
      {
        k = -1;
        iterX = iX;
        break;
      }
      step = std::isinf(step_u) ? 2.0 * step : 0.5 * (step_l + step_u);
    }
    return k;
  }
} // namespace BGAL
