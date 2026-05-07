#pragma once
#include <algorithm>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace BGAL
{
  class _LBFGS
  {
  public:
    class _Parameter
    {
    public:
      int m;
      int max_iteration;
      int max_linearsearch;
      double min_step;
      double min_xtol;
      double min_ftol;
      double epsilon;
      bool is_show;
      double wolfe;
      double max_time;
      std::function<void(int, int, double, double, double)> iteration_callback;
      _Parameter();
      _Parameter(const _Parameter &in_parameter);
    };
    _Parameter _parameter;
    _LBFGS();
    _LBFGS(const _Parameter &in_parameter);

    template <class fun>
    int minimize(fun &f, Eigen::VectorXd &iterX);

    template <class fun>
    int minimizeI(fun &f, Eigen::VectorXd &iterX, int IterNum);

  private:
    template <class fun>
    int linear_search_(fun &f,
                       double &fval,
                       Eigen::VectorXd &iterX,
                       Eigen::VectorXd &gradient,
                       double &step,
                       const Eigen::VectorXd &direction);

    template <class fun>
    int linear_searchI_(fun &f,
                        double &fval,
                        Eigen::VectorXd &iterX,
                        Eigen::VectorXd &gradient,
                        double &step,
                        const Eigen::VectorXd &direction,
                        int IterNum);
  };

  template <class fun>
  int _LBFGS::minimize(fun &f, Eigen::VectorXd &iterX)
  {
    const clock_t start_t = clock();
    const int n = iterX.size();
    std::vector<Eigen::VectorXd> s(_parameter.m);
    std::vector<Eigen::VectorXd> y(_parameter.m);
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(_parameter.m);
    Eigen::VectorXd ys = Eigen::VectorXd::Zero(_parameter.m);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd oldX, oldGradient;
    double fval = f(iterX, gradient);
    double grad_norm = gradient.norm();
      if (_parameter.is_show)
      {
        std::cout << 0 << "\t" << 0 << "\t" << (clock() - start_t) * 1.0 / CLOCKS_PER_SEC << "\t" << grad_norm << "\t"
                  << fval << std::endl;
      }
      if (_parameter.iteration_callback)
      {
        _parameter.iteration_callback(0, 0, (clock() - start_t) * 1.0 / CLOCKS_PER_SEC, grad_norm, fval);
      }
    Eigen::VectorXd direction = -gradient;
    int k = 0;
    int l = 0;
    int cursor = 0;
    double step = std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
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
      if (_parameter.max_time > 0 && _parameter.max_time < (clock() - start_t) * 1000.0 / CLOCKS_PER_SEC)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the max time" << std::endl;
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
      if (grad_norm < _parameter.epsilon)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the gradient tolerance" << std::endl;
        }
        return k;
      }
      if (_parameter.is_show)
      {
        std::cout << k << "\t" << l << "\t" << (clock() - start_t) * 1.0 / CLOCKS_PER_SEC << "\t" << grad_norm
                  << "\t" << fval << std::endl;
      }
      if (_parameter.iteration_callback)
      {
        _parameter.iteration_callback(k, l, (clock() - start_t) * 1.0 / CLOCKS_PER_SEC, grad_norm, fval);
      }
      s[cursor] = iterX - oldX;
      y[cursor] = gradient - oldGradient;
      const double curvature = y[cursor].dot(s[cursor]);
      const double yy = y[cursor].dot(y[cursor]);
      if (curvature <= 1e-14 * s[cursor].norm() * y[cursor].norm() || yy <= 0.0)
      {
        direction = -gradient;
        step = std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
        continue;
      }
      ys[cursor] = curvature;
      cursor = (cursor + 1) % _parameter.m;
      const int bound = std::min(k, _parameter.m);
      direction = -gradient;
      int j = cursor;

      for (int i = 0; i < bound; i++)
      {
        j = (j + _parameter.m - 1) % _parameter.m;
        alpha(j) = s[j].dot(direction) / ys(j);
        direction -= alpha(j) * y[j];
      }
      direction *= (curvature / yy);
      for (int i = 0; i < bound; i++)
      {
        const double beta = y[j].dot(direction) / ys(j);
        direction += (alpha(j) - beta) * s[j];
        j = (j + 1) % _parameter.m;
      }
      step = 1.0;
    }
  }

  template <class fun>
  int _LBFGS::minimizeI(fun &f, Eigen::VectorXd &iterX, int IterNum)
  {
    const clock_t start_t = clock();
    const int n = iterX.size();
    std::vector<Eigen::VectorXd> s(_parameter.m);
    std::vector<Eigen::VectorXd> y(_parameter.m);
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(_parameter.m);
    Eigen::VectorXd ys = Eigen::VectorXd::Zero(_parameter.m);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd oldX, oldGradient;
    double fval = f(iterX, gradient, IterNum);
    double grad_norm = gradient.norm();
    if (_parameter.is_show)
    {
      std::cout << 0 << "\t" << 0 << "\t" << (clock() - start_t) * 1.0 / CLOCKS_PER_SEC << "\t" << grad_norm << "\t"
                << fval << std::endl;
    }
    Eigen::VectorXd direction = -gradient;
    int k = 0;
    int l = 0;
    int cursor = 0;
    double step = std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
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
      if (_parameter.max_time > 0 && _parameter.max_time < (clock() - start_t) * 1000.0 / CLOCKS_PER_SEC)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the max time" << std::endl;
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
      int num_linear = linear_searchI_(f, fval, iterX, gradient, step, direction, IterNum);
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
      if (grad_norm < _parameter.epsilon)
      {
        if (_parameter.is_show)
        {
          std::cout << "reach the gradient tolerance" << std::endl;
        }
        return k;
      }
      if (_parameter.is_show)
      {
        std::cout << k << "\t" << l << "\t" << (clock() - start_t) * 1.0 / CLOCKS_PER_SEC << "\t" << grad_norm
                  << "\t" << fval << std::endl;
      }
      s[cursor] = iterX - oldX;
      y[cursor] = gradient - oldGradient;
      const double curvature = y[cursor].dot(s[cursor]);
      const double yy = y[cursor].dot(y[cursor]);
      if (curvature <= 1e-14 * s[cursor].norm() * y[cursor].norm() || yy <= 0.0)
      {
        direction = -gradient;
        step = std::min(1.0, 1.0 / std::max(grad_norm, 1e-12));
        continue;
      }
      ys[cursor] = curvature;
      cursor = (cursor + 1) % _parameter.m;
      const int bound = std::min(k, _parameter.m);
      direction = -gradient;
      int j = cursor;

      for (int i = 0; i < bound; i++)
      {
        j = (j + _parameter.m - 1) % _parameter.m;
        alpha(j) = s[j].dot(direction) / ys(j);
        direction -= alpha(j) * y[j];
      }
      direction *= (curvature / yy);
      for (int i = 0; i < bound; i++)
      {
        const double beta = y[j].dot(direction) / ys(j);
        direction += (alpha(j) - beta) * s[j];
        j = (j + 1) % _parameter.m;
      }
      step = 1.0;
    }
  }

  template <class fun>
  int _LBFGS::linear_searchI_(fun &f,
                              double &fval,
                              Eigen::VectorXd &iterX,
                              Eigen::VectorXd &gradient,
                              double &step,
                              const Eigen::VectorXd &direction,
                              int IterNum)
  {
    if (step < 0)
    {
      std::cout << "error! step<0" << std::endl;
      throw std::runtime_error("error! step<0");
    }
    const double ifval = fval;
    const Eigen::VectorXd iX = iterX;
    const double idg = gradient.dot(direction);
    if (idg >= 0.0)
    {
      return -1;
    }
    double step_l = 0.0;
    double step_u = std::numeric_limits<double>::infinity();
    int k = 1;
    while (1)
    {
      iterX = iX + step * direction;
      fval = f(iterX, gradient, IterNum);
      if (gradient.norm() < _parameter.epsilon)
      {
        break;
      }
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

  template <class fun>
  int _LBFGS::linear_search_(fun &f,
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
    if (idg >= 0.0)
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
      if (gradient.norm() < _parameter.epsilon)
      {
        break;
      }
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
