//#include "BKKDTree.h"
#include "BGAL/BaseShape/KDTree.h"
#include <numeric>
#include <algorithm>
#include <stack>
#include <queue>
#include <tuple>
namespace BGAL
{
	_KDTree::_KDTree() : _root(nullptr)
	{
		_points.clear();
	}
	_KDTree::_KDTree(const std::vector<_Point3> &in_points) : _root(nullptr)
	{
		build_(in_points);
	}
	_KDTree::~_KDTree()
	{
		clear_();
	}
	int _KDTree::search_(const _Point3 &in_p) const
	{
		double _min_dist;
		int guess = search_(in_p, _min_dist);
		return guess;
	}
	int _KDTree::search_(const _Point3 &in_p, double &min_dist) const
	{
		int guess;
		min_dist = std::numeric_limits<double>::max();
		std::vector<bool> visited(_points.size(), false);
		std::stack<_Node *> S;
		S.push(_root);
		while (!S.empty())
		{
			if (!S.top())
			{
				S.pop();
				continue;
			}
			const int axis = S.top()->_axis;
			//std::cout << axis << std::endl;
			//std::cout << S.top()->_id << std::endl;
			const int dir = in_p[axis] < _points[S.top()->_id][axis] ? 0 : 1;
			if (!S.top()->_next[dir] || visited[S.top()->_next[dir]->_id])
			{
				const _Point3 &tp = _points[S.top()->_id];
				double dist = (in_p - tp).length_();
				if (dist < min_dist)
				{
					min_dist = dist;
					guess = S.top()->_id;
				}
				double diff = fabs(in_p[axis] - tp[axis]);
				if (diff < min_dist)
				{
					if (!S.top()->_next[!dir] || visited[S.top()->_next[!dir]->_id])
					{
						visited[S.top()->_id] = true;
						S.pop();
					}
					else
					{
						S.push(S.top()->_next[!dir]);
					}
				}
				else
				{
					visited[S.top()->_id] = true;
					S.pop();
				}
			}
			else
			{
				S.push(S.top()->_next[dir]);
			}
		}
		return guess;
	}
	std::vector<int> _KDTree::rsearch_(const std::vector<_Point3> &in_ps, const double &in_r) const
	{
		std::vector<int> res;
		if (!_root || _points.empty() || in_ps.empty() || in_r < 0.0)
		{
			return res;
		}

		const double radius2 = in_r * in_r;
		std::vector<bool> selected(_points.size(), false);
		for (const auto &query : in_ps)
		{
			std::stack<_Node *> S;
			S.push(_root);
			while (!S.empty())
			{
				_Node *node = S.top();
				S.pop();
				if (!node)
				{
					continue;
				}

				const _Point3 &tp = _points[node->_id];
				if ((query - tp).sqlength_() <= radius2 && !selected[node->_id])
				{
					selected[node->_id] = true;
					res.push_back(node->_id);
				}

				const int axis = node->_axis;
				const double diff = query[axis] - tp[axis];
				if (diff <= in_r)
				{
					S.push(node->_next[0]);
				}
				if (diff >= -in_r)
				{
					S.push(node->_next[1]);
				}
			}
		}
		return res;
	}
	std::vector<int> _KDTree::nsearch_(const std::vector<_Point3> &in_ps, const int &k) const
	{
		std::vector<int> res;
		if (!_root || _points.empty() || in_ps.empty() || k <= 0)
		{
			return res;
		}

		const _Point3 &query = in_ps.front();
		using DistId = std::pair<double, int>;
		auto cmp = [](const DistId &lhs, const DistId &rhs) {
			return lhs.first < rhs.first;
		};
		std::priority_queue<DistId, std::vector<DistId>, decltype(cmp)> best(cmp);

		std::stack<_Node *> S;
		S.push(_root);
		while (!S.empty())
		{
			_Node *node = S.top();
			S.pop();
			if (!node)
			{
				continue;
			}

			const _Point3 &tp = _points[node->_id];
			const double dist2 = (query - tp).sqlength_();
			if ((int)best.size() < k)
			{
				best.push(std::make_pair(dist2, node->_id));
			}
			else if (dist2 < best.top().first)
			{
				best.pop();
				best.push(std::make_pair(dist2, node->_id));
			}

			const int axis = node->_axis;
			const double diff = query[axis] - tp[axis];
			_Node *near_child = diff < 0.0 ? node->_next[0] : node->_next[1];
			_Node *far_child = diff < 0.0 ? node->_next[1] : node->_next[0];
			const double bound =
				(int)best.size() < k ? std::numeric_limits<double>::infinity()
				                     : best.top().first;

			if (far_child && diff * diff <= bound)
			{
				S.push(far_child);
			}
			if (near_child)
			{
				S.push(near_child);
			}
		}

		std::vector<DistId> ordered;
		ordered.reserve(best.size());
		while (!best.empty())
		{
			ordered.push_back(best.top());
			best.pop();
		}
		std::sort(ordered.begin(), ordered.end(),
				  [](const DistId &lhs, const DistId &rhs) {
					  return lhs.first < rhs.first;
				  });
		for (const auto &entry : ordered)
		{
			res.push_back(entry.second);
		}
		return res;
	}
	void _KDTree::build_(const std::vector<_Point3> &in_points)
	{
		clear_();
		_points = in_points;
		std::vector<int> ids(_points.size());
		std::iota(std::begin(ids), std::end(ids), 0);
		std::stack<std::tuple<int *, int, int, _Node *, int>> S;
		_root = new _Node();
		_root->_axis = 0;
		int mid = ((int)(ids.size()) - 1) / 2;
		std::nth_element(ids.data(), ids.data() + mid, ids.data() + (int)(ids.size()),
						 [&](int lhs, int rhs) {
							 return _points[lhs][0] < _points[rhs][0];
						 });
		_root->_id = ids[mid];
		S.push(std::make_tuple(ids.data(), mid, 1, _root, 0));
		S.push(std::make_tuple(ids.data() + mid + 1, (int)(ids.size()) - mid - 1, 1, _root, 1));
		while (!S.empty())
		{
			if (std::get<1>(S.top()) <= 0)
			{
				auto state = S.top();
				S.pop();
				std::get<3>(state)->_next[std::get<4>(state)] = nullptr;
			}
			else
			{
				auto state = S.top();
				S.pop();
				const int axis = std::get<2>(state) % 3;
				const int mid = (std::get<1>(state) - 1) / 2;
				int *sid = std::get<0>(state);
				std::nth_element(sid, sid + mid, sid + std::get<1>(state),
								 [&](int lhs, int rhs) {
									 return _points[lhs][axis] < _points[rhs][axis];
								 });
				_Node *_node = new _Node();
				_node->_id = sid[mid];
				_node->_axis = axis;
				std::get<3>(state)->_next[std::get<4>(state)] = _node;
				S.push(std::make_tuple(sid, mid, std::get<2>(state) + 1, _node, 0));
				S.push(std::make_tuple(sid + mid + 1, std::get<1>(state) - mid - 1, std::get<2>(state) + 1, _node, 1));
			}
		}
	}
	void _KDTree::clear_()
	{
		std::stack<_Node *> S;
		S.push(_root);
		while (!S.empty())
		{
			if (!S.top())
			{
				S.pop();
			}
			else
			{
				_Node *node = S.top();
				S.pop();
				S.push(node->_next[0]);
				S.push(node->_next[1]);
				delete node;
			}
		}
		_root = nullptr;
		_points.clear();
	}
} // namespace BGAL
