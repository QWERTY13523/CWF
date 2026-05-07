#include "BGAL/BaseShape/KDTree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stack>
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
		double min_dist = std::numeric_limits<double>::max();
		return search_(in_p, min_dist);
	}

	int _KDTree::search_(const _Point3 &in_p, double &min_dist) const
	{
		if (!_root || _points.empty())
		{
			min_dist = std::numeric_limits<double>::max();
			return -1;
		}
		double best_sq = std::numeric_limits<double>::max();
		int best_id = -1;
		search_recursive_(_root, in_p, best_sq, best_id);
		min_dist = std::sqrt(best_sq);
		return best_id;
	}

	void _KDTree::search_recursive_(const _Node *node, const _Point3 &query, double &best_sq, int &best_id) const
	{
		if (!node)
			return;

		const _Point3 &tp = _points[node->_id];
		const double dist_sq = (query - tp).sqlength_();
		if (dist_sq < best_sq)
		{
			best_sq = dist_sq;
			best_id = node->_id;
		}

		const int axis = node->_axis;
		const double diff = query[axis] - tp[axis];
		const _Node *near_child = diff < 0.0 ? node->_next[0] : node->_next[1];
		const _Node *far_child = diff < 0.0 ? node->_next[1] : node->_next[0];

		search_recursive_(near_child, query, best_sq, best_id);
		if (diff * diff <= best_sq)
		{
			search_recursive_(far_child, query, best_sq, best_id);
		}
	}

	std::vector<int> _KDTree::rsearch_(const _Point3 &in_p, const double &in_r) const
	{
		std::vector<int> res;
		if (!_root || _points.empty() || in_r < 0.0)
		{
			return res;
		}
		res.reserve(32);
		rsearch_recursive_(_root, in_p, in_r * in_r, in_r, res);
		return res;
	}

	void _KDTree::rsearch_recursive_(const _Node *node, const _Point3 &query, double radius_sq, double radius, std::vector<int> &res) const
	{
		if (!node)
			return;
		const _Point3 &tp = _points[node->_id];
		if ((query - tp).sqlength_() <= radius_sq)
		{
			res.push_back(node->_id);
		}
		const int axis = node->_axis;
		const double diff = query[axis] - tp[axis];
		if (diff <= radius)
			rsearch_recursive_(node->_next[0], query, radius_sq, radius, res);
		if (diff >= -radius)
			rsearch_recursive_(node->_next[1], query, radius_sq, radius, res);
	}

	std::vector<int> _KDTree::rsearch_(const std::vector<_Point3> &in_ps, const double &in_r) const
	{
		if (in_ps.size() == 1)
		{
			return rsearch_(in_ps.front(), in_r);
		}

		std::vector<int> res;
		if (!_root || _points.empty() || in_ps.empty() || in_r < 0.0)
		{
			return res;
		}

		const double radius_sq = in_r * in_r;
		std::vector<char> selected(_points.size(), 0);
		for (const auto &query : in_ps)
		{
			std::vector<int> local;
			local.reserve(32);
			rsearch_recursive_(_root, query, radius_sq, in_r, local);
			for (int id : local)
			{
				if (!selected[static_cast<std::size_t>(id)])
				{
					selected[static_cast<std::size_t>(id)] = 1;
					res.push_back(id);
				}
			}
		}
		return res;
	}

	void _KDTree::knn_recursive_(const _Node *node, const _Point3 &query, int k, _MaxHeap &best) const
	{
		if (!node)
			return;
		const _Point3 &tp = _points[node->_id];
		const double dist_sq = (query - tp).sqlength_();
		if (static_cast<int>(best.size()) < k)
		{
			best.emplace(dist_sq, node->_id);
		}
		else if (dist_sq < best.top().first)
		{
			best.pop();
			best.emplace(dist_sq, node->_id);
		}

		const int axis = node->_axis;
		const double diff = query[axis] - tp[axis];
		const _Node *near_child = diff < 0.0 ? node->_next[0] : node->_next[1];
		const _Node *far_child = diff < 0.0 ? node->_next[1] : node->_next[0];
		knn_recursive_(near_child, query, k, best);
		const double bound = static_cast<int>(best.size()) < k ? std::numeric_limits<double>::infinity() : best.top().first;
		if (diff * diff <= bound)
		{
			knn_recursive_(far_child, query, k, best);
		}
	}

	std::vector<int> _KDTree::nsearch_(const std::vector<_Point3> &in_ps, const int &k) const
	{
		std::vector<int> res;
		if (!_root || _points.empty() || in_ps.empty() || k <= 0)
		{
			return res;
		}

		const _Point3 &query = in_ps.front();
		_MaxHeap best;
		knn_recursive_(_root, query, k, best);

		std::vector<_HeapItem> ordered;
		ordered.reserve(best.size());
		while (!best.empty())
		{
			ordered.push_back(best.top());
			best.pop();
		}
		std::sort(ordered.begin(), ordered.end(), [](const _HeapItem &lhs, const _HeapItem &rhs) {
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
		if (_points.empty())
		{
			_root = nullptr;
			return;
		}
		std::vector<int> ids(_points.size());
		std::iota(std::begin(ids), std::end(ids), 0);
		std::stack<std::tuple<int *, int, int, _Node *, int>> S;
		_root = new _Node();
		_root->_axis = 0;
		int mid = (static_cast<int>(ids.size()) - 1) / 2;
		std::nth_element(ids.data(), ids.data() + mid, ids.data() + static_cast<int>(ids.size()), [&](int lhs, int rhs) {
			return _points[lhs][0] < _points[rhs][0];
		});
		_root->_id = ids[mid];
		S.push(std::make_tuple(ids.data(), mid, 1, _root, 0));
		S.push(std::make_tuple(ids.data() + mid + 1, static_cast<int>(ids.size()) - mid - 1, 1, _root, 1));
		while (!S.empty())
		{
			auto state = S.top();
			S.pop();
			if (std::get<1>(state) <= 0)
			{
				std::get<3>(state)->_next[std::get<4>(state)] = nullptr;
				continue;
			}

			const int axis = std::get<2>(state) % 3;
			const int local_mid = (std::get<1>(state) - 1) / 2;
			int *sid = std::get<0>(state);
			std::nth_element(sid, sid + local_mid, sid + std::get<1>(state), [&](int lhs, int rhs) {
				return _points[lhs][axis] < _points[rhs][axis];
			});
			_Node *_node = new _Node();
			_node->_id = sid[local_mid];
			_node->_axis = axis;
			std::get<3>(state)->_next[std::get<4>(state)] = _node;
			S.push(std::make_tuple(sid, local_mid, std::get<2>(state) + 1, _node, 0));
			S.push(std::make_tuple(sid + local_mid + 1, std::get<1>(state) - local_mid - 1, std::get<2>(state) + 1, _node, 1));
		}
	}

	void _KDTree::clear_()
	{
		std::stack<_Node *> S;
		S.push(_root);
		while (!S.empty())
		{
			_Node *node = S.top();
			S.pop();
			if (!node)
				continue;
			S.push(node->_next[0]);
			S.push(node->_next[1]);
			delete node;
		}
		_root = nullptr;
		_points.clear();
	}
} // namespace BGAL
