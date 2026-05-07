#include "BGAL/Geodesic/Dijkstra/Dijkstra.h"

#include <limits>
#include <queue>
#include <tuple>
#include <vector>

namespace BGAL
{
	namespace Geodesic
	{
		void _Dijkstra::initialize_()
		{
			_Abstract_Method::initialize_();
		}

		void _Dijkstra::implement_()
		{
			using _QueueNode = std::tuple<double, int, int, int, int>; // dist, self, parent, root, level
			std::priority_queue<_QueueNode, std::vector<_QueueNode>, std::greater<_QueueNode>> evtQue;
			const int nv = _model.number_vertices_();
			std::vector<int> parent(nv, -1);
			std::vector<int> root(nv, -1);
			std::vector<int> level(nv, std::numeric_limits<int>::max());
			std::vector<char> fixed(nv, 0);

			for (auto it = _sources.begin(); it != _sources.end(); ++it)
			{
				const int s = it->first;
				const double d = it->second;
				if (s < 0 || s >= nv)
					continue;
				if (d < _distances[static_cast<std::size_t>(s)])
				{
					_distances[static_cast<std::size_t>(s)] = d;
					root[static_cast<std::size_t>(s)] = s;
					level[static_cast<std::size_t>(s)] = 0;
					evtQue.emplace(d, s, -1, s, 0);
				}
			}

			_result.assign(static_cast<std::size_t>(nv), std::make_tuple(-1, -1, std::numeric_limits<int>::max(), std::numeric_limits<double>::max()));

			while (!evtQue.empty())
			{
				if (static_cast<int>(evtQue.size()) > _max_queue_length)
					_max_queue_length = static_cast<int>(evtQue.size());
				auto [cur_dist, u, par, rt, dep] = evtQue.top();
				evtQue.pop();

				if (fixed[static_cast<std::size_t>(u)])
					continue;
				if (cur_dist > _distances[static_cast<std::size_t>(u)])
					continue;

				fixed[static_cast<std::size_t>(u)] = 1;
				parent[static_cast<std::size_t>(u)] = par;
				root[static_cast<std::size_t>(u)] = rt;
				level[static_cast<std::size_t>(u)] = dep;
				_result[static_cast<std::size_t>(u)] = std::make_tuple(par, rt, dep, cur_dist);
				if (dep > _max_result_depth)
					_max_result_depth = dep;

				for (auto veit = _model.ve_begin(u); veit != _model.ve_end(u); ++veit)
				{
					const int v = (*veit)._id_right_vertex;
					if (fixed[static_cast<std::size_t>(v)])
						continue;
					const double cand = cur_dist + (*veit).length_();
					if (cand < _distances[static_cast<std::size_t>(v)])
					{
						_distances[static_cast<std::size_t>(v)] = cand;
						evtQue.emplace(cand, v, u, rt, dep + 1);
					}
				}
			}
		}

		_Dijkstra::_Dijkstra(const _ManifoldModel &in_model, const std::map<int, double> &in_sources)
			: _Abstract_Method(in_model, in_sources)
		{
			_method = 1;
		}

	} // namespace Geodesic
} // namespace BGAL
