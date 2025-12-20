#ifndef CACHE_H
#define CACHE_H

#include "dataview.h"
#include "tree.h"
#include <vector>
#include <map>


struct CacheEntry {

	CacheEntry(int depth) : depth(depth) {
		solution = std::make_shared<Tree>();
	}

	CacheEntry(int depth, const std::shared_ptr<Tree>& solution) : depth(depth), solution(solution) { }
	bool is_set() const { return solution->is_initialized(); }

	int depth;
	std::shared_ptr<Tree> solution;
};

class Cache {
public:

	Cache(int max_depth, int num_instances);
	Cache() : Cache(0, 0) {}


	bool is_cached(const Dataview& data, int depth);
	void store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree);
	std::shared_ptr<Tree> retrieve(const Dataview& data, int depth);

	void disable() { use_caching = false; }

	static Cache global_cache;

private:
	bool use_caching{ true };

	std::vector<std::vector<std::unordered_map<DataviewBitset, CacheEntry>>> _cache;
};

#endif // CACHE_H