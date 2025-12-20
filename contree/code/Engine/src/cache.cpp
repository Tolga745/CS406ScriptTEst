#include "cache.h"

Cache Cache::global_cache = Cache();

Cache::Cache(int max_depth, int num_instances) :
	use_caching(true), 
	_cache(size_t(max_depth)+1, std::vector<std::unordered_map<DataviewBitset, CacheEntry>>(size_t(num_instances) + 1)) { }

bool Cache::is_cached(const Dataview& data, int depth) {
	if (!use_caching) return false;

	auto& depth_cache = _cache[depth];
	auto& size_cache = depth_cache[data.get_dataset_size()];
	auto& bitset = data.get_bitset();
	if (!bitset.is_hash_set()) bitset.set_hash(std::hash<DataviewBitset>()(bitset));

	const auto& it = size_cache.find(bitset);
	if (it == size_cache.end()) return false;
	if (it->second.is_set()) return true;
	return false;
}

void Cache::store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree) {
	if (!use_caching) return;
	if (!tree->is_initialized()) return;

	auto& depth_cache = _cache[depth];
	auto& size_cache = depth_cache[data.get_dataset_size()];
	auto& bitset = data.get_bitset();
	if (!bitset.is_hash_set()) bitset.set_hash(std::hash<DataviewBitset>()(bitset));

	size_cache.insert(std::pair<DataviewBitset, CacheEntry>(bitset, CacheEntry(depth, tree)));
}


std::shared_ptr<Tree> Cache::retrieve(const Dataview& data, int depth) {
	if (!use_caching) return std::make_shared<Tree>();

	auto& depth_cache = _cache[depth];
	auto& size_cache = depth_cache[data.get_dataset_size()];
	auto& bitset = data.get_bitset();

	const auto& it = size_cache.find(bitset);
	if (it == size_cache.end()) return std::make_shared<Tree>();
	if (it->second.is_set()) return it->second.solution;
	return std::make_shared<Tree>();
}