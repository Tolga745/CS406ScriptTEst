#include "cache.h"

Cache Cache::global_cache = Cache();

Cache::Cache(int max_depth, int num_instances) :
    use_caching(true), 
    _cache(size_t(max_depth)+1, std::vector<std::unordered_map<DataviewBitset, CacheEntry>>(size_t(num_instances) + 1)) { }

bool Cache::is_cached(const Dataview& data, int depth) {
    if (!use_caching) return false;

    // Acquire Shared Lock (Reader) - Multiple threads can enter here simultaneously
    std::shared_lock<std::shared_mutex> lock(_mutex);

    auto& depth_cache = _cache[depth];
    auto& size_cache = depth_cache[data.get_dataset_size()];
    auto& bitset = data.get_bitset();
    
    // Note: bitset.set_hash() modifies the bitset inside the dataview. 
    // If dataview is shared across threads, this specific line is still a race condition 
    // regardless of the cache lock. Ideally, hash calculation should be done before 
    // entering the parallel region or protected by a dataview-specific lock.
    if (!bitset.is_hash_set()) bitset.set_hash(std::hash<DataviewBitset>()(bitset));

    const auto& it = size_cache.find(bitset);
    if (it == size_cache.end()) return false;
    if (it->second.is_set()) return true;
    return false;
}

void Cache::store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree) {
    if (!use_caching) return;
    if (!tree->is_initialized()) return;

    // Acquire Unique Lock (Writer) - Only one thread can write at a time
    // This blocks all readers until finished
    std::unique_lock<std::shared_mutex> lock(_mutex);

    auto& depth_cache = _cache[depth];
    auto& size_cache = depth_cache[data.get_dataset_size()];
    auto& bitset = data.get_bitset();
    if (!bitset.is_hash_set()) bitset.set_hash(std::hash<DataviewBitset>()(bitset));

    size_cache.insert(std::pair<DataviewBitset, CacheEntry>(bitset, CacheEntry(depth, tree)));
}


std::shared_ptr<Tree> Cache::retrieve(const Dataview& data, int depth) {
    if (!use_caching) return std::make_shared<Tree>();

    // Acquire Shared Lock (Reader)
    std::shared_lock<std::shared_mutex> lock(_mutex);

    auto& depth_cache = _cache[depth];
    auto& size_cache = depth_cache[data.get_dataset_size()];
    auto& bitset = data.get_bitset();

    const auto& it = size_cache.find(bitset);
    if (it == size_cache.end()) return std::make_shared<Tree>();
    if (it->second.is_set()) return it->second.solution;
    return std::make_shared<Tree>();
}
void Cache::resize(int max_depth, int num_instances) {
    // Acquire lock just in case, though main() is usually serial here
    std::unique_lock<std::shared_mutex> lock(_mutex); 
    
    use_caching = true;
    _cache.clear();
    
    // Re-allocate the vector structure
    _cache.resize(size_t(max_depth) + 1, 
        std::vector<std::unordered_map<DataviewBitset, CacheEntry>>(size_t(num_instances) + 1));
}