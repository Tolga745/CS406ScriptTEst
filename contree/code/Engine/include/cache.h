#ifndef CACHE_H
#define CACHE_H

#include "dataview.h"
#include "tree.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex> // Required for Read-Write Locks

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
    void resize(int max_depth, int num_instances); 

    bool is_cached(const Dataview& data, int depth);
    void store(const Dataview& data, int depth, std::shared_ptr<Tree>& tree);
    std::shared_ptr<Tree> retrieve(const Dataview& data, int depth);

    void disable() { use_caching = false; }

    static Cache global_cache;

private:
    bool use_caching{ true };
    
    // The main data structure
    std::vector<std::vector<std::unordered_map<DataviewBitset, CacheEntry>>> _cache;
    
    // The Read-Write Lock
    // mutable allows it to be modified even in const functions (though yours are non-const)
    mutable std::shared_mutex _mutex; 
};

#endif // CACHE_H