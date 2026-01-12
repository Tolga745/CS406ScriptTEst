#ifndef INTERVALS_PRUNER_H
#define INTERVALS_PRUNER_H

#include <algorithm>
#include <vector>
#include <utility>
#include <shared_mutex> // Added for Reader-Writer Locks

class IntervalsPruner {
public:
    IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap);

    struct Bound {
        int left_bound;                
        int right_bound;               
        int last_split_left_index;     
        int last_split_right_index;    
    };

    std::pair<int, int> neighbourhood_pruning(int score_difference, int left, int right, int split_index);

    bool subinterval_pruning(const Bound& current_bounds, int current_best_score);

    void interval_shrinking(Bound& current_bounds, int current_best_score);

    void add_result(int index, int left_score, int right_score);

private:
    const std::vector<int>& possible_split_indexes; 
    int possible_split_size;                    
    int rightmost_zero_index;                      
    int leftmost_zero_index;                       
    int max_gap;                                   
    
    // Changed from unordered_map to vector for thread-safe random access
    std::vector<std::pair<int, int>> evaluated_indices_record; 
    
    // Synchronization primitive
    mutable std::shared_mutex pruner_mutex;
};

#endif // INTERVALS_PRUNER_H