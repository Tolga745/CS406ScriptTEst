#ifndef INTERVALS_PRUNER_H
#define INTERVALS_PRUNER_H

#include <algorithm>
#include <utility>
#include <vector>
#include <shared_mutex>  // RW lock

class IntervalsPruner {
public:
    /**
     * Constructor for IntervalsPruner.
     *
     * Initializes the IntervalsPruner with a reference to a vector of possible split indexes
     * and a maximum allowable gap, which is used to control the permissibility of suboptimal solutions.
     *
     * Thread-safety notes:
     *  - evaluated_indices_record is RESIZED in the ctor to possible_split_size, so random access is always valid.
     *  - Reads use shared_lock (many readers), writes use unique_lock (single writer).
     */
    IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap);

    struct Bound {
        int left_bound;
        int right_bound;
        int last_split_left_index;
        int last_split_right_index;
    };

    /**
     * Performs neighborhood pruning by evaluating the interval around a given split index.
     *
     * @param score_difference The difference in scores used to determine pruning.
     * @param left The left boundary of the interval.
     * @param right The right boundary of the interval.
     * @param split_index The index at which the split is evaluated.
     * @return A pair of integers representing the new pruned interval bounds.
     */
    std::pair<int, int> neighbourhood_pruning(int score_difference, int left, int right, int split_index);

    /**
     * Applies subinterval pruning to determine if a given interval can be entirely pruned.
     *
     * @param current_bounds The current bounds of the interval being evaluated.
     * @param current_best_score The best score obtained so far, used as a reference for pruning.
     * @return True if the subinterval can be pruned, otherwise false.
     */
    bool subinterval_pruning(const Bound& current_bounds, int current_best_score);

    /**
     * Performs interval shrinking by narrowing the bounds of the interval based on the current best score.
     *
     * @param current_bounds The current bounds of the interval to be updated by shrinking
     * @param current_best_score The best score to compare against during the shrinking process.
     */
    void interval_shrinking(Bound& current_bounds, int current_best_score);

    /**
     * Records the result of a split, storing the index and associated scores.
     *
     * @param index The index at which the split was performed.
     * @param left_score The score associated with the left subinterval.
     * @param right_score The score associated with the right subinterval.
     */
    void add_result(int index, int left_score, int right_score);

private:
    const std::vector<int>& possible_split_indexes;
    int possible_split_size;
    int rightmost_zero_index;
    int leftmost_zero_index;
    int max_gap;

    // Thread-safe random access because:
    //  - fully sized once in ctor (no reallocation/resizing later)
    //  - guarded by pruner_mutex for reads/writes
    std::vector<std::pair<int, int>> evaluated_indices_record;

    // Reader-writer lock: shared for reads, unique for writes
    mutable std::shared_mutex pruner_mutex;
};

#endif // INTERVALS_PRUNER_H
