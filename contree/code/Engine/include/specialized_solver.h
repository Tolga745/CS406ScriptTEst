#ifndef SPECIALIZED_SOLVER_H
#define SPECIALIZED_SOLVER_H

#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>

#include "configuration.h"
#include "dataset.h"
#include "dataview.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"

class Depth1ScoreHelper;

class SpecializedSolver {
public:
    /**
     * Creates the optimal decision tree for the given dataset and solution configuration.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataset The dataset to create the decision tree from.
     * @param solution_config The configuration for the solution.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataset, const Configuration& solution_config, std::shared_ptr<Tree>& current_optimal_tree, int upper_bound);

private:
    /**
     * Creates the optimal decision tree for the given dataset, solution configuration and the first feature to split on.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataset The dataset to create the decision tree from.
     * @param solution_config The configuration for the solution.
     * @param feature_index The index of the feature to split on.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataset, const Configuration& solution_config, int feature_index, std::shared_ptr<Tree>& current_optimal_tree, int upper_bound);

    /**
     * Calculates the misclassification scores for both the left and right splits of the dataset using only one dataset traversal.
     * 
     * @param dataset The dataset to calculate the scores from.
     * @param feature_index The index of the feature to split on.
     * @param split_point The split point for the feature.
     * @param threshold The threshold value for the split.
     * @param left_optimal_tree The optimal tree for the left split.
     * @param right_optimal_tree The optimal tree for the right split.
     * @param upper_bound The upper bound for the scores.
     */
    static void get_best_left_right_scores(const Dataview& dataset, int feature_index, int split_point, float threshold, std::shared_ptr<Tree>& left_optimal_tree, std::shared_ptr<Tree>& right_optimal_tree, int upper_bound);

    /**
    * Calculates the best depth one split for both the left and right splits for a feature to split on
    *
    * @param dataset The dataset to calculate the scores from.
    * @param feature_index The index of the feature (f1) to split on.
    * @param split_point The split point for the feature (f1).
    * @param current_feature_index The index of the second feature (f2) to split on
    * @param split_index The index of the split point (f1)
    * @param left_tree the depth one score helper for the left subtree
    * @param right_tree the depth one score helper for the right subtree
    * @param split_feature_split_indices A mapping from datapoint_indices to unique value indices
    * @param upper_bound The upper bound for the scores.
    */
    template <bool is_same_feature>
    static void process_depth_one_feature(const Dataview& dataview,
        const int feature_index, const int split_point, const int current_feature_index, const int split_index,
        Depth1ScoreHelper& left_tree, Depth1ScoreHelper& right_tree,
        const std::vector<int>& split_feature_split_indices, int& upper_bound);
};

#endif // SPECIALIZED_SOLVER_H
