#ifndef GENERAL_SOLVER_H
#define GENERAL_SOLVER_H

#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "cache.h"
#include "dataset.h"
#include "dataview.h"
#include "general_solver.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"


class GeneralSolver {
public:
    /**
     * Creates the optimal decision tree for the given dataset and solution configuration.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataview The dataset to create the decision tree from.
     * @param solution_config The configuration for the solution.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_config, std::shared_ptr<Tree>& current_optimal_tree, int upper_bound);

private:
    /**
     * Creates the optimal decision tree for the given dataset, solution configuration and the first feature to split on.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataview The dataset to create the decision tree from.
     * @param solution_config The configuration for the solution.
     * @param feature_index The index of the feature to split on.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_config, int feature_index, std::shared_ptr<Tree>& current_optimal_tree, int upper_bound);

    /**
     * Calculates the misclassification score if the current node is a leaf node.
     * 
     * @param dataset The dataset to calculate the scores from.
     * @param feature_index The index of the feature to split on.
     * @param split_point The split point for the feature.
     * @param threshold The threshold value for the split.
     * @param left_optimal_tree The optimal tree for the left split.
     * @param right_optimal_tree The optimal tree for the right split.
     * @param upper_bound The upper bound for the scores.
     */
    static void calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree);
};

#endif // GENERAL_SOLVER_H
