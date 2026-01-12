#ifndef GENERAL_SOLVER_H
#define GENERAL_SOLVER_H

#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "cache.h"
#include "dataset.h"
#include "dataview.h"
//#include "general_solver.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"
#include <mutex>


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
    static void create_optimal_decision_tree(
        const Dataview& dataview,
        const Configuration& solution_config,
        int feature_index,
        std::shared_ptr<Tree>& current_optimal_tree,
        int upper_bound
    );

    static void calculate_leaf_node(
        int class_number,
        int instance_number,
        const std::vector<int>& label_frequency,
        std::shared_ptr<Tree>& current_optimal_decision_tree
    );

    static std::mutex tree_mutex;
};

#endif // GENERAL_SOLVER_H
