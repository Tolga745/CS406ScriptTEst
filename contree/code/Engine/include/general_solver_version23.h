#ifndef GENERAL_SOLVER_H
#define GENERAL_SOLVER_H

#include <iostream>
#include <memory>
#include <queue>
#include <vector>
#include <mutex>

#include "cache.h"
#include "dataset.h"
#include "dataview.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"

// Forward declaration to avoid including CUDA headers here
struct GPUDataview;

class GeneralSolver {
public:
    static std::mutex tree_mutex;

    /**
     * TOP-LEVEL DISPATCHER:
     * Integrated with GPU support.
     * * @param dataview The dataset to create the decision tree from.
     * @param solution_config The configuration for the solution.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     * @param gpu_view (Optional) Pointer to the GPU data view for this node. 
     * If provided, the solver may attempt to use the GPU.
     */
    static void create_optimal_decision_tree(const Dataview& dataview, 
                                             const Configuration& solution_config, 
                                             std::shared_ptr<Tree>& current_optimal_tree, 
                                             int upper_bound,
                                             GPUDataview* gpu_view = nullptr);

private:
    static void create_optimal_decision_tree_internal(const Dataview& dataview, 
                                                      const Configuration& solution_config, 
                                                      int feature_index, 
                                                      std::shared_ptr<Tree>& current_optimal_tree, 
                                                      int upper_bound, 
                                                      bool run_in_parallel);

    static void calculate_leaf_node(int class_number, 
                                    int instance_number, 
                                    const std::vector<int>& label_frequency, 
                                    std::shared_ptr<Tree>& current_optimal_decision_tree);
};

#endif // GENERAL_SOLVER_H