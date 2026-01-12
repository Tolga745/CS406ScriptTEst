#include "specialized_solver.h"
#include <mutex>
#include <atomic>
#include <omp.h>
#include <limits>
#include "intervals_pruner.h"
#include <queue>
// Mutex for merging results in the top-level function
std::mutex spec_tree_mutex;

// Helper class for Depth 1 scoring
// (Keep class definition as is, ensuring it has a copy constructor for thread locality)
class Depth1ScoreHelper {
public:
    Depth1ScoreHelper(const int size, const int CLASS_NUMBER)
        : label_frequency(std::vector<int>(CLASS_NUMBER, 0)), 
          current_label_frequency(std::vector<int>(CLASS_NUMBER, 0)), 
          size(size) {}

    // Copy constructor is essential for thread-local copies
    Depth1ScoreHelper(const Depth1ScoreHelper& other) = default;

    void reset_label_frequency() {
        std::fill(current_label_frequency.begin(), current_label_frequency.end(), 0);
        previous_value = 0.0f;
        previous_unique_value_index = -1;
        is_zero = false;
        can_skip = 0;
        current_element_count = 0;
    }

    // State variables
    int classification_score = -1;
    int best_feature_index = -1;
    float best_threshold = -1.0f;
    int best_left_label = -1;
    int best_right_label = -1;

    float previous_value = 0.0f;
    int previous_unique_value_index = -1;
    bool is_zero = false;

    int can_skip = 0;
    int current_element_count = 0;

    const int size;
    int max_label_frequency{ 0 };
    int max_label{ 0 };

    std::vector<int> label_frequency;
    std::vector<int> current_label_frequency;
};

// --- Top Level Parallelization ---
void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    
    // 1. Shared Atomic UB to share the best score instantly across threads
    std::atomic<int> shared_ub(std::min(upper_bound, current_optimal_decision_tree->misclassification_score));
    bool already_parallel = omp_in_parallel();
    // 2. Parallelize the Feature Loop
    // dynamic schedule handles load imbalance (some features are harder to split)
    #pragma omp parallel for schedule(dynamic) if(!already_parallel)
    for (int feature_index = 0; feature_index < dataview.get_feature_number(); feature_index++) {
        
        // Check time limit
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) continue;
        
        // Fast Pruning: If global UB is 0, we are done.
        int current_global_ub = shared_ub.load();
        if (current_global_ub == 0) continue;

        // Thread-Local Tree
        std::shared_ptr<Tree> thread_local_tree = std::make_shared<Tree>(*current_optimal_decision_tree);

        // Run Solver
        create_optimal_decision_tree(dataview, solution_configuration, feature_index, thread_local_tree, current_global_ub);

        // 3. Merge Results
        if (thread_local_tree->misclassification_score <= solution_configuration.max_gap) {
            // Found a "perfect" solution relative to gap
             std::lock_guard<std::mutex> lock(spec_tree_mutex);
             if (thread_local_tree->misclassification_score < current_optimal_decision_tree->misclassification_score) {
                 current_optimal_decision_tree = thread_local_tree;
                 shared_ub.store(current_optimal_decision_tree->misclassification_score);
             }
             // Cannot 'return' from parallel loop, just continue and let others finish or check UB
             continue; 
        }

        // Standard merge for non-perfect improvements
        if (thread_local_tree->misclassification_score < current_global_ub) {
            std::lock_guard<std::mutex> lock(spec_tree_mutex);
            if (thread_local_tree->misclassification_score < current_optimal_decision_tree->misclassification_score) {
                current_optimal_decision_tree = thread_local_tree;
                shared_ub.store(current_optimal_decision_tree->misclassification_score);
            }
        }
    }
}

// --- Inner Loop Parallelization (Reduction Strategy) ---
void SpecializedSolver::get_best_left_right_scores(const Dataview& dataview, int feature_index, int split_point, float threshold, std::shared_ptr<Tree> &left_optimal_dt, std::shared_ptr<Tree> &right_optimal_dt, int upper_bound) {
    const auto& split_feature = dataview.get_sorted_dataset_feature(feature_index);
    const auto& unsorted_split_feature = dataview.get_unsorted_dataset_feature(feature_index);
    std::vector<int> split_feature_split_indices(unsorted_split_feature.size());
    int split_index = -1;
    
    for (const auto& split_feature_data : split_feature) {
        split_feature_split_indices[split_feature_data.data_point_index] = split_feature_data.unique_value_index;
        if (split_index == -1 && split_feature_data.value >= threshold) {
            split_index = split_feature_data.unique_value_index;
        }
    }
    RUNTIME_ASSERT(split_index != -1, "Split index not found.");

    const int dataset_size = dataview.get_dataset_size();
    const int class_number = dataview.get_class_number();

    // Prepare Base Helpers
    Depth1ScoreHelper base_left_tree(split_point, class_number);
    Depth1ScoreHelper base_right_tree(dataset_size - split_point, class_number);

    base_left_tree.classification_score = std::max(0, base_left_tree.size - upper_bound);
    base_right_tree.classification_score = std::max(0, base_right_tree.size - upper_bound);

    Dataview::initialize_split_parameters(split_feature, class_number, dataview.get_label_frequency(), split_point, base_left_tree.label_frequency, base_right_tree.label_frequency);

    // Initial max frequency calculation
    base_left_tree.max_label_frequency = 0;
    base_right_tree.max_label_frequency = 0;
    for (int label = 0; label < class_number; label++) {
        if (base_left_tree.label_frequency[label] > base_left_tree.max_label_frequency) {
            base_left_tree.max_label_frequency = base_left_tree.label_frequency[label];
            base_left_tree.max_label = label;
        }
        if (base_right_tree.label_frequency[label] > base_right_tree.max_label_frequency) {
            base_right_tree.max_label_frequency = base_right_tree.label_frequency[label];
            base_right_tree.max_label = label;
        }
    }
    base_left_tree.classification_score = std::max(base_left_tree.classification_score, base_left_tree.max_label_frequency);
    base_right_tree.classification_score = std::max(base_right_tree.classification_score, base_right_tree.max_label_frequency);


    // PARALLEL REGION: Thread-Local Reduction
    // We create a vector of helpers (one pair per thread) to avoid race conditions
    // Then we merge them at the end.
    int max_threads = omp_get_max_threads();
    bool run_parallel = !omp_in_parallel();
    std::vector<Depth1ScoreHelper> local_left_trees(max_threads, base_left_tree);
    std::vector<Depth1ScoreHelper> local_right_trees(max_threads, base_right_tree);

    #pragma omp parallel if(run_parallel)
    {
        int tid = omp_get_thread_num();
        auto& my_left_tree = local_left_trees[tid];
        auto& my_right_tree = local_right_trees[tid];
        
        // This loop is the expensive part (scanning all features for Depth 1 splits)
        #pragma omp for
        for (int current_feature_index = 0 ; current_feature_index < dataview.get_feature_number(); current_feature_index++) {
            
            // Optimization: If we already found a perfect split locally, we could skip (optional)
            if (my_left_tree.classification_score + my_right_tree.classification_score == dataset_size) continue;

            if (current_feature_index == feature_index) {
                process_depth_one_feature<true>(dataview, feature_index, split_point, current_feature_index, split_index,
                    my_left_tree, my_right_tree, split_feature_split_indices, upper_bound);
            } else {
                process_depth_one_feature<false>(dataview, feature_index, split_point, current_feature_index, split_index,
                    my_left_tree, my_right_tree, split_feature_split_indices, upper_bound);
            }
        }
    }

    // REDUCTION STEP: Find the best result among all threads
    // We start with the base (which holds default scores)
    Depth1ScoreHelper* best_left = &base_left_tree;
    Depth1ScoreHelper* best_right = &base_right_tree;

    for(int i = 0; i < max_threads; ++i) {
        int current_total_score = local_left_trees[i].classification_score + local_right_trees[i].classification_score;
        int best_total_score = best_left->classification_score + best_right->classification_score;

        if (current_total_score > best_total_score) {
            best_left = &local_left_trees[i];
            best_right = &local_right_trees[i];
        }
    }

    // Apply Best Results
    if (best_left->classification_score == best_left->max_label_frequency) {
        left_optimal_dt->make_leaf(best_left->max_label, best_left->size - best_left->classification_score);
    } else {
        left_optimal_dt->update_split(best_left->best_feature_index, best_left->best_threshold, 
            std::make_shared<Tree>(best_left->best_left_label, -1), std::make_shared<Tree>(best_left->best_right_label, -1));
    }
    left_optimal_dt->misclassification_score = best_left->size - best_left->classification_score;

    if (best_right->classification_score == best_right->max_label_frequency) {
        right_optimal_dt->make_leaf(best_right->max_label, best_right->size - best_right->classification_score);
    } else {
        right_optimal_dt->update_split(best_right->best_feature_index, best_right->best_threshold, 
            std::make_shared<Tree>(best_right->best_left_label, -1), std::make_shared<Tree>(best_right->best_right_label, -1));
    }
    right_optimal_dt->misclassification_score = best_right->size - best_right->classification_score;
}
template <bool is_same_feature>
void SpecializedSolver::process_depth_one_feature(const Dataview& dataview,
    const int feature_index, const int split_point, const int current_feature_index, const int split_index,
    Depth1ScoreHelper& left_tree, Depth1ScoreHelper& right_tree,
    const std::vector<int>& split_feature_split_indices, int& upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(current_feature_index);
    const int class_number = dataview.get_class_number();
    const int dataset_size = dataview.get_dataset_size();

    left_tree.reset_label_frequency();
    right_tree.reset_label_frequency();

    Depth1ScoreHelper* tree_p = &left_tree;
    int index = 0;
    for (const auto& current_feature_data : current_feature) {
        if constexpr (!is_same_feature) {
            int cur_split_index = split_feature_split_indices[current_feature_data.data_point_index];
            bool is_left_tree = (cur_split_index < split_index);
            tree_p = is_left_tree ? &left_tree : &right_tree;
        } else {
            if (index++ == split_point) {
                tree_p = &right_tree;
            };
        }
        auto& tree = *tree_p;

        if (tree.is_zero) {
            continue;
        }

        tree.can_skip--;

        if (current_feature_data.unique_value_index == tree.previous_unique_value_index || tree.can_skip > 0){
            tree.current_element_count++;
            tree.current_label_frequency[current_feature_data.label]++;
            tree.previous_value = current_feature_data.value;
            tree.previous_unique_value_index = current_feature_data.unique_value_index;
            continue;
        }

        int left_classification_score = -1;
        int right_classification_score = -1;

        int left_label = -1;
        int right_label = -1;

        for (int label_value = 0; label_value < class_number; label_value++) {
            if (tree.current_label_frequency[label_value] > left_classification_score) {
                left_classification_score = tree.current_label_frequency[label_value];
                left_label = label_value;
            }
            if (tree.label_frequency[label_value] - tree.current_label_frequency[label_value] > right_classification_score) {
                right_classification_score = tree.label_frequency[label_value] - tree.current_label_frequency[label_value];
                right_label = label_value;
            }
        }

        if (left_classification_score + right_classification_score > tree.classification_score) {
            RUNTIME_ASSERT(tree.classification_score <= tree.size, "LR - Classification score cannot exceed the number of instances.");
            tree.classification_score = left_classification_score + right_classification_score;
            tree.best_feature_index = current_feature_index;
            tree.best_threshold = (current_feature_data.value + tree.previous_value) / 2.0f;

            upper_bound = std::min(dataset_size - (left_tree.classification_score + right_tree.classification_score), upper_bound);

            tree.best_left_label = left_label;
            tree.best_right_label = right_label;
        } else {
            tree.can_skip = tree.classification_score - (left_classification_score + right_classification_score);
        }

        int remaining_size = tree.size - tree.current_element_count;
        tree.is_zero |= (right_classification_score == remaining_size) || (tree.can_skip >= remaining_size);

        if (left_tree.is_zero && right_tree.is_zero) {
            break;
        }

        if (left_tree.can_skip >= left_tree.size - left_tree.current_element_count && right_tree.can_skip >= right_tree.size - right_tree.current_element_count) {
            break;
        }

        tree.current_element_count++;
        tree.current_label_frequency[current_feature_data.label]++;
        tree.previous_value = current_feature_data.value;
        tree.previous_unique_value_index = current_feature_data.unique_value_index;
    }
}

void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);

    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, solution_configuration.max_gap);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if(left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  

        std::shared_ptr<Tree> left_optimal_dt = std::make_shared<Tree>();
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>();

        statistics::total_number_of_specialized_solver_calls += 1;
        get_best_left_right_scores(dataview, feature_index, split_point, threshold, left_optimal_dt, right_optimal_dt, current_optimal_decision_tree->misclassification_score);
        RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "D2 - Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "D2 - Right tree should have non-negative misclassification score.");

        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {

            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

            upper_bound = std::min(upper_bound, current_best_score);

            if (current_best_score == 0) {
                return;
            }

            if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root) {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }

        }

        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = current_best_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}