#include "general_solver_version23.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <omp.h>
#include <queue>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono> // Added for timing

// GPU Integration
#include "gpu_solver.cuh"
#include "gpu_dataview.h"

// --- TUNING PARAMETERS ---
// If a node has fewer than this many rows, we skip the GPU and use CPU.
// GPU overhead (malloc, kernel launch) usually kills performance below 20k-50k rows.
static const int GPU_MIN_INSTANCES = 45000; 

// Set to true to see timing logs in your console
static const bool DEBUG_PERFORMANCE = true; 

std::mutex GeneralSolver::tree_mutex;

void GeneralSolver::create_optimal_decision_tree(
    const Dataview& dataview,
    const Configuration& solution_configuration,
    std::shared_ptr<Tree>& current_optimal_decision_tree,
    int upper_bound,
    GPUDataview* gpu_view
) {
    // 1. Base Cases & Early Pruning (CPU)
    if (current_optimal_decision_tree->misclassification_score == 0 || dataview.get_dataset_size() == 0) {
        return;
    }

    if (Cache::global_cache.is_cached(dataview, solution_configuration.max_depth)) {
        current_optimal_decision_tree = Cache::global_cache.retrieve(dataview, solution_configuration.max_depth);
        return;
    }

    calculate_leaf_node(
        dataview.get_class_number(),
        dataview.get_dataset_size(),
        dataview.get_label_frequency(),
        current_optimal_decision_tree
    );

    if (solution_configuration.max_depth == 0) return;
    if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap ||
        dataview.get_dataset_size() == 1) {
        return;
    }

    // ───────────────────────────────────────────────────────────────
    // GPU EXECUTION PATH
    // ───────────────────────────────────────────────────────────────
    bool gpu_path_taken = false;
    bool is_gpu_root_alloc = false;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // A. GPU Initialization (At Root)
    // Only allocate if the ROOT dataset is large enough.
    if (solution_configuration.is_root && gpu_view == nullptr) {
        if (dataview.get_dataset_size() > GPU_MIN_INSTANCES) { 
            allocate_recursion_buffers(solution_configuration.max_depth, dataview.get_dataset_size(), dataview.get_feature_number());
            global_gpu_dataset.initialize(dataview);
            
            static GPUDataview root_view_storage;
            prepare_gpu_view(dataview, root_view_storage);
            
            gpu_view = &root_view_storage;
            is_gpu_root_alloc = true;
            
            if (DEBUG_PERFORMANCE) {
                std::cout << "[GPU] Initialized for dataset size: " << dataview.get_dataset_size() << std::endl;
            }
        }
    }

    // B. GPU Execution
    // Crucial Fix: Added '&& gpu_view->num_instances > GPU_MIN_INSTANCES'
    // This prevents using the GPU for small nodes deep in the tree.
    if (gpu_view != nullptr && gpu_view->num_instances > GPU_MIN_INSTANCES) {
        
        int num_features = dataview.get_feature_number();

        // Host containers
        std::vector<int> h_scores_L(num_features), h_leaf_scores_L(num_features);
        std::vector<int> h_scores_R(num_features), h_leaf_scores_R(num_features);
        std::vector<float> h_thresh_L(num_features), h_thresh_R(num_features);
        // ... (We skip other detailed vectors for clarity unless needed) ...
        // We only really need these for decision making right now, others can be allocated if split found
        
        // Detailed allocs (needed for API)
        std::vector<int> h_lbl_L_L(num_features), h_lbl_L_R(num_features), h_lbl_R_L(num_features), h_lbl_R_R(num_features);
        std::vector<int> h_cs_L_L(num_features), h_cs_L_R(num_features), h_cs_R_L(num_features), h_cs_R_R(num_features);
        std::vector<int> h_leaf_lbl_L(num_features), h_leaf_lbl_R(num_features);

        // 1. Run Solver
        auto t1 = std::chrono::high_resolution_clock::now();
        
        run_specialized_solver_gpu(
            *gpu_view, -1, 0.0f, upper_bound,
            h_scores_L.data(), h_thresh_L.data(), h_lbl_L_L.data(), h_lbl_L_R.data(), h_cs_L_L.data(), h_cs_L_R.data(), h_leaf_scores_L.data(), h_leaf_lbl_L.data(),
            h_scores_R.data(), h_thresh_R.data(), h_lbl_R_L.data(), h_lbl_R_R.data(), h_cs_R_L.data(), h_cs_R_R.data(), h_leaf_scores_R.data(), h_leaf_lbl_R.data()
        );

        auto t2 = std::chrono::high_resolution_clock::now();

        // 2. Find Best Split
        int best_feature_idx = -1;
        float best_threshold = 0.0f;
        int global_best_score = std::min(upper_bound, current_optimal_decision_tree->misclassification_score);
        
        for (int f = 0; f < num_features; ++f) {
            int score = h_scores_L[f];
            if (score < global_best_score) {
                global_best_score = score;
                best_feature_idx = f;
                best_threshold = h_thresh_L[f];
            }
        }

        if (best_feature_idx != -1) {
            // 3. Physically Split on GPU
            GPUDataview left_gpu, right_gpu;
            split_gpu_dataview(*gpu_view, left_gpu, right_gpu, best_feature_idx, best_threshold, solution_configuration.max_depth, 0);
            
            auto t3 = std::chrono::high_resolution_clock::now();

            // 4. Prepare CPU Dataviews (Parallelized on CPU)
            // Note: This duplicates work but is required for the current architecture
            Dataview left_dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
            Dataview right_dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
            
            const auto& feature_vec = dataview.get_sorted_dataset_feature(best_feature_idx);
            
            auto it = std::lower_bound(feature_vec.begin(), feature_vec.end(), best_threshold, 
                [](const Dataset::FeatureElement& a, float val){ return a.value < val; });
            int split_point = std::distance(feature_vec.begin(), it);
            int split_val_idx = (it != feature_vec.end()) ? it->unique_value_index : -1;
            
            Dataview::split_data_points(dataview, best_feature_idx, split_point, split_val_idx, left_dataview, right_dataview, solution_configuration.max_depth);

            auto t4 = std::chrono::high_resolution_clock::now();

            // 5. Construct & Recurse
            std::shared_ptr<Tree> left_dt = std::make_shared<Tree>(-1, global_best_score);
            std::shared_ptr<Tree> right_dt = std::make_shared<Tree>(-1, global_best_score);

            auto& larger_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_dataview : left_dataview;
            auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_dataview : right_dataview;
            
            auto& larger_gpu = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_gpu : left_gpu;
            auto& smaller_gpu = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_gpu : right_gpu;
            
            auto& larger_dt = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_dt : left_dt;
            auto& smaller_dt = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_dt : right_dt;

            Configuration left_conf = solution_configuration.GetLeftSubtreeConfig();
            GeneralSolver::create_optimal_decision_tree(larger_data, left_conf, larger_dt, global_best_score, &larger_gpu);

            Configuration right_conf = solution_configuration.GetRightSubtreeConfig(left_conf.max_gap);
            GeneralSolver::create_optimal_decision_tree(smaller_data, right_conf, smaller_dt, global_best_score - larger_dt->misclassification_score, &smaller_gpu);

            current_optimal_decision_tree->update_split(best_feature_idx, best_threshold, left_dt, right_dt);
            current_optimal_decision_tree->misclassification_score = left_dt->misclassification_score + right_dt->misclassification_score;

            if (DEBUG_PERFORMANCE && gpu_view->num_instances > 50000) {
                auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                auto duration_split = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
                auto duration_cpu_prep = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
                std::cout << "Depth " << (solution_configuration.is_root ? 0 : "N") 
                          << " | Size: " << gpu_view->num_instances 
                          << " | GPU Solve: " << duration_gpu << "ms"
                          << " | GPU Split: " << duration_split << "ms"
                          << " | CPU Prep: " << duration_cpu_prep << "ms" << std::endl;
            }
        }

        gpu_path_taken = true;
    }

    if (is_gpu_root_alloc) {
        global_gpu_dataset.free(); 
        free_recursion_buffers();
    }

    if (gpu_path_taken) {
        if (current_optimal_decision_tree->misclassification_score <= upper_bound) {
            Cache::global_cache.store(dataview, solution_configuration.max_depth, current_optimal_decision_tree);
        }
        return;
    }

    // ───────────────────────────────────────────────────────────────
    // CPU FALLBACK (Optimized OpenMP)
    // ───────────────────────────────────────────────────────────────
    
    // Depth-2 special solver (Pure CPU)
    if (solution_configuration.max_depth == 2) {
        SpecializedSolver::create_optimal_decision_tree(
            dataview,
            solution_configuration,
            current_optimal_decision_tree,
            std::min(upper_bound, current_optimal_decision_tree->misclassification_score)
        );
        return;
    }

    const bool allow_parallel = !omp_in_parallel();
    const int max_threads = omp_get_max_threads();
    std::atomic<int> shared_best(std::min(upper_bound, current_optimal_decision_tree->misclassification_score));

    // For smaller datasets deep in the tree, features are few and parallelizing per-feature is very fast
    const bool use_feature_parallel = allow_parallel && (dataview.get_feature_number() > max_threads);

    if (use_feature_parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
            if (!solution_configuration.stopwatch.IsWithinTimeLimit()) continue;

            const int ub_snapshot = shared_best.load(std::memory_order_relaxed);
            if (ub_snapshot == 0) continue;

            const int feature_index = dataview.gini_values[feature_nr].second;
            std::shared_ptr<Tree> thread_local_tree = std::make_shared<Tree>(-1, ub_snapshot);

            calculate_leaf_node(
                dataview.get_class_number(),
                dataview.get_dataset_size(),
                dataview.get_label_frequency(),
                thread_local_tree
            );

            create_optimal_decision_tree_internal(
                dataview,
                solution_configuration,
                feature_index,
                thread_local_tree,
                ub_snapshot,
                false 
            );

            const int local_score = thread_local_tree->misclassification_score;
            if (local_score < ub_snapshot) {
                std::lock_guard<std::mutex> lock(GeneralSolver::tree_mutex);
                if (local_score < current_optimal_decision_tree->misclassification_score) {
                    current_optimal_decision_tree = thread_local_tree;
                    shared_best.store(local_score, std::memory_order_relaxed);
                }
            }
        }
    } else {
        // Serial / Block Parallel
        for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
            if (!solution_configuration.stopwatch.IsWithinTimeLimit()) break;
            const int ub_snapshot = shared_best.load(std::memory_order_relaxed);
            if (ub_snapshot == 0) break;

            const int feature_index = dataview.gini_values[feature_nr].second;
            create_optimal_decision_tree_internal(
                dataview,
                solution_configuration,
                feature_index,
                current_optimal_decision_tree,
                ub_snapshot,
                allow_parallel
            );
            shared_best.store(current_optimal_decision_tree->misclassification_score, std::memory_order_relaxed);
            if (current_optimal_decision_tree->misclassification_score == 0) break;
        }
    }

    if (current_optimal_decision_tree->misclassification_score <= upper_bound) {
        Cache::global_cache.store(dataview, solution_configuration.max_depth, current_optimal_decision_tree);
    }
}

// ... (Rest of file: create_optimal_decision_tree_internal and calculate_leaf_node remain identical) ...
// Ensure you copy the bottom half of the file (from previous turn) here if replacing manual edits.
void GeneralSolver::create_optimal_decision_tree_internal(
    const Dataview& dataview,
    const Configuration& solution_configuration,
    int feature_index,
    std::shared_ptr<Tree>& current_optimal_decision_tree,
    int upper_bound,
    bool run_in_parallel
) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);

    const int num_splits = (int)possible_split_indices.size();
    if (num_splits <= 0) return;

    IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);
    const bool allow_parallel = run_in_parallel && !omp_in_parallel();
    std::atomic<int> best_score_atomic(current_optimal_decision_tree->misclassification_score);

    #pragma omp parallel if(allow_parallel)
    {
        const int num_threads = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        const int chunk_size = (num_splits + num_threads - 1) / num_threads;
        const int start_idx = thread_id * chunk_size;
        const int end_idx = std::min(start_idx + chunk_size - 1, num_splits - 1);

        std::queue<IntervalsPruner::Bound> local_queue;
        if (start_idx <= end_idx) {
            local_queue.push({start_idx, end_idx, -1, -1});
        }

        while (!local_queue.empty()) {
            if (!solution_configuration.stopwatch.IsWithinTimeLimit()) break;
            const int global_best = best_score_atomic.load(std::memory_order_relaxed);
            if (global_best == 0) break;

            auto current_interval = local_queue.front();
            local_queue.pop();

            if (interval_pruner.subinterval_pruning(current_interval, global_best)) continue;

            interval_pruner.interval_shrinking(current_interval, global_best);
            const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
            if (left > right) continue;

            const int mid = (left + right) / 2;
            const int split_point = possible_split_indices[mid];
            const int split_unique_value_index = current_feature[split_point].unique_value_index;

            const float threshold = (mid > 0)
                ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f
                : (current_feature[split_point].value + current_feature[0].value) / 2.0f;

            Dataview left_dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
            Dataview right_dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());

            Dataview::split_data_points(
                dataview,
                feature_index,
                split_point,
                split_unique_value_index,
                left_dataview,
                right_dataview,
                solution_configuration.max_depth
            );

            std::shared_ptr<Tree> left_optimal_dt = std::make_shared<Tree>(-1, global_best);
            std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>(-1, global_best);

            auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_dataview : right_dataview;
            auto& larger_data  = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_dataview : left_dataview;
            auto& smaller_dt = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_optimal_dt : right_optimal_dt;
            auto& larger_dt  = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_optimal_dt : left_optimal_dt;

            const int ub_for_larger = solution_configuration.use_upper_bound ? std::min(upper_bound, global_best) : global_best;
            const Configuration left_conf = solution_configuration.GetLeftSubtreeConfig();
            
            GeneralSolver::create_optimal_decision_tree(larger_data, left_conf, larger_dt, ub_for_larger, nullptr);

            const int best_after_larger = best_score_atomic.load(std::memory_order_relaxed);
            const int interval_half_distance = std::max(split_point - possible_split_indices[left], possible_split_indices[right] - split_point);
            const int ub_for_smaller = solution_configuration.use_upper_bound 
                ? std::max(std::min(best_after_larger, upper_bound) - larger_dt->misclassification_score, interval_half_distance)
                : best_after_larger;

            if (ub_for_smaller > 0 || (ub_for_smaller == 0 && best_after_larger == larger_dt->misclassification_score)) {
                const Configuration right_conf = solution_configuration.GetRightSubtreeConfig(left_conf.max_gap);
                GeneralSolver::create_optimal_decision_tree(smaller_data, right_conf, smaller_dt, ub_for_smaller, nullptr);

                const int candidate_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;
                
                if (candidate_score < best_score_atomic.load(std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(GeneralSolver::tree_mutex);
                    if (candidate_score < current_optimal_decision_tree->misclassification_score) {
                        current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);
                        current_optimal_decision_tree->misclassification_score = candidate_score;
                        best_score_atomic.store(candidate_score, std::memory_order_relaxed);
                    }
                }
            } else {
                smaller_dt->misclassification_score = -1;
            }

            interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

            if (left == right) continue;
            const int best_now = best_score_atomic.load(std::memory_order_relaxed);
            const int score_diff = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score - best_now;
            const auto [new_l, new_r] = interval_pruner.neighbourhood_pruning(score_diff, left, right, mid);
            if (new_l <= right) local_queue.push({new_l, right, mid, current_right_bound});
            if (left <= new_r) local_queue.push({left, new_r, current_left_bound, mid});
        }
    }
}

void GeneralSolver::calculate_leaf_node(
    int class_number,
    int instance_number,
    const std::vector<int>& label_frequency,
    std::shared_ptr<Tree>& current_optimal_decision_tree
) {
    int best_classification_score = -1;
    int best_classification_label = -1;
    for (int label = 0; label < class_number; label++) {
        if (label_frequency[label] > best_classification_score) {
            best_classification_score = label_frequency[label];
            best_classification_label = label;
        }
    }
    const int best_misclassification_score = instance_number - best_classification_score;
    if (best_misclassification_score < current_optimal_decision_tree->misclassification_score) {
        current_optimal_decision_tree->make_leaf(best_classification_label, best_misclassification_score);
    }
}