#include "intervals_pruner.h"

IntervalsPruner::IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap)
    : possible_split_indexes(possible_split_indexes_ref), possible_split_size(int(possible_split_indexes.size())), max_gap(max_gap) {
        evaluated_indices_record.reserve(possible_split_indexes.size() / 8);
        rightmost_zero_index = possible_split_size;
        leftmost_zero_index = -1;
    }

bool IntervalsPruner::subinterval_pruning(const IntervalsPruner::Bound& current_bounds, int current_best_score) {
    int left_bound_score_left = 0;
    int right_bound_score_right = 0;

    if (current_bounds.last_split_left_index != -1) {
        left_bound_score_left = evaluated_indices_record[current_bounds.last_split_left_index].first;
    }

    if (current_bounds.last_split_right_index != -1) {
        right_bound_score_right = evaluated_indices_record[current_bounds.last_split_right_index].second;
    }

    return left_bound_score_left + right_bound_score_right + max_gap >= current_best_score;
}

void IntervalsPruner::interval_shrinking(IntervalsPruner::Bound& current_bounds, int current_best_score) {
    current_bounds.left_bound = std::max(current_bounds.left_bound, leftmost_zero_index + 1);
    current_bounds.right_bound = std::min(current_bounds.right_bound, rightmost_zero_index - 1);

    if (current_bounds.last_split_left_index != -1) {
        int updated_score_difference = evaluated_indices_record[current_bounds.last_split_left_index].first + evaluated_indices_record[current_bounds.last_split_left_index].second - current_best_score + max_gap;
        if (updated_score_difference > 0) {
            int extended_left_bound = possible_split_indexes[current_bounds.last_split_left_index] + updated_score_difference + 1;

            if (extended_left_bound <= possible_split_indexes[current_bounds.right_bound]) {
                int new_left_bound = int(std::lower_bound(possible_split_indexes.begin() + current_bounds.left_bound, possible_split_indexes.begin() + current_bounds.right_bound, extended_left_bound) - possible_split_indexes.begin());
                current_bounds.left_bound = std::max(current_bounds.left_bound, new_left_bound);
            } else {
                current_bounds.left_bound = 1;
                current_bounds.right_bound = 0;
                current_bounds.last_split_left_index = -1;
                current_bounds.last_split_right_index = -1;
                return;
            }
        }
    }

    if (current_bounds.last_split_right_index != -1) {
        int updated_score_difference = evaluated_indices_record[current_bounds.last_split_right_index].first + evaluated_indices_record[current_bounds.last_split_right_index].second - current_best_score + max_gap;
        if (updated_score_difference > 0) {
            int extended_right_bound = possible_split_indexes[current_bounds.last_split_right_index] - updated_score_difference - 1;

            if (extended_right_bound >= possible_split_indexes[current_bounds.left_bound]) {
                int new_right_bound = int(std::upper_bound(possible_split_indexes.begin() + current_bounds.left_bound, possible_split_indexes.begin() + current_bounds.right_bound, extended_right_bound) - possible_split_indexes.begin());
                current_bounds.right_bound = std::min(current_bounds.right_bound, new_right_bound);
            } else {
                current_bounds.left_bound = 1;
                current_bounds.right_bound = 0;
                current_bounds.last_split_left_index = -1;
                current_bounds.last_split_right_index = -1;
                return;
            }
        }
    }
}

std::pair<int, int> IntervalsPruner::neighbourhood_pruning(int score_difference, int left, int right, int split_index) {
    score_difference += max_gap;
    if (score_difference <= 0) {
        return {split_index + 1, split_index};
    }

    int new_bound_left = std::max(split_index + 1, leftmost_zero_index + 1);
    int minimum_right_value = possible_split_indexes[split_index] + score_difference + 1;
    if (minimum_right_value < possible_split_indexes[right]) {
        new_bound_left = int(std::lower_bound(possible_split_indexes.begin() + new_bound_left, possible_split_indexes.begin() + right, minimum_right_value) - possible_split_indexes.begin());
    } else {
        new_bound_left = right + 1; 
    }

    int new_bound_right = std::min(split_index - 1, rightmost_zero_index - 1);
    int minimum_left_value = possible_split_indexes[split_index] - score_difference - 1;
    if (minimum_left_value > possible_split_indexes[left]) {
        new_bound_right = int(std::upper_bound(possible_split_indexes.begin() + left, possible_split_indexes.begin() + new_bound_right, minimum_left_value) - possible_split_indexes.begin());
    } else {
        new_bound_right = left - 1;
    }

    return {new_bound_left, new_bound_right};
}

void IntervalsPruner::add_result(int index, int left_score, int right_score) {
    if (left_score == 0) {
        leftmost_zero_index = std::max(leftmost_zero_index, index);
    }

    if (right_score == 0) {
        rightmost_zero_index = std::min(rightmost_zero_index, index);
    }

    if (left_score == -1) {
        left_score = 0;
    }

    if (right_score == -1) {
        right_score = 0;
    }

    evaluated_indices_record[index] = {left_score, right_score};
}
