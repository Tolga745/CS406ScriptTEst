#include "configuration.h"

std::chrono::high_resolution_clock::time_point starting_time = std::chrono::high_resolution_clock::now();

Configuration Configuration::GetLeftSubtreeConfig() const {
	Configuration left_config(*this);
	left_config.max_depth -= 1;
	left_config.max_gap = (max_gap - (max_gap+1)/2) / 2;
	left_config.is_root = false;
	return left_config;
}

Configuration Configuration::GetRightSubtreeConfig(int left_gap) const {
	Configuration right_config(*this);
	right_config.max_depth -= 1;
	right_config.max_gap = (max_gap - (max_gap + 1) / 2) - left_gap;
	right_config.is_root = false;
	return right_config;
}