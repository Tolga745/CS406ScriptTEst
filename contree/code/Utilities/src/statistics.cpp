#include <iostream>

#include "statistics.h"

unsigned long long statistics::total_number_of_specialized_solver_calls = 0;
unsigned long long statistics::total_number_of_general_solver_calls = 0;
unsigned long long statistics::total_number_cache_hits = 0;

bool statistics::should_print = false;

void statistics::print_statistics() {
    std::cout << "Total number of specialized solver calls: " << total_number_of_specialized_solver_calls << std::endl;
    std::cout << "Total number of general solver calls: " << total_number_of_general_solver_calls << std::endl;
    std::cout << "Total number cache hits: " << total_number_cache_hits << std::endl;
}
