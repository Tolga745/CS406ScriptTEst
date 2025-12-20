#ifndef STATISTICS_H
#define STATISTICS_H

class statistics {
public:
    static unsigned long long total_number_of_specialized_solver_calls;
    static unsigned long long total_number_of_general_solver_calls;
    static unsigned long long total_number_cache_hits;

    static bool should_print;

    static void print_statistics();
};

#endif // STATISTICS_H


