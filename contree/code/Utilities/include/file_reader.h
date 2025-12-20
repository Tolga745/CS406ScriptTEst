#ifndef FILE_READER_H
#define FILE_READER_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "configuration.h"
#include "dataset.h"


class file_reader {
public:
    static void read_file(const std::string& file_name, Dataset& data, int& class_number);
private:
};


#endif // FILE_READER_H
