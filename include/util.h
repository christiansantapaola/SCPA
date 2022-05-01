//
// Created by 9669c on 20/04/2022.
//

#ifndef SPARSEMATRIX_UTIL_H
#define SPARSEMATRIX_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

void print_status_bar(int used, int total,char *file);
int count_file_in_directory(const char *dirpath);

#endif //SPARSEMATRIX_UTIL_H
