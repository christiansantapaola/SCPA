#include "util.h"
#include "ELLMatrix.h"

void print_status_bar(int used, int total,char *file) {
    fprintf(stderr, "\33[2K\r[");
    for (int i = 0; i < used; i++) {
        fprintf(stderr, "#");
    }
    for (int i = used; i < total; i++) {
        fprintf(stderr, "-");
    }
    fprintf(stderr, "] %.2f %s", (double) used / (double) total, file);
}

int count_file_in_directory(const char *dirpath) {
    int count = 0;
    struct dirent *entry;
    DIR *dir = opendir(dirpath);
    if (!dir) {
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        count++;
    }
    closedir(dir);
    return count;
}

COOMatrix *read_matrix_from_file(const char *path) {
    if (!path) return NULL;
    MTXParser *mtxParser = MTXParser_new(path);
    if (!mtxParser) {
        return NULL;
    }
    COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
    MTXParser_free(mtxParser);
    return cooMatrix;
}

CSRMatrix *read_csrMatrix_from_file(const char *path) {
    COOMatrix *cooMatrix = read_matrix_from_file(path);
    if (!cooMatrix) return NULL;
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    COOMatrix_free(cooMatrix);
    return csrMatrix;
}

double compute_FLOPS(u_int64_t nz, float time) {
    return 2.0 * (double) nz / time;
}

double compute_mean(float *obs, unsigned int size) {
    if (!obs || size == 0) return NAN;
    double sum = 0.0;
    for (unsigned int i = 0; i < size; i++) {
        sum += obs[i];
    }
    return sum / (double) size;
}

double compute_var(float *obs, unsigned int size, double mean) {
    if (!obs || size == 0 || size == 1) return NAN;
    double square_sum = 0.0;
    for (unsigned int i = 0; i < size; i++) {
        double centered_obs = (obs[i] - mean);
        square_sum +=  centered_obs * centered_obs;
    }
    return square_sum / (double) (size - 1);
}