#include "util.h"
#include "ELLMatrix.h"

int forEachFile(const char *path, int (*op)(char *, struct dirent *, void *), void *args ) {
    DIR *dir = opendir(path);
    if (!dir) {
        return -1;
    }
    char old_dir[4096] = {0};
    if (getcwd(old_dir, 4096) == NULL) {
        return -1;
    }
    chdir(path);
    int fileProcessed = 0;
    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        // if file is not a regular file then skip.
        if (entry->d_type != DT_REG) {
            continue;
        }
        // update status bar
        // print_status_bar(fileProcessed, numDir, entry->d_name);
        fileProcessed++;
        // get the fullpath of the current file.
        char *absolutePath = realpath(entry->d_name, NULL);
        if (!absolutePath) {
            perror(entry->d_name);
            closedir(dir);
            chdir(old_dir);
            exit(EXIT_FAILURE);
        }
        int ret = op(absolutePath, entry, args);
        if (ret < 0) {
            closedir(dir);
            chdir(old_dir);
            return ret;
        }
        free(absolutePath);
    }
    closedir(dir);
    chdir(old_dir);
    return 0;
}

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

void transposef(float *dest, const float *src, u_int64_t num_row, u_int64_t num_col) {
    for (size_t row = 0; row < num_row; row++) {
        for (size_t col = 0; col < num_col; col++) {
            size_t srcIdx = row * num_col + col;
            size_t dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}

void transpose_u_int64_t(u_int64_t *dest, const u_int64_t *src, u_int64_t num_row, u_int64_t num_col) {
    for (size_t row = 0; row < num_row; row++) {
        for (size_t col = 0; col < num_col; col++) {
            size_t srcIdx = row * num_col + col;
            size_t dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}
