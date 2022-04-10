#include "MTXParser.h"


MTXParser *MTXParser_new(char *path) {
    if (!path) return NULL;
    MTXParser *mtxParser = malloc(sizeof(*mtxParser));
    if (!mtxParser) {
        return NULL;
    }
    mtxParser->filename = strdup(path);
    mtxParser->file = fopen(path, "r");
    if (!mtxParser->file) {
        return NULL;
    }
    mtxParser->currentLine = 0;
    mtxParser->invalidToken = NULL;
    return mtxParser;
}

void MTXParser_free(MTXParser *mtxParserStatus) {
    if (!mtxParserStatus) return;
    free(mtxParserStatus->filename);
    free(mtxParserStatus->line);
    free(mtxParserStatus->invalidToken);
    fclose(mtxParserStatus->file);
    free(mtxParserStatus);
}

COOMatrix *MTXParser_parse(MTXParser *parser) {
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;
    u_int64_t i, *I, *J;
    double *val;
    COOMatrix *matrix = NULL;

    if (mm_read_banner(parser->file, &matcode) != 0) {
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode) ) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(parser->file, &M, &N, &nz)) !=0) {
        exit(1);
    }


    /* reserve memory for matrices */
    matrix = (COOMatrix *) malloc(sizeof(COOMatrix));
    if (!matrix) {
        perror("MTXParser_parse(): ");
        return NULL;
    }
    matrix->row_size = M;
    matrix->col_size = N;
    matrix->num_non_zero_elements = nz;
    matrix->row_index = (u_int64_t *) malloc(matrix->num_non_zero_elements * sizeof(u_int64_t));
    if (!matrix->row_index) {
        perror("MTXParser_parse(): ");
        free(matrix);
        return NULL;
    }
    matrix->col_index = (u_int64_t *) malloc(matrix->num_non_zero_elements * sizeof(u_int64_t));
    if (!matrix->col_index) {
        perror("MTXParser_parse(): ");
        free(matrix->row_index);
        free(matrix);
        return NULL;
    }
    matrix->data =(float *) malloc(matrix->num_non_zero_elements * sizeof(float));
    if (!matrix->data) {
        free(matrix->col_index);
        free(matrix->row_index);
        free(matrix);
        perror("MTXParser_parse(): ");
        return NULL;
    }

    for (i=0; i < nz ; i++) {
        int res = MTXParser_parseLine(parser, &matrix->col_index[i], &matrix->row_index[i], &matrix->data[i]);
        if (res == -1) {
            fprintf(stderr, "[CRITICAL FAIL] cannot parse file %s at line %u\n%s is invalid!\n %s unrecognized\n", parser->filename, parser->currentLine, parser->line, parser->invalidToken);
            free(parser->line);
            exit(EXIT_FAILURE);
        }
        /* adjust from 1-based to 0-based */
        matrix->row_index[i]--;
        matrix->col_index[i]--;
    }
    return matrix;

}

// tokenize split the inputstring using delim as delimitator, and put up to maxtokens splitted string in argv.
size_t tokenize(char *inputString, const char *delim, char **argv, size_t maxtokens)
{
    size_t ntokens = 0;
    char *tokenized = strdup(inputString);
    //char *tokenized = inputString;
    if(tokenized) {
        argv[0] = tokenized;
        while(*tokenized) {
            if(strchr(delim, *tokenized)) {
                *tokenized = 0;
                ntokens++;
                if(ntokens == maxtokens) {
                    break;
                }
                argv[ntokens] = tokenized + 1;
            }
            tokenized++;
        }
    }
    return ntokens + 1;
}

size_t parseToken(char **token, size_t numToken, u_int64_t *row, u_int64_t *col, float * data) {
    char *endptr = NULL;
    int consumedToken = 0;
    *row = strtoll(token[0], &endptr, 10);
    if (!endptr) {
            return 1;
    }
    consumedToken++;
    *col = strtoll(token[1], &endptr, 10);
    if (!endptr) {
        return 2;
    }
    consumedToken++;
    if (consumedToken == numToken) {
        *data = 1.0f;
    } else {
        *data = strtof(token[2], &endptr);
        if (endptr != NULL && *endptr != '\n') {
            return 3;
        }
        consumedToken++;
    }
    return 0;
}

int MTXParser_parseLine(MTXParser *mtxParser, u_int64_t *row, u_int64_t *col, float * data) {
    if (!mtxParser || !row || !col || !data) return 0;
    char * line = NULL;
    size_t len = 0, token = 0;
    ssize_t read = 0;
    char *tok[3] = {NULL, NULL, NULL};
    mtxParser->currentLine++;
    read = getline(&line, &len, mtxParser->file);
    if (read == -1) {
        return -1;
    }
    fprintf(stderr, "%s\n", line);
    free(mtxParser->line);
    mtxParser->line = strdup(line);
    token = tokenize(line, " ", tok, 3);
    if (token == 0) {
        free(line);
        return -1;
    }
    token = parseToken(tok, token, row, col, data);
    if (token != 0) {
        mtxParser->invalidToken = strdup(tok[token - 1]);
        free(line);
        return -1;
    }
    free(line);
    return 0;
}