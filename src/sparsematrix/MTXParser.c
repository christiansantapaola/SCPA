#include "MTXParser.h"

/**
 * tokenize() will split the inputstring into token delimited by delim and put them into argv,
 * maxtokens indicated the max number of token to split.
 * @param inputString: string to tokenize
 * @param delim : the token to split.
 * @param argv : where the token are put.
 * @param maxtokens : the max number of token to extract from inputString.
 * @return the number of token put in argv.
 */
size_t tokenize(char *inputString, const char *delim, char **argv, size_t maxtokens)
{
    size_t ntokens = 0;
    // char *tokenized = strdup(inputString);
    char *tokenized = inputString;
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


/**
 * parseToken() will parse the token into the appropriate value.
 * @param token: array containing the token.
 * @param numToken: the number of token in input
 * @param row: where the row value should be stored, can't be NULL.
 * @param col: where the column value should be stored, can't be NULL.
 * @param data: where the element at pos (row, col) value should be stored, can't be NULL.
 * @return 0 if successfull, -1 if fail.
 */
int parseToken(char **token, u_int64_t numToken, u_int64_t *row, u_int64_t *col, float * data) {
    char *endptr = NULL;
    *row = strtoll(token[0], &endptr, 10);
    if (!endptr) {
        return -1;
    }
    *col = strtoll(token[1], &endptr, 10);
    if (!endptr) {
        return -1;
    }
    if (numToken == 2) {
        *data = 1.0f;
    } else if (numToken == 3) {
        *data = strtof(token[2], &endptr);
        if (endptr != NULL && *endptr != '\n') {
            return -1;
        }
    } else {
        return -1;
    }
    return 0;
}

/**
 * parseHeader() will parse the token into the appropriate value.
 * @param token: array containing the token.
 * @param numToken: the number of token in input
 * @param row: where the row value should be stored, can't be NULL.
 * @param col: where the column value should be stored, can't be NULL.
 * @param numNonZero: where the number of non zero elements is stored, can't be NULL.
 * @return 0 if successfull, the index of the token that cannot be parsed  -1 if fail.
 */
int parseHeader(char **token, u_int64_t numToken, u_int64_t *row, u_int64_t *col, u_int64_t * numNonZero) {
    char *endptr = NULL;
    if (numToken != 3) {
        return -1;
    }
    *row = strtoll(token[0], &endptr, 10);
    if (!endptr) {
        return -1;
    }
    *col = strtoll(token[1], &endptr, 10);
    if (!endptr) {
        return -1;
    }
    *numNonZero = strtoll(token[2], &endptr, 10);
    if (!endptr) {
        return -1;
    }
    return 0;
}

/**
 *
 * @param path: path to the mtx file to parse
 * @return a new instance of the MTXParser struct allocated on the heap
 */
MTXParser *MTXParser_new(const char *path) {
    if (!path) return NULL;
    MTXParser *mtxParser = malloc(sizeof(*mtxParser));
    if (!mtxParser) {
        return NULL;
    }
    mtxParser->filename = strdup(path);
    if (!mtxParser->filename) {
        fclose(mtxParser->file);
        free(mtxParser);
        return NULL;
    }
    mtxParser->file = fopen(path, "r");
    if (!mtxParser->file) {
        free(mtxParser->filename);
        free(mtxParser);
        return NULL;
    }
    mtxParser->currentLine = 0;
    mtxParser->line = malloc(sizeof(char) * MTXPARSER_MAX_LINE_SIZE);
    if (!mtxParser->line) {
        free(mtxParser->filename);
        fclose(mtxParser->file);
        free(mtxParser);
        return NULL;
    }
    mtxParser->lineSize = sizeof(char) * MTXPARSER_MAX_LINE_SIZE;
    return mtxParser;
}

/**
 *
 * @param mtxParser: instance of the mtxParser struct to free.
 */
void MTXParser_free(MTXParser *mtxParser) {
    if (!mtxParser) return;
    free(mtxParser->filename);
    free(mtxParser->line);
    fclose(mtxParser->file);
    free(mtxParser);
}

int MTXParser_read_mtx_crd_size(MTXParser *parser, u_int64_t *M, u_int64_t *N, u_int64_t *nz )
{
    if (!parser || !M || !N || !nz) return -1;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;
    /* now continue scanning until you reach the end-of-comments */
    int ret = 0;
    char *token[3] = {0};
    while ((ret = getline(&parser->line, &parser->lineSize, parser->file)) != -1) {
        parser->currentLine++;
        int linelen = strlen(parser->line);
        if (linelen < 1) {
            continue;
        }
        // parse comment;
        if (parser->line[0] == '%') {
            continue;
        } else {
            char *line = strdup(parser->line);
            if (!line) {
                return -1;
            }
            size_t tok = tokenize(line, " ", token, 3);
            if (tok != 3) {
                free(line);
                return  -1;
            }
            int parseHeaderRet = parseHeader(token, tok, M, N, nz);
            if (parseHeaderRet == -1) {
                free(line);
                return -1;
            }
            return 0;
        }
    }
    return -1;
}

/**
 *
 * @param parser: instance of the mtxParser
 * @return the content of the mtx file stored in a COO sparse matrix format.
 */
COOMatrix *MTXParser_parse(MTXParser *parser) {
    if (!parser) return NULL;
    int ret_code;
    MM_typecode matcode;
    u_int64_t M, N, nz;
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

    if ((ret_code = MTXParser_read_mtx_crd_size(parser, &M, &N, &nz)) == -1) {
        fprintf(stderr, "[CRITICAL FAIL] cannot parse file %s\nerror at line %u: %s", parser->filename, parser->currentLine, parser->line);
        exit(EXIT_FAILURE);
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

    for (u_int64_t i=0; i < nz ; i++) {
        int res = MTXParser_parseLine(parser, &matrix->col_index[i], &matrix->row_index[i], &matrix->data[i]);
        if (res == -1) {
            fprintf(stderr, "[CRITICAL FAIL] cannot parse file %s\nerror at line %u: %s", parser->filename, parser->currentLine, parser->line);
            exit(EXIT_FAILURE);
        }
        /* adjust from 1-based to 0-based */
        matrix->row_index[i]--;
        matrix->col_index[i]--;
    }
    return matrix;

}



/**
* MTXParser_parseLine() will parse a line of the mtx data file.
* @param mtxParser: istance of the MTXParser
* @param row: where the row value should be stored, can't be NULL.
* @param col: where the column value should be stored, can't be NULL.
* @param data: where the element at pos (row, col) value should be stored, can't be NULL.
* @return 0 if successfull, -1 if fail.
*/
int MTXParser_parseLine(MTXParser *mtxParser, u_int64_t *row, u_int64_t *col, float * data) {
    if (!mtxParser || !row || !col || !data) return 0;
    char * line = NULL;
    size_t token = 0;
    ssize_t read = 0;
    char *tok[3] = {NULL, NULL, NULL};
    read = getline(&mtxParser->line, &mtxParser->lineSize, mtxParser->file);
    if (read == -1) {
        return -1;
    }
    mtxParser->currentLine++;
    line = strdup(mtxParser->line);
    if (!line) {
        return -1;
    }
    token = tokenize(line, " ", tok, 3);
    if (token < 2 || token > 3) {
        free(line);
        return -1;
    }
    token = parseToken(tok, token, row, col, data);
    if (token != 0) {
        free(line);
        return -1;
    }
    free(line);
    return 0;
}