#include "MTXParser.h"


// tokenize split the inputstring using delim as delimitator, and put up to maxtokens splitted string in argv.
size_t tokenize(char *inputString, const char *delim, char **argv, size_t maxtokens)
{
    size_t ntokens = 0;
    //char *tokenized = strdup(inputString);
    char *tokenized = inputString;
    if(tokenized)
    {
        argv[0] = tokenized;
        while(*tokenized)
        {
            if(strchr(delim, *tokenized))
            {
                *tokenized = 0;
                ntokens++;
                if(ntokens == maxtokens - 1)
                {
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
    if (numToken == 2) {
        *row = strtoll(token[0], &endptr, 10);
        if (!endptr) {
            return 0;
        }
        *col = strtoll(token[1], &endptr, 10);
        if (!endptr) {
            return 0;
        }
        *data = 1.0f;
        return 2;
    } else if (numToken == 3) {
        *row = strtoll(token[0], &endptr, 10);
        if (endptr != NULL) {
            return 0;
        }
        *col = strtoll(token[1], &endptr, 10);
        if (endptr != NULL) {
            return 0;
        }
        *data = strtof(token[2], &endptr);
        if (endptr != NULL) {
            return 0;
        }
        return 3;
    } else {
        return 0;
    }
}

int MTXParseLine(FILE *f, u_int64_t *row, u_int64_t *col, float * data) {
    if (!f || !row || !col || !data) return 0;
    char * line = NULL;
    size_t len = 0, token = 0;
    ssize_t read;
    char *tok[3] = {NULL, NULL, NULL};
    read = getline(&line, &len, f);
    if (read == -1) {
        return -1;
    }
    token = tokenize(line, " ", tok, 3);
    if (token == 0) {
        free(line);
        return -1;
    }
    token = parseToken(tok, token, row, col, data);
    if (token == 0) {
        free(line);
        return -1;
    }
    free(line);
    return 0;
}