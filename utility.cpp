#include <iostream>
#include <string>
#include "utility.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using std::string;
using std::cout;
using std::endl;

char *line=NULL;
int max_line_len =0;

char* readline(FILE *input)
{
    int len;
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

