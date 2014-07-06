#ifndef _UTILITY_H
#define _UTILITY_H

#include <iostream>
#include <string.h>
#include <stdio.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

using boost::numeric::ublas::matrix;
using boost::numeric::ublas::vector;
using std::string;
using std::cout;
using std::endl;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
extern char *line;
extern int max_line_len;

/**
 * @brief copy the [head tail] entries from the c-th column of src to dst
 *
 * @param dst the destination vector
 * @param src the source matrix
 * @param head the head index of c-th column
 * @param tail the tail index of c-th column
 * @param c the c-th colum of src
 */
template <class S> static inline void get_column(S*& dst, S** src,int head, int tail, int c)
{
    dst = Malloc(S,tail-head+1);
    if(head<0||src[tail][c]==NULL){
        cout<<"head "<<head<<"or tail "<<tail<<" out of range."<<endl;
        exit(1);
    }
    for(int i=head;i<=tail;i++)
        dst[i-head]=src[i][c];
}

// read each line from the file
char* readline(FILE *input);

// read matrix from file
/**
 * @brief read matrix from file
 *
 * @param filename the file name
 * @param mat the loaded matrix
 */
template<class T> void read_matrix(string filename,matrix<T> & mat)
{
    unsigned int i, j;
    FILE *fp = fopen(filename.c_str(),"r");

    if(fp == NULL)
    {
        cout<<"can't open input file" << filename<<endl;
        exit(1);
    }

    max_line_len = 1024;
    line = Malloc(char,max_line_len);

    // read data
    int n,d;
    n=0;
    while(readline(fp)!=NULL)
    {
        d=0;
        char *p = strtok(line," \t");
        while(1)
        {
            d++;
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n')// check '\n' as ' ' may be after the last feature
                break;
        }
        ++n;
    }

    rewind(fp);

    // read data
    mat.resize(n,d,false);
    i=0;
    while(readline(fp)!=NULL)
    {
        j=0;
        char *p = strtok(line," \t");
        while(1)
        {
            T temp=(T)atof(p);
            mat(i,j++) = temp;
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n')// check '\n' as ' ' may be after the last feature
                break;
        }
        ++i;
    }
    cout<<mat.size1()<<" "<<mat.size2();
}

// read vector from file
/**
 * @brief read vector from the file
 *
 * @param filename file name
 * @return vec return the loaded vector
 */
template<class T> void read_vector(string filename,vector<T> & vec)
{
    matrix<T> temp;
    read_matrix(filename,temp);
    vec = column(temp,0);
}

/* Matrix inversion routine.
Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
/**
 * @brief Get the inverse of the input matrix
 *
 * @param input input matrix
 * @param inverse the inverse of the input matrix
 * @return bool indicate whether the inverse of the matrix is existed or not
 */
bool InvertMatrix(const matrix<T>& input, matrix<T>& inverse)
{
   typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;

   // create a working copy of the input
   matrix<T> A(input);

   // create a permutation matrix for the LU-factorization
   pmatrix pm(A.size1());

   // perform LU-factorization
   int res = lu_factorize(A, pm);
   if (res != 0)
       return false;

   // create identity matrix of "inverse"
   inverse.assign(boost::numeric::ublas::identity_matrix<T> (A.size1()));

   // backsubstitute to get the inverse
   lu_substitute(A, pm, inverse);

   return true;
}



template<class T>
/**
 * @brief Like Matlab function min(), return the minimum number and the index along the chosen dimension
 *
 * @param mat the target matrix
 * @return val the minimum value along the dimension
 * @return idx the corresponding index for the minimum value
 * @param dimension like Matlab, choose the dimension to measure
 */
void min_matrix(const matrix<T>& mat, vector<T>& val, vector<int>& idx, int dimension=1)
{
    int n = mat.size1();
    int m = mat.size2();
    if(dimension == 1){
        val = boost::numeric::ublas::column(mat,0);
        idx = vector<int>(n,0);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(val(i)>mat(i,j)){
                    val(i) = mat(i,j);
                    idx(i) = j;
                }
            }
        }
    }
    else if(dimension == 2)
    {
        val = boost::numeric::ublas::row(mat,0);
        idx = vector<int>(m,0);
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(val(i)>mat(j,i)){
                    val(i) = mat(j,i);
                    idx(i) = j;
                }
            }
        }
    }
}

template<class T>
/**
 * @brief Like Matlab function max(), return the maximum number and the index along the chosen dimension
 *
 * @param mat the target matrix
 * @return val the maximum value along the dimension
 * @return idx the corresponding index for the maximum value
 * @param dimension like Matlab, choose the dimension to measure
 */
void max_matrix(const matrix<T>& mat, vector<T>& val, vector<int>& idx, int dimension=1)
{
    int n = mat.size1();
    int m = mat.size2();
    if(dimension == 1){
        val = boost::numeric::ublas::column(mat,0);
        idx = vector<int>(n,0);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(val(i)<mat(i,j)){
                    val(i) = mat(i,j);
                    idx(i) = j;
                }
            }
        }
    }
    else if(dimension == 2)
    {
        val = boost::numeric::ublas::row(mat,0);
        idx = vector<int>(m,0);
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(val(i)<mat(j,i)){
                    val(i) = mat(j,i);
                    idx(i) = j;
                }
            }
        }
    }
}

template<typename T>
/**
 * @brief count the number of the distinct elements of the vector
 *
 * @param vec  the target vector for the counting
 * @return dif the boost vector containing all the distinct elements of vec
 * @return cnt the boost vector, cnt(i) counts the times that dif(i) appears in vec
 */
void count_difference(const vector<T>& vec, vector<T>& dif, vector<int>& cnt)
{
    int j = 0;
    dif = vector<T> (vec.size(),INFINITY);
    cnt = vector<int> (vec.size(),0);

    for(int i=0;i<vec.size();i++){
        typename vector<T>::iterator it = find(dif.begin(),dif.end(),vec(i));
        if(it == dif.end()) {
            dif(j) = vec(i);
            cnt(j++) = 1;
//            cout<<"value"<<dif<<endl;
//            cout<<"cnt"<<cnt<<endl;
        }else{
            cnt(it-dif.begin()) += 1;
        }
    }
    dif.resize(j,true);
    cnt.resize(j,true);
}

template<typename T>
/**
 * @brief count the number of the distinct elements of the vector
 *
 * @param vec the target vector for the counting
 * @return dif the std vector containing all the distinct elements of vec
 * @return cnt the std vector, cnt(i) counts the times that dif(i) appears in vec
 */
void count_difference(const vector<T>& vec, std::vector<T>& dif, std::vector<int>& cnt)
{
    dif.clear();
    cnt.clear();
    for(int i=0;i<vec.size();i++){
        typename std::vector<T>::iterator it = find(dif.begin(),dif.end(),vec(i));
        if(it == dif.end()) {
            dif.push_back(vec(i));
            cnt.push_back(1);
//            cout<<"value"<<vec(i)<<endl;
//            cout<<"cnt"<<*(cnt.end()-1)<<endl;
        }else{
            cnt[it-dif.begin()] += 1;
        }
    }
}


template <typename T>
/**
 * @brief sort vector and return the indices
 *
 * @param values a vector that needs to be sorted.
 * @return boost::numeric::ublas::vector<size_t>
 */
boost::numeric::ublas::vector<size_t> ordered(boost::numeric::ublas::vector<T> const& values) {
    boost::numeric::ublas::vector<size_t> indices(values.size());
    for(int i=0;i<indices.size();i++)
        indices(i)=i;
    std::sort(
        begin(indices), end(indices),
        [&](size_t a, size_t b) { return values[a] < values[b]; }
    );
    return indices;
}
#endif
