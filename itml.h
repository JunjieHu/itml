/*!
 * @mainpage ITML
 * @version 0.1
 * @brief ITML C++ version, implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk).
 * CreateDate 2014-07-05
 */

#ifndef ITML_H
#define ITML_H
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>

using boost::numeric::ublas::matrix;
using boost::numeric::ublas::vector;

/**
 * @brief the struct storing the parameters of ITML
 *
 */
struct itml_param
{
    itml_param(): thresh(10e-3), gamma(1), max_iters(100000){}
    float thresh;
    float gamma;
    int max_iters;
};



/**
 * @brief the struct to store the raw data for the Distance Metric Learning
 *
 */
struct dml_problem
{
    matrix<float> X;
    vector<float> y;
    matrix<float> C;
    vector<float> label;
};

/**
 * @brief ITML class containing all the supporting functions
 *
 */
class ITML
{
public:

    /**
     * @brief The main function of ITML
     *
     * @param C mx4 matrix, 1:x1 2:x2 3:+/-1 similar/disimilar 4:l/u lower/upper bound
     * @param X nxd matrix, n samples with d dimensions
     * @param M_0 dxd matrix, identity matrix
     * @param param itml paramers
     * @param M dxd matrix, the learned matrix that is returned as a reference parameter
     */
    void itml_alg(const matrix<float>& C,const matrix<float>& X,const matrix<float>& M_0,const itml_param& param,
                  matrix<float>& M );


    /**
     * @brief
     *
     * @param y 1xn vector, the label
     * @param X nxd matrix, n samples with d dimensions
     * @param M dxd matrix, the learned matrix that is returned as a reference parameter
     * @param k the number of the nearest neighbors
     * @param Xt mxd matrix, m testing samples with d dimensions
     * @param pred_y 1xm vector, the predicted label for testing samples (return)
     */
    void KNN(const vector<float>& y, const matrix<float>& X, const matrix<float>& M, int k, const matrix<float>& Xt,
             vector<float>& pred_y);


    /**
     * @brief This function is to compute the distance extreme value for the similar/dissimilar pairs.
     *
     * @param X nxd matrix, n samples with d dimensions
     * @param lpercent the lower percent of the distance range
     * @param upercent the upper percent of the distance range
     * @param M dxd matrix, the learned matrix that is returned as a reference parameter
     * @param l the lower bound for the constrains
     * @param u the upper bound for the constrains
     */
    void ComputeDistanceExtreme(const matrix<float>& X,int lpercent, int upercent, const matrix<float>& M,
                                float& l, float& u);

    /**
     * @brief This function is to compute the distance extreme value for the similar/dissimilar pairs.
     *
     * @param X nxd matrix, n samples with d dimensions
     * @param lpercent the lower percent of the distance range
     * @param upercent the upper percent of the distance range
     * @param l the lower bound for the constrains
     * @param u the upper bound for the constrains
     */
    void ComputeDistanceExtreme(const matrix<float>& X,int lpercent, int upercent,
                                float& l, float& u);


    /**
     * @brief This function is to generate the similar/dissimilar pairs of constraints.
     *
     * @param y 1xn vector, the label
     * @param contraints_num int, the number of comstraints
     * @param l float, the lower bound for the constrains
     * @param u float, the upper bound for the constrains
     * @param C mx4 matrix, 1:x1 2:x2 3:+/-1 similar/disimilar 4:l/u lower/upper bound
     */
    void GetConstraints(const vector<float>& y, int contraints_num, float l, float u,
                        matrix<float>& C);
};

#endif // ITML_H
