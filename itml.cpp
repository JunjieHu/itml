#include "itml.h"
#include "utility.h"
#include <iostream>
#include <ctime>
#include <math.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/container/vector.hpp>


/***************************************************************************************//**
 * Input:
 *      C: mx4 matrix, 1:x1 2:x2 3:+/-1 similar/disimilar 4:l/u lower/upper bound
 *      X: nxd matrix, n samples with d dimensions
 *    A_0: dxd matrix, identity matrix
 *  param: itml paramers
 * Output:
 *      A: dxd matrix, the learned matrix
 * ******************************************************************************************/
void ITML::itml_alg(const matrix<float>& C,const matrix<float>& X,const matrix<float>& A_0,const itml_param& param,
              matrix<float>& A )
{
//    pitml_alg(C,X,X,A_0,param,A);
    float tol = param.thresh;
    float gamma = param.gamma;
    //float gamma = 1;
    int max_iters = param.max_iters;

    int i, i1, i2, iter, c;
    float gamma_proj;
    double conv;
    boost::numeric::ublas::vector<float> lambda, lambdaold, bhat, x1, x2, v;

    c = C.size1();
    lambda = boost::numeric::ublas::vector<float> (C.size1(),0);
    bhat = column(C,3);
    lambdaold = boost::numeric::ublas::vector<float> (C.size1(),0);
    conv = INFINITY;
    A = A_0;


    i=0;
    iter = 0;
    while(1)
    {
        i1 = C(i,0);
        i2 = C(i,1);
        x1 = row(X,i1);
        x2 = row(X,i2);   // 1xd
        v = x1-x2;
        float wtw = inner_prod(prod(v,A),v);     // 1x1

        if(std::abs(bhat(i))<10e-10)
            std::cerr<<"bhat should never be 0!"<<std::endl;

        if(INFINITY==gamma)
            gamma_proj = 1;
        else
            gamma_proj = gamma/(gamma+1);

        float alpha,beta;
        if(C(i,2)==1)
        {
            alpha = std::min(lambda(i),gamma_proj*(1/wtw - 1/bhat(i)));
            lambda(i) = lambda(i) - alpha;
            beta = alpha/(1-alpha*wtw);
            bhat(i) = 1/( (1/bhat(i)) + (alpha/gamma));

        }else if(C(i,2)==-1)
        {
            alpha = std::min(lambda(i),gamma_proj*(1/bhat(i) - 1/wtw));
            lambda(i) = lambda(i) - alpha;
            beta = -1*alpha/( 1+alpha*wtw );
            bhat(i) = 1/( (1/bhat(i)) - (alpha/gamma));
        }

        vector<float> tempvec = prod(v,A);
        matrix<float> tempmat = outer_prod( tempvec,v );
        A  = A + beta* prod( tempmat, A );

        if(i == c-1)
        {
            double normsum = norm_2(lambda) + norm_2(lambdaold);
            if(normsum==0)
                break;
            else{
                conv = norm_1(lambdaold-lambda)/normsum;
                if(conv<tol || iter>max_iters)
                    break;
            }
            lambdaold = lambda;
        }
        i = (i+1)%c;
        iter = iter +1;
        if( iter%5000 ==0)
            std::cout<<"itml iter: "<<iter<<", conv ="<<conv<<std::endl;
    }
}

/*****************************************************************************************//**
 * Input:
 *      y: the class labels of the samples
 *      X: n1xd matrix, n1 training samples with d dimensions
 *      M: dxd matrix, distance matrix
 *      k: the k-nearest neighbors
 *     Xt: n2xd matrix, n2 testing samples with d dimensions
 * Output:
 * pred_y: 1xn2 vector, predicted labels for the testing samples
 * ******************************************************************************************/
void ITML::KNN(const vector<float>& y, const matrix<float>& X, const matrix<float>& M, int k, const matrix<float>& Xt,
         vector<float>& pred_y)
{
    vector<float> dif;
    vector<int> cnt;
    count_difference(y,dif,cnt);
    cout<<dif<<endl<<cnt<<endl;
    int num_class =  dif.size();

    int n,nt;
    n = X.size1();
    nt = Xt.size1();


//    matrix<float> D(n,nt,0);
//    pdistance(X,Xt,M,D);
    matrix<float> temp1 = prod(X,M);
    matrix<float> temp2 = prod(trans(M),trans(Xt));
    matrix<float> K = prod( temp1, temp2 );  //K:nxnt
    vector<float> l(n,0);
    vector<float> lt(nt,0);
    matrix<float> D(n,nt,0);

    // calculate the nxnt distance matrix:D(i,j) = distance between X(i) and Xt(j)
    for(int i=0;i<n;i++){
        l(i) = norm_2( prod(row(X,i),M) );
        l(i) *= l(i);
    }
    for(int i=0;i<nt;i++){
        lt(i) = norm_2( prod(row(Xt,i),M) );
        lt(i) *= lt(i);
    }
    for(int i=0;i<n;i++)
        for(int j=0;j<nt;j++)
            D(i,j) = l(i) + lt(j) -2 * K(i,j);

//    pKNN(y,D,k,pred_y);
    // count number of different labels in k-nearest neighbors
    matrix<int> count(num_class,nt,0);
    vector<float>::iterator it;
    for(int i=0;i<nt;i++){
        vector<float> colum_i = column(D,i);
        vector<int> sidx = ordered(colum_i);
        for(int j=0;j<k;j++){
            it = find(dif.begin(),dif.end(),y(sidx(j)));
            count(it-dif.begin(),i)++;
        }
    }
    for(int i=0;i<10;i++){
       cout<<column(count,i)<<endl;
    }

    // predict the label by largest number of neighbors
    vector<int> val;
    vector<int> idx;
    max_matrix(count,val,idx,2);
    pred_y = vector<float>(nt,0);
    for(int i=0;i<nt;i++)
        pred_y(i) = dif(idx(i));
}

/*****************************************************************************************//**
 * \brief Compute the distance extreme for similar/dissimilar pairs
 * Input:
 * \param        X nxd matrix, n samples with d dimensions
 * \param lpercent lower percent of sorted distance
 * \param upercent upper percent of sorted distance
 * \param        M dxd matrix, distance matrix
 * Output:
 * \return       l lower extreme
 * \return       u upper extreme
 * ******************************************************************************************/
void ITML::ComputeDistanceExtreme(const matrix<float>& X,int lpercent, int upercent, const matrix<float>& M,
                            float& l, float& u)
{
    std::srand(unsigned(std::time(0)));

    if(lpercent<1||lpercent>100)
        std::cerr<<"low percent should between 1 and 100"<<std::endl;
    if(upercent<1||upercent>100)
        std::cerr<<"up percent should between 1 and 100"<<std::endl;
    int n = X.size1();
    int num_trials = std::min(100, n*(n-1)/2);

    boost::numeric::ublas::vector<float> dists(num_trials,0);
    for(int i=0;i<num_trials;i++){
        int j1 = std::ceil(std::rand()%n);
        int j2 = std::ceil(std::rand()%n);
        boost::numeric::ublas::vector<float> v = row(X,j1)-row(X,j2);
        dists(i) = inner_prod( prod(v,M), v);
    }
    std::sort(dists.begin(),dists.end());
    //std::cout<<dists<<std::endl;
    l = dists(std::ceil(num_trials* lpercent/100));
    u = dists(std::ceil(num_trials* upercent/100));
}

/*****************************************************************************************//**
 * Input:
 *        X: nxd matrix, n samples with d dimensions
 * lpercent: lower percent of sorted distance
 * upercent: upper percent of sorted distance
 * Output:
 *        l: lower extreme
 *        u: upper extreme
 * ******************************************************************************************/
void ITML::ComputeDistanceExtreme(const matrix<float>& X,int lpercent, int upercent,
                            float& l, float& u)
{
    std::srand(unsigned(std::time(0)));

    if(lpercent<1||lpercent>100)
        std::cerr<<"low percent should between 1 and 100"<<std::endl;
    if(upercent<1||upercent>100)
        std::cerr<<"up percent should between 1 and 100"<<std::endl;
    int n = X.size1();
    int num_trials = std::min(100, n*(n-1)/2);

    boost::numeric::ublas::vector<float> dists(num_trials,0);
    for(int i=0;i<num_trials;i++){
        int j1 = std::ceil(std::rand()%n);
        int j2 = std::ceil(std::rand()%n);
        while(j2==j1){
            j2 = std::ceil(std::rand()%n);
        }
        boost::numeric::ublas::vector<float> v = row(X,j1)-row(X,j2);
        dists(i) = inner_prod( v, v);
    }
    std::sort(dists.begin(),dists.end());
    //std::cout<<dists<<std::endl;
    l = dists(std::ceil(num_trials* lpercent/100));
    u = dists(std::ceil(num_trials* upercent/100));
}

/*****************************************************************************************//**
 * Input:
 *        y: the class labels of the samples
 * constrains_num: number of the generated constraints
 *        l: lower extreme
 *        u: upper extreme
 * Output:
 *        C: mx4 matrix, 1:x1 2:x2 3:+/-1 similar/disimilar 4:l/u lower/upper bound
 * ******************************************************************************************/
/**
 * @brief
 *
 * @param y the class labels of the samples
 * @param constraints_num number of the generated constraints
 * @param l lower extreme
 * @param u upper extreme
 * @param C mx4 matrix, 1:x1 2:x2 3:+/-1 similar/disimilar 4:l/u lower/upper bound
 */
void ITML::GetConstraints(const vector<float>& y, int constraints_num, float l, float u,
                    matrix<float>& C)
{
    std::srand(unsigned(std::time(0)));
    int m = y.size();
    C = boost::numeric::ublas::zero_matrix<float> (constraints_num,4);

    for( int k =0;k<constraints_num;k++){
        int i = std::ceil(std::rand() % m );
        int j = std::ceil(std::rand() % m );
        while(i==j){
            j = std::ceil(std::rand() % m );
        }
        C(k,0) = i;
        C(k,1) = j;
        if( y(i) ==y(j) ){
            C(k,2) = 1;
            C(k,3) = l;
        }else
        {
            C(k,2) = -1;
            C(k,3) = u;
        }
    }

}





