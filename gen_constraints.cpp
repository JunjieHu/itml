/*!
 * @mainpage Generate the Constraints
 * @version 0.1
 * @brief ITML C++ version, implemented by Junjie Hu.
 * CreateDate 2014-07-05
 */
#include <iostream>
#include <fstream>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <mutex>
#include <memory>
#include <map>
#include <math.h>

#include "utility.h"
#include "itml.h"


using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using std::cout;
using std::endl;
using std::string;


int main()
{
    ITML itml;
    dml_problem prob;
    read_matrix("dataset/synthetics10.mtx", prob.X);

    read_vector("dataset/synthetics10.truth", prob.y);


    float l,u;

    // compute the l and u
    itml.ComputeDistanceExtreme(prob.X,5,95,l,u);
    cout<<endl<<"l="<<l<<endl;
    cout<<"u="<<u<<endl;

    // get the constraints C
    boost::numeric::ublas::matrix<float> C;
    itml.GetConstraints(prob.y,80,l,u,C);
    //cout<<C<<endl;
    std::ofstream ofc("dataset/synthetics10_C.txt");
    for(int i=0;i<C.size1();i++){
        for(int j=0;j<C.size2();j++){
            ofc<<C(i,j)<<"\t";
        }
        ofc<<endl;
    }
    ofc.close();
    cout<<"generate constraints sucessfully."<<endl;
    return 0;
}
