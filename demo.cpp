#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <memory>
#include <map>
#include <math.h>
#include <ctime>
#include <fstream>

#include "utility.h"
#include "itml.h"

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using std::cout;
using std::endl;
using std::map;
using std::string;

ITML itml;

void run_itml(string xtrain, string ytrain, string xtest, string ytest, string Cfile,
              double& run_time,double& Accuracy)
{

    dml_problem train_prob, test_prob;
    read_matrix(xtrain, train_prob.X);
    read_vector(ytrain, train_prob.y);

    read_matrix(xtest, test_prob.X);
    read_vector(ytest, test_prob.y);

    int d = train_prob.X.size2();

    boost::numeric::ublas::matrix<float> A_0(d,d,0);
    for(int i=0;i<d;i++)
        A_0(i,i)=1;

    // load C in the txt file
    boost::numeric::ublas::matrix<float> C;
    read_matrix(Cfile,C);

    itml_param param;
    cout<<param.gamma<<endl;
    cout<<param.max_iters<<endl;
    cout<<param.thresh<<endl;
    boost::numeric::ublas::matrix<float> A;

    clock_t start, end;
    start = clock();
    // run ITML
    itml.itml_alg(C,train_prob.X,A_0,param,A);
    //cout<<"A="<<A<<endl;

    // use KNN to predict
    vector<float> pred_y;
    itml.KNN(train_prob.y,train_prob.X,A,4,test_prob.X,pred_y);
    end = clock();

    //cout<<"predict"<<pred_y<<endl;

    int cor = 0;
    for(int i=0;i<pred_y.size();i++){
        if(pred_y(i) == test_prob.y(i))
            cor ++;
    }
    Accuracy = 1.0*cor/pred_y.size();
    cout<<"Accuracy:"<<Accuracy<<endl;
    run_time = (double)(end-start)/CLOCKS_PER_SEC;
    cout<<"Run time:"<<run_time<<endl;
}

void mean_std(double* x,int n, double & m, double sig){
    m=0;
    for(int i=0;i<n;i++)
        m += x[i];
    m/=n;

    sig = 0;
    for(int i=0;i<n;i++){
        sig += (x[i]-m)*(x[i]-m);
    }
    sig = std::sqrt(sig/n);
}

int main()
{
    const int fold = 10;
    double* Accuracy = Malloc(double,fold);
    double* Run_time = Malloc(double,fold);
    double mean_acc,std_acc,mean_time,std_time;
    string prefix = "dataset/iris";

    for(int i=0;i<fold;i++){
        // You can change the training and testing set for cross validation
        run_itml(prefix+".mtx",prefix+".truth",prefix+".mtx",prefix+".truth",prefix+"_C.txt",
                 Accuracy[i],Run_time[i]);
    }

    mean_std(Accuracy,fold,mean_acc,std_acc);
    mean_std(Run_time,fold,mean_time,std_time);

    // record
    std::ofstream ofa(prefix+"accuracy_result.txt");
    std::ofstream oft(prefix+"time_result.txt");
    for(int i=0;i<fold;i++){
        ofa<<Accuracy[i]<<std::endl;
        oft<<Run_time[i]<<std::endl;
    }
    ofa<<"mean:"<<mean_acc<<endl<<"std:"<<std_acc<<endl;
    oft<<"mean:"<<mean_time<<endl<<"std:"<<std_time<<endl;


    ofa.close();
    oft.close();

    return 0;
}

