/*!
 * @mainpage Generate the Crossvalidation Index
 * @version 0.1
 * @brief ITML C++ version, implemented by Junjie Hu.
 * CreateDate 2014-07-05
 */
#include<fstream>
#include<iostream>
#include<algorithm>
#include<string>
#include<sstream>
#include<vector>
#include<boost/numeric/ublas/matrix.hpp>
#include<boost/numeric/ublas/matrix_proxy.hpp>
#include "utility.h"
#include "itml.h"

using namespace std;
int main(int argc, char** argv)
{
    if(argc != 6)
    {
        cout<<"Argument format : "<<argv[0]<<" <Num of Samples> <Num of Folds> <prefix of filename> <filename of matrix> <filename of label>"<<endl;
        return 0;
    }
    ITML itml;
    int number_of_samples = atoi(argv[1]);
    int number_of_folds = atoi(argv[2]);
    int slide = number_of_samples / number_of_folds;
    string prefix(argv[3]);
    string C_matrix(argv[4]);
    string label_vector(argv[5]);

    //read matrix
    using namespace boost::numeric::ublas;
    matrix<float> X;
    read_matrix(C_matrix, X);
    boost::numeric::ublas::vector<float> Y;
    read_vector(label_vector, Y);
    //create the while index list
    std::vector<int> index(number_of_samples);
    for(int i = 0; i < number_of_samples; i++)
    {
        index[i] = i;
    }
    std::random_shuffle(index.begin(), index.end());
    for(int i = 0; i < number_of_folds; i++)
    {
        //generate file name
        stringstream temp;
        temp<<prefix<<i<<"_of_"<<number_of_folds;
        ofstream test_file(temp.str() + "_test.dat");
        ofstream train_file(temp.str() + "_train.dat");
        ofstream test_label_file(temp.str() + "_test_label.dat");
        ofstream train_label_file(temp.str() + "_train_label.dat");
        //test range
        int start_of_test = slide * i;
        for(int j = 0; j < index.size(); j++)
        {
            matrix_row<matrix<float>> row(X, index[j]);
            if(start_of_test <= j && j < start_of_test + slide)
            {
                test_label_file<<Y(index[j])<<endl;
                for(int k = 0; k < row.size(); k++)
                {
                    test_file<<row(k)<<"\t";
                }
                test_file<<endl;
            }
            else
            {
                train_label_file<<Y(index[j])<<endl;
                for(int k = 0; k < row.size(); k++)
                {
                    train_file<<row(k)<<"\t";
                }
                train_file<<endl;
            }
        }
        test_file.close();
        train_file.close();
        test_label_file.close();
        train_label_file.close();


        dml_problem prob;
        read_matrix(temp.str() + "_train.dat", prob.X);
        read_vector(temp.str() + "_train_label.dat", prob.y);


        float l,u;

        // compute the l and u
        itml.ComputeDistanceExtreme(prob.X,5,95,l,u);
        cout<<endl<<"l="<<l<<endl;
        cout<<"u="<<u<<endl;

        // get the constraints C
        boost::numeric::ublas::matrix<float> C;
        itml.GetConstraints(prob.y,800,l,u,C);
        //cout<<C<<endl;
        std::ofstream ofc(temp.str() + "_C.txt");
        for(int i=0;i<C.size1();i++){
            for(int j=0;j<C.size2();j++){
                ofc<<C(i,j)<<"\t";
            }
            ofc<<endl;
        }
        ofc.close();
        cout<<"generate constraints sucessfully."<<endl;


    }
    return 0;
}
