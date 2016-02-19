#include <Rcpp.h>
#include "types.h"
#include "deepboost_C.h"

using namespace Rcpp;

class Examples_R {
  public:
    Examples_R(DataFrame data){

      int example_number = data.nrows();
      NumericVector labels   = data["label"];
      NumericVector weights = data["weight"];

      StringVector names = data.names();
      StringVector feature_names;
      for (int i = 0; i < names.length(); ++i) {
        if (names[i] != "label" && names[i] !="weight")
        {
          feature_names.push_back(names[i]);
        }
      }

      for (int i = 0; i < example_number; ++i) {
        Example *example = new Example;

        example -> label = labels[i];
        example -> weight = labels[i];
        example -> values = vector<Value>();

        examples.push_back(*example);
      }

      for (int j = 0; j < feature_names.size(); ++j) {

        NumericVector current_feature = data[String(feature_names[j])];

        for (int i = 0; i < example_number; ++i) {
          examples[i].values.push_back(current_feature[i]);
        }
      }
    }

    // TODO
//   get_examples =
//   set_examples=

  private:
    vector<Example> examples;
};

class Model_R {
  public:
    Model_R(){
      vector<pair<Weight, Tree> > tree_vec;
      model = tree_vec;
    }
  private:
    Model model;
};


RCPP_MODULE(mod_Examples_R) {
  class_<Examples_R>( "Examples_R" )
  .constructor<DataFrame>()
  ;
}

RCPP_MODULE(mod_Model_R) {
  class_<Model_R>( "Model_R" )
  .constructor()
  ;
}

//’ Trains a deepboost ensemble model
//’
//’ @param x input character vector
//’ @return characters in each element of the vector
// [[Rcpp::export]]
Model_R Train_R(DataFrame data,
                   int tree_depth, int num_iter,
                   double beta, double lambda, char loss_type,
                   bool verbose) {
  Examples_R *examples_R = new Examples_R(data);
  Model_R *model_R = new Model_R();

  // get examples + get model - adresses
  //Train();

  return *model_R;
}

//’ Predicts instances labels based on a deepboost model
//’
//’ @param x input character vector
//’ @return characters in each element of the vector
// [[Rcpp::export]]
NumericVector Predict_R(NumericVector x) {

  return x*2;
}

//’ Evaluates and prints statistics for a deepboost model
//’
//’ @param x input character vector
//’ @return characters in each element of the vector
// [[Rcpp::export]]
NumericVector Evaluate_R(NumericVector x) {

  //Evaluate();
  NumericVector y = x*2;
  return y;
}
