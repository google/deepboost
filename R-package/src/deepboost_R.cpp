#include <Rcpp.h>
#include "types.h"
#include "deepboost_converters.hpp"
#include "deepboost_C.h"

using namespace Rcpp;

//’ Trains a deepboost ensemble model
//’
//’ @param x input character vector
//’ @return characters in each element of the vector
// [[Rcpp::export]]
Rcpp::List Train_R(DataFrame data,
                   int tree_depth, int num_iter,
                   double beta, double lambda, char loss_type,
                   bool verbose) {
  // Train with inner model
  List model_R =
    train_not_exported(data,
                   tree_depth, num_iter, beta, lambda, loss_type, verbose);

  return model_R;
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
