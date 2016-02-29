#include <Rcpp.h>
#include "types.h"
#include "deepboost_converters.h"
#include "deepboost_C.h"

using namespace Rcpp;

//’ Trains a deepboost ensemble model
//’
//’ @param data input data.frame as training for model
//’ @param tree_depth maximum depth for a single decision tree in the model
//’ @param num_iter number of iterations = number of trees in ensemble
//’ @param beta regularisation for scores (L1)
//’ @param lambda regularisation for tree depth
//’ @param loss_type - "l" logistic, "e" exponential
//’ @param verbose - print extra data while training TRUE / FALSE
//’ @return a trained Deepboost model
// [[Rcpp::export]]
Rcpp::List Train_R(DataFrame data,
                   int tree_depth, int num_iter,
                   double beta, double lambda, char loss_type,
                   bool verbose) {
  // Train with inner model
  List model_R =
    Train_C(data,
            tree_depth, num_iter, beta, lambda, loss_type, verbose);

  return model_R;
}

//’ Predicts instances labels based on a deepboost model
//’
//’ @param newdata input data.frame to predict labels for
//’ @param model trained Deepboost model
//’ @return a list with labels for all instances in newdata
// [[Rcpp::export]]
Rcpp::List Predict_R(DataFrame newdata,
                        Rcpp::List model) {
  List labels = Predict_C(newdata, model);
  return labels;
}

//’ Evaluates and prints statistics for a deepboost model
//’
//’ @param data input data.frame as training for model
//’ @param model trained Deepboost model
//’ @return a list with model statistics - error, avg_tree_size, num_trees
// [[Rcpp::export]]
Rcpp::List Evaluate_R(DataFrame data,
                         Rcpp::List model) {

  List model_stats = Evaluate_C(data, model);
  return model_stats;
}
