/*
 Written by:
Daniel Marcous, Yotam Sandbank
*/

#ifndef DEEPBOOST_CONVERTERS_H_
#define DEEPBOOST_CONVERTERS_H_

#include <Rcpp.h>
#include "types.h"

using namespace Rcpp;

// Train a deepboost model on the given examples, using
// numIter iterations (which not necessarily means numIter trees)
Rcpp::List Train_C(DataFrame data,
                              int tree_depth, int num_iter,
                              double beta, double lambda, char loss_type,
                              bool verbose);

// Compute the error of model on examples. Also compute the number of trees in
// model and their average size.
// Returns a list with the model's error, avg_tree_size, num_trees
Rcpp::List Evaluate_C(DataFrame data, Rcpp::List model);

// Classify examples using model
Rcpp::List Predict_C(DataFrame data, Rcpp::List model);

#endif  // DEEPBOOST_CONVERTERS_H_
