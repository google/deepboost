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
Rcpp::List train_not_exported(DataFrame data,
                              int tree_depth, int num_iter,
                              double beta, double lambda, char loss_type,
                              bool verbose);

#endif  // DEEPBOOST_CONVERTERS_H_
