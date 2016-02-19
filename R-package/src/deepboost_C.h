/*
Written by:
Daniel Marcous, Yotam Sandbank
*/

#ifndef DEEPBOOST_C_H_
#define DEEPBOOST_C_H_

#include "types.h"

// Train a deepboost model on the given examples, using
// numIter iterations (which not necessarily means numIter trees)
void Train(vector<Example>* train_examples, Model* model, int tree_depth,
 int num_iter, double beta, double lambda, char loss_type, bool verbose);


// Classify examples using model
vector<Label> Predict(const vector<Example>& examples, const Model& model);


// Compute the error of model on examples. Also compute the number of trees in
// model and their average size.
void Evaluate(const vector<Example>& examples, const Model& model,
                   float* error, float* avg_tree_size, int* num_trees);

#endif  // DEEPBOOST_C_H_
