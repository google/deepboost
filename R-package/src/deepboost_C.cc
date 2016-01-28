/*
Written by:
Daniel Marcous, Yotam Sandbank
*/

#include "deepboost_C.h"
#include "../../src/boost.h"

#include <float.h>
#include <math.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "../../src/tree.h"

DECLARE_int32(tree_depth);
DECLARE_double(beta);
DECLARE_double(lambda);
DECLARE_string(loss_type);

// Train a deepboost model on the given examples, using
// numIter iterations (which not necessarily means numIter trees)
void Train(vector<Example>& train_examples, Model* model, int tree_depth,
 int num_iter, float beta, float lambda, char loss_type, bool verbose) {

	// Set flags
	FLAGS_tree_depth = tree_depth;
  FLAGS_beta = beta;
  FLAGS_lambda = lambda;
	if (loss_type == 'e') {
	  FLAGS_loss_type = 'exponential';
	} else if (loss_type == 'l') {
	  FLAGS_loss_type = 'logistic';
	}
	// Train the model
	for (int iter = 1; iter <= num_iter; ++iter) {
		AddTreeToModel(train_examples, model);
		if (verbose) {
			float error, avg_tree_size;
			int num_trees;
			EvaluateModel(train_examples, model, &error, &avg_tree_size,
						  &num_trees);
			printf("Iteration: %d, error: %g, "
				   "avg tree size: %g, num trees: %d\n",
				   iter, error, avg_tree_size, num_trees);
		}
	}
}


// Classify examples using model
vector<Label> Predict(const vector<Example>& examples, const Model& model){
	//TODO::initiate labels
	vector<Label> labels;
    labels.resize(examples.size(), 0);
	for (unsigned i=0; i<examples.size(); i++){
		labels[i] = ClassifyExample(examples[i], model);
    }
	return labels;
}


// Compute the error of model on examples. Also compute the number of trees in
// model and their average size.
void Evaluate(const vector<Example>& examples, const Model& model,
                   float* error, float* avg_tree_size, int* num_trees){
	EvaluateModel(examples, model, error, avg_tree_size, num_trees);
}
