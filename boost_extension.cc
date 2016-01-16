/*
Written by:
Daniel Marcous, Yotam Sandbank
*/

#include "boost_extension.h"
#include "boost.h"

#include <float.h>
#include <math.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "tree.h"

DECLARE_int32(tree_depth);
DECLARE_double(beta);
DECLARE_double(lambda);
DECLARE_string(loss_type);

// Train a deepboost model on the given examples, using
// numIter iterations (which not necessarily means numIter trees) 
void Train(vector<Example>& train_examples, vector<Example>& cv_examples,
 vector<Example>& test_examples, Model* model, int tree_depth,
 int num_iter, float beta, float lambda, char loss_type, bool verboose) {
	
	// Set flags
	flags_tree_depth = tree_depth;
	flags_beta = beta;
	flags_lambda = lambda;
	if (loss_type == 'e') {
		flags_loss_type = 'exponential';
	} else if (loss_type == 'l') {
		flags_loss_type = 'logistic';
	}
	// Train the model
	for (int i = 1; i <= num_iter; ++i) {
		AddTreeToModel(train_examples, &model);
		if (verboose) {
			float cv_error, test_error, avg_tree_size;
			int num_trees;
			EvaluateModel(cv_examples, model, &cv_error, &avg_tree_size,
						  &num_trees);
			EvaluateModel(test_examples, model, &test_error, &avg_tree_size,
						  &num_trees);
			printf("Iteration: %d, test error: %g, cv error: %g, "
				   "avg tree size: %g, num trees: %d\n",
				   iter, test_error, cv_error, avg_tree_size, num_trees);
		}
	}
}


// Classify examples using model
void Predict(const vector<Example>& examples, const Model& model, vector<Label>& labels){
	for (unsigned i=0; i<examples.size(); i++){	
		labels[i] = ClassifyExample(examples[i], model);
    }
}


// Compute the error of model on examples. Also compute the number of trees in
// model and their average size.
void EvaluateModel(const vector<Example>& examples, const Model& model,
                   float* error, float* avg_tree_size, int* num_trees){
	EvaluateModel(examples, model, error, avg_tree_size, num_trees);
}
