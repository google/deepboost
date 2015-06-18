/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "boost.h"
#include "io.h"
#include "types.h"

DECLARE_int32(tree_depth);
DECLARE_string(data_set);
DECLARE_string(data_filename);
DECLARE_int32(num_folds);
DECLARE_int32(fold_to_cv);
DECLARE_int32(fold_to_test);
DECLARE_double(beta);
DECLARE_double(lambda);
DECLARE_string(loss_type);
DEFINE_int32(num_iter, -1,
             "Number of boosting iterations. Required: num_iter >= 1.");
DEFINE_int32(seed, -1,
             "Seed for random number generator. Required: seed >= 0.");

void ValidateFlags() {
  CHECK_GE(FLAGS_tree_depth, 0);
  CHECK_GE(FLAGS_num_iter, 1);
  CHECK(!FLAGS_data_filename.empty());
  CHECK(FLAGS_data_set == "breastcancer" || FLAGS_data_set == "ion" ||
        FLAGS_data_set == "ocr17" || FLAGS_data_set == "ocr49" ||
        FLAGS_data_set == "splice" || FLAGS_data_set == "german" ||
        FLAGS_data_set == "ocr17-princeton" ||
        FLAGS_data_set == "ocr49-princeton" || FLAGS_data_set == "mnist17" ||
        FLAGS_data_set == "mnist49" || FLAGS_data_set == "pima");
  CHECK_GE(FLAGS_num_folds, 3);
  CHECK_GE(FLAGS_fold_to_cv, 0);
  CHECK_GE(FLAGS_fold_to_test, 0);
  CHECK_LE(FLAGS_fold_to_cv, FLAGS_num_folds - 1);
  CHECK_LE(FLAGS_fold_to_test, FLAGS_num_folds - 1);
  CHECK_GE(FLAGS_seed, 0);
  CHECK_GE(FLAGS_beta, 0.0);
  CHECK_GE(FLAGS_lambda, 0.0);
  CHECK(FLAGS_loss_type == "exponential" || FLAGS_loss_type == "logistic");
}

int main(int argc, char** argv) {
  ValidateFlags();

  srand(FLAGS_seed);

  vector<Example> train_examples, cv_examples, test_examples;
  ReadData(&train_examples, &cv_examples, &test_examples);

  Model model;
  for (int iter = 1; iter <= FLAGS_num_iter; ++iter) {
    AddTreeToModel(train_examples, &model);
    // TODO(usyed): Evaluating every iteration might be very expensive. Add an
    // option to evaluate every K iterations, where K is a command-line
    // parameter.
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
