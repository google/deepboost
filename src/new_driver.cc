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
#include "../R-package/src/deepboost_C.h"
#include "io.h"
#include "types.h"

DECLARE_string(dataset);
DECLARE_string(data_filename);
DECLARE_int32(num_folds);
DECLARE_int32(fold_to_cv);
DECLARE_int32(fold_to_test);
DECLARE_double(noise_prob);
DEFINE_int32(new_seed, -1,
             "Seed for random number generator. Required: seed >= 0.");



int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_data_filename = "../testdata/breast-cancer-wisconsin.data";
  FLAGS_dataset = "breastcancer";
  FLAGS_num_folds = 5;
  FLAGS_fold_to_cv = 1;
  FLAGS_fold_to_test = 1;
  FLAGS_new_seed = 1;
  FLAGS_noise_prob = 0.1;

  SetSeed(FLAGS_new_seed);

  vector<Example> train_examples, cv_examples, test_examples;
  ReadData(&train_examples, &cv_examples, &test_examples);
  Model model;
  int tree_depth = 3;
  int num_iter = 10;
  float beta = 1;
  float lambda = 0.5;
  char loss_type = 'e';
  bool verbose = 1;
  Train(&train_examples, &model, tree_depth, num_iter, beta, lambda, loss_type, verbose);
  float error, avg_tree_size;
  int num_trees;
  Evaluate(cv_examples, model, &error, &avg_tree_size, &num_trees);
  printf("Error: %g, avg tree size: %g, num trees: %d\n", error, avg_tree_size, num_trees);
}
