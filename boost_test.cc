/*
Copyright 2014 Google Inc. All rights reserved.

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

#include <math.h>

#include "boost.h"
#include "tree.h"  // TODO(usyed): Figure out how not to have to include this.
#include "srm_test.h"

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DECLARE_int32(tree_depth);
DECLARE_double(beta);
DECLARE_double(lambda);
DECLARE_string(loss_type);

class BoostTest : public SrmTest {
 protected:
  virtual void SetUp() {
    SrmTest::SetUp();
    InitializeTreeData(examples_, examples_.size());
  }
};

TEST_F(BoostTest, TestAddTreeToModel) {
  FLAGS_tree_depth = 1;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  // Train a model with a single tree. The tree's weighted error will be 0.2,
  // and it will only get example 3 wrong.
  AddTreeToModel(examples_, &model);
  // Every example is originally weighted equally.
  const float original_wgt = 0.2;
  // alpha = 0.5 * log((1 - error) / error), where error = 0.2.
  float alpha = 0.69314718056;
  // Normalizer is sum of all adjusted weights.
  float normalizer = 4 * original_wgt * exp(-alpha) + original_wgt * exp(alpha);
  // Adjust weights and normalize.
  float correct_wgt = original_wgt * exp(-alpha) / normalizer;
  float incorrect_wgt = original_wgt * exp(alpha) / normalizer;
  EXPECT_NEAR(correct_wgt, examples_[0].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[1].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[2].weight, kTolerance);
  EXPECT_NEAR(incorrect_wgt, examples_[3].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[4].weight, kTolerance);

  // Add another tree to the model. The tree's weighted error will be 0.125, and
  // it will only get example 4 wrong.
  AddTreeToModel(examples_, &model);
  // alpha = 0.5 * log((1 - error) / error), where error = 0.125.
  alpha = 0.97295507452;
  // Normalizer is sum of all adjusted weights.
  normalizer = 3 * correct_wgt * exp(-alpha) + correct_wgt * exp(alpha) +
               incorrect_wgt * exp(-alpha);
  float both_correct_wgt = correct_wgt * exp(-alpha) / normalizer;
  float first_correct_wgt = correct_wgt * exp(alpha) / normalizer;
  float second_correct_wgt = incorrect_wgt * exp(-alpha) / normalizer;
  EXPECT_NEAR(both_correct_wgt, examples_[0].weight, kTolerance);
  EXPECT_NEAR(both_correct_wgt, examples_[1].weight, kTolerance);
  EXPECT_NEAR(both_correct_wgt, examples_[2].weight, kTolerance);
  EXPECT_NEAR(second_correct_wgt, examples_[3].weight, kTolerance);
  EXPECT_NEAR(first_correct_wgt, examples_[4].weight, kTolerance);
}

TEST_F(BoostTest, TestClassifyExampleDepthOne) {
  FLAGS_tree_depth = 1;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  AddTreeToModel(examples_, &model);
  // By the previous test, the first tree gets example 3 wrong and has weight
  // 0.69314718056, and the second tree has weight gets example 4 wrong and has
  // weight 0.97295507452. Since 0.97295507452 > 0.69314718056, the second tree
  // outvotes the first on all examples, so their combination gets example 4
  // wrong.
  EXPECT_EQ(examples_[0].label, ClassifyExample(examples_[0], model));
  EXPECT_EQ(examples_[1].label, ClassifyExample(examples_[1], model));
  EXPECT_EQ(examples_[2].label, ClassifyExample(examples_[2], model));
  EXPECT_EQ(examples_[3].label, ClassifyExample(examples_[3], model));
  EXPECT_EQ(-examples_[4].label, ClassifyExample(examples_[4], model));
}

TEST_F(BoostTest, TestClassifyExampleDepthTwo) {
  FLAGS_tree_depth = 2;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  // Depth 2 trees can classify all examples perfectly.
  EXPECT_EQ(examples_[0].label, ClassifyExample(examples_[0], model));
  EXPECT_EQ(examples_[1].label, ClassifyExample(examples_[1], model));
  EXPECT_EQ(examples_[2].label, ClassifyExample(examples_[2], model));
  EXPECT_EQ(examples_[3].label, ClassifyExample(examples_[3], model));
  EXPECT_EQ(examples_[4].label, ClassifyExample(examples_[4], model));
  const float alpha = model[0].first;
  // Won't actually add trees, will just increase weight on current tree.
  for (int i = 0; i < 99; ++i) {
    AddTreeToModel(examples_, &model);
  }
  EXPECT_EQ(1, model.size());
  EXPECT_NEAR(alpha, model[0].first / 100, kTolerance * 100);
  EXPECT_EQ(examples_[0].label, ClassifyExample(examples_[0], model));
  EXPECT_EQ(examples_[1].label, ClassifyExample(examples_[1], model));
  EXPECT_EQ(examples_[2].label, ClassifyExample(examples_[2], model));
  EXPECT_EQ(examples_[3].label, ClassifyExample(examples_[3], model));
  EXPECT_EQ(examples_[4].label, ClassifyExample(examples_[4], model));
}

TEST_F(BoostTest, TestClassifyExampleEmptyModel) {
  Model model;
  // Empty model classifies every example as positive
  EXPECT_EQ(1, ClassifyExample(examples_[0], model));
  EXPECT_EQ(1, ClassifyExample(examples_[1], model));
  EXPECT_EQ(1, ClassifyExample(examples_[2], model));
  EXPECT_EQ(1, ClassifyExample(examples_[3], model));
  EXPECT_EQ(1, ClassifyExample(examples_[4], model));
}

TEST_F(BoostTest, TestEvaluateModelDepthOne) {
  FLAGS_tree_depth = 1;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  AddTreeToModel(examples_, &model);
  float error, avg_tree_size;
  int num_trees;
  EvaluateModel(examples_, model, &error, &avg_tree_size, &num_trees);
  EXPECT_NEAR(0.2, error, kTolerance);
  EXPECT_EQ(2, num_trees);
  EXPECT_NEAR(3, avg_tree_size, kTolerance);
}

TEST_F(BoostTest, TestEvaluateModelDepthTwo) {
  FLAGS_tree_depth = 2;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  float error, avg_tree_size;
  int num_trees;
  EvaluateModel(examples_, model, &error, &avg_tree_size, &num_trees);
  EXPECT_NEAR(0.0, error, kTolerance);
  EXPECT_EQ(1, num_trees);
  EXPECT_NEAR(5, avg_tree_size, kTolerance);
}

TEST_F(BoostTest, ComputeEtaTest) {
  FLAGS_beta = 1;
  FLAGS_lambda = 1;
  float eta = ComputeEta(1, 10, 1);
  EXPECT_NEAR(-1, eta, kTolerance);

  FLAGS_beta = 1;
  FLAGS_lambda = 0;
  eta = ComputeEta(0.1, 5, 2);
  float ratio = ComplexityPenalty(5) / 0.1;
  EXPECT_NEAR(log(-ratio + sqrt(ratio * ratio + (0.9 / 0.1))), eta, kTolerance);

  FLAGS_beta = 0;
  FLAGS_lambda = 1;
  eta = ComputeEta(0.75, 10, -10);
  ratio = ComplexityPenalty(10) / 0.75;
  EXPECT_NEAR(log(ratio + sqrt(ratio * ratio + (0.25 / 0.75))),
              eta, kTolerance);
}

TEST_F(BoostTest, TestAddTreeToModelLargeBeta) {
  // Large beta penalty means two trees with zero weight and depth 0.
  FLAGS_tree_depth = 1;
  FLAGS_beta = 100;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  AddTreeToModel(examples_, &model);
  EXPECT_EQ(2, model.size());
  EXPECT_LT(model[0].first, kTolerance);
  EXPECT_LT(model[1].first, kTolerance);
  EXPECT_EQ(1, model[0].second.size());
  EXPECT_EQ(1, model[1].second.size());
}

TEST_F(BoostTest, TestAddTreeToModelLargeLambda) {
  // Large lambda penalty means two trees with zero weight and depth 0.
  FLAGS_tree_depth = 1;
  FLAGS_beta = 0;
  FLAGS_lambda = 100;
  FLAGS_loss_type = "exponential";
  Model model;
  AddTreeToModel(examples_, &model);
  AddTreeToModel(examples_, &model);
  EXPECT_EQ(2, model.size());
  EXPECT_LT(model[0].first, kTolerance);
  EXPECT_LT(model[1].first, kTolerance);
  EXPECT_EQ(1, model[0].second.size());
  EXPECT_EQ(1, model[1].second.size());
}

TEST_F(BoostTest, TestAddTreeToModelLogisticLoss) {
  FLAGS_tree_depth = 1;
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_loss_type = "logistic";
  Model model;
  // Train a model with a single tree. The tree's weighted error will be 0.2,
  // and it will only get example 3 wrong.
  AddTreeToModel(examples_, &model);
  // alpha1 = 0.5 * log((1 - error) / error), where error = 0.2.
  float alpha1 = 0.69314718056;
  // Normalizer is sum of all adjusted weights.
  float normalizer =
      4 * (1 / (1 + exp(alpha1 - 1))) + (1 / (1 + exp(-alpha1 - 1)));
  // Adjust weights and normalize.
  float correct_wgt = (1 / (1 + exp(alpha1 - 1))) / normalizer;
  float incorrect_wgt = (1 / (1 + exp(-alpha1 - 1))) / normalizer;
  EXPECT_NEAR(correct_wgt, examples_[0].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[1].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[2].weight, kTolerance);
  EXPECT_NEAR(incorrect_wgt, examples_[3].weight, kTolerance);
  EXPECT_NEAR(correct_wgt, examples_[4].weight, kTolerance);

  // Add another tree to the model. The tree's weighted error will be
  // 0.182946235, and it will only get example 4 wrong.
  AddTreeToModel(examples_, &model);
  // alpha2 = 0.5 * log((1 - error) / error), where error = 0.182946235.
  float alpha2 = 0.7482563445;
  // Normalizer is sum of all adjusted weights.
  normalizer = 3 * (1 / (1 + exp(alpha1 + alpha2 - 1))) +
               (1 / (1 + exp(alpha1 - alpha2 - 1))) +
               (1 / (1 + exp(-alpha1 + alpha2 - 1)));
  float both_correct_wgt = (1 / (1 + exp(alpha1 + alpha2 - 1))) / normalizer;
  float first_correct_wgt = (1 / (1 + exp(alpha1 - alpha2 - 1))) / normalizer;
  float second_correct_wgt = (1 / (1 + exp(-alpha1 + alpha2 - 1))) / normalizer;
  EXPECT_NEAR(both_correct_wgt, examples_[0].weight, kTolerance);
  EXPECT_NEAR(both_correct_wgt, examples_[1].weight, kTolerance);
  EXPECT_NEAR(both_correct_wgt, examples_[2].weight, kTolerance);
  EXPECT_NEAR(second_correct_wgt, examples_[3].weight, kTolerance);
  EXPECT_NEAR(first_correct_wgt, examples_[4].weight, kTolerance);
}
