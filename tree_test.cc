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

#include "srm_test.h"
#include "tree.h"

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DECLARE_int32(tree_depth);
DECLARE_double(beta);
DECLARE_double(lambda);

class TreeTest : public SrmTest {
 protected:
  virtual void SetUp() {
    SrmTest::SetUp();
    InitializeTreeData(examples_, examples_.size());
  }
};

TEST_F(TreeTest, TestMakeRootNode) {
  Node root = MakeRootNode(examples_);
  EXPECT_EQ(5, root.examples.size());
  EXPECT_NEAR(0.6, root.positive_weight, kTolerance);
  EXPECT_NEAR(0.4, root.negative_weight, kTolerance);
  EXPECT_TRUE(root.leaf);
  EXPECT_EQ(0, root.depth);
}

TEST_F(TreeTest, TestMakeValueToWeightsMap) {
  Node root = MakeRootNode(examples_);
  map<Value, pair<Weight, Weight>> value_to_weights;

  // Sort by first feature
  value_to_weights = MakeValueToWeightsMap(root, 0);
  vector<Value> values_for_0 = {1.0, 2.0, 3.0, 4.0, 5.0};
  vector<Weight> positive_weights_for_0 = {0.2, 0.0, 0.2, 0.0, 0.2};
  vector<Weight> negative_weights_for_0 = {0.0, 0.2, 0.0, 0.2, 0.0};
  int i = 0;
  for (const pair<Value, pair<Weight, Weight>>& elem : value_to_weights) {
    EXPECT_NEAR(values_for_0[i], elem.first, kTolerance);
    EXPECT_NEAR(positive_weights_for_0[i], elem.second.first, kTolerance);
    EXPECT_NEAR(negative_weights_for_0[i], elem.second.second, kTolerance);
    ++i;
  }

  // Sort by second feature
  value_to_weights = MakeValueToWeightsMap(root, 1);
  vector<Value> values_for_1 = {0.1, 0.2, 0.3, 0.4, 0.5};
  vector<Weight> positive_weights_for_1 = {0.2, 0.0, 0.2, 0.2, 0.0};
  vector<Weight> negative_weights_for_1 = {0.0, 0.2, 0.0, 0.0, 0.2};
  i = 0;
  for (const pair<Value, pair<Weight, Weight>>& elem : value_to_weights) {
    EXPECT_NEAR(values_for_1[i], elem.first, kTolerance);
    EXPECT_NEAR(positive_weights_for_1[i], elem.second.first, kTolerance);
    EXPECT_NEAR(negative_weights_for_1[i], elem.second.second, kTolerance);
    ++i;
  }
}

TEST_F(TreeTest, TestBestSplitValue) {
  Node root = MakeRootNode(examples_);
  map<Value, pair<Weight, Weight>> value_to_weights;
  Value split_value;
  float delta_gradient;

  FLAGS_tree_depth = 1;
  FLAGS_lambda = 0;
  FLAGS_beta = 0;

  // Split on first feature, which is useless.
  value_to_weights = MakeValueToWeightsMap(root, 0);
  BestSplitValue(value_to_weights, root, 1, &split_value, &delta_gradient);
  EXPECT_NEAR(0, delta_gradient, kTolerance);

  // Split on second feature, which is useful.
  value_to_weights = MakeValueToWeightsMap(root, 1);
  BestSplitValue(value_to_weights, root, 1, &split_value, &delta_gradient);
  EXPECT_NEAR(0.2, delta_gradient, kTolerance);
  EXPECT_NEAR(0.4, split_value, kTolerance);

  // Don't split on second feature if complexity penalty is very high.
  FLAGS_lambda = 100;
  value_to_weights = MakeValueToWeightsMap(root, 1);
  BestSplitValue(value_to_weights, root, 1, &split_value, &delta_gradient);
  EXPECT_NEAR(delta_gradient, 0, kTolerance);
}

TEST_F(TreeTest, TestMakeChildNodes) {
  Node root = MakeRootNode(examples_);
  Tree tree;

  tree.push_back(root);
  MakeChildNodes(0, 3.0, &tree[0], &tree);
  EXPECT_EQ(3, tree.size());
  // Check root node
  EXPECT_EQ(5, tree[0].examples.size());
  EXPECT_EQ(0, tree[0].split_feature);
  EXPECT_EQ(1, tree[0].left_child_id);
  EXPECT_EQ(2, tree[0].right_child_id);
  EXPECT_NEAR(3.0, tree[0].split_value, kTolerance);
  EXPECT_NEAR(0.6, tree[0].positive_weight, kTolerance);
  EXPECT_NEAR(0.4, tree[0].negative_weight, kTolerance);
  EXPECT_FALSE(tree[0].leaf);
  EXPECT_EQ(0, tree[0].depth);
  // Check left child node
  EXPECT_EQ(3, tree[1].examples.size());
  EXPECT_NEAR(0.4, tree[1].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[1].negative_weight, kTolerance);
  EXPECT_TRUE(tree[1].leaf);
  EXPECT_EQ(1, tree[1].depth);
  // Check right child node
  EXPECT_EQ(2, tree[2].examples.size());
  EXPECT_NEAR(0.2, tree[2].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[2].negative_weight, kTolerance);
  EXPECT_TRUE(tree[2].leaf);
  EXPECT_EQ(1, tree[2].depth);

  tree.clear();
  tree.push_back(root);
  MakeChildNodes(1, 0.4, &tree[0], &tree);
  EXPECT_EQ(3, tree.size());
  // Check root node
  EXPECT_EQ(5, tree[0].examples.size());
  EXPECT_EQ(1, tree[0].split_feature);
  EXPECT_NEAR(0.4, tree[0].split_value, kTolerance);
  EXPECT_EQ(1, tree[0].left_child_id);
  EXPECT_EQ(2, tree[0].right_child_id);
  EXPECT_NEAR(0.6, tree[0].positive_weight, kTolerance);
  EXPECT_NEAR(0.4, tree[0].negative_weight, kTolerance);
  EXPECT_FALSE(tree[0].leaf);
  EXPECT_EQ(0, tree[0].depth);
  // Check left child node
  EXPECT_EQ(4, tree[1].examples.size());
  EXPECT_NEAR(0.6, tree[1].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[1].negative_weight, kTolerance);
  EXPECT_TRUE(tree[1].leaf);
  EXPECT_EQ(1, tree[1].depth);
  // Check right child node
  EXPECT_EQ(1, tree[2].examples.size());
  EXPECT_NEAR(0.0, tree[2].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[2].negative_weight, kTolerance);
  EXPECT_TRUE(tree[2].leaf);
  EXPECT_EQ(1, tree[2].depth);
}

TEST_F(TreeTest, TestTrainTree) {
  FLAGS_beta = 0;
  FLAGS_lambda = 0;

  FLAGS_tree_depth = 1;
  Tree tree = TrainTree(examples_);
  EXPECT_EQ(3, tree.size());

  FLAGS_tree_depth = 2;
  tree = TrainTree(examples_);
  EXPECT_EQ(5, tree.size());

  // Check all the nodes
  // Node 0
  EXPECT_EQ(1, tree[0].split_feature);
  EXPECT_NEAR(0.4, tree[0].split_value, kTolerance);
  EXPECT_EQ(1, tree[0].left_child_id);
  EXPECT_EQ(2, tree[0].right_child_id);
  EXPECT_NEAR(0.6, tree[0].positive_weight, kTolerance);
  EXPECT_NEAR(0.4, tree[0].negative_weight, kTolerance);
  EXPECT_FALSE(tree[0].leaf);
  EXPECT_EQ(0, tree[0].depth);
  // Node 1
  EXPECT_EQ(2, tree[1].split_feature);
  EXPECT_NEAR(11.0, tree[1].split_value, kTolerance);
  EXPECT_EQ(3, tree[1].left_child_id);
  EXPECT_EQ(4, tree[1].right_child_id);
  EXPECT_NEAR(0.6, tree[1].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[1].negative_weight, kTolerance);
  EXPECT_FALSE(tree[1].leaf);
  EXPECT_EQ(1, tree[1].depth);
  // Node 2
  EXPECT_NEAR(0.0, tree[2].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[2].negative_weight, kTolerance);
  EXPECT_TRUE(tree[2].leaf);
  EXPECT_EQ(1, tree[2].depth);
  // Node 3
  EXPECT_NEAR(0.6, tree[3].positive_weight, kTolerance);
  EXPECT_NEAR(0.0, tree[3].negative_weight, kTolerance);
  EXPECT_TRUE(tree[3].leaf);
  EXPECT_EQ(2, tree[3].depth);
  // Node 4
  EXPECT_NEAR(0.0, tree[4].positive_weight, kTolerance);
  EXPECT_NEAR(0.2, tree[4].negative_weight, kTolerance);
  EXPECT_TRUE(tree[4].leaf);
  EXPECT_EQ(2, tree[4].depth);

  // Very high complexity penalty causes tree to never split
  FLAGS_lambda = 100;
  tree = TrainTree(examples_);
  EXPECT_EQ(1, tree.size());
}

TEST_F(TreeTest, TestComplexityPenalty) {
  FLAGS_beta = 1;
  FLAGS_lambda = 1;

  float complexity_penalty = ComplexityPenalty(10);
  EXPECT_NEAR(2.48087078356, complexity_penalty, kTolerance);
}

TEST_F(TreeTest, GradientTest) {
  FLAGS_beta = 0;
  FLAGS_lambda = 0;

  float gradient = Gradient(0.25, 100, 4, -1);
  EXPECT_NEAR(0.25 - 0.5, gradient, kTolerance);

  FLAGS_beta = 1;
  FLAGS_lambda = 1;

  gradient = Gradient(0.25, 10, 1, 1);
  EXPECT_NEAR(0.25 - 0.5 + ComplexityPenalty(10), gradient, kTolerance);

  gradient = Gradient(0.25, 10, -1, 1);
  EXPECT_NEAR(0.25 - 0.5 - ComplexityPenalty(10), gradient, kTolerance);

  gradient = Gradient(0.25, 10, 0, 1);
  EXPECT_NEAR(0, gradient, kTolerance);

  FLAGS_beta = 0;
  FLAGS_lambda = 0.1;

  gradient = Gradient(0.2, 10, 0, 1);
  EXPECT_NEAR(0.2 - 0.5 - ComplexityPenalty(10), gradient, kTolerance);

  gradient = Gradient(0.2, 10, 0, -1);
  EXPECT_NEAR(0.2 - 0.5 + ComplexityPenalty(10), gradient, kTolerance);
}

TEST_F(TreeTest, TestClassifyExample) {
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_tree_depth = 2;
  Tree tree = TrainTree(examples_);

  EXPECT_EQ(1, ClassifyExample(examples_[0], tree));
  EXPECT_EQ(1, ClassifyExample(examples_[1], tree));
  EXPECT_EQ(1, ClassifyExample(examples_[2], tree));
  EXPECT_EQ(-1, ClassifyExample(examples_[3], tree));
  EXPECT_EQ(-1, ClassifyExample(examples_[4], tree));
}

TEST_F(TreeTest, TestEvaluateTreeWgtd) {
  FLAGS_beta = 0;
  FLAGS_lambda = 0;
  FLAGS_tree_depth = 1;
  Tree tree = TrainTree(examples_);
  EXPECT_NEAR(0.2, EvaluateTreeWgtd(examples_, tree), kTolerance);
}
