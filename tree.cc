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

#include <math.h>

#include "tree.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_double(beta, -1.0, "beta parameter for gradient.");
DEFINE_double(lambda, -1.0, "lambda parameter for gradient.");
DEFINE_int32(tree_depth, -1,
             "Maximum depth of each decision tree. The root node has depth 0. "
             "Required: tree_depth >= 0.");

// TODO(usyed): Global variables are bad style.
static int num_features;
static int num_examples;
static float the_normalizer;
static bool is_initialized = false;

void InitializeTreeData(const vector<Example>& examples, float normalizer) {
  CHECK_GE(examples.size(), 1);
  num_examples = examples.size();
  num_features = examples[0].values.size();
  the_normalizer = normalizer;
  is_initialized = true;
}

Node MakeRootNode(const vector<Example>& examples) {
  Node root;
  root.examples = examples;
  root.positive_weight = root.negative_weight = 0;
  for (const Example& example : examples) {
    if (example.label == 1) {
      root.positive_weight += example.weight;
    } else {  // label == -1
      root.negative_weight += example.weight;
    }
  }
  root.leaf = true;
  root.depth = 0;
  return root;
}

map<Value, pair<Weight, Weight>> MakeValueToWeightsMap(const Node& node,
                                                       Feature feature) {
  map<Value, pair<Weight, Weight>> value_to_weights;
  for (const Example& example : node.examples) {
    if (example.label == 1) {
      value_to_weights[example.values[feature]].first += example.weight;
    } else {  // label = -1
      value_to_weights[example.values[feature]].second += example.weight;
    }
  }
  return value_to_weights;
}

void BestSplitValue(const map<Value, pair<Weight, Weight>>& value_to_weights,
                    const Node& node, int tree_size, Value* split_value,
                    float* delta_gradient) {
  *delta_gradient = 0;
  Weight left_positive_weight = 0, left_negative_weight = 0,
         right_positive_weight = node.positive_weight,
         right_negative_weight = node.negative_weight;
  float old_error = fmin(left_positive_weight + right_positive_weight,
                         left_negative_weight + right_negative_weight);
  float old_gradient = Gradient(old_error, tree_size, 0, -1);
  for (const pair<Value, pair<Weight, Weight>>& elem : value_to_weights) {
    left_positive_weight += elem.second.first;
    right_positive_weight -= elem.second.first;
    left_negative_weight += elem.second.second;
    right_negative_weight -= elem.second.second;
    float new_error = fmin(left_positive_weight, left_negative_weight) +
                      fmin(right_positive_weight, right_negative_weight);
    float new_gradient = Gradient(new_error, tree_size + 2, 0, -1);
    if (fabs(new_gradient) - fabs(old_gradient) >
        *delta_gradient + kTolerance) {
      *delta_gradient = fabs(new_gradient) - fabs(old_gradient);
      *split_value = elem.first;
    }
  }
}

void MakeChildNodes(Feature split_feature, Value split_value, Node* parent,
                    Tree* tree) {
  parent->split_feature = split_feature;
  parent->split_value = split_value;
  parent->leaf = false;
  Node left_child, right_child;
  left_child.depth = right_child.depth = parent->depth + 1;
  left_child.leaf = right_child.leaf = true;
  left_child.positive_weight = left_child.negative_weight =
      right_child.positive_weight = right_child.negative_weight = 0;
  for (const Example& example : parent->examples) {
    Node* child;
    if (example.values[split_feature] <= split_value) {
      child = &left_child;
    } else {
      child = &right_child;
    }
    // TODO(usyed): Moving examples around is inefficient.
    child->examples.push_back(example);
    if (example.label == 1) {
      child->positive_weight += example.weight;
    } else {  // label == -1
      child->negative_weight += example.weight;
    }
  }
  parent->left_child_id = tree->size();
  parent->right_child_id = tree->size() + 1;
  tree->push_back(left_child);
  tree->push_back(right_child);
}

Tree TrainTree(const vector<Example>& examples) {
  CHECK(is_initialized);
  Tree tree;
  tree.push_back(MakeRootNode(examples));
  NodeId node_id = 0;
  while (node_id < tree.size()) {
    Node& node = tree[node_id];  // TODO(usyed): Too bad this can't be const.
    Feature best_split_feature;
    Value best_split_value;
    float best_delta_gradient = 0;
    for (Feature split_feature = 0; split_feature < num_features;
         ++split_feature) {
      const map<Value, pair<Weight, Weight>> value_to_weights =
          MakeValueToWeightsMap(node, split_feature);
      Value split_value;
      float delta_gradient;
      BestSplitValue(value_to_weights, node, tree.size(), &split_value,
                     &delta_gradient);
      if (delta_gradient > best_delta_gradient + kTolerance) {
        best_delta_gradient = delta_gradient;
        best_split_feature = split_feature;
        best_split_value = split_value;
      }
    }
    if (node.depth < FLAGS_tree_depth && best_delta_gradient > kTolerance) {
      MakeChildNodes(best_split_feature, best_split_value, &node, &tree);
    }
    ++node_id;
  }
  return tree;
}

Label ClassifyExample(const Example& example, const Tree& tree) {
  CHECK_GE(tree.size(), 1);
  const Node* node = &tree[0];
  while (node->leaf == false) {
    if (example.values[node->split_feature] <= node->split_value) {
      node = &tree[node->left_child_id];
    } else {
      node = &tree[node->right_child_id];
    }
  }
  if (node->positive_weight >= node->negative_weight) {
    return 1;
  } else {
    return -1;
  }
}

float Gradient(float wgtd_error, int tree_size, float alpha, int sign_edge) {
  // TODO(usyed): Can we make some mild assumptions and get rid of sign_edge?
  const float complexity_penalty = ComplexityPenalty(tree_size);
  const float edge = wgtd_error - 0.5;
  const int sign_alpha = (alpha >= 0) ? 1 : -1;
  if (fabs(alpha) > kTolerance) {
    return edge + sign_alpha * complexity_penalty;
  } else if (fabs(edge) <= complexity_penalty) {
    return 0;
  } else {
    return edge - sign_edge * complexity_penalty;
  }
}

float EvaluateTreeWgtd(const vector<Example>& examples, const Tree& tree) {
  float wgtd_error = 0;
  for (const Example& example : examples) {
    if (ClassifyExample(example, tree) != example.label) {
      wgtd_error += example.weight;
    }
  }
  return wgtd_error;
}

float ComplexityPenalty(int tree_size) {
  CHECK(is_initialized);
  float rademacher =
      sqrt(((2 * tree_size + 1) * (log(num_features + 2) / log(2)) *
            log(num_examples)) /
           num_examples);
  return ((FLAGS_lambda * rademacher + FLAGS_beta) * num_examples) /
         (2 * the_normalizer);
}
