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

#ifndef TREE_H_
#define TREE_H_

#include "types.h"

// Initialize some global variables.
void InitializeTreeData(const vector<Example>& examples, float normalizer);

// Return root node for a tree.
Node MakeRootNode(const vector<Example>& examples);

// Return a tree trained on examples.
Tree TrainTree(const vector<Example>& examples);

// Make child nodes using split feature/value and add them to the tree. Also
// update info in the parent node, like child pointers.
void MakeChildNodes(Feature split_feature, Value split_value, Node* parent,
                    Tree* tree);

// Return a map from each value of feature to a pair of weights. The first
// weight in the pair is the total weight of positive examples at node that have
// that value for feature, and the second weight in the pair is the total weight
// of negative examples at node that have that value for feature. This map is
// used to determine the best split feature/value.
map<Value, pair<Weight, Weight>> MakeValueToWeightsMap(const Node& node,
                                                       Feature feature);

// Given a value-to-weights map for a feature (constructed by
// MakeValueToWeightsMap()), determine the best split value for the feature and
// the improvement in the gradient of the objective if we split on that value.
// Note that delta_gradient <= 0 indicates that we should not split on this
// feature.
void BestSplitValue(const map<Value, pair<Weight, Weight>>& value_to_weights,
                    const Node& node, int tree_size, Value* split_value,
                    float* delta_gradient);

// Given an example and a tree, classify the example with the tree.
// NB: This function assumes that if an example has a feature value that is
// _less than or equal to_ a node's split value then the example should be sent
// to the left child, and otherwise sent to the right child.
Label ClassifyExample(const Example& example, const Tree& tree);

// Return the (sub)gradient of the objective with respect to a tree.
float Gradient(float wgtd_error, int tree_size, float alpha, int sign_edge);

// Given a set of examples and a tree, return the weighted error of tree on
// the examples.
float EvaluateTreeWgtd(const vector<Example>& examples, const Tree& tree);

// Return complexity penalty.
float ComplexityPenalty(int tree_size);

#endif  // TREE_H_
