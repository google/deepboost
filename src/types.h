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

#ifndef TYPES_H_
#define TYPES_H_

#include <map>
#include <vector>

using std::map;
using std::pair;
using std::vector;

// Used in many places as the minimum possible difference between two distinct
// numbers. Helps make code stable, tests predictable, etc.
static const float kTolerance = 1e-7;

typedef int Feature;
typedef int Label;
typedef int NodeId;
typedef float Value;
typedef float Weight;

// An example consists of a vector of feature values, a label and a weight.
// Note that this is a dense feature representation; the value of every
// feature is contained in the vector, listed in a canonical order.
typedef struct Example {
  vector<Value> values;
  Label label;
  Weight weight;
} Example;

// A tree node.
typedef struct Node {
  vector<Example> examples;  // Examples at this node.
  Feature split_feature;  // Split feature.
  Value split_value;  // Split value.
  NodeId left_child_id;  // Pointer to left child, if any.
  NodeId right_child_id;  // Pointer to right child, if any.
  Weight positive_weight;  // Total weight of positive examples at this node.
  Weight negative_weight;  // Total weight of negative examples at this node.
  bool leaf;  // Is this node is a leaf?
  int depth;  // Depth of the node in the tree. Root node has depth 0.
} Node;

// A tree is a vector of nodes.
typedef vector<Node> Tree;

// A model is a vector of (weight, tree) pairs, i.e., a weighted combination of
// trees.
typedef vector<pair<Weight, Tree> > Model;

#endif  // TYPES_H_
