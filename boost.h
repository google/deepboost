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

#ifndef BOOST_H_
#define BOOST_H_

#include "types.h"

// Either add a new tree to model or update the weight of an existing tree in
// model. The tree and weight are selected via approximate coordinate descent on
// the objective, where the "approximate" indicates that we do not search all
// trees but instead grow trees greedily.
void AddTreeToModel(vector<Example>& examples, Model* model);

// Classify example with model.
Label ClassifyExample(const Example& example, const Model& model);

// Compute the error of model on examples. Also compute the number of trees in
// model and their average size.
void EvaluateModel(const vector<Example>& examples, const Model& model,
                   float* error, float* avg_tree_size, int* num_trees);

// Return the optimal weight to add to a tree that will maximally decrease the
// objective.
float ComputeEta(float wgtd_error, float tree_size, float alpha);

#endif  // BOOST_H_
