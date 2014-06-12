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

#ifndef SRM_TEST_H_
#define SRM_TEST_H_

#include "types.h"
#include "gtest/gtest.h"

class SrmTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // Three positive examples, two negative examples, three features. Best
    // split for feature 0 is useless, and for the other features divides
    // examples into the ratios +3/-1 (left) and +0/-1 (right). Two splits (on
    // features 1 and 2) perfectly classify all examples.
    // TODO(usyed): For more interesting tests, weights should be different from
    // uniform.
    Example examples_arr[5];
    examples_arr[0].values = {1.0, 0.1, 11.0};
    examples_arr[0].label = 1;
    examples_arr[0].weight = 0.2;
    examples_arr[1].values = {3.0, 0.3, 11.0};
    examples_arr[1].label = 1;
    examples_arr[1].weight = 0.2;
    examples_arr[2].values = {5.0, 0.4, 11.0};
    examples_arr[2].label = 1;
    examples_arr[2].weight = 0.2;
    examples_arr[3].values = {2.0,  0.2, 22.0};
    examples_arr[3].label = -1;
    examples_arr[3].weight = 0.2;
    examples_arr[4].values = {4.0, 0.5, 11.0};
    examples_arr[4].label = -1;
    examples_arr[4].weight = 0.2;
    examples_.assign(examples_arr, examples_arr + 5);
  }

  vector<Example> examples_;
};

#endif  // SRM_TEST_H_
