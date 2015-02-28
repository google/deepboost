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

#ifndef IO_H_
#define IO_H_

#include <string>

#include "types.h"

using std::string;

// Split text into the tokens vector, using sep as a delimiter. Consecutive
// delimiters are ignored.
void SplitString(const string &text, char sep, vector<string>* tokens);

// The following functions each parse one line of a data set.

bool ParseLineBreastCancer(const string& line, Example* example);

bool ParseLineIon(const string& line, Example* example);

bool ParseLineGerman(const string& line, Example* example);

bool ParseLineOcr49(const string& line, Example* example);

bool ParseLineOcr17(const string& line, Example* example);

bool ParseLineOcr49Princeton(const string& line, Example* example);

bool ParseLineOcr17Princeton(const string& line, Example* example);

bool ParseLineMnist49(const string& line, Example* example);

bool ParseLineMnist17(const string& line, Example* example);

bool ParseLinePima(const string& line, Example* example);

// Read data set into training set, cross-validation set and test set.
void ReadData(vector<Example>* train_examples,
              vector<Example>* cv_examples,
              vector<Example>* test_examples);

#endif  // IO_H_
