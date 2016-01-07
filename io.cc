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

#include "io.h"

#include <algorithm>
#include <fstream>
#include <random>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(data_set, "",
              "Name of data set. Required: One of breastcancer, ion, ocr17, "
              "ocr49");
DEFINE_string(data_filename, "",
              "Filename containing data. Required: data_filename not empty.");
DEFINE_int32(num_folds, -1,
             "(num_folds - 2)/num_folds of data used for training, 1/num_folds "
             "of data used for cross-validation, 1/num_folds of data used for "
             "testing. Required: num_folds >= 3.");
DEFINE_int32(fold_to_cv, -1,
             "Zero-indexed fold used for cross-validation. Required: "
             "0 <= fold_to_cv <= num_folds - 1.");
DEFINE_int32(fold_to_test, -1,
             "Zero-indexed fold used for testing. Required: 0 <= fold_to_test "
             "<= num_folds - 1.");
DEFINE_double(noise_prob, 0,
              "Noise probability. Required: 0 <= noise_prob <= 1.");

static std::mt19937 rng;

void SetSeed(uint_fast32_t seed) { rng.seed(seed); }

void SplitString(const string &text, char sep, vector<string>* tokens) {
  int start = 0, end = 0;
  string token;
  while ((end = text.find(sep, start)) != string::npos) {
    token = text.substr(start, end - start);
    if (!token.empty()) {
      tokens->push_back(token);
    }
    start = end + 1;
  }
  token = text.substr(start);
  if (!token.empty()) {
    tokens->push_back(token);
  }
}

bool ParseLineBreastCancer(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == 0) {
      continue;  // Skip ID
    } else if (i == values.size() - 1) {
      if (values[i] == "2") {  // Benign
        example->label = -1;
      } else if (values[i] == "4") {  // Malignant
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else if (values[i] == "?") {
      return false;
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineIon(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "b") {  // Bad
        example->label = -1;
      } else if (values[i] == "g") {  // Good
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineGerman(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Good
        example->label = -1;
      } else if (values[i] == "2") {  // Bad
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr17(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Digit 1
        example->label = -1;
      } else if (values[i] == "7") {  // Digit 7
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr49(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "4") {  // Digit 4
        example->label = -1;
      } else if (values[i] == "9") {  // Digit 9
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr17Princeton(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Digit 1
        example->label = -1;
      } else if (values[i] == "7") {  // Digit 7
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr49Princeton(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "4") {  // Digit 4
        example->label = -1;
      } else if (values[i] == "9") {  // Digit 9
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineMnist17(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == 0) {
      if (values[i] == "1") {  // Digit 1
        example->label = -1;
      } else if (values[i] == "7") {  // Digit 7
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineMnist49(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == 0) {
      if (values[i] == "4") {  // Digit 4
        example->label = -1;
      } else if (values[i] == "9") {  // Digit 9
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLinePima(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "0") {
        example->label = -1;
      } else if (values[i] == "1") {
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}


void ReadData(vector<Example>* train_examples,
              vector<Example>* cv_examples,
              vector<Example>* test_examples) {
  train_examples->clear();
  cv_examples->clear();
  test_examples->clear();
  vector<Example> examples;
  std::ifstream file(FLAGS_data_filename);
  CHECK(file.is_open());
  string line;
  while (!std::getline(file, line).eof()) {
    Example example;
    bool keep_example;
    if (FLAGS_data_set == "breastcancer") {
      keep_example = ParseLineBreastCancer(line, &example);
    } else if (FLAGS_data_set == "ion") {
      keep_example = ParseLineIon(line, &example);
    } else if (FLAGS_data_set == "german") {
      keep_example = ParseLineGerman(line, &example);
    } else if (FLAGS_data_set == "ocr17") {
      keep_example = ParseLineOcr17(line, &example);
    } else if (FLAGS_data_set == "ocr49") {
      keep_example = ParseLineOcr49(line, &example);
    } else if (FLAGS_data_set == "ocr17-princeton") {
      keep_example = ParseLineOcr17Princeton(line, &example);
    } else if (FLAGS_data_set == "ocr49-princeton") {
      keep_example = ParseLineOcr49Princeton(line, &example);
    } else if (FLAGS_data_set == "mnist17") {
      keep_example = ParseLineMnist17(line, &example);
    } else if (FLAGS_data_set == "mnist49") {
      keep_example = ParseLineMnist49(line, &example);
    } else if (FLAGS_data_set == "pima") {
      keep_example = ParseLinePima(line, &example);
    } else {
      LOG(FATAL) << "Unknown data set: " << FLAGS_data_set;
    }
    if (keep_example) examples.push_back(example);
  }
  std::shuffle(examples.begin(), examples.end(), rng);
  std::uniform_real_distribution<double> dist;
  int fold = 0;
  // TODO(usyed): Two loops is inefficient
  for (Example& example : examples) {
    double r = dist(rng);
    if (r < FLAGS_noise_prob) {
      example.label = -example.label;
    }
    if (fold == FLAGS_fold_to_test) {
      test_examples->push_back(example);
    } else if (fold == FLAGS_fold_to_cv) {
      cv_examples->push_back(example);
    } else {
      train_examples->push_back(example);
    }
    ++fold;
    if (fold == FLAGS_num_folds) fold = 0;
  }
  const float initial_wgt = 1.0 / train_examples->size();
  // TODO(usyed): Three loops is _really_ inefficient
  for (Example& example : *train_examples) {
    example.weight = initial_wgt;
  }
  return;
}
