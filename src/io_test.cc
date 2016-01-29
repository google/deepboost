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
#include "io.h"

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DECLARE_string(dataset);
DECLARE_string(data_filename);
DECLARE_int32(fold_to_cv);
DECLARE_int32(fold_to_test);
DECLARE_int32(num_folds);
DECLARE_double(noise_prob);

class IoTest : public SrmTest {};

TEST_F(IoTest, SplitStringTest) {
  string text = ",,2,3,,14,,";
  vector<string> tokens;
  SplitString(text, ',', &tokens);
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ("2", tokens[0]);
  EXPECT_EQ("3", tokens[1]);
  EXPECT_EQ("14", tokens[2]);

  text = "   55 71     90 1 ";
  tokens.clear();
  SplitString(text, ' ', &tokens);
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ("55", tokens[0]);
  EXPECT_EQ("71", tokens[1]);
  EXPECT_EQ("90", tokens[2]);
  EXPECT_EQ("1", tokens[3]);
}

TEST_F(IoTest, ParseLineBreastCancerTest) {
  Example example;
  string line = "1000025,5,1,1,1,2,1,3,1,1,2";
  EXPECT_TRUE(ParseLineBreastCancer(line, &example));
  EXPECT_EQ(-1, example.label);
  EXPECT_EQ(9, example.values.size());
  // Spot check features
  EXPECT_NEAR(5, example.values[0], kTolerance);
  EXPECT_NEAR(3, example.values[6], kTolerance);
  EXPECT_NEAR(1, example.values[8], kTolerance);
  line = "1017122,8,10,10,8,7,10,9,7,1,4";
  EXPECT_TRUE(ParseLineBreastCancer(line, &example));
  EXPECT_EQ(1, example.label);
  line = "1057013,8,4,5,1,2,?,7,3,1,4";
  EXPECT_FALSE(ParseLineBreastCancer(line, &example));
}

TEST_F(IoTest, ParseLineIonTest) {
  Example example;
  string line =
      "1,0,1,-0.15899,0.72314,0.27686,0.83443,-0.58388,1,-0.28207,1,-0.49863,0."
      "79962,-0.12527,0.76837,0.14638,1,0.39337,1,0.26590,0.96354,-0.01891,0."
      "92599,-0.91338,1,0.14803,1,-0.11582,1,-0.11129,1,0.53372,1,-0.57758,g";
  EXPECT_TRUE(ParseLineIon(line, &example));
  EXPECT_EQ(1, example.label);
  EXPECT_EQ(34, example.values.size());
  // Spot check features
  EXPECT_NEAR(1, example.values[0], kTolerance);
  EXPECT_NEAR(0.83443, example.values[6], kTolerance);
  EXPECT_NEAR(-0.11129, example.values[29], kTolerance);
  line =
      "1,0,1,-0.18829,0.93035,-0.36156,-0.10868,-0.93597,1,-0.04549,0.50874,-0."
      "67743,0.34432,-0.69707,-0.51685,-0.97515,0.05499,-0.62237,0.33109,-1,-0."
      "13151,-0.45300,-0.18056,-0.35734,-0.20332,-0.26569,-0.20468,-0.18401,-0."
      "19040,-0.11593,-0.16626,-0.06288,-0.13738,-0.02447,b";
  EXPECT_TRUE(ParseLineIon(line, &example));
  EXPECT_EQ(-1, example.label);
}

TEST_F(IoTest, ParseLineGermanTest) {
  Example example;
  string line =
      "   2  48   2  60   1   3   2   2   1  22   3   1   1   1   1   0   0   "
      "1   0   0   1   0   0   1   2 ";
  EXPECT_TRUE(ParseLineGerman(line, &example));
  EXPECT_EQ(1, example.label);
  EXPECT_EQ(24, example.values.size());
  line =
      "   1   6   4  12   5   5   3   4   1  67   3   2   1   2   1   0   0   "
      "1   0   0   1   0   0   1   1 ";
  EXPECT_TRUE(ParseLineGerman(line, &example));
  EXPECT_EQ(-1, example.label);
  // Spot check features
  EXPECT_NEAR(1, example.values[0], kTolerance);
  EXPECT_NEAR(3, example.values[6], kTolerance);
  EXPECT_NEAR(1, example.values[8], kTolerance);
  EXPECT_NEAR(67, example.values[9], kTolerance);
}

TEST_F(IoTest, ParseLineOcr17Test) {
  Example example;
  string line =
      "0,0,0,3,16,11,1,0,0,0,0,8,16,16,1,0,0,0,0,9,16,14,0,0,0,1,7,16,16,11,0,"
      "0,0,9,16,16,16,8,0,0,0,1,8,6,16,7,0,0,0,0,0,5,16,9,0,0,0,0,0,2,14,14,1,"
      "0,1";
  EXPECT_TRUE(ParseLineOcr17(line, &example));
  EXPECT_EQ(-1, example.label);
  EXPECT_EQ(64, example.values.size());
  line =
      "0,0,8,15,16,13,0,0,0,1,11,9,11,16,1,0,0,0,0,0,7,14,0,0,0,0,3,4,14,12,2,"
      "0,0,1,16,16,16,16,10,0,0,2,12,16,10,0,0,0,0,0,2,16,4,0,0,0,0,0,9,14,0,0,"
      "0,0,7";
  EXPECT_TRUE(ParseLineOcr17(line, &example));
  EXPECT_EQ(1, example.label);
  // Spot check features
  EXPECT_NEAR(0, example.values[0], kTolerance);
  EXPECT_NEAR(0, example.values[6], kTolerance);
  EXPECT_NEAR(0, example.values[8], kTolerance);
  EXPECT_NEAR(9, example.values[58], kTolerance);
  line =
      "0,0,0,3,11,16,0,0,0,0,5,16,11,13,7,0,0,3,15,8,1,15,6,0,0,11,16,16,16,16,"
      "10,0,0,1,4,4,13,10,2,0,0,0,0,0,15,4,0,0,0,0,0,3,16,0,0,0,0,0,0,1,15,2,0,"
      "0,4";
  EXPECT_FALSE(ParseLineOcr17(line, &example));
}

TEST_F(IoTest, ParseLineOcr49Test) {
  Example example;
  string line =
      "0,0,0,3,11,16,0,0,0,0,5,16,11,13,7,0,0,3,15,8,1,15,6,0,0,11,16,16,16,16,"
      "10,0,0,1,4,4,13,10,2,0,0,0,0,0,15,4,0,0,0,0,0,3,16,0,0,0,0,0,0,1,15,2,0,"
      "0,4";
  EXPECT_TRUE(ParseLineOcr49(line, &example));
  EXPECT_EQ(-1, example.label);
  // Spot check features
  EXPECT_NEAR(0, example.values[0], kTolerance);
  EXPECT_NEAR(0, example.values[6], kTolerance);
  EXPECT_NEAR(0, example.values[8], kTolerance);
  EXPECT_NEAR(0, example.values[58], kTolerance);
  EXPECT_NEAR(15, example.values[60], kTolerance);
  EXPECT_EQ(64, example.values.size());
  line =
      "0,0,0,4,13,16,16,3,0,0,8,16,9,12,16,4,0,7,16,3,3,15,13,0,0,9,15,14,16,"
      "16,6,0,0,1,8,7,12,15,0,0,0,0,0,0,13,10,0,0,0,0,0,3,15,6,0,0,0,0,0,5,15,"
      "4,0,0,9";
  EXPECT_TRUE(ParseLineOcr49(line, &example));
  EXPECT_EQ(1, example.label);
  line =
      "0,0,8,15,16,13,0,0,0,1,11,9,11,16,1,0,0,0,0,0,7,14,0,0,0,0,3,4,14,12,2,"
      "0,0,1,16,16,16,16,10,0,0,2,12,16,10,0,0,0,0,0,2,16,4,0,0,0,0,0,9,14,0,0,"
      "0,0,7";
  EXPECT_FALSE(ParseLineOcr49(line, &example));
}

TEST_F(IoTest, ParseLineOcr17PrincetonTest) {
  Example example;
  string line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 3 3 3 3 3 3 0 0 0 0 0 "
      "0 0 3 2 0 0 0 3 1 0 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 0 0 0 0 0 0 0 2 3 0 "
      "0 0 0 0 0 0 0 0 0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 "
      "0 3 1 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 7";
  EXPECT_TRUE(ParseLineOcr17Princeton(line, &example));
  EXPECT_EQ(1, example.label);
  EXPECT_EQ(196, example.values.size());
  line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 "
      "0 0 0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 0 2 3 0 0 "
      "0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 0 0 0 0 0 0 "
      "0 0 3 2 0 0 0 0 0 0 0 0 0 0 0 0 3 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1";
  EXPECT_TRUE(ParseLineOcr17Princeton(line, &example));
  EXPECT_EQ(-1, example.label);
  // Spot check features
  EXPECT_NEAR(0, example.values[0], kTolerance);
  EXPECT_NEAR(0, example.values[6], kTolerance);
  EXPECT_NEAR(0, example.values[8], kTolerance);
  EXPECT_NEAR(3, example.values[35], kTolerance);
  line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 "
      "0 0 0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 0 2 3 0 0 "
      "0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 0 0 0 0 0 0 "
      "0 0 3 2 0 0 0 0 0 0 0 0 0 0 0 0 3 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4";
  EXPECT_FALSE(ParseLineOcr17Princeton(line, &example));
}

TEST_F(IoTest, ParseLineOcr49PrincetonTest) {
  Example example;
  string line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 2 3 3 3 1 0 0 0 0 0 0 "
      "0 0 2 3 1 0 2 3 0 0 0 0 0 0 0 0 2 1 0 1 3 3 0 0 0 0 0 0 0 0 0 2 3 3 3 2 "
      "0 0 0 0 0 0 0 0 0 0 1 3 3 0 0 0 0 0 0 0 0 0 0 0 1 3 1 0 0 0 0 0 0 0 0 0 "
      "0 0 2 3 0 0 0 0 0 0 0 0 0 0 0 0 3 1 0 0 0 0 0 0 0 0 0 0 0 2 3 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 9";
  EXPECT_TRUE(ParseLineOcr49Princeton(line, &example));
  EXPECT_EQ(1, example.label);
  EXPECT_EQ(196, example.values.size());
  line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 2 0 0 1 3 2 0 0 0 0 "
      "0 0 1 3 1 0 2 3 2 0 0 0 0 0 0 2 3 2 0 2 3 2 1 1 0 0 0 0 1 3 3 3 3 3 3 3 "
      "3 2 0 0 0 0 0 2 2 3 3 3 1 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 0 0 0 0 0 0 0 "
      "0 3 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4";
  EXPECT_TRUE(ParseLineOcr49Princeton(line, &example));
  EXPECT_EQ(-1, example.label);
  // Spot check features
  EXPECT_NEAR(0, example.values[0], kTolerance);
  EXPECT_NEAR(0, example.values[6], kTolerance);
  EXPECT_NEAR(0, example.values[8], kTolerance);
  EXPECT_NEAR(1, example.values[54], kTolerance);
  EXPECT_NEAR(0, example.values[58], kTolerance);
  line =
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 2 0 0 1 3 2 0 0 0 0 "
      "0 0 1 3 1 0 2 3 2 0 0 0 0 0 0 2 3 2 0 2 3 2 1 1 0 0 0 0 1 3 3 3 3 3 3 3 "
      "3 2 0 0 0 0 0 2 2 3 3 3 1 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 0 0 0 0 0 0 0 "
      "0 3 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7";
  EXPECT_FALSE(ParseLineOcr49Princeton(line, &example));
}

TEST_F(IoTest, ParseLinePimaTest) {
  Example example;
  string line = "6,148,72,35,0,33.6,0.627,50,1";
  EXPECT_TRUE(ParseLinePima(line, &example));
  EXPECT_EQ(1, example.label);
  EXPECT_EQ(8, example.values.size());
  // Spot check features
  EXPECT_NEAR(6, example.values[0], kTolerance);
  EXPECT_NEAR(0.627, example.values[6], kTolerance);
  line = "1,85,66,29,0,26.6,0.351,31,0";
  EXPECT_TRUE(ParseLinePima(line, &example));
  EXPECT_EQ(-1, example.label);
}

TEST_F(IoTest, ReadDataTest) {
  FLAGS_dataset = "breastcancer";
  FLAGS_data_filename = "./testdata/breast-cancer-wisconsin.data";
  FLAGS_num_folds = 4;
  FLAGS_fold_to_cv = 1;
  FLAGS_fold_to_test = 0;
  SetSeed(123456);

  vector<Example> train_examples, cv_examples, test_examples;
  ReadData(&train_examples, &cv_examples, &test_examples);
  EXPECT_EQ(2, train_examples.size());
  EXPECT_EQ(1, cv_examples.size());
  EXPECT_EQ(1, test_examples.size());
  EXPECT_NEAR(0.5, train_examples[0].weight, kTolerance);
  EXPECT_NEAR(0.5, train_examples[1].weight, kTolerance);
}

TEST_F(IoTest, ReadDataTestWithNoise) {
  FLAGS_dataset = "breastcancer";
  FLAGS_data_filename = "./testdata/breast-cancer-wisconsin.data";
  FLAGS_num_folds = 4;
  FLAGS_fold_to_cv = 1;
  FLAGS_fold_to_test = 0;
  FLAGS_noise_prob = 0;
  SetSeed(123456);

  vector<Example> train_examples, cv_examples, test_examples;
  ReadData(&train_examples, &cv_examples, &test_examples);
  Label train_label_0 = train_examples[0].label;
  Label train_label_1 = train_examples[1].label;
  Label cv_label_0 = cv_examples[0].label;
  Label test_label_0 = test_examples[0].label;

  FLAGS_noise_prob = 1;
  ReadData(&train_examples, &cv_examples, &test_examples);
  EXPECT_EQ(-train_label_0, train_examples[0].label);
  EXPECT_EQ(-train_label_1, train_examples[1].label);
  EXPECT_EQ(-cv_label_0, cv_examples[0].label);
  EXPECT_EQ(-test_label_0, test_examples[0].label);

  FLAGS_noise_prob = 0.5;
  const int kIterations = 100;
  double sum_labels = 0.0;
  for (int i = 0; i < kIterations; ++i) {
    ReadData(&train_examples, &cv_examples, &test_examples);
    sum_labels += train_examples[0].label;
    sum_labels += train_examples[1].label;
    sum_labels += cv_examples[0].label;
    sum_labels += test_examples[0].label;
  }
  // The average of uniformly random +1/-1 labels should be about 0
  EXPECT_NEAR(0, sum_labels / (4 * kIterations), 1e-2);
}
