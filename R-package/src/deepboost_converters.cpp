#include <Rcpp.h>
#include "types.h"
#include "deepboost_converters.hpp"
#include "deepboost_C.h"

using namespace Rcpp;

vector<Example> DFtoTrainEXamples(DataFrame data)
{
    vector<Example> examples;
    int example_number = data.nrows();
    NumericVector labels   = data["label"];
    NumericVector weights = data["weight"];

    StringVector names = data.names();
    StringVector feature_names;
    for (int i = 0; i < names.length(); ++i) {
      if (names[i] != "label" && names[i] !="weight")
      {
        feature_names.push_back(names[i]);
      }
    }

    for (int i = 0; i < example_number; ++i) {
      Example *example = new Example;

      example -> label = labels[i];
      example -> weight = labels[i];
      example -> values = vector<Value>();

      examples.push_back(*example);
    }

    for (int j = 0; j < feature_names.size(); ++j) {

      NumericVector current_feature = data[String(feature_names[j])];

      for (int i = 0; i < example_number; ++i) {
        examples[i].values.push_back(current_feature[i]);
      }
    }

    return examples;
}

Rcpp::List modelToList(Model model)
{
  Rcpp::List model_R = List(1);
  return model_R;
}

Model listToModel(Rcpp::List model_R)
{
  Model model;
  return model;
}

Rcpp::List train_not_exported(DataFrame data,
                        int tree_depth, int num_iter,
                        double beta, double lambda, char loss_type,
                        bool verbose)
{
  vector<Example> train_examples = DFtoTrainEXamples(data);
  Model model;

  Train(&train_examples,
        &model,
        tree_depth, num_iter, beta, lambda, loss_type, verbose);

  Rcpp::List model_R = modelToList(model);

  return model_R;
}
