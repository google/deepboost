//#include <sstream>
//#include <cereal/archives/binary.hpp>
#include <Rcpp.h>
#include "types.h"
#include "deepboost_converters.h"
#include "deepboost_C.h"

// [[Rcpp::plugins("cpp11")]]

using namespace Rcpp;

vector<Example> createExampleVectorFromDataFrame(DataFrame data)
{
    vector<Example> examples;
    int example_number = data.nrows();
    StringVector labels;
    NumericVector weights;
    bool labelsExist=false;
    bool weightsExist=false;

    StringVector names = data.names();
    StringVector feature_names;
    for (int i = 0; i < names.length(); ++i) {
      if (names[i] == "label")
      {
        labels = data["label"];
        labelsExist = true;
      }
      else if (names[i] == "weight")
      {
        weights = data["weight"];
        weightsExist = true;
      }
      else
      {
        feature_names.push_back(names[i]);
      }
    }

    for (int i = 0; i < example_number; ++i) {
      Example *example = new Example;

      if (labelsExist)
      {
        example -> label = std::stoi(as<std::string>(labels[i]));
      }
      else
      {
        example -> label = (int)NULL;
      }

      if(weightsExist)
      {
        example -> weight = weights[i];
      }
      else
      {
        example -> weight = 1.0/example_number;
      }
      example -> values = vector<Value>();

      examples.push_back(*example);
    }

    for (int j = 0; j < feature_names.size(); ++j)
    {
      NumericVector current_feature = data[String(feature_names[j])];

      for (int i = 0; i < example_number; ++i) {
        examples[i].values.push_back(current_feature[i]);
      }
    }

    return examples;
}

// struct Serialisable_Example : Example {
//   Serialisable_Example(Example ex_){
//     values = ex_.values;
//     label = ex_.label;
//     weight = ex_.weight;
//   }
//
//   // This method lets cereal know which data members to serialize
//   template<class Archive>
//   void serialize(Archive & archive) {
//     archive( values, label, weight ); // serialize things by passing them to the archive
//   }
// } ;
//
// struct Serialisable_Node : Node {
//   Serialisable_Node(Node nd_){
//     split_feature = nd_.split_feature;
//     split_value = nd_.split_value;
//     left_child_id = nd_.left_child_id;
//     right_child_id = nd_.right_child_id;
//     positive_weight = nd_.positive_weight;
//     negative_weight = nd_.negative_weight;
//     leaf = nd_.leaf;
//     depth = nd_.depth;
//   }
//   vector<Example> examples;  // Examples at this node.
// };
//
// RawVector serialize_model(vector<Example> exx_) {
//   //vector<Serialisable_Example> vExR = exx_;
//   std::stringstream ss;
//   {
//     cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
//     oarchive(exx_);
//   }
//   ss.seekg(0, ss.end);
//   RawVector retval(ss.tellg());
//   ss.seekg(0, ss.beg);
//   ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
//   return retval;
// }

Rcpp::List modelToList(Model model_)
{
  List model = List();
  for (pair<Weight, Tree> pair_: model_) {
    vector<Node> tree_ = pair_.second;
    List nodes = List::create();
    for(Node node_: tree_) {
      vector<Example> examples_ = node_.examples;
      List examples = List::create();
//       TODO : check without initializing values
//       for(Example example_: examples_){
//         examples.push_back(
//           List::create(
//             Named("values",example_.values),
//             Named("label",example_.label),
//             Named("weight",example_.weight)
//           )
//         );
//       }
      nodes.push_back(
          List::create(
                      Named("examples",examples),
                      Named("split_feature",node_.split_feature),
                      Named("split_value",node_.split_value),
                      Named("left_child_id",node_.left_child_id),
                      Named("right_child_id",node_.right_child_id),
                      Named("positive_weight",node_.positive_weight),
                      Named("negative_weight",node_.negative_weight),
                      Named("leaf",node_.leaf),
                      Named("depth",node_.depth)
                      )
                      );
    }
    model.push_back(
      List::create(
        Named("tree_weight",pair_.first),
        Named("tree",nodes)
        )
      );
  }
  return model;
}

Model listToModel(Rcpp::List model_)
{
  Model model;
  for(List pair_ : model_){
    Weight tree_weight = Rcpp::as<Weight>(pair_["tree_weight"]);
    List nodes_ = Rcpp::as<List>(pair_["tree"]);

    vector<Node> tree;
    for (List node_ : nodes_){
      List examples_ = Rcpp::as<List>(node_["examples"]);
      Feature split_feature = Rcpp::as<Feature>(node_["split_feature"]);
      Value split_value = Rcpp::as<Value>(node_["split_value"]);
      NodeId left_child_id = Rcpp::as<NodeId>(node_["left_child_id"]);
      NodeId right_child_id = Rcpp::as<NodeId>(node_["right_child_id"]);
      Weight positive_weight = Rcpp::as<Weight>(node_["positive_weight"]);
      Weight negative_weight = Rcpp::as<Weight>(node_["negative_weight"]);
      bool leaf = Rcpp::as<bool>(node_["leaf"]);
      int depth = Rcpp::as<int>(node_["depth"]);

      vector<Example> examples;
      // TODO : check with empty example vector
//       for (List example_ : examples_){
//         vector<Value> values = Rcpp::as<vector<Value>>(example_["values"]);
//         Label label = Rcpp::as<Label>(example_["label"]);
//         Weight weight = Rcpp::as<Weight>(example_["weight"]);
//
//         Example *example = new Example;
//
//         example -> values = values;
//         example -> label = label;
//         example -> weight = weight;
//
//         examples.push_back(*example);
//     }

      Node *node = new Node;

      node -> examples = examples;
      node -> split_feature = split_feature;
      node -> split_value = split_value;
      node -> left_child_id = left_child_id;
      node -> right_child_id = right_child_id;
      node -> positive_weight = positive_weight;
      node -> negative_weight = negative_weight;
      node -> leaf = leaf;
      node -> depth = depth;

      tree.push_back(*node);
    }

    pair<Weight, Tree> pair = make_pair(tree_weight, tree);
    model.push_back(pair);
  }

  return model;
}

Rcpp::List Train_C(DataFrame data,
                        int tree_depth, int num_iter,
                        double beta, double lambda, char loss_type,
                        bool verbose)
{
  vector<Example> train_examples = createExampleVectorFromDataFrame(data);
  Model model_;

  Train(&train_examples,
        &model_,
        tree_depth, num_iter, beta, lambda, loss_type, verbose);

  List model = modelToList(model_);

  return model;
}

Rcpp::List Evaluate_C(DataFrame data, Rcpp::List model)
{
  vector<Example> examples = createExampleVectorFromDataFrame(data);

  Model model_ = listToModel(model);

  float error;
  float avg_tree_size;
  int num_trees;

  Evaluate(examples, model_,
                &error, &avg_tree_size, &num_trees);

  List model_stats = List::create(
    Named("error",error),
    Named("avg_tree_size",avg_tree_size),
    Named("num_trees",num_trees)
    );

  return model_stats;
}

Rcpp::List Predict_C(DataFrame data, Rcpp::List model)
{
  List labels;

  // Create datasets for predict
  vector<Example> examples = createExampleVectorFromDataFrame(data);
  Model model_ = listToModel(model);

  // predict
  vector<Label> labels_ = Predict(examples, model_);

  // adjust return value
  for (Label label_ : labels_){
    labels.push_back(label_);
  }

  return (labels);
}


