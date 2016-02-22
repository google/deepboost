// #include <Rcpp.h>
// #include "types.h"
// #include "deepboost_C.h"
//
// using namespace Rcpp;
//
// class Examples_R {
// public:
//   Examples_R(DataFrame data){
//
//     int example_number = data.nrows();
//     NumericVector labels   = data["label"];
//     NumericVector weights = data["weight"];
//
//     StringVector names = data.names();
//     StringVector feature_names;
//     for (int i = 0; i < names.length(); ++i) {
//       if (names[i] != "label" && names[i] !="weight")
//       {
//         feature_names.push_back(names[i]);
//       }
//     }
//
//     for (int i = 0; i < example_number; ++i) {
//       Example *example = new Example;
//
//       example -> label = labels[i];
//       example -> weight = labels[i];
//       example -> values = vector<Value>();
//
//       examples.push_back(*example);
//     }
//
//     for (int j = 0; j < feature_names.size(); ++j) {
//
//       NumericVector current_feature = data[String(feature_names[j])];
//
//       for (int i = 0; i < example_number; ++i) {
//         examples[i].values.push_back(current_feature[i]);
//       }
//     }
//   }
//
//   vector<Example> get_examples() { return examples; }
//   void set_examples( vector<Example> examples_ ) { examples = examples_; }
//
// private:
//   vector<Example> examples;
// };
//
// class Model_R {
// public:
//   Model_R(){
//     vector<pair<Weight, Tree> > tree_vec;
//     model = tree_vec;
//   }
//   vector<pair<Weight, Tree> > get_tree_vec() { return model; }
//   void set_tree_vec( vector<pair<Weight, Tree> > model_ ) { model = model_; }
// private:
//   //Model model;
//   vector<pair<Weight, Tree> > model;
// };
//
//
// RCPP_MODULE(mod_Examples_R) {
//   class_<Examples_R>( "Examples_R" )
//   .constructor<DataFrame>()
//   .property( "examples", &Examples_R::get_examples, &Examples_R::set_examples )
//   ;
// }
//
// RCPP_MODULE(mod_Model_R) {
//   class_<Model_R>( "Model_R" )
//   .constructor()
//   .property( "model", &Model_R::get_tree_vec, &Model_R::set_tree_vec )
//   ;
// }
