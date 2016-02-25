#' @useDynLib deepboost
#' @importFrom Rcpp evalCpp
NULL

#' An S4 class to represent a deepboost model.
#'
#' @slot lambda parameter something
#' @slot train deepboost model training function
#' @slot predict deepboost model instance prediction function
#' @slot print deepboost model evaluation statistics function
#' @slot error deepboost model training error
setClass("Deepboost",
         slots = list(
           tree_depth = "numeric",
           num_iter = "numeric",
           beta = "numeric",
           lambda= "numeric",
           loss_type = "character",
           verbose = "logical",
           examples = "data.frame",
           model = "list",
           classes = "character",
           error = "numeric"
         ))

#' Trains a deepboost model
#'
#' @param object A Deepboost S4 class object
#’ @param data input data.frame as training for model
#’ @param tree_depth maximum depth for a single decision tree in the model
#’ @param num_iter number of iterations = number of trees in ensemble
#’ @param beta regularisation for scores (L1)
#’ @param lambda regularisation for tree depth
#’ @param loss_type - "l" logistic, "e" exponential
#’ @param verbose - print extra data while training TRUE / FALSE
#' @details (beta,lambda) = (0,0) - adaboost, (>0,0) - L1, (0,>0) deepboost, (>0, >0) deepbost+L1
#' @return A trained Deepbost model
#' @export
#setMethod("train", signature = "deepboost.train",
#definition =
deepboost.train <- function(object, data,
                            tree_depth,
                            num_iter,
                            beta,
                            lambda,
                            loss_type,
                            verbose,
                            classes) {
  # set slots
  RET = new("Deepboost")

  # Check parameter validity
  if (!(is.numeric(tree_depth)) || tree_depth <= 0 || !(tree_depth%%1==0))
  {
    stop("ERROR_paramter_setting : tree_depth must be >= 1 and integer (Default : 5)" )
  }
  RET@tree_depth = as.integer(tree_depth)

  # Check parameter validity
  if (!(is.numeric(num_iter)) || num_iter <= 0 || !(num_iter%%1==0))
  {
    stop("ERROR_paramter_setting : num_iter must be >= 1 and integer (Default : 1)" )
  }
  RET@num_iter = as.integer(num_iter)

  # (beta, lambda) =
  # (0,0) - adaboost, (>0,0) - L1, (0,>0) deepboost, (>0, >0) deepbost+L1

  # Check parameter validity
  if (!(is.numeric(beta)) || beta < 0.0)
  {
    stop("ERROR_paramter_setting : beta must be >= 0 and double (Default : 0.0)" )
  }
  RET@beta = as.double(beta)

  # Check parameter validity
  if (!(is.numeric(lambda)) || lambda < 0.0)
  {
    stop("ERROR_paramter_setting : lambda must be >= 0 and double (Default : 0.05)" )
  }
  RET@lambda = as.double(lambda)

  # Check parameter validity
  if (!(is.character(loss_type)) || (loss_type != "l" && loss_type != "e"))
  {
    stop("ERROR_paramter_setting : loss_type must be \"l\" - logistic or \"e\" - exponential (Default : \"l\")" )
  }
  RET@loss_type = as.character(loss_type)

  if (!(is.logical(verbose)))
  {
    stop("ERROR_paramter_setting : verbose must be boolean (True / False) (Default : TRUE)" )
  }
  RET@verbose = verbose

  RET@examples = data
  RET@classes = classes

  # call training
  model =  Train_R(RET@examples,
                   RET@tree_depth, RET@num_iter, RET@beta, RET@lambda, RET@loss_type, RET@verbose)

  RET@model = model

  return(RET)
}

#' Predicts instances responses based on a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param newdata A data.frame to predict responses for
#' @return A vector of respones
#' @export
deepboost.predict <- function(object, newdata) {
  labels <-
    Predict_R(newdata,
               object@model)
  labels <- unlist(labels)
  labels[labels==1] <- object@classes[1]
  labels[labels==-1] <- object@classes[2]
  return (labels)
}

#' Evaluates and prints statistics for a deepboost model on the train set
#'
#' @param object A Deepboost S4 class object
#' @return List with model_statistics to console the model evaluation string
#' @export
deepboost.print <- function(object) {
  model_stats <- deepboost.evaluate(object, object@examples)
  print(paste("Model error:",model_stats[["error"]]))
  print(paste("Average tree size:",model_stats[["avg_tree_size"]]))
  print(paste("Number of trees:",model_stats[["num_trees"]]))
  return (model_stats)
}

#' Evaluates and prints statistics for a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param a \code{data.frame} object to evaluate with the model
#' @return a list with model statistics - error, avg_tree_size, num_trees
#' @export
deepboost.evaluate <- function(object, data) {
  model_stats <-
    Evaluate_R(data,
               object@model)
  return (model_stats)
}

#' Empty Deepboost S4 class object with default settings
Deepboost <- new("Deepboost",
                 examples = data.frame(),
                 model = list()
)

#' Main function for deepboost model creation
#'
#' @param x A data.frame of samples' values
#' @param y A data.frame of samples's labels
#' @param weights The weight of each example
#' @param controls parameters
#' @return A trained Deepbost model
#' @export
deepboost.default <- function(x, y, weights = NULL,
                              tree_depth = 5,
                              num_iter = 1,
                              beta = 0.0,
                              lambda= 0.05,
                              loss_type = "l",
                              verbose = TRUE
                              ) {
  # initialize weights
  n <- dim(x)[1]
  if(is.null(weights))
  {
    weights <- rep(1/n, n)
  }
  # make response either 1 or -1
  y <- factor(y)
  if (length(levels(y))!=2)
  {
    stop("ERROR_data : response must be binary" )
  }
  classes = levels(y)
  levels(y) <- c(1,-1)
  # create data
  data <- data.frame(x)
  data['label'] <- y
  data['weight'] <- weights

  print("training deepboost model")
  fit <- deepboost.train(Deepboost, data,
                         tree_depth,
                         num_iter,
                         beta,
                         lambda,
                         loss_type,
                         verbose,
                         classes)
  print("evaluating deepboost model")
  deepboost.print(fit)

  return (fit)
}

#' Main function for deepboost model creation
#'
#' @param formula A R Formula object see : ?formula
#' @param data A data.frame of samples to train on
#' @param weights The weight of each example
#' @param controls parameters
#' @return A trained Deepbost model
#' @export
deepboost <- function(formula, data, ...) {
  deepboost.formula(formula, data, ...)
}


#' Main function for deepboost model creation, using a formula
#'
#' @param formula A R Formula object see : ?formula
#' @param data A data.frame of samples to train on
#' @param weights The weight of each example
#' @param controls parameters
#' @return A trained Deepbost model
#' @export
deepboost.formula <- function(formula, data, weights = NULL,
                              tree_depth = 5,
                              num_iter = 1,
                              beta = 0.0,
                              lambda= 0.05,
                              loss_type = "l",
                              verbose = TRUE) {
  # initialize weights
  n <- dim(data)[1]
  if(is.null(weights))
  {
    weights <- rep(1/n, n)
  }
  # parse formula
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- factor(model.response(mf))
  x <- model.matrix(mt, mf, contrasts)
  # make response either 1 or -1
  if (length(levels(y))!=2)
  {
    stop("ERROR_data : response must be binary" )
  }
  classes = levels(y)
  levels(y) <- c(1,-1)
  # create data
  data <- data.frame(x[,-1])
  data['label'] <- y
  data['weight'] <- weights

  print("training deepboost model")
  fit <- deepboost.train(Deepboost, data,
                         tree_depth,
                         num_iter,
                         beta,
                         lambda,
                         loss_type,
                         verbose,
                         classes)
  print("evaluating deepboost model")
  deepboost.print(fit)

  return (fit)
}

#' Predict method for Deepboost model
#'
#' Predicted values based on deepboost model object.
#'
#' @param object Object of class "Deepboost"
#' @param newdata takes \code{data.frame}.
#'
#' @details
#' The option \code{ntreelimit} purpose is to let the user train a model with lots
#' of trees but use only the first trees for prediction to avoid overfitting
#' (without having to train a new model with less trees).
#' @export
setMethod("predict", signature = "Deepboost",
          definition = function(object, newdata) {
            deepboost.predict(object, newdata)
})

#' Print method for Deepboost model
#'
#' Evaluates a trained deepboost model object.
#'
#' @details
#' Evalutes a Deepboost model object.
#' Prints : bla \code{error}
#' @export
setMethod("show", signature = "Deepboost",
          definition = function(object) {
            deepboost.print(object)
          })
