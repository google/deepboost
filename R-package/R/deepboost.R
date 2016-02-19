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
           verbose = "boolean",
           train = "function",
           predict = "function",
           print = "function",
           error = "numeric",
           examples = "Examples_R",
           model = "Model_R"
         ))

#' Trains a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param data A data.frame to train on
#' @param controls Paramters
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
                            verbose) {
  # set slots
  RET = new("Deepboost",
            train = deepboost.train,
            predict = deepboost.predict,
            print = deepboost.print)

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

  if (!(is.boolean(verbose)))
  {
    stop("ERROR_paramter_setting : verbose must be boolean (True / False) (Default : TRUE)" )
  }
  RET@verbose = as.character(verbose)

  RET@error = 0.0


#   examples = "Examples_R",
#   model = "Model_R"

  return(RET)
}

#' Predicts instances responses based on a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param newdata A data.frame to predict responses for
#' @param controls Paramters
#' @return A vector of respones
#' @export
deepboost.predict <- function(object, newdata, controls = NULL) {
}

#' Evaluates and prints statistics for a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param controls Paramters
#' @return Prints to console the model evaluation string
#' @export
deepboost.print <- function(object) {
  # call Evlaute_R from RcppExports.R
  print(paste("error ",object@error))
  x <- c(1,2,3)
  print(x)
  y <- Evaluate_R(x)
  print(y)
}

#' Empty Deepboost S4 class object with default settings
Deepboost <- new("Deepboost",
#                  tree_depth = 5,
#                  num_iter = 1,
#                  # (0,0) - adaboost, (>0,0) - L1, (0,>0) deepboost, (>0, >0) deepbost+L1
#                  beta = 0.0,
#                  lambda= 0.05,
#                  loss_type = "l", #l - logistic, #e exponential
#                  verbose = TRUE,
                 train = deepboost.train,
                 predict = deepboost.predict,
                 print = deepboost.print, #evaluate
                 error = -1.0,
                 examples = NULL,
                 model = NULL
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
  if(length(levels(y))!=2)
  {
    print("ERROR: response must be binary")
    return()
  }
  print(paste("1 for",levels(y)[1],"and -1 for",levels(y)[2]))
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
                         verbose)
  print("evaluating deepboost model")
  deepboost.print(fit)
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
  if(length(levels(y))!=2)
  {
    print("ERROR: response must be binary")
    return()
  }
  print(paste("1 for",levels(y)[1],"and -1 for",levels(y)[2]))
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
                         verbose)
  print("evaluating deepboost model")
  deepboost.print(fit)
}
