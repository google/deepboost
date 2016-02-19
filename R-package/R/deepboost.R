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
           lambda= "numeric",
           train = "function",
           predict = "function",
           print = "function",
           error = "numeric"
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
deepboost.train <- function(object, data, controls = NULL) {
  # set slots
  RET = new("Deepboost",
            train = deepboost.train,
            predict = deepboost.predict,
            print = deepboost.print)
  RET@lambda = object@lambda
  RET@error = 0.0
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
                 lambda=0.1,
                 train = deepboost.train,
                 predict = deepboost.predict,
                 print = deepboost.print, #evaluate
                 error = -1.0
)

#' Main function for deepboost moel creation
#'
#' @param formula A R Formula object see : ?formula
#' @param data A data.frame of samples to train on
#' @param controls parameters
#' @return A trained Deepbost model
#' @export
deepboost <- function(formula, data = list(),
                      controls = NULL) {

  # parse formula
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  
  mt <- attr(mf, "terms")
  y <- model.response(mf, "numeric")
  x <- model.matrix(mt, mf, contrasts)
  
  
  print("training deepboost model")
  fit <- deepboost.train(Deepboost, data, controls)
  print("evaluating deepboost model")
  deepboost.print(fit)
}


