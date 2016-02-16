#' An S4 class to represent a bank account.
#'
#' @slot lambda parameter something
#' @slot train deepboost model training function
#' @slot predict deepboost model instance prediction function
#' @slot print deepboost model evaluation statistics function
setClass("Deepboost",
         slots = list(
           lambda= "numeric",
           train = "function",
           predict = "function",
           print = "function"
         ))

#' Trains a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param data A data.frame to train on
#' @param controls Paramters
#' @return A trained Deepbost model
deepboost.train <- function(object, data, controls, weights = NULL, fitmem = NULL, ...) {
  # set slots
}

#' Predicts instances responses based on a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param newdata A data.frame to predict responses for
#' @param controls Paramters
#' @return A vector of respones
deepboost.predict <- function(object, newdata, controls, weights = NULL, fitmem = NULL, ...) {
}

#' Evaluates and prints statistics for a deepboost model
#'
#' @param object A Deepboost S4 class object
#' @param controls Paramters
#' @return Prints to console the model evaluation string
deepboost.print <- function(object, controls, weights = NULL, fitmem = NULL, ...) {
  # call evlaute_R from RcppExports.R
}

#' Empty Deepboost S4 class object with default settings
Deepboost <- new("Deepboost",
                 lambda=0.1,
                 train = deepboost.train,
                 predict = deepboost.predict,
                 print = deepboost.print #evaluate
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

  deepboost.train(Deepboost, data, ...)
  print("in deepboost")
}


