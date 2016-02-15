setClass("Deepboost",
         slots = list(
           data = "data.frame",
           nobs = "numeric",
           lambda= "numeric",
           train = "function",
           predict = "function",
           print = "function"
         ))

deepboost.train <- function(object, controls, weights = NULL, fitmem = NULL, ...) {
  # set slots
}

deepboost.predict <- function(object, controls, weights = NULL, fitmem = NULL, ...) {
}

deepboost.print <- function(object, controls, weights = NULL, fitmem = NULL, ...) {
  # call evlaute_R
}

Deepboost <- new("Deepboost",
                 data=data.frame(),
                 nobs=0,
                 lambda=0.1,
                 train = deepboost.train,
                 predict = deepboost.predict,
                 print = deepboost.print #evaluate
)

#' @export
deepboost <- function(formula, data = list(), subset = NULL, weights = NULL,
                    controls = cforest_unbiased(),
                    xtrafo = ptrafo, ytrafo = ptrafo, scores = NULL) {

  # parse formula

  deepboost.train(Deepboost, ...)
}


