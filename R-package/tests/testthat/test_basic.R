library(deepboost)

context("basic functions")

data(adult, package='deepboost')

formula <- X..50K ~ X39 + X77516 + X13 + X2174 +  X0 + X40
levels(adult[,length(adult)]) <- c(1,-1)

train <- adult[1:29000,]
test <- adult[29001:32560,]

set.seed(666)

test_that("train and predict", {
  bst <- deepboost.formula(formula, train, num_iter = 10)
  pred <- predict(bst, test)
  expect_equal(length(pred), 3560)
})
