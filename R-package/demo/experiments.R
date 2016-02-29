# library(caret)
# library(ada)
# library(deepboost)
#
# #setwd('D:/MSC/Learnig algorithms for information systems/project')
# setwd('/Users/dmarcous/git/deepboost/R-package/tests')
#
# datadir <- "./datasets/"
#
# # read datasets
# adult <- read.csv(paste0(datadir,"adult.data"))
# # X..50K ~ X39 + X77516 + X13 + X2174 +  X0 + X40
# aust <- read.csv(paste0(datadir,"australian.dat"))
# # X0.3 ~ .
# banana <- read.csv(paste0(datadir,"banana.dat"))
# # X.1.0 ~ .
# bupa <- read.csv(paste0(datadir,"bupa.dat"))
# # X1 ~ .
# coli <- read.csv(paste0(datadir,"coil2000.dat"))
# # X0.45 ~ .
# haber <- read.csv(paste0(datadir,"haberman.dat"))
# # negative ~ .
# heart <- read.csv(paste0(datadir,"heart.dat"))
# # X2.2 ~ .
# magic <- read.csv(paste0(datadir,"magic.dat"))
# # g ~ .
# pima <- read.csv(paste0(datadir,"pima.dat"))
# # tested_positive ~ .
# sonar <- read.csv(paste0(datadir,"sonar.dat"))
# # R ~ .
#
# # create lists of datasets and formulas
# datasets <- list(adult=adult, aust=aust, banana=banana, bupa=bupa, coli=coli,
#                  haber=haber, heart=heart, magic=magic, pima=pima, sonar=sonar)
# formulas <- list(X..50K ~ X39 + X77516 + X13 + X2174 +  X0 + X40,
#                  X0.3 ~ .,
#                  X.1.0 ~ .,
#                  X1 ~ .,
#                  X0.45 ~ .,
#                  negative ~ .,
#                  X2.2 ~ .,
#                  g ~ .,
#                  tested_positive ~ .,
#                  R ~ .)
#
# results <- data.frame(dataset = numeric(0), ensemble_size = numeric(0), ada_acc = numeric(0), ada_sd = numeric(0),
#                      ada_time = numeric(0), deep_acc = numeric(0), deep_sd = numeric(0), deep_time = numeric(0),
#                      t_test = numeric(0))
# # for each number of iterations
# for(num_iter in c(5,10,50,250)){
#   # for each data set
#   for(i in 1:10){
#     ds <- datasets[[i]]
#     levels(ds[,length(ds)]) <- c(1,-1)
#     formula <- formulas[[i]]
#     ada_acc <- rep(0,5)
#     deep_acc <- rep(0,5)
#     #ada_auc <- rep(0,5)
#     #deep_auc <- rep(0,5)
#     ada_t <- 0
#     deep_t <- 0
#     # 5 different 10folds
#     for(j in 1:5){
#       flds <- createFolds(1:nrow(ds), k = 10)
#       for(k in 1:10){
#         train <- ds[-flds[[k]],]
#         test <- ds[flds[[k]],]
#
#         # train models and calculate accurcy
#         t <- Sys.time()
#         ab_model <- ada(formula, train, iter = num_iter)
#         ada_acc[j] <- ada_acc[j] + sum(predict(ab_model, test) == test[,length(test)]) / nrow(test)
#         ada_t <- ada_t + round(difftime(Sys.time(), t, units = "secs"), 2)
#
#         t <- Sys.time()
#         db_model <- deepboost.formula(formula, train, num_iter = num_iter, lambda=0, loss_type="e")
#         deep_acc[j] <- deep_acc[j] + sum(predict(db_model, test) == test[,length(test)]) / nrow(test)
#         deep_t <- deep_t + round(difftime(Sys.time(), t, units = "secs"), 2)
#
#       }
#       ada_acc[j] <- ada_acc[j]/10.0
#       deep_acc[j] <- deep_acc[j]/10.0
#     }
#     # caluculate results
#     ada_acc_mean <- round(mean(ada_acc), 4)
#     #ada_auc_mean <- mean(ada_auc)
#     deep_acc_mean <- round(mean(deep_acc), 4)
#     #deep_auc_mean <- mean(deep_auc)
#     ada_acc_sd <- round(sd(ada_acc), 6)
#     #ada_auc_sd <- sd(ada_auc)
#     deep_acc_sd <- round(sd(deep_acc), 6)
#     #deep_auc_sd <- sd(deep_auc)
#     acc_t_test <- t.test(ada_acc, deep_acc, paired=TRUE)$p.value < 0.05
#     #auc_t_test <- t.test(ada_auc, deep_auc, paired=TRUE)$p.value < 0.05
#
#     # print to file
#     fname <- paste('Results/', names(datasets)[i], num_iter, ".res", sep='')
#     res <- data.frame(dataset = names(datasets)[i], ensemble_size = num_iter, ada_acc = ada_acc_mean,
#                       ada_sd = ada_acc_sd, ada_time = ada_t, deep_acc = deep_acc_mean,
#                       deep_sd = deep_acc_sd, deep_time = deep_t,  t_test = acc_t_test)
#     write.csv(res, fname, row.names = FALSE)
#     print(paste(ada_t+deep_t, 'seconds for dataset:', names(datasets)[i], ',ensemble size:', num_iter))
#     results <- rbind(results, res)
#   }
# }
# write.csv(results, 'Results/results.txt', row.names = FALSE)
#
