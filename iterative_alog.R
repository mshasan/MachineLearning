getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
source("all_plots_onepage.R")
source("h2o_automl2.R")

#install.packages("arulesViz")
library("tidyverse")
library(arulesViz)
library(devtools)
# use this to get latest version of h2o
#install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
library(h2o)
library(lares)
library(mlbench)
library(magrittr)   # for %>% operator
#library(plyr)       # rename function
#library(dplyr)       # rename function
library(pROC)
library(beepr)
#library(ggplot2)
library(gridExtra)
library(haven)
#library(readr)


######################################################################################################
## Model for AE==========================================================================

# 
# h2o.init(nthreads=-1, max_mem_size="12G")
# #h2o.shutdown()
# h2o.removeAll()
# 
# bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
# 
# #split the data
# splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
# train  <- h2o.assign(splits[[1]], "train.hex") 
# test  <- h2o.assign(splits[[2]], "valid.hex") 
#  
# predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
# 
# 
# iter_seeds <- function(seed)
# {
#   aml <- h2o.automl(x = predictors, 
#                     y = "AEFL",
#                     training_frame = as.h2o(train),
#                     validation_frame = test,
#                     leaderboard_frame = as.h2o(test),
#                     max_runtime_secs = 5*60,
#                     max_models = 10,
#                     exclude_algos = c("GLM","DRF","StackedEnsemble","DeepLearning"),
#                     nfolds = 10, 
#                     seed = seed)
#   auc <- h2o.auc(h2o.getModel(as.vector(aml@leaderboard$model_id[1])), xval=TRUE)
# 
#   return(cbind(seed, auc))
# }
# 
# best_seed_AE <- sapply(c(123,652,485,32145,1359,13245,13254), iter_seeds)
# 
# 
# 
# print(aml@leaderboard[,1:3])
# 
# best_model_AE <- h2o.getModel(as.vector(aml@leaderboard$model_id[1]))
# h2o.auc(best_model_AE, train=TRUE, valid=TRUE, xval=TRUE)
# 
# 
# 
# # split orginal data--------
# 
# 
# iter_algo <- function(iter, train)
#   {
#     # split from the split data for iterative algo
#     train2 <- h2o.splitFrame(train, 0.2)[[1]]
#     gbm <- h2o.gbm(
#       training_frame = train2,     
#       x = predictors,                     
#       y = "AEFL",                        
#       ntrees = 30,                
#       learn_rate = 0.1,           
#       max_depth = 10,              
#       sample_rate = 1,          
#       col_sample_rate = 0.7,       
#       stopping_rounds = 2,        
#       stopping_tolerance = 0.01,  
#       score_each_iteration = TRUE,
#       min_rows = 2,
#       seed = 2000000)
#     
#     return(h2o.auc(gbm, train=TRUE))
#   }
# 
# 
# iter_algo_results <- sapply(1:100, iter_algo, train=train)
# mean(iter_algo_results)




# iterative Model for AE by DL============================================================

h2o.init(nthreads=-1, max_mem_size="12G")
#h2o.shutdown()
h2o.removeAll()

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))

#split the data
splits <- h2o.splitFrame(bmk, c(.6,.1), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex")
test  <- h2o.assign(splits[[3]], "test.hex")


#predictors <- c("LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA23") 
predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL", "KDA88", "AGE", "SEX"))



iter_algo_dl_AE <- function(iter)
{
  #train2 <- h2o.splitFrame(train, 0.4)[[1]]
  dl_AE <- h2o.deeplearning(
    #model_id = "dl_bmk_AE", 
    #checkpoint = "dl_check_AE", 
    training_frame = train, 
    validation_frame=valid, 
    x = predictors, 
    y = "AEFL", 
    hidden = c(300, 300),            ## more hidden layers -> more complex interactions
    epochs = 100,                     ## hopefully long enough to converge (otherwise restart again)
    stopping_metric = "logloss",     ## logloss is directly optimized by Deep Learning
    stopping_tolerance = 1e-2,         ## stop when validation logloss does not improve by >=1% for 2 scoring events
    stopping_rounds = 2,
    #score_validation_samples=10000, ## downsample validation set for faster scoring
    score_duty_cycle = 0.025,          ## don't score more than 2.5% of the wall time
    adaptive_rate = FALSE,            ## manually tuned learning rate
    rate = 0.01, 
    rate_annealing = 2e-6,            
    momentum_start = 0.2,            ## manually tuned momentum
    momentum_stable = 0.4, 
    momentum_ramp = 1e7,
    #input_dropout_ratio = .2,
    #hidden_dropout_ratios = .7,
    l1 = 1e-5,                       ## add some L1/L2 regularization
    l2 = 1e-5,
    activation = c("MaxoutWithDropout"),
    max_w2 = 10                       ## helps stability for Rectifier
  )               
  return(list(iterNo=iter, AUC=h2o.auc(dl_AE, train=TRUE), modelInfo=dl_AE))
}

iter_algo_dl_AE_results <- map(1:1000,iter_algo_dl_AE)

#max(unlist(lapply(iter_algo_dl_results,"[[",n=2)))
# max(unlist(iter_algo_dl_AE_results %>% map(2)))
# iter_algo_dl_AE_results %>% map(2) %>% unlist %>% max

auc_all_AE <- map_dbl(iter_algo_dl_AE_results, "AUC")
summary(auc_all_AE)
hist(auc_all_AE)
dl_best_model_AE <- iter_algo_dl_AE_results[[which.max(auc_all_AE)]]$modelInfo


# save the model
h2o.saveModel(dl_best_model_AE, path="./best_iter_model", force=TRUE)

# load the previously saved mdoel
dl_model_loaded_AE <- h2o.loadModel("best_iter_model\\DeepLearning_model_R_1535729785288_2799")
#dl_model_loaded_AE 

#plot(h2o.performance(dl_model_loaded_AE))

# training data
tag_AE = as.vector(train$AEFL)
score_AE = as.vector(h2o.predict(dl_model_loaded_AE, as.h2o(train))$Y)
pROC::roc(tag_AE, score_AE, ci=TRUE)
mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_dl_full.png")


# validation data
tag_AE_v = as.vector(valid$AEFL)
score_AE_v = as.vector(h2o.predict(dl_model_loaded_AE, as.h2o(valid))$Y)
pROC::roc(tag_AE_v, score_AE_v, ci=TRUE)
mplot_full(tag = tag_AE_v, 
           score = score_AE_v,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_dl_full_v.png")


# test data
tag_AE2 = as.vector(test$AEFL)
score_AE2 = as.vector(h2o.predict(dl_model_loaded_AE, as.h2o(test))$Y)
pROC::roc(tag_AE2, score_AE2, ci=TRUE)
mplot_full(tag = tag_AE2, 
           score = score_AE2,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_dl_full2.png")


#prediction on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX=="Male","M","F")
#names(bmk_extra)
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID=="4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID=="4590",]

# use one of them
bmk_extra <- bmk_extra
bmk_extra <- bmk_extra[bmk_extra$group=="NEOPHYTE",]
bmk_extra <- bmk_extra_CR4550
bmk_extra <- bmk_extra_CR4590

bmk_extra2 <- bmk_extra[,c(5,6,12:14,17:28)]
bmk_extra2 <- bmk_extra2[,c(3:17,1,2)]
#names(bmk_extra2)
names(bmk_extra2) <- c("AEFL","CIEFL", "SSCIEFL","LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                       "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO","AGE","SEX")


tag_AE3 = as.vector(bmk_extra2$AEFL)
score_AE3 = as.vector(h2o.predict(dl_model_loaded_AE, as.h2o(bmk_extra2[,c(-3,-2)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci=TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)





# iterative Model for CIE by DL============================================================

h2o.init(nthreads=-1, max_mem_size="12G")
#h2o.shutdown()
h2o.removeAll()

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))

#split the data
splits <- h2o.splitFrame(bmk, c(.6,.1), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex")
test  <- h2o.assign(splits[[3]], "test.hex")


#predictors <- c("LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA23") 
predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL", "KDA88", "AGE", "SEX"))



iter_algo_dl_CIE <- function(iter)
{
  #train2 <- h2o.splitFrame(train, 0.4)[[1]]
  dl_CIE <- h2o.deeplearning(
    #model_id = "dl_bmk_CIE", 
    #checkpoint = "dl_check_CIE", 
    training_frame = train, 
    validation_frame=valid, 
    x = predictors, 
    y = "CIEFL", 
    hidden = c(300),                ## more hidden layers -> more complex interactions
    epochs = 100,                     ## hopefully long enough to converge (otherwise restart again)
    stopping_metric = "logloss",     ## logloss is directly optimized by Deep Learning
    stopping_tolerance = 1e-2,         ## stop when validation logloss does not improve by >=1% for 2 scoring events
    stopping_rounds = 2,
    #score_validation_samples=10000, ## downsample validation set for faster scoring
    score_duty_cycle = 0.025,          ## don't score more than 2.5% of the wall time
    adaptive_rate = FALSE,            ## manually tuned learning rate
    rate = 0.01, 
    rate_annealing = 2e-6,            
    momentum_start = 0.2,            ## manually tuned momentum
    momentum_stable = 0.4, 
    momentum_ramp = 1e7,
    #input_dropout_ratio = .2,
    #hidden_dropout_ratios = .7,
    l1 = 1e-5,                       ## add some L1/L2 regularization
    l2 = 1e-5,
    activation = c("MaxoutWithDropout"),
    max_w2 = 10                       ## helps stability for Rectifier
  )               
  return(list(iterNo=iter, AUC=h2o.auc(dl_CIE, train=TRUE), modelInfo=dl_CIE))
}

iter_algo_dl_CIE_results <- map(1:1000,iter_algo_dl_CIE)

#max(unlist(lapply(iter_algo_dl_CIE_results,"[[",n=2)))
# max(unlist(iter_algo_dl_CIE_results %>% map(2)))
# iter_algo_dl_CIE_results %>% map(2) %>% unlist %>% max

auc_all_CIE <- map_dbl(iter_algo_dl_CIE_results, "AUC")
summary(auc_all_CIE)
hist(auc_all_CIE)
dl_best_model_CIE <- iter_algo_dl_CIE_results[[which.max(auc_all_CIE)]]$modelInfo


# save the model
h2o.saveModel(dl_best_model_CIE, path="./best_iter_model", force=TRUE)

# load the previously saved mdoel
dl_model_loaded_CIE <- h2o.loadModel("best_iter_model\\DeepLearning_model_R_1536154762111_2258")


#plot(h2o.performance(dl_model_loaded_CIE))

# training data
tag_CIE = as.vector(train$CIEFL)
score_CIE = as.vector(h2o.predict(dl_model_loaded_CIE, as.h2o(train))$Y)
pROC::roc(tag_CIE, score_CIE, ci=TRUE)
mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_dl_full.png")


# validation data
tag_CIE_v = as.vector(valid$CIEFL)
score_CIE_v = as.vector(h2o.predict(dl_model_loaded_CIE, as.h2o(valid))$Y)
pROC::roc(tag_CIE_v, score_CIE_v, ci=TRUE)
mplot_full(tag = tag_CIE_v, 
           score = score_CIE_v,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_dl_full_v.png")


# test data
tag_CIE2 = as.vector(test$CIEFL)
score_CIE2 = as.vector(h2o.predict(dl_model_loaded_CIE, as.h2o(test))$Y)
pROC::roc(tag_CIE2, score_CIE2, ci=TRUE)
mplot_full(tag = tag_CIE2, 
           score = score_CIE2,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_dl_full2.png")


#prediction on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX=="Male","M","F")
#names(bmk_extra)
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID=="4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID=="4590",]

# use one of them
bmk_extra <- bmk_extra
bmk_extra <- bmk_extra[bmk_extra$group=="NEOPHYTE",]
bmk_extra <- bmk_extra_CR4550
bmk_extra <- bmk_extra_CR4590

bmk_extra2 <- bmk_extra[,c(5,6,12:14,17:28)]
bmk_extra2 <- bmk_extra2[,c(3:17,1,2)]
#names(bmk_extra2)
names(bmk_extra2) <- c("AEFL","CIEFL", "SSCIEFL","LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                       "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO","AGE","SEX")


tag_CIE3 = as.vector(bmk_extra2$CIEFL)
score_CIE3 = as.vector(h2o.predict(dl_model_loaded_CIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
pROC::roc(tag_CIE3, score_CIE3, ci=TRUE)
mplot_full(tag = tag_CIE3, 
           score = score_CIE3,
           subtitle = "Distribution by CIE group - extra data",
           save = TRUE,
           file_name = "CIE_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)








# iterative Model for SSCIE by DL============================================================

h2o.init(nthreads=-1, max_mem_size="12G")
#h2o.shutdown()
h2o.removeAll()

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))

#split the data
splits <- h2o.splitFrame(bmk, c(.6,.1), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex")
test  <- h2o.assign(splits[[3]], "test.hex")


#predictors <- c("LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA23") 
predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL", "KDA88", "AGE", "SEX"))



iter_algo_dl_SSCIE <- function(iter)
{
  #train2 <- h2o.splitFrame(train, 0.4)[[1]]
  dl_SSCIE <- h2o.deeplearning(
    #model_id = "dl_bmk_SSCIE", 
    #checkpoint = "dl_check_SSCIE", 
    training_frame = train, 
    validation_frame=valid, 
    x = predictors, 
    y = "SSCIEFL", 
    hidden = c(300),            ## more hidden layers -> more complex interactions
    epochs = 100,                     ## hopefully long enough to converge (otherwise restart again)
    stopping_metric = "logloss",     ## logloss is directly optimized by Deep Learning
    stopping_tolerance = 1e-2,         ## stop when validation logloss does not improve by >=1% for 2 scoring events
    stopping_rounds = 2,
    #score_validation_samples=10000, ## downsample validation set for faster scoring
    score_duty_cycle = 0.025,          ## don't score more than 2.5% of the wall time
    adaptive_rate = FALSE,            ## manually tuned learning rate
    rate = 0.01, 
    rate_annealing = 2e-6,            
    momentum_start = 0.2,            ## manually tuned momentum
    momentum_stable = 0.4, 
    momentum_ramp = 1e7,
    #input_dropout_ratio = .2,
    #hidden_dropout_ratios = .7,
    l1 = 1e-5,                       ## add some L1/L2 regularization
    l2 = 1e-5,
    activation = c("MaxoutWithDropout"),
    max_w2 = 10                       ## helps stability for Rectifier
  )               
  return(list(iterNo=iter, AUC=h2o.auc(dl_SSCIE, train=TRUE), modelInfo=dl_SSCIE))
}

iter_algo_dl_SSCIE_results <- map(1:1000,iter_algo_dl_SSCIE)

#max(unlist(lapply(iter_algo_dl_SSCIE_results,"[[",n=2)))
# max(unlist(iter_algo_dl_SSCIE_results %>% map(2)))
# iter_algo_dl_SSCIE_results %>% map(2) %>% unlist %>% max

auc_all_SSCIE <- map_dbl(iter_algo_dl_SSCIE_results, "AUC")
summary(auc_all_SSCIE)
hist(auc_all_SSCIE)
dl_best_model_SSCIE <- iter_algo_dl_SSCIE_results[[which.max(auc_all_SSCIE)]]$modelInfo


# save the model
h2o.saveModel(dl_best_model_SSCIE, path="./best_iter_model", force=TRUE)

# load the previously saved mdoel
dl_model_loaded_SSCIE <- h2o.loadModel("best_iter_model\\DeepLearning_model_R_1536163975668_8357")


#plot(h2o.performance(dl_model_loaded_SSCIE))

# training data
tag_SSCIE = as.vector(train$SSCIEFL)
score_SSCIE = as.vector(h2o.predict(dl_model_loaded_SSCIE, as.h2o(train))$Y)
pROC::roc(tag_SSCIE, score_SSCIE, ci=TRUE)
mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_dl_full.png")


# validation data
tag_SSCIE_v = as.vector(valid$SSCIEFL)
score_SSCIE_v = as.vector(h2o.predict(dl_model_loaded_SSCIE, as.h2o(valid))$Y)
pROC::roc(tag_SSCIE_v, score_SSCIE_v, ci=TRUE)
mplot_full(tag = tag_SSCIE_v, 
           score = score_SSCIE_v,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_dl_full_v.png")


# test data
tag_SSCIE2 = as.vector(test$SSCIEFL)
score_SSCIE2 = as.vector(h2o.predict(dl_model_loaded_SSCIE, as.h2o(test))$Y)
pROC::roc(tag_SSCIE2, score_SSCIE2, ci=TRUE)
mplot_full(tag = tag_SSCIE2, 
           score = score_SSCIE2,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_dl_full2.png")


#prediction on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX=="Male","M","F")
#names(bmk_extra)
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID=="4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID=="4590",]

# use one of them
bmk_extra <- bmk_extra
bmk_extra <- bmk_extra[bmk_extra$group=="NEOPHYTE",]
bmk_extra <- bmk_extra_CR4550
bmk_extra <- bmk_extra_CR4590

bmk_extra2 <- bmk_extra[,c(5,6,12:14,17:28)]
bmk_extra2 <- bmk_extra2[,c(3:17,1,2)]
#names(bmk_extra2)
names(bmk_extra2) <- c("AEFL","CIEFL", "SSCIEFL","LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                       "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO","AGE","SEX")


tag_SSCIE3 = as.vector(bmk_extra2$SSCIEFL)
score_SSCIE3 = as.vector(h2o.predict(dl_model_loaded_SSCIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
pROC::roc(tag_SSCIE3, score_SSCIE3, ci=TRUE)
mplot_full(tag = tag_SSCIE3, 
           score = score_SSCIE3,
           subtitle = "Distribution by SSCIE group - extra data",
           save = TRUE,
           file_name = "SSCIE_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)






