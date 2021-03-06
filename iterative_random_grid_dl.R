getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
#setwd("Z:/Projects/5201/PROD/Post-Hoc/Program/Analysis/Machine Learning")
source("all_plots_onepage.R")

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
library(plyr)       # rename function
library(plyr)       # rename function
library(pROC)
library(beepr)
library(ggplot2)
library(gridExtra)



######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
dim(bmk)
#bmk <- as.data.frame(bmk)



# par(mfrow=c(2,2)) #set up the canvas for 2x2 plots
# model_bmk <- h2o.deeplearning(4:17,1,bmk,epochs=1e5)
# summary(model_bmk)
#plot(bmk[,c(4,6)],pch=19,col=bmk[,1],cex=0.5)

#split the data
#splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# Model for AE============================================================
#### Random Hyper-Parameter Search
hyper_params_bmk_AE <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(30,30),c(30,30,30)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_AE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                              max_models = 100, seed=1234567, stopping_rounds=5, 
                              stopping_tolerance=1e-2)


iter_randGrid_dl_AE <- function(iter)
{
  splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
  train  <- h2o.assign(splits[[1]], "train.hex") 
  valid  <- h2o.assign(splits[[2]], "valid.hex")
  
  dl_random_grid_bmk_AE <- h2o.grid(
    algorithm="deeplearning",
    grid_id = "dl_grid_rand_AE",
    training_frame=train,
    validation_frame=valid, 
    x=predictors, 
    y="AEFL",
    epochs=1,
    stopping_metric="logloss",
    stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
    stopping_rounds=2,
    #score_validation_samples=33, ## downsample validation set for faster scoring
    score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
    max_w2=10,                      ## can help improve stability for Rectifier
    hyper_params = hyper_params_bmk_AE,
    search_criteria = search_criteria_bmk_AE,
    nfolds=3,
    rate = 0.01, 
    rate_annealing = 2e-6,            
    momentum_start = 0.2,            ## manually tuned momentum
    momentum_stable = 0.4, 
    momentum_ramp = 1e7,
    fold_assignment="Modulo",
    keep_cross_validation_predictions = TRUE
  )                                
grid_AE <- h2o.getGrid("dl_grid_rand_AE",sort_by="auc",decreasing=TRUE)
best_model_perIter_AE <- h2o.getModel(grid_AE@model_ids[[1]])

return(list(iterNo=iter, 
            AUC=h2o.auc(best_model_perIter_AE, train=TRUE, valid=TRUE, xval=TRUE),
            modelInfo=best_model_perIter_AE))
}


iter_randGrid_dl_AE_results <- map(1:100,iter_randGrid_dl_AE)

auc_all_AE <- as.tibble(t(map_dfc(iter_randGrid_dl_AE_results, "AUC")))
names(auc_all_AE) <- c("train","valid","xval")

summary(auc_all_AE)
hist(auc_all_AE$xval)
dl_best_IterRandGrid_model_AE <- iter_randGrid_dl_AE_results[[which.max(auc_all_AE$xval)]]$modelInfo


# save the model
h2o.saveModel(dl_best_IterRandGrid_model_AE, path="./best_IterRandGrid_model", force=TRUE)


# load the previously saved mdoel
dl_IterRandGrid_model_loaded_AE <- h2o.loadModel("best_IterRandGrid_model\\dl_grid_rand_AE_model_114")

h2o.cross_validation_predictions(dl_IterRandGrid_model_loaded_AE)

h2o.getFrame(dl_IterRandGrid_model_loaded_AE@model[["cross_validation_holdout_predictions_frame_id"]][["name"]])


mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_dl_full.png")











#prediction of AE on new data================
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
score_AE3 = as.vector(predict(model_loaded_AE, as.h2o(bmk_extra2[,c(-3,-2)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci=TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)




######################################################################################################
## Model for CIE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
dim(bmk)
#bmk <- as.data.frame(bmk)



# par(mfrow=c(2,2)) #set up the canvas for 2x2 plots
# model_bmk <- h2o.deeplearning(4:17,1,bmk,epochs=1e5)
# summary(model_bmk)
#plot(bmk[,c(4,6)],pch=19,col=bmk[,1],cex=0.5)

#split the data
#splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors





## Model for CIE==========================================================================
hyper_params_bmk_CIE <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(30,30), c(30,30,30),c(30,30,30,30)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_CIE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                               max_models = 100, seed=1234567, stopping_rounds=5, 
                               stopping_tolerance=1e-2)

dl_random_grid_bmk_CIE <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random_bmk_CIE",
  training_frame=train,
  validation_frame=valid, 
  x=predictors, 
  y="CIEFL",
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  #score_validation_samples=133, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_CIE,
  search_criteria = search_criteria_bmk_CIE,
  nfolds=10,
  fold_assignment="Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123
)                                
grid_CIE <- h2o.getGrid("dl_grid_random_bmk_CIE",sort_by="auc",decreasing=TRUE)

#grid_CIE@summary_table[1,]
best_model_CIE <- h2o.getModel(grid_CIE@model_ids[[1]]) ## model with lowest logloss
#best_model_CIE
plot(best_model_CIE)
h2o.auc(best_model_CIE, train=TRUE, valid=TRUE, xval=TRUE)

# save the model
h2o.saveModel(best_model_CIE, path="./best_model_CIE", force=TRUE)

# load the previously saved mdoel
# print(path_CIE)
model_loaded_CIE <- h2o.loadModel("best_model_CIE\\dl_grid_random_bmk_CIE_model_22")
# summary(model_loaded_CIE)
h2o.auc(model_loaded_CIE, train=TRUE, valid=TRUE, xval=TRUE)

# # see best parameters
# best_params_CIE <- model_loaded_CIE@allparameters
# best_params_CIE$activation
# best_params_CIE$hidden
# best_params_CIE$epochs
# best_params_CIE$input_dropout_ratio
# best_params_CIE$l1
# best_params_CIE$l2


cvpreds_id_CIE <- model_loaded_CIE@model$cross_validation_holdout_predictions_frame_id$name
cvpreds_CIE <- h2o.getFrame(cvpreds_id_CIE)

# nfolds <- model_loaded_CIE@parameters$nfolds
# predlist <- sapply(1:nfolds, function(v) h2o.getFrame(model_loaded_CIE@model$cross_validation_predictions[[v]]$name)$predict, simplify = FALSE)
# cvpred_sparse <- h2o.cbind(predlist)  # N x V Hdf with rows that are all zeros, except corresponding to the v^th fold if that rows is associated with v
# pred <- apply(cvpred_sparse, 1, sum)


tag_CIE = as.vector(train$CIEFL)
score_CIE = as.vector(cvpreds_CIE$Y)
write.csv(data.frame(tag_CIE, score_CIE), file = "tag_score_CIE.csv")

# cv_auc <- model_loaded_CIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_CIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_CIE@algorithm)

mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_dl_full.png")




#prediction of CIE on new data================
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
score_CIE3 = as.vector(predict(model_loaded_CIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
pROC::roc(tag_CIE3, score_CIE3, ci=TRUE)
mplot_full(tag = tag_CIE3, 
           score = score_CIE3,
           subtitle = "Distribution by SIE group - extra data",
           save = TRUE,
           file_name = "SIE_dl_full3.png")








#shutdown H2O
h2o.shutdown(prompt=FALSE)






######################################################################################################
## Model for SSCIE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
dim(bmk)
#bmk <- as.data.frame(bmk)



# par(mfrow=c(2,2)) #set up the canvas for 2x2 plots
# model_bmk <- h2o.deeplearning(4:17,1,bmk,epochs=1e5)
# summary(model_bmk)
#plot(bmk[,c(4,6)],pch=19,col=bmk[,1],cex=0.5)

#split the data
#splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors




## Model for SSCIE==========================================================================
hyper_params_bmk_SSCIE <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(30,30,30),c(30,30,30,30),c(30,30,30,30,30)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_SSCIE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                                 max_models = 100, seed=1652483, stopping_rounds=5, 
                                 stopping_tolerance=1e-2)

dl_random_grid_bmk_SSCIE <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random_bmk_SSCIE",
  training_frame=train,
  validation_frame=valid, 
  x=predictors, 
  y="SSCIEFL",
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=133, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_SSCIE,
  search_criteria = search_criteria_bmk_SSCIE,
  nfolds=10,
  fold_assignment="Modulo",
  keep_cross_validation_predictions = TRUE
)                                
grid_SSCIE <- h2o.getGrid("dl_grid_random_bmk_SSCIE",sort_by="auc",decreasing=TRUE)

#grid_SSCIE@summary_table[1,]
best_model_SSCIE <- h2o.getModel(grid_SSCIE@model_ids[[1]]) ## model with lowest logloss
#best_model_SSCIE
plot(best_model_SSCIE)
h2o.auc(best_model_SSCIE, train=TRUE, valid=TRUE, xval=TRUE)

# save the model
h2o.saveModel(best_model_SSCIE, path="./best_model_SSCIE", force=TRUE)

# load the previously saved mdoel
# print(path_SSCIE)
model_loaded_SSCIE <- h2o.loadModel("best_model_SSCIE\\dl_grid_random_bmk_SSCIE_model_33")
# summary(model_loaded_SSCIE)
h2o.auc(model_loaded_SSCIE, train=TRUE, valid=TRUE, xval=TRUE)


cvpreds_id_SSCIE <- model_loaded_SSCIE@model$cross_validation_holdout_predictions_frame_id$name
cvpreds_SSCIE <- h2o.getFrame(cvpreds_id_SSCIE)


tag_SSCIE = as.vector(train$SSCIEFL)
score_SSCIE = as.vector(cvpreds_SSCIE$Y)
write.csv(data.frame(tag_SSCIE, score_SSCIE), file = "tag_score_SSCIE.csv")

# cv_auc <- model_loaded_SSCIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_SSCIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_SSCIE@algorithm)

mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_dl_full.png")





#prediction of SSCIE on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX=="Male","M","F")
names(bmk_extra)
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID=="4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID=="4590",]

# use one of them
bmk_extra <- bmk_extra
bmk_extra <- bmk_extra[bmk_extra$group=="NEOPHYTE",]
bmk_extra <- bmk_extra_CR4550
bmk_extra <- bmk_extra_CR4590

bmk_extra2 <- bmk_extra[,c(5,6,12:14,17:28)]
bmk_extra2 <- bmk_extra2[,c(3:17,1,2)]
names(bmk_extra2)
names(bmk_extra2) <- c("AEFL","CIEFL", "SSCIEFL","LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                       "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO","AGE","SEX")


tag_SSCIE3 = as.vector(bmk_extra2$SSCIEFL)
score_SSCIE3 = as.vector(predict(model_loaded_SSCIE, as.h2o(bmk_extra2[,c(-1,-2)]))$Y)
pROC::roc(tag_SSCIE3, score_SSCIE3, ci=TRUE)
mplot_full(tag = tag_SSCIE3, 
           score = score_SSCIE3,
           subtitle = "Distribution by SSCIE group - extra data",
           save = TRUE,
           file_name = "SSCIE_dl_full3.png")






#shutdown H2O
h2o.shutdown(prompt=FALSE)


########################################################################################################









































