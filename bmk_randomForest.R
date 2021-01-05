getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
source("all_plots_onepage.R")

#install.packages("arulesViz")
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


# # random forest
# h2o.init(nthreads=-1, max_mem_size="12G")
# h2o.removeAll() ## clean slate - just in case the cluster was already running
# 
# bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
# dim(bmk)
# #bmk <- as.data.frame(bmk)
# 
# 
# #split the data
# #splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
# splits <- h2o.splitFrame(bmk, 0.8, seed=1234)
# train  <- h2o.assign(splits[[1]], "train.hex") 
# valid  <- h2o.assign(splits[[2]], "valid.hex") 
# #test   <- h2o.assign(splits[[3]], "test.hex")
# 
# predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
# #predictors
# 
# AE_rf <- h2o.randomForest(        
#   training_frame = train,       
#   validation_frame = valid,     
#   x=predictors,                       
#   y="AEFL",                         
#   model_id = "rf_covType2",
#   nfolds=10,
#   ntrees = 200,                 
#   max_depth = 50,                
#   stopping_rounds = 2,          
#   stopping_tolerance = 1e-2,    
#   score_each_iteration = TRUE,     
#   seed=3000000)   
# 
# summary(AE_rf)
# # AE_rf@model$validation_metrics
# # h2o.hit_ratio_table(AE_rf,valid = TRUE)[1,2]
# 
# plot(AE_rf)
# 
# #shutdown H2O
# h2o.shutdown(prompt=FALSE)


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
  hidden=list(c(20,20),c(30,30),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_AE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                           max_models = 100, seed=1234567, stopping_rounds=5, 
                           stopping_tolerance=1e-2)


dl_random_grid_bmk_AE <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random_bmk_AE",
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
  nfolds=30,
  fold_assignment="Modulo",
  keep_cross_validation_predictions = TRUE
)                                
grid_AE <- h2o.getGrid("dl_grid_random_bmk_AE",sort_by="auc",decreasing=TRUE)

#grid_AE@summary_table[1,]
best_model_AE <- h2o.getModel(grid_AE@model_ids[[1]]) ## model with lowest logloss
#best_model_AE
plot(best_model_AE)

# m <- h2o.getModel(grid_AE@model_ids[[1]])
# h2o.varimp(m)

# prediction
# pred_AE <- h2o.predict(best_model_AE, valid)
# pred_AE
# test$Accuracy_AE <- pred_AE$predict == test$AEFL
# 1-mean(test$Accuracy_AE)

# save the model
#h2o.saveModel(best_model_AE, path="./best_model_AE", force=TRUE)

# load the previously saved mdoel
model_loaded_AE <- h2o.loadModel("Z:\\X-Drive\\Common\\MHasan\\ML\\best_model_AE\\dl_grid_random_bmk_AE_model_36")
# summary(model_loaded_AE)

cvpreds_id_AE <- model_loaded_AE@model$cross_validation_holdout_predictions_frame_id$name
cvpreds_AE <- h2o.getFrame(cvpreds_id_AE)

# pred_AE <- h2o.predict(model_loaded_AE, bmk)[,3]
# h2o.make_metrics(pred_AE,bmk$AEFL)
# fpr = model_loaded_AE@model$training_metrics@metrics$thresholds_and_metric_scores$fpr
# tpr = model_loaded_AE@model$training_metrics@metrics$thresholds_and_metric_scores$tpr
# fpr_val = model_loaded_AE@model$validation_metrics@metrics$thresholds_and_metric_scores$fpr
# tpr_val = model_loaded_AE@model$validation_metrics@metrics$thresholds_and_metric_scores$tpr
# plot(fpr,tpr, type='l')
# title('AUC')
# lines(fpr_val,tpr_val,type='l',col='red')
# legend("bottomright",c("Train", "Validation"),col=c("black","red"),lty=c(1,1),lwd=c(3,3))




# # see best parameters
# best_params_AE <- model_loaded_AE@allparameters
# best_params_AE$activation
# best_params_AE$hidden
# best_params_AE$epochs
# best_params_AE$input_dropout_ratio
# best_params_AE$l1
# best_params_AE$l2

tag_AE = as.vector(train$AEFL)
score_AE = as.vector(cvpreds_AE$Y)

# cv_auc <- model_loaded_AE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_AE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_AE@algorithm)

# density plots--------
mplot_density(tag = tag_AE, 
                   score = score_AE, 
                   #model_name = "Deeplearning", 
                   subtitle = "Distribution by AE group", 
                   save = TRUE, 
                   file_name = "AE_viz_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_AE, 
          score = score_AE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "AE_viz_roc.png")


mplot_cuts(score = score_AE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "AE_viz_ncuts.png")

mplot_splits(tag = tag_AE, 
             score = score_AE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "AE_viz_splits.png")


mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Area Under ROC Curve",
           save = TRUE,
           file_name = "AE_viz_full.png")



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
  nfolds=30,
  fold_assignment="Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123
)                                
grid_CIE <- h2o.getGrid("dl_grid_random_bmk_CIE",sort_by="auc",decreasing=TRUE)

#grid_CIE@summary_table[1,]
best_model_CIE <- h2o.getModel(grid_CIE@model_ids[[1]]) ## model with lowest logloss
#best_model_CIE
plot(best_model_CIE)

# save the model
h2o.saveModel(best_model_CIE, path="./best_model_CIE", force=TRUE)

# load the previously saved mdoel
# print(path_CIE)
model_loaded_CIE <- h2o.loadModel("Z:\\X-Drive\\Common\\MHasan\\ML\\best_model_CIE\\dl_grid_random_bmk_CIE_model_22")
# summary(model_loaded_CIE)


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


tag_CIE = as.vector(train$CIEFL)
score_CIE = as.vector(cvpreds_CIE$Y)

# cv_auc <- model_loaded_CIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_CIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_CIE@algorithm)

# density plots--------
mplot_density(tag = tag_CIE, 
              score = score_CIE, 
              #model_name = "Deeplearning", 
              subtitle = "Distribution by CIE group", 
              save = TRUE, 
              file_name = "CIE_viz_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_CIE, 
          score = score_CIE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "CIE_viz_roc.png")


mplot_cuts(score = score_CIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "CIE_viz_ncuts.png")

mplot_splits(tag = tag_CIE, 
             score = score_CIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "CIE_viz_splits.png")


mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Area Under ROC Curve",
           save = TRUE,
           file_name = "CIE_viz_full.png")



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
  nfolds=30,
  fold_assignment="Modulo",
  keep_cross_validation_predictions = TRUE
)                                
grid_SSCIE <- h2o.getGrid("dl_grid_random_bmk_SSCIE",sort_by="auc",decreasing=TRUE)

#grid_SSCIE@summary_table[1,]
best_model_SSCIE <- h2o.getModel(grid_SSCIE@model_ids[[1]]) ## model with lowest logloss
#best_model_SSCIE
plot(best_model_SSCIE)
# save the model
h2o.saveModel(best_model_SSCIE, path="./best_model_SSCIE", force=TRUE)

# load the previously saved mdoel
# print(path_SSCIE)
model_loaded_SSCIE <- h2o.loadModel("Z:\\X-Drive\\Common\\MHasan\\ML\\best_model_SSCIE\\dl_grid_random_bmk_SSCIE_model_16")
# summary(model_loaded_SSCIE)


cvpreds_id_SSCIE <- model_loaded_SSCIE@model$cross_validation_holdout_predictions_frame_id$name
cvpreds_SSCIE <- h2o.getFrame(cvpreds_id_SSCIE)


tag_SSCIE = as.vector(train$SSCIEFL)
score_SSCIE = as.vector(cvpreds_SSCIE$Y)

# cv_auc <- model_loaded_SSCIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_SSCIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_SSCIE@algorithm)

# density plots--------
mplot_density(tag = tag_SSCIE, 
              score = score_SSCIE, 
              #model_name = "Deeplearning", 
              subtitle = "Distribution by SSCIE group", 
              save = TRUE, 
              file_name = "SSCIE_viz_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_SSCIE, 
          score = score_SSCIE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "SSCIE_viz_roc.png")


mplot_cuts(score = score_SSCIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "SSCIE_viz_ncuts.png")

mplot_splits(tag = tag_SSCIE, 
             score = score_SSCIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "SSCIE_viz_splits.png")


mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "SSCIE Distribution by group",
           save = TRUE,
           file_name = "SSCIE_viz_full.png")





#shutdown H2O
h2o.shutdown(prompt=FALSE)


########################################################################################################
#===========================================================================

# aml_AE <- h2o.automl(x = predictors, y = 'AEFL',
#                   training_frame = train,
#                   validation_frame = test,
#                   nfolds = 10,
#                   max_models = 10,
#                   seed = 1234)
# 
# lb <- aml_AE@leaderboard
# print(lb, n = nrow(lb))











































