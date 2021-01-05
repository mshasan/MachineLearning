getwd()
setwd("C:/Users/mhasan7/mhasan7/ML")
source("all_plots_onepage.R")
source("h2o_automl2.R")

#install.packages("arulesViz")
library(arulesViz)
library(devtools)
# use this to get latest version of h2o
#install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
library(h2o)
library(lares)
library(mlbench)
library(magrittr)   # for %>% operator
#library(plyr)       # rename function
library(dplyr)       # rename function
library(pROC)
library(beepr)
library(ggplot2)
library(gridExtra)
library(readr)


######################################################################################################
## Model for AE==========================================================================

h2o.init(nthreads = -1, max_mem_size = "12G")
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
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors


gbm_AE <- h2o.gbm(
  training_frame = train,     
  validation_frame = valid,   
  x = predictors,                     
  y = "AEFL",
  distribution = "bernoulli",
  ntrees = 30,
  min_rows = 5,
  learn_rate = .001,           
  max_depth = 8,             
  sample_rate = .9,          
  col_sample_rate = .7,      
  stopping_rounds = 2,        
  stopping_tolerance = .05,  
  score_each_iteration = TRUE,
  score_tree_interval = 5,
  model_id = "gbm_covType_AE",  
  seed = 1,
  nfolds = 5,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions =  TRUE
  )                 

plot(gbm_AE)
h2o.auc(gbm_AE, train = TRUE, valid = TRUE, xval = TRUE)

pred_gbm_AE <- h2o.predict(gbm_AE, newdata = valid)
tag_AE = as.vector(valid$AEFL)
score_ensem_AE = as.vector(pred_gbm_AE$Y)
pROC::roc(tag_AE, score_ensem_AE, ci = TRUE)



bmk_AE <- read_csv("base_bmk.csv")[,-c(2,3)]


df_AE <- bmk_AE %>% dplyr::rename("tag" = "AEFL")
results <- lares::h2o_automl(df_AE, seed = 1)
#results <- h2o_automl2(df_AE, seed = 1)
mplot_full(tag = results$scores$tag,
           score = results$scores$score,
           subtitle = "Distribution by AE group",
           model_name = results$model_name,
           save = TRUE,
           file_name = "AE_gbm_full.png")


best_autoML_AE <- h2o_selectmodel(results, which_model = 1)

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
                   file_name = "AE_gbm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_AE, 
          score = score_AE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "AE_gbm_roc.png")


mplot_cuts(score = score_AE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "AE_gbm_ncuts.png")

mplot_splits(tag = tag_AE, 
             score = score_AE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "AE_gbm_splits.png")


mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_gbm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)




######################################################################################################
## Model for CIE==========================================================================

bmk_CIE <- read_csv("base_bmk.csv")[,-c(1,3)]


df_CIE <- bmk_CIE %>% dplyr::rename("tag" = "CIEFL")
results <- lares::h2o_automl(df_CIE, seed = 1)
#results <- h2o_automl2(df_CIE, seed = 1)
mplot_full(tag = results$scores$tag,
           score = results$scores$score,
           subtitle = "Summary plots",
           model_name = results$model_name,
           save = TRUE,
           file_name = "CIE_gbm_full.png")


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
              file_name = "CIE_gbm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_CIE, 
          score = score_CIE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "CIE_gbm_roc.png")


mplot_cuts(score = score_CIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "CIE_gbm_ncuts.png")

mplot_splits(tag = tag_CIE, 
             score = score_CIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "CIE_gbm_splits.png")


mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "CIE_gbm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)





######################################################################################################
## Model for SSCIE==========================================================================


bmk_SSCIE <- read_csv("base_bmk.csv")[,-c(2,1)]


df_SSCIE <- bmk_SSCIE %>% dplyr::rename("tag" = "SSCIEFL")
results <- lares::h2o_automl(df_SSCIE, seed = 1)
#results <- h2o_automl2(df_SSCIE, seed = 1)
mplot_full(tag = results$scores$tag,
           score = results$scores$score,
           subtitle = "Summary plots",
           model_name = results$model_name,
           save = TRUE,
           file_name = "SSCIE_gbm_full.png")


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
              file_name = "SSCIE_gbm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_SSCIE, 
          score = score_SSCIE, 
          #model_name = "Deeplearning", 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "SSCIE_gbm_roc.png")


mplot_cuts(score = score_SSCIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "SSCIE_gbm_ncuts.png")

mplot_splits(tag = tag_SSCIE, 
             score = score_SSCIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "SSCIE_gbm_splits.png")


mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "SSCIE_gbm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)

########################################################################################################
#===========================================================================

# aml_SSCIE <- h2o.automl(x = predictors, y = 'SSCIEFL',
#                   training_frame = train,
#                   validation_frame = test,
#                   nfolds = 10,
#                   max_models = 10,
#                   seed = 1234)
# 
# lb <- aml_SSCIE@leaderboard
# print(lb, n = nrow(lb))











































