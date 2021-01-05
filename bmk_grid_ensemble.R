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


h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
#dim(bmk)
#bmk <- as.data.frame(bmk)



#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# Model for AE============================================================

###########################################################################################################
# Deep learning models
###########################################################################################################
#### Random Hyper-Parameter Search
hyper_params_bmk_AE <- list(
  activation = c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden = list(c(20,20),c(30,30),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio = c(0,0.05),
  l1 = seq(0,1e-4,1e-6),
  l2 = seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_AE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                              max_models = 100, seed = 1234567, stopping_rounds = 5, 
                              stopping_tolerance = 1e-2)


dl_random_grid_bmk_AE <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random_bmk_AE",
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = "AEFL",
  epochs = 10,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  #score_validation_samples=33, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_AE,
  search_criteria = search_criteria_bmk_AE,
  nfolds = 5,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  seed = 123
)                                
grid_AE <- h2o.getGrid("dl_grid_random_bmk_AE", sort_by = "auc", decreasing = TRUE)

#grid_AE@summary_table[1,]
best_dl_model_AE <- h2o.getModel(grid_AE@model_ids[[1]]) 
#best_model_AE
plot(best_dl_model_AE)
h2o.auc(best_dl_model_AE, train = TRUE, valid = TRUE, xval = TRUE)


# save the model
h2o.saveModel(best_dl_model_AE, path = "./grid_ensemble_models", force = TRUE)

# load the previously saved mdoel
model_dl_loaded_AE <- h2o.loadModel("grid_ensemble_models\\dl_grid_random_bmk_AE_model_20")

h2o.auc(model_dl_loaded_AE, train = TRUE, valid = TRUE, xval = TRUE)



###########################################################################################################
# gbm model
###########################################################################################################


hyper_params_gbm_AE <- list(learn_rate = c(0.01, 0.05, 0.1),
                            learn_rate_annealing = c(.99, 1),
                            min_rows = c(1, 5, 10),
                            max_depth = c(1, 3, 5),
                            sample_rate = c(.5, .75, 1),
                            col_sample_rate = c(.8, .9, 1))

search_criteria_gbm_AE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                              max_models = 100, seed = 1234567, stopping_rounds = 5, 
                              stopping_tolerance = 1e-2)

gbm_grid_AE <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid_binomial",
  x = predictors,
  y = "AEFL",
  training_frame = train,
  validation_frame = valid,
  distribution = "bernoulli",
  ntrees = 10,
  stopping_metric = "logloss",
  nfolds = 30,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  hyper_params = hyper_params_gbm_AE,
  search_criteria = search_criteria_gbm_AE,
  seed = 1 
)

best_gbm_model_AE <- h2o.getModel(gbm_grid_AE@model_ids[[1]]) ## model with lowest logloss
#best_model_AE
plot(best_gbm_model_AE)

h2o.auc(best_gbm_model_AE, train = TRUE, valid = TRUE, xval = TRUE)

# save the model
h2o.saveModel(best_gbm_model_AE, path = "./grid_ensemble_models", force = TRUE)

# load the previously saved mdoel
model_gbm_loaded_AE <- h2o.loadModel("grid_ensemble_models\\gbm_grid_binomial_model_162")

h2o.auc(model_gbm_loaded_AE, train = TRUE, valid = TRUE, xval = TRUE)





models_list <- list(dl_grid_load_AE, model_gbm_loaded_AE)

# ensemble models
grid_ensemble_AE <- h2o.stackedEnsemble(
  x = predictors,
  y = "AEFL",
  training_frame = train,
  validation_frame = valid,
  metalearner_algorithm = "AUTO",
  metalearner_nfolds = 5,
  metalearner_fold_assignment = "Modulo",
  seed = 1,
  base_models = models_list
)

h2o.auc(grid_ensemble_AE, train = TRUE, valid = TRUE, xval = TRUE)







#shutdown H2O
h2o.shutdown(prompt = FALSE)


