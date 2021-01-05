getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
#setwd("Z:/Projects/5201/PROD/Post-Hoc/Program/Analysis/Machine Learning")
source("all_plots_onepage.R")
source("feature_selection.R")

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
library(haven)
library(cowplot)


######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
dim(bmk)



#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


# # Add Features by categorizing continuous variables
# data_AE = list(train, valid)
# data_AE_ext <- add_features(data_AE)
# data_AE_ext$Train <- h2o.assign(data_AE_ext$Train, "train_b_ext")
# data_AE_ext$Valid <- h2o.assign(data_AE_ext$Valid, "valid_b_ext")


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# Model for AE============================================================
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
  grid_id = "dl_grid_random_AE",
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = "AEFL",
  epochs = 10,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  #stopping_rounds = 2,
  #score_validation_samples=33, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_AE,
  search_criteria = search_criteria_bmk_AE,
  nfolds = 30,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  seed = 123
)                


# best grid seach model performance
#grid_AE <- h2o.getGrid("dl_grid_random_AE", sort_by = "auc", decreasing = TRUE)
grid_AE <- h2o.getGrid("dl_grid_random_AE", sort_by = "logloss", decreasing = FALSE)
dl_grid_AE <- h2o.getModel(grid_AE@model_ids[[1]]) 
h2o.auc(dl_grid_AE, train = TRUE, valid = TRUE, xval = TRUE)

# save grid search model
h2o.saveModel(dl_grid_AE, path = "./dl_models", force = TRUE)

# load the previously grid search saved mdoels
#dl_models\\dl_grid_random_AE_model_67
#dl_models/dl_grid_random_AE_model_9
#dl_models/dl_grid_random_AE_model_70
dl_grid_load_AE <- h2o.loadModel("dl_models\\dl_grid_random_AE_model_46")
h2o.auc(dl_grid_load_AE, train = TRUE, valid = TRUE, xval = TRUE)


# best grid search models are loaded to assemble
baseModels <- list(
  dl_grid_load_AE1 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_0"),
  dl_grid_load_AE2 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_6"),
  dl_grid_load_AE3 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_9"),
  dl_grid_load_AE4 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_16"),
  dl_grid_load_AE5 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_35"),
  dl_grid_load_AE6 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_38"),
  dl_grid_load_AE7 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_52"),
  dl_grid_load_AE8 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_53"),
  dl_grid_load_AE9 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_61"),
  dl_grid_load_AE10 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_67"),
  dl_grid_load_AE11 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_70"),
  dl_grid_load_AE12 = h2o.loadModel("dl_models\\dl_grid_random_AE_model_74")
)

# ensemble the best grid searched models
dl_ensemble_AE <- h2o.stackedEnsemble(
  x = predictors,
  y = "AEFL",
  training_frame = train,
  validation_frame = valid,
  metalearner_algorithm = "AUTO",
  metalearner_nfolds = 30,
  metalearner_fold_assignment = "Modulo",
  #base_models = dl_random_grid_bmk_AE@model_ids,
  seed = 123,
  base_models = baseModels
)

# Evaluate ensemble model performance 
h2o.auc(h2o.performance(dl_ensemble_AE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_AE, newdata = valid))

# save the esemble model
h2o.saveModel(dl_ensemble_AE, path = "./dl_models", force = TRUE)

# load the previously saved esemble mdoel
# dl_models\\StackedEnsemble_model_R_1538142377183_18179
# dl_models\\StackedEnsemble_model_R_1538164909720_18045
dl_ensemble_load_AE <- h2o.loadModel("dl_models\\StackedEnsemble_model_R_1538164909720_18045")

# verify ensemble performance 
h2o.auc(h2o.performance(dl_ensemble_load_AE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_load_AE, newdata = valid))


# training data
tag_AE = as.vector(train$AEFL)
score_AE = as.vector(h2o.predict(dl_ensemble_load_AE, as.h2o(train))$Y)
pROC::roc(tag_AE, score_AE, ci = TRUE)
mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_ensem_dl_full.png")


# validation data
tag_AE_v = as.vector(valid$AEFL)
score_AE_v = as.vector(h2o.predict(dl_ensemble_load_AE, as.h2o(valid))$Y)
pROC::roc(tag_AE_v, score_AE_v, ci = TRUE)
mplot_full(tag = tag_AE_v, 
           score = score_AE_v,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_ensem_dl_full_v.png")




#prediction of AE on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX == "Male","M","F")

#names(bmk_extra)
bmk_extra_neo    <- bmk_extra[bmk_extra$group == "NEOPHYTE",]
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID == "4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID == "4590",]


# data processing function
auc_test <- function(bmk_extra)
{
  bmk_extra_b <- bmk_extra[,c(5,6,12:14,17:28)]
  bmk_extra_b <- bmk_extra_b[,c(3:17,1,2)]
  names(bmk_extra_b) <- c("AEFL","CIEFL", "SSCIEFL",
                         "LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                         "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO",
                         "AGE","SEX")
 return(bmk_extra_b) 
}



bmk_extra2 <- auc_test(bmk_extra_CR4550)

tag_AE3 = as.vector(bmk_extra2$AEFL)
score_AE3 = as.vector(h2o.predict(dl_grid_load_AE, as.h2o(bmk_extra2[,c(-2,-3)]))$Y)
#score_AE3 = as.vector(h2o.predict(dl_ensemble_load_AE, as.h2o(bmk_extra2[,c(-2,-3)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci = TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_ensem_dl_full3.png")



# final plots dl_grid or dl_grid_ensemble
p_train <- mplot_density2(tag = tag_AE, score = score_AE, xlower = 70, title = "Training")
p_valid <- mplot_density2(tag = tag_AE_v, score = score_AE_v, xlower = 70, title = "Validation")
p_test  <- mplot_density2(tag = tag_AE3, score = score_AE3, xlower = 70, title = "Testing")

roc_train <- mplot_roc(tag = tag_AE,   score = score_AE)
roc_valid <- mplot_roc(tag = tag_AE_v, score = score_AE_v)
roc_test  <- mplot_roc(tag = tag_AE3,  score = score_AE3)

legend <- get_legend(p_train + 
                    theme(legend.direction = "horizontal", legend.position = "bottom"))
p_train = p_train + theme(legend.position = "none")
p_valid = p_valid + theme(legend.position = "none")
p_test  = p_test  + theme(legend.position = "none")


plots <- plot_grid(p_train, p_valid, p_test, roc_train, roc_valid, roc_test, 
                       labels = paste0(letters[1:3],')'), ncol = 3, align = 'hv')
title1 <- ggdraw() + draw_label("Summary Plots of Adverse Events")
plot_grid(title1, legend, plots, ncol = 1, rel_heights = c(.1, .1, 1))




#shutdown H2O
h2o.shutdown(prompt = FALSE)


##############################################################################################################

######################################################################################################
## Model for CIE==========================================================================


h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
#dim(bmk)



#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


# # Add Features by categorizing continuous variables
# data_CIE = list(train, valid)
# data_CIE_ext <- add_features(data_CIE)
# data_CIE_ext$Train <- h2o.assign(data_CIE_ext$Train, "train_b_ext")
# data_CIE_ext$Valid <- h2o.assign(data_CIE_ext$Valid, "valid_b_ext")


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# Model for CIE============================================================
#### Random Hyper-Parameter Search
hyper_params_bmk_CIE <- list(
  activation = c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden = list(c(30,30),c(30,30,30),c(30,30,30,30)),
  input_dropout_ratio = c(0,0.05),
  l1 = seq(0,1e-4,1e-6),
  l2 = seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_CIE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                               max_models = 100, seed = 1234567, stopping_rounds = 5, 
                               stopping_tolerance = 1e-2)



dl_random_grid_bmk_CIE <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random_CIE",
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = "CIEFL",
  epochs = 10,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  #stopping_rounds = 2,
  #score_validation_samples=33, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_CIE,
  search_criteria = search_criteria_bmk_CIE,
  nfolds = 30,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  seed = 123
)                



# best grid seach model performance
#grid_CIE <- h2o.getGrid("dl_grid_random_CIE", sort_by = "auc", decreasing = TRUE)
grid_CIE <- h2o.getGrid("dl_grid_random_CIE", sort_by = "logloss", decreasing = FALSE)
dl_grid_CIE <- h2o.getModel(grid_CIE@model_ids[[1]]) 
h2o.auc(dl_grid_CIE, train = TRUE, valid = TRUE, xval = TRUE)

# save grid search model
h2o.saveModel(dl_grid_CIE, path = "./dl_models", force = TRUE)

# load the previously grid search saved mdoels
#dl_models\\dl_grid_random_CIE_model_67
dl_grid_load_CIE <- h2o.loadModel("dl_models\\dl_grid_random_CIE_model_39")
h2o.auc(dl_grid_load_CIE, train = TRUE, valid = TRUE, xval = TRUE)


# best grid search models are loaded to assemble
baseModels <- list(
  dl_grid_load_CIE1 = h2o.loadModel("dl_models\\dl_grid_random_CIE_model_51"),
  dl_grid_load_CIE2 = h2o.loadModel("dl_models\\dl_grid_random_CIE_model_83"),
  dl_grid_load_CIE3 = h2o.loadModel("dl_models\\dl_grid_random_CIE_model_2"),
  dl_grid_load_CIE4 = h2o.loadModel("dl_models\\dl_grid_random_CIE_model_61"),
  dl_grid_load_CIE5 = h2o.loadModel("dl_models\\dl_grid_random_CIE_model_39")
)

# ensemble the best grid searched models
dl_ensemble_CIE <- h2o.stackedEnsemble(
  x = predictors,
  y = "CIEFL",
  training_frame = train,
  validation_frame = valid,
  metalearner_algorithm = "AUTO",
  metalearner_nfolds = 30,
  metalearner_fold_assignment = "Modulo",
  #base_models = dl_random_grid_bmk_CIE@model_ids,
  seed = 123,
  base_models = baseModels
)

# Evaluate ensemble model performance 
h2o.auc(h2o.performance(dl_ensemble_CIE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_CIE, newdata = valid))

# save the esemble model
h2o.saveModel(dl_ensemble_CIE, path = "./dl_models", force = TRUE)

# load the previously saved esemble mdoel
# dl_models\\StackedEnsemble_model_R_1538142377183_18179
# dl_models\\StackedEnsemble_model_R_1538164909720_18045
dl_ensemble_load_CIE <- h2o.loadModel("dl_models\\StackedEnsemble_model_R_1538169369925_18289")

# verify ensemble performance 
h2o.auc(h2o.performance(dl_ensemble_load_CIE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_load_CIE, newdata = valid))


# training data
tag_CIE = as.vector(train$CIEFL)
score_CIE = as.vector(h2o.predict(dl_ensemble_load_CIE, as.h2o(train))$Y)
pROC::roc(tag_CIE, score_CIE, ci = TRUE)
mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_ensem_dl_full.png")


# validation data
tag_CIE_v = as.vector(valid$CIEFL)
score_CIE_v = as.vector(h2o.predict(dl_ensemble_load_CIE, as.h2o(valid))$Y)
pROC::roc(tag_CIE_v, score_CIE_v, ci = TRUE)
mplot_full(tag = tag_CIE_v, 
           score = score_CIE_v,
           subtitle = "Distribution by CIE group",
           save = TRUE,
           file_name = "CIE_ensem_dl_full_v.png")




#prediction of CIE on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX == "Male","M","F")

#names(bmk_extra)
bmk_extra_neo    <- bmk_extra[bmk_extra$group == "NEOPHYTE",]
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID == "4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID == "4590",]


# data processing function
auc_test <- function(bmk_extra)
{
  bmk_extra_b <- bmk_extra[,c(5,6,12:14,17:28)]
  bmk_extra_b <- bmk_extra_b[,c(3:17,1,2)]
  names(bmk_extra_b) <- c("AEFL","CIEFL", "SSCIEFL",
                          "LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                          "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO",
                          "AGE","SEX")
  return(bmk_extra_b) 
}



bmk_extra2 <- auc_test(bmk_extra)

tag_CIE3 = as.vector(bmk_extra2$CIEFL)
score_CIE3 = as.vector(h2o.predict(dl_grid_load_CIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
#score_CIE3 = as.vector(h2o.predict(dl_ensemble_load_CIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
pROC::roc(tag_CIE3, score_CIE3, ci = TRUE)
mplot_full(tag = tag_CIE3, 
           score = score_CIE3,
           subtitle = "Distribution by CIE group - extra data",
           save = TRUE,
           file_name = "CIE_ensem_dl_full3.png")



# final plots dl_grid or dl_grid_ensemble
p_train <- mplot_density2(tag = tag_CIE, score = score_CIE, xlower = 70, title = "Training")
p_valid <- mplot_density2(tag = tag_CIE_v, score = score_CIE_v, xlower = 70, title = "Validation")
p_test  <- mplot_density2(tag = tag_CIE3, score = score_CIE3, xlower = 70, title = "Testing")

roc_train <- mplot_roc(tag = tag_CIE,   score = score_CIE)
roc_valid <- mplot_roc(tag = tag_CIE_v, score = score_CIE_v)
roc_test  <- mplot_roc(tag = tag_CIE3,  score = score_CIE3)

legend <- get_legend(p_train + 
                       theme(legend.direction = "horizontal", legend.position = "bottom"))
p_train = p_train + theme(legend.position = "none")
p_valid = p_valid + theme(legend.position = "none")
p_test  = p_test  + theme(legend.position = "none")


plots <- plot_grid(p_train, p_valid, p_test, roc_train, roc_valid, roc_test, 
                   labels = paste0(letters[1:3],')'), ncol = 3, align = 'hv')
title1 <- ggdraw() + draw_label("Summary Plots of Adverse Events")
plot_grid(title1, legend, plots, ncol = 1, rel_heights = c(.1, .1, 1))




#shutdown H2O
h2o.shutdown(prompt = FALSE)



##############################################################################################################

######################################################################################################
## Model for SSCIE==========================================================================


h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
#dim(bmk)



#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


# # Add Features by categorizing continuous variables
# data_SSCIE = list(train, valid)
# data_SSCIE_ext <- add_features(data_SSCIE)
# data_SSCIE_ext$Train <- h2o.assign(data_SSCIE_ext$Train, "train_b_ext")
# data_SSCIE_ext$Valid <- h2o.assign(data_SSCIE_ext$Valid, "valid_b_ext")


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# Model for SSCIE============================================================
#### Random Hyper-Parameter Search
hyper_params_bmk_SSCIE <- list(
  activation = c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden = list(c(30,30),c(30,30,30),c(30,30,30,30)),
  input_dropout_ratio = c(0,0.05),
  l1 = seq(0,1e-4,1e-6),
  l2 = seq(0,1e-4,1e-6)
)
#hyper_params_bmk

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria_bmk_SSCIE = list(strategy = "RandomDiscrete", max_runtime_secs = 360, 
                               max_models = 100, seed = 1234567, stopping_rounds = 5, 
                               stopping_tolerance = 1e-2)



dl_random_grid_bmk_SSCIE <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random_SSCIE",
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = "SSCIEFL",
  epochs = 10,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  #stopping_rounds = 2,
  #score_validation_samples=33, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params_bmk_SSCIE,
  search_criteria = search_criteria_bmk_SSCIE,
  nfolds = 30,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  seed = 123
)                



# best grid seach model performance
#grid_SSCIE <- h2o.getGrid("dl_grid_random_SSCIE", sort_by = "auc", decreasing = TRUE)
grid_SSCIE <- h2o.getGrid("dl_grid_random_SSCIE", sort_by = "logloss", decreasing = FALSE)
dl_grid_SSCIE <- h2o.getModel(grid_SSCIE@model_ids[[1]]) 
h2o.auc(dl_grid_SSCIE, train = TRUE, valid = TRUE, xval = TRUE)

# save grid search model
h2o.saveModel(dl_grid_SSCIE, path = "./dl_models", force = TRUE)

# load the previously grid search saved mdoels
#dl_models\\dl_grid_random_SSCIE_model_67
dl_grid_load_SSCIE <- h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_39")
h2o.auc(dl_grid_load_SSCIE, train = TRUE, valid = TRUE, xval = TRUE)


# best grid search models are loaded to assemble
baseModels <- list(
  dl_grid_load_SSCIE1 = h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_51"),
  dl_grid_load_SSCIE2 = h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_83"),
  dl_grid_load_SSCIE3 = h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_2"),
  dl_grid_load_SSCIE4 = h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_61"),
  dl_grid_load_SSCIE5 = h2o.loadModel("dl_models\\dl_grid_random_SSCIE_model_39")
)

# ensemble the best grid searched models
dl_ensemble_SSCIE <- h2o.stackedEnsemble(
  x = predictors,
  y = "SSCIEFL",
  training_frame = train,
  validation_frame = valid,
  metalearner_algorithm = "AUTO",
  metalearner_nfolds = 30,
  metalearner_fold_assignment = "Modulo",
  #base_models = dl_random_grid_bmk_SSCIE@model_ids,
  seed = 123,
  base_models = baseModels
)

# Evaluate ensemble model performance 
h2o.auc(h2o.performance(dl_ensemble_SSCIE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_SSCIE, newdata = valid))

# save the esemble model
h2o.saveModel(dl_ensemble_SSCIE, path = "./dl_models", force = TRUE)

# load the previously saved esemble mdoel
# dl_models\\StackedEnsemble_model_R_1538142377183_18179
# dl_models\\StackedEnsemble_model_R_1538164909720_18045
dl_ensemble_load_SSCIE <- h2o.loadModel("dl_models\\StackedEnsemble_model_R_1538169369925_18289")

# verify ensemble performance 
h2o.auc(h2o.performance(dl_ensemble_load_SSCIE, newdata = train))
h2o.auc(h2o.performance(dl_ensemble_load_SSCIE, newdata = valid))


# training data
tag_SSCIE = as.vector(train$SSCIEFL)
score_SSCIE = as.vector(h2o.predict(dl_ensemble_load_SSCIE, as.h2o(train))$Y)
pROC::roc(tag_SSCIE, score_SSCIE, ci = TRUE)
mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_ensem_dl_full.png")


# validation data
tag_SSCIE_v = as.vector(valid$SSCIEFL)
score_SSCIE_v = as.vector(h2o.predict(dl_ensemble_load_SSCIE, as.h2o(valid))$Y)
pROC::roc(tag_SSCIE_v, score_SSCIE_v, ci = TRUE)
mplot_full(tag = tag_SSCIE_v, 
           score = score_SSCIE_v,
           subtitle = "Distribution by SSCIE group",
           save = TRUE,
           file_name = "SSCIE_ensem_dl_full_v.png")




#prediction of SSCIE on new data================
bmk_extra <- read_sas("Z:/Projects/5201/PROD/Post-Hoc/Data/Derived/bmk_extra.sas7bdat", NULL)
#View(bmk_extra)
bmk_extra$SEX <- ifelse(bmk_extra$SEX == "Male","M","F")

#names(bmk_extra)
bmk_extra_neo    <- bmk_extra[bmk_extra$group == "NEOPHYTE",]
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID == "4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID == "4590",]


# data processing function
auc_test <- function(bmk_extra)
{
  bmk_extra_b <- bmk_extra[,c(5,6,12:14,17:28)]
  bmk_extra_b <- bmk_extra_b[,c(3:17,1,2)]
  names(bmk_extra_b) <- c("AEFL","CIEFL", "SSCIEFL",
                          "LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                          "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO",
                          "AGE","SEX")
  return(bmk_extra_b) 
}



bmk_extra2 <- auc_test(bmk_extra)

tag_SSCIE3 = as.vector(bmk_extra2$SSCIEFL)
score_SSCIE3 = as.vector(h2o.predict(dl_grid_load_SSCIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
#score_SSCIE3 = as.vector(h2o.predict(dl_ensemble_load_SSCIE, as.h2o(bmk_extra2[,c(-1,-3)]))$Y)
pROC::roc(tag_SSCIE3, score_SSCIE3, ci = TRUE)
mplot_full(tag = tag_SSCIE3, 
           score = score_SSCIE3,
           subtitle = "Distribution by SSCIE group - extra data",
           save = TRUE,
           file_name = "SSCIE_ensem_dl_full3.png")



# final plots dl_grid or dl_grid_ensemble
p_train <- mplot_density2(tag = tag_SSCIE, score = score_SSCIE, xlower = 70, title = "Training")
p_valid <- mplot_density2(tag = tag_SSCIE_v, score = score_SSCIE_v, xlower = 70, title = "Validation")
p_test  <- mplot_density2(tag = tag_SSCIE3, score = score_SSCIE3, xlower = 70, title = "Testing")

roc_train <- mplot_roc(tag = tag_SSCIE,   score = score_SSCIE)
roc_valid <- mplot_roc(tag = tag_SSCIE_v, score = score_SSCIE_v)
roc_test  <- mplot_roc(tag = tag_SSCIE3,  score = score_SSCIE3)

legend <- get_legend(p_train + 
                       theme(legend.direction = "horizontal", legend.position = "bottom"))
p_train = p_train + theme(legend.position = "none")
p_valid = p_valid + theme(legend.position = "none")
p_test  = p_test  + theme(legend.position = "none")


plots <- plot_grid(p_train, p_valid, p_test, roc_train, roc_valid, roc_test, 
                   labels = paste0(letters[1:3],')'), ncol = 3, align = 'hv')
title1 <- ggdraw() + draw_label("Summary Plots of Adverse Events")
plot_grid(title1, legend, plots, ncol = 1, rel_heights = c(.1, .1, 1))




#shutdown H2O
h2o.shutdown(prompt = FALSE)











