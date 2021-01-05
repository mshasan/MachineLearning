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
library(haven)



######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))


#split the data
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


dl_rand_ensem_iter_AE <- function(iter)
  {
      train2 <- h2o.splitFrame(train, 0.7)[[1]]
  
      dl_rand_iter_AE <- h2o.grid(
      algorithm="deeplearning",
      training_frame=train2,
      x=predictors, 
      y="AEFL",
      epochs=10,
      stopping_metric="logloss",
      stopping_tolerance=1e-2,        
      stopping_rounds=2,
      score_duty_cycle=0.025,         
      max_w2=10,                      
      hyper_params = hyper_params_bmk_AE,
      search_criteria = search_criteria_bmk_AE,
      nfolds=10,
      fold_assignment="Modulo",
      keep_cross_validation_predictions = TRUE,
      keep_cross_validation_fold_assignment = TRUE
    ) 
    
    # ensemble random grid searched models
    dl_ensem_AE <- h2o.stackedEnsemble(x = predictors,
                                       y = "AEFL",
                                       training_frame = train2,
                                       metalearner_algorithm = "gbm",
                                       base_models = dl_rand_iter_AE@model_ids)
    return(dl_ensem_AE)
  }


# obtain ensemble models for each iteration
dl_rand_ensem_iter_AE_results <- map(1:3, dl_rand_ensem_iter_AE)

# final ensemble model using all ensembles models
dl_ensem_final_AE <- h2o.stackedEnsemble(x = predictors,
                                   y = "AEFL",
                                   training_frame = train,
                                   metalearner_algorithm = "gbm",
                                   base_models = as.list(dl_rand_ensem_iter_AE_results))




# Evaluate ensemble performance 
h2o.auc(h2o.performance(dl_ensem_AE, newdata = train))
h2o.auc(h2o.performance(dl_ensem_AE, newdata = valid))


# best grid seach model performance
grid_AE <- h2o.getGrid("dl_rand_iter_AE",sort_by="auc",decreasing=TRUE)
best_model_AE <- h2o.getModel(grid_AE@model_ids[[1]]) 
h2o.auc(best_model_AE, train=TRUE, valid=FALSE, xval=TRUE)


# save the model
h2o.saveModel(dl_ensem_AE, path="./ensemble_dl_model", force=TRUE)

# load the previously saved mdoel
dl_ensem_load_AE <- h2o.loadModel("ensemble_dl_model\\StackedEnsemble_model_R_1536353732607_6049")


# training data
tag_AE = as.vector(train$AEFL)
score_AE = as.vector(h2o.predict(dl_ensem_load_AE, as.h2o(train))$Y)
pROC::roc(tag_AE, score_AE, ci=TRUE)
mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_ensem_dl_full.png")


# validation data
tag_AE_v = as.vector(valid$AEFL)
score_AE_v = as.vector(h2o.predict(dl_ensem_load_AE, as.h2o(valid))$Y)
pROC::roc(tag_AE_v, score_AE_v, ci=TRUE)
mplot_full(tag = tag_AE_v, 
           score = score_AE_v,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_ensem_dl_full_v.png")




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
score_AE3 = as.vector(h2o.predict(dl_ensem_load_AE, as.h2o(bmk_extra2[,c(-2,-3)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci=TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_ensem_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)
















