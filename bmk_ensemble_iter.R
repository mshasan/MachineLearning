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
#install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
library(h2o)
library(h2oEnsemble)
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
library(glue)
library("SuperLearner")




######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads = -1, max_mem_size = "20G")
h2o.removeAll() 

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))


#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex")
valid  <- h2o.assign(splits[[2]], "valid.hex") 


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors


# Model for AE============================================================



ensemble_iter_AE <- function(niter, nfolds = 10, seed = 1)
  {
    splits2 <- h2o.splitFrame(train, 0.7)
    train2  <- h2o.assign(splits2[[1]], "train2.hex") 
    valid2  <- h2o.assign(splits2[[2]], "valid2.hex")
    
    # deepllearning model
    dl_AE <- h2o.deeplearning(
        training_frame = train2, 
        validation_frame = valid2, 
        x = predictors, 
        y = "AEFL", 
        hidden = c(300, 300),            
        epochs = 10,                    
        stopping_metric = "logloss",     
        stopping_tolerance = 1e-2,         
        stopping_rounds = 2,
        score_duty_cycle = 0.025,          
        adaptive_rate = FALSE,            
        rate = 0.01, 
        rate_annealing = 2e-6,            
        momentum_start = 0.2,            
        momentum_stable = 0.4, 
        momentum_ramp = 1e7,
        l1 = 1e-5,                       
        l2 = 1e-5,
        activation = c("MaxoutWithDropout"),
        max_w2 = 10,
        nfolds = nfolds,
        fold_assignment = "Modulo",
        keep_cross_validation_predictions = TRUE,
        seed = seed
      )
    
    # gradient boosting model
    gbm_AE <- h2o.gbm(
        x = predictors,
        y = "AEFL",
        training_frame = train2,
        validation_frame = valid2,
        distribution = "bernoulli",
        ntrees = 10,
        max_depth = 3,
        min_rows = 2,
        learn_rate = 0.2,
        nfolds = nfolds,
        fold_assignment = "Modulo",
        keep_cross_validation_predictions = TRUE,
        seed = seed
      )
    
    # randomfores model
    rf_AE <- h2o.randomForest(
      x = predictors,
      y = "AEFL",
      training_frame = train2,
      validation_frame = valid2,
      ntrees = 50,
      nfolds = nfolds,
      fold_assignment = "Modulo",
      keep_cross_validation_predictions = TRUE,
      seed = seed
      )
    
    
    # Add Features by categorizing continuous variables
    data_AE = list(train2, valid2)
    data_AE_ext <- add_features(data_AE)
    data_AE_ext$Train2 <- h2o.assign(data_AE_ext$Train, "train_b_ext")
    data_AE_ext$Valid2 <- h2o.assign(data_AE_ext$Valid, "valid_b_ext")

    # generalized linear model
    glm_AE <- h2o.glm(
         training_frame = data_AE_ext$Train2, 
         validation_frame = data_AE_ext$Valid2, 
         x = predictors, 
         y = "AEFL", 
         family = 'binomial',
         solver = 'L_BFGS',
         lambda_search = TRUE,
         nfolds = nfolds,
         fold_assignment = "Modulo",
         keep_cross_validation_predictions = TRUE,
         seed = seed
        )
    
    
    # ensemble models
    ensemble_AE <- h2o.stackedEnsemble(
         x = predictors,
         y = "AEFL",
         training_frame = train2,
         validation_frame = valid2,
         metalearner_algorithm = "AUTO",
         metalearner_nfolds = nfolds,
         metalearner_fold_assignment = "Modulo",
         seed = seed,
         base_models = list(dl_AE, gbm_AE, rf_AE, glm_AE)
        )
    
    return(list(
      niter = niter,
      dl_AE  = dl_AE,
      gbm_AE = gbm_AE,
      rf_AE  = rf_AE,
      glm_AE = glm_AE,
      ensemble_AE = ensemble_AE))
  }


# obtain ensemble models for each iteration
niter = 100
ensemble_iter_AE_results <- map(1:niter, ensemble_iter_AE, nfolds = 5, seed = 1)



# # combine all ensemble models
# ensem_iter_models_AE <- map(ensemble_iter_AE_results, "ensemble_AE")
# 
# # ensemble models
# ensemble_final_AE <- h2o.stackedEnsemble(
#   x = predictors,
#   y = "AEFL",
#   training_frame = train[1:91,],
#   validation_frame = valid,
#   metalearner_algorithm = "AUTO",
#   metalearner_nfolds = 0,
#   metalearner_fold_assignment = "Modulo",
#   seed = 1,
#   base_models = ensem_iter_models_AE
# )



# find the AUC of validation set
model_auc <- function(n_model)
{
  auc_train <- ensemble_iter_AE_results[[n_model]]$ensemble_AE %>%
    h2o.auc(train = TRUE) 
  
  auc_valid <- ensemble_iter_AE_results[[n_model]]$ensemble_AE %>%
    h2o.auc(valid = TRUE)
  
  auc_xval <- ensemble_iter_AE_results[[n_model]]$ensemble_AE %>%
    h2o.auc(xval = TRUE)
  
  auc_train_vs_valid <- ifelse(auc_train > auc_valid, auc_valid, NA)
  auc_train_vs_xval  <- ifelse(auc_train > auc_xval,  auc_xval,  NA)
  
  return(list(auc_train_vs_valid = auc_train_vs_valid,
              auc_train_vs_xval = auc_train_vs_xval))
}

auc_all_AE0 <- map(1:niter, model_auc)
#use onlyone
auc_all_AE <- map_dbl(auc_all_AE0, "auc_train_vs_valid")
#auc_all_AE <- map_dbl(auc_all_AE0, "auc_train_vs_xval")

summary(auc_all_AE)
hist(auc_all_AE)
ensem_best_model_AE <- ensemble_iter_AE_results[[which.max(auc_all_AE)]]$ensemble_AE



# save the model
h2o.saveModel(ensem_best_model_AE, path = "./ensemble_iter_model", force = TRUE)

# load the previously saved mdoel
ensem_load_AE <- h2o.loadModel("ensemble_iter_model\\StackedEnsemble_model_R_1537900734747_75538")
#h2o.auc(ensem_load_AE, train = TRUE, valid = TRUE, xval = TRUE)

pred_data_by_ensem <- h2o.predict(ensem_load_AE, newdata = valid)
tag_AE = as.vector(valid$AEFL)
score_ensem_AE = as.vector(pred_data_by_ensem$Y)
pROC::roc(tag_AE, score_ensem_AE, ci = TRUE)

# Evaluate ensemble performance 
h2o.auc(h2o.performance(ensem_load_AE, newdata = train))
h2o.auc(h2o.performance(ensem_load_AE, newdata = valid))


# pred_tbl <- function(algorithm)
# {
#   auc_valid_dl <- identity_transformer(text = ensemble_iter_AE_results[[n_model]]$algorithm) %>%
#     h2o.auc(valid = TRUE)
#   pred_dl0 <- ensemble_iter_AE_results[[n_model]]$algorithm %>%
#     h2o.predict(newdata = valid) %>%
#     tbl_df()
#   pred_dl <- if (auc_valid_dl > aucThres) {pred_dl0} else{pred_dl0[NA,]}
# 
#   return(pred_dl)
# }
# 
# pred_tbl(algorithm = "gbm_AE")



#valid <- as.h2o(bmk_extra2[,c(-2,-3)])

# find average predicted probability from all models
model_pred <- function(n_model, aucThres = 0.6)
{
  auc_valid_dl <- ensemble_iter_AE_results[[n_model]]$dl_AE %>%
    h2o.auc(valid = TRUE)
  pred_dl0 <- ensemble_iter_AE_results[[n_model]]$dl_AE %>% 
    h2o.predict(newdata = valid) %>% 
    tbl_df()
  pred_dl <- if (auc_valid_dl > aucThres) {pred_dl0} else{pred_dl0[NA,]}
  
  auc_valid_gbm <- ensemble_iter_AE_results[[n_model]]$gbm_AE %>%
    h2o.auc(valid = TRUE)
  pred_gbm0 <- ensemble_iter_AE_results[[n_model]]$gbm_AE %>% 
    h2o.predict(newdata = valid) %>% 
    tbl_df()
  pred_gbm <- if (auc_valid_gbm > aucThres) {pred_gbm0} else{pred_gbm0[NA,]}
  
  auc_valid_rf <- ensemble_iter_AE_results[[n_model]]$rf_AE %>%
    h2o.auc(valid = TRUE)
  pred_rf0 <- ensemble_iter_AE_results[[n_model]]$rf_AE %>% 
    h2o.predict(newdata = valid) %>% 
    tbl_df()
  pred_rf <- if (auc_valid_rf > aucThres) {pred_rf0} else{pred_rf0[NA,]}
  
  auc_valid_glm <- ensemble_iter_AE_results[[n_model]]$glm_AE %>%
    h2o.auc(valid = TRUE)
  pred_glm0 <- ensemble_iter_AE_results[[n_model]]$glm_AE %>% 
    h2o.predict(newdata = valid) %>% 
    tbl_df()
  pred_glm <- if (auc_valid_glm > aucThres) {pred_glm0} else{pred_glm0[NA,]}
  
  auc_valid_ensem <- ensemble_iter_AE_results[[n_model]]$ensemble_AE %>%
    h2o.auc(valid = TRUE)
  pred_ensem0 <- ensemble_iter_AE_results[[n_model]]$ensemble_AE %>% 
    h2o.predict(newdata = valid) %>% 
    tbl_df()
  pred_ensem <- if (auc_valid_ensem > aucThres) {pred_ensem0} else{pred_ensem0[NA,]}
  
  predProb <- bind_cols(dl = pred_dl$Y, 
                         gbm = pred_gbm$Y, 
                         rf = pred_rf$Y, 
                         glm = pred_glm$Y, 
                         ensem = pred_ensem$Y)
  
  #avg_pred <- rowMeans(predProb, na.rm = TRUE)

  return(predProb)
}

avg_pred <- map(1:niter, model_pred, aucThres = 0.5)
avg_pred_by_alg <- aaply(laply(avg_pred, as.matrix), c(2, 3), mean, na.rm = TRUE)

summary(avg_pred_by_alg)

# pred_data_by_avg <- tibble(
#                     predict = ifelse(avg_pred_by_alg[, "dl"] > .5, "Y", "N"),
#                     N = 1 - avg_pred, 
#                     Y = avg_pred 
#                     ) 

tag_AE = as.vector(valid$AEFL)
# use one model
score_avg_AE = as.vector(avg_pred_by_alg[,"ensem"])
pROC::roc(tag_AE, score_avg_AE, ci = TRUE)





# training data
tag_AE = as.vector(train$AEFL)
score_AE = as.vector(h2o.predict(ensem_load_AE, as.h2o(train))$Y)
pROC::roc(tag_AE, score_AE, ci = TRUE)
mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Distribution by AE group",
           save = TRUE,
           file_name = "AE_ensem_full.png")


# validation data
tag_AE_v = as.vector(valid$AEFL)
score_AE_v = as.vector(h2o.predict(ensem_load_AE, as.h2o(valid))$Y)
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
bmk_extra_CR4550 <- bmk_extra[bmk_extra$STUDYID == "4550",]
bmk_extra_CR4590 <- bmk_extra[bmk_extra$STUDYID == "4590",]

# use one of them
bmk_extra <- bmk_extra
bmk_extra <- bmk_extra[bmk_extra$group == "NEOPHYTE",]
bmk_extra <- bmk_extra_CR4550
bmk_extra <- bmk_extra_CR4590

bmk_extra2 <- bmk_extra[,c(5,6,12:14,17:28)]
bmk_extra2 <- bmk_extra2[,c(3:17,1,2)]
#names(bmk_extra2)
names(bmk_extra2) <- c("AEFL","CIEFL", "SSCIEFL","LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA88","KDA23",
                       "LYSO_LIPO","LYSO_LACTO", "LACTO_LIPO","LYSO_sIgA","ALB_LYSO","AGE","SEX")


tag_AE3 = as.vector(bmk_extra2$AEFL)
score_AE3 = as.vector(h2o.predict(ensem_load_AE, as.h2o(bmk_extra2[,c(-2,-3)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci = TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_ensem_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt = FALSE)
















