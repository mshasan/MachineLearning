getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
source("all_plots_onepage.R")
source("feature_selection.R")

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



######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))[,-c(2,3)]
dim(bmk)
#h2o.summary(bmk)



# split to train/test/validation again
data_AE = h2o.splitFrame(bmk,ratios=.8, destination_frames = c("train_b","valid_b"))
#data_AE = h2o.splitFrame(bmk,ratios=c(.7,.2),destination_frames = c("train_b","valid_b", "test_b"))
names(data_AE) <- c("Train","Valid")
#names(data_AE) <- c("Train","Valid", "Test")
y = "AEFL"
x = names(data_AE$Train)
x = x[-which(x==y)]

m_AE = h2o.glm(training_frame = data_AE$Train, 
                     validation_frame = data_AE$Valid, 
                     x = x, y = y,
                     nfolds = 10,
                     family='binomial',
                     lambda=0,
                     keep_cross_validation_predictions = TRUE)
h2o.confusionMatrix(m_AE, valid = FALSE)
h2o.confusionMatrix(m_AE, valid = TRUE)
h2o.auc(m_AE,train=TRUE, valid=TRUE, xval=TRUE) 


# Add Features by categorizing continuous variables
data_AE_ext <- add_features(data_AE)
data_AE_ext$Train <- h2o.assign(data_AE_ext$Train,"train_b_ext")
data_AE_ext$Valid <- h2o.assign(data_AE_ext$Valid,"valid_b_ext")
#data_AE_ext$Test <- h2o.assign(data_AE_ext$Test,"test_b_ext")
y = "AEFL"
x = names(data_AE_ext$Train)
x = x[-which(x==y)]


m_AE_1_ext = try(h2o.glm(training_frame = data_AE_ext$Train, 
                               validation_frame = data_AE_ext$Valid, 
                               x = x, y = y, 
                               family='binomial',
                               solver='L_BFGS',lambda_search = TRUE,
                               keep_cross_validation_predictions = TRUE))
h2o.confusionMatrix(m_AE_1_ext,valid=FALSE)
h2o.confusionMatrix(m_AE_1_ext,valid=TRUE)
h2o.auc(m_AE_1_ext,train=TRUE, valid=TRUE)



# save the model
#h2o.saveModel(m_AE_1_ext, path="./best_model_AE", force=TRUE)

# load the previously saved mdoel
model_loaded_AE <- h2o.loadModel("best_model_AE\\GLM_model_R_1534679547936_19")
# summary(model_loaded_AE)
#predict(model_loaded_AE, as.h2o(train))
h2o.auc(model_loaded_AE, train=TRUE, valid=TRUE)


tag_AE = as.vector(data_AE$Valid$AEFL)
score_AE = as.vector(h2o.predict(model_loaded_AE, data_AE_ext$Valid)[,3])

# cv_auc <- model_loaded_AE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_AE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_AE@algorithm)

# density plots--------
mplot_density(tag = tag_AE, 
                   score = score_AE, 
                   #model_name = "Deeplearning", 
                   subtitle = "Distribution by AE group", 
                   save = TRUE, 
                   file_name = "AE_glm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_AE, 
          score = score_AE, 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "AE_glm_roc.png")


mplot_cuts(score = score_AE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "AE_glm_ncuts.png")

mplot_splits(tag = tag_AE, 
             score = score_AE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "AE_glm_splits.png")


mplot_full(tag = tag_AE, 
           score = score_AE,
           subtitle = "Area Under ROC Curve",
           save = TRUE,
           file_name = "SSCIE_glm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)




######################################################################################################
## Model for CIE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))[,-c(1,3)]
dim(bmk)
#h2o.summary(bmk)



# split to train/test/validation again
data_CIE = h2o.splitFrame(bmk,ratios=.8, destination_frames = c("train_b","valid_b"))
#data_CIE = h2o.splitFrame(bmk,ratios=c(.7,.2),destination_frames = c("train_b","valid_b", "test_b"))
names(data_CIE) <- c("Train","Valid")
#names(data_CIE) <- c("Train","Valid", "Test")
y = "CIEFL"
x = names(data_CIE$Train)
x = x[-which(x==y)]

m_CIE = h2o.glm(training_frame = data_CIE$Train, 
               validation_frame = data_CIE$Valid, 
               x = x, y = y,
               nfolds = 10,
               family='binomial',
               lambda=0,
               keep_cross_validation_predictions = TRUE)
h2o.confusionMatrix(m_CIE, valid = FALSE)
h2o.confusionMatrix(m_CIE, valid = TRUE)
h2o.auc(m_CIE,train=TRUE, valid=TRUE, xval=TRUE) 



# Add Features by categorizing continuous variables
data_CIE_ext <- add_features(data_CIE)
data_CIE_ext$Train <- h2o.assign(data_CIE_ext$Train,"train_b_ext")
data_CIE_ext$Valid <- h2o.assign(data_CIE_ext$Valid,"valid_b_ext")
#data_CIE_ext$Test <- h2o.assign(data_CIE_ext$Test,"test_b_ext")
y = "CIEFL"
x = names(data_CIE_ext$Train)
x = x[-which(x==y)]


m_CIE_1_ext = try(h2o.glm(training_frame = data_CIE_ext$Train, 
                         validation_frame = data_CIE_ext$Valid, 
                         x = x, y = y, 
                         family='binomial',
                         solver='L_BFGS',lambda_search = TRUE,
                         keep_cross_validation_predictions = TRUE))
h2o.confusionMatrix(m_CIE_1_ext,valid=FALSE)
h2o.confusionMatrix(m_CIE_1_ext,valid=TRUE)
h2o.auc(m_CIE_1_ext,train=TRUE, valid=TRUE)



# save the model
h2o.saveModel(m_CIE_1_ext, path="./best_model_CIE", force=TRUE)

# load the previously saved mdoel
model_loaded_CIE <- h2o.loadModel("best_model_CIE\\GLM_model_R_1534682805437_38")
# summary(model_loaded_CIE)
#predict(model_loaded_CIE, as.h2o(train))
h2o.auc(model_loaded_CIE, train=TRUE, valid=TRUE)


tag_CIE = as.vector(data_CIE$Valid$CIEFL)
score_CIE = as.vector(h2o.predict(model_loaded_CIE, data_CIE_ext$Valid)[,3])

# cv_auc <- model_loaded_CIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_CIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_CIE@algorithm)

# density plots--------
mplot_density(tag = tag_CIE, 
              score = score_CIE, 
              #model_name = "Deeplearning", 
              subtitle = "Distribution by CIE group", 
              save = TRUE, 
              file_name = "CIE_glm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_CIE, 
          score = score_CIE, 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "CIE_glm_roc.png")


mplot_cuts(score = score_CIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "CIE_glm_ncuts.png")

mplot_splits(tag = tag_CIE, 
             score = score_CIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "CIE_glm_splits.png")


mplot_full(tag = tag_CIE, 
           score = score_CIE,
           subtitle = "Area Under ROC Curve",
           save = TRUE,
           file_name = "CIE_glm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)






######################################################################################################
## Model for SSCIE==========================================================================


h2o.init(nthreads=-1, max_mem_size="12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))[,-c(2,1)]
dim(bmk)
#h2o.summary(bmk)



# split to train/test/validation again
data_SSCIE = h2o.splitFrame(bmk,ratios=.8, destination_frames = c("train_b","valid_b"))
#data_SSCIE = h2o.splitFrame(bmk,ratios=c(.7,.2),destination_frames = c("train_b","valid_b", "test_b"))
names(data_SSCIE) <- c("Train","Valid")
#names(data_SSCIE) <- c("Train","Valid", "Test")
y = "SSCIEFL"
x = names(data_SSCIE$Train)
x = x[-which(x==y)]

m_SSCIE = h2o.glm(training_frame = data_SSCIE$Train, 
               validation_frame = data_SSCIE$Valid, 
               x = x, y = y,
               nfolds = 10,
               family='binomial',
               lambda=0,
               keep_cross_validation_predictions = TRUE)
h2o.confusionMatrix(m_SSCIE, valid = FALSE)
h2o.confusionMatrix(m_SSCIE, valid = TRUE)
h2o.auc(m_SSCIE, train=TRUE, valid=TRUE, xval=TRUE)


# Add Features
data_SSCIE_ext <- add_features(data_SSCIE)
data_SSCIE_ext$Train <- h2o.assign(data_SSCIE_ext$Train,"train_b_ext")
data_SSCIE_ext$Valid <- h2o.assign(data_SSCIE_ext$Valid,"valid_b_ext")
#data_SSCIE_ext$Test <- h2o.assign(data_SSCIE_ext$Test,"test_b_ext")
y = "SSCIEFL"
x = names(data_SSCIE_ext$Train)
x = x[-which(x==y)]


m_SSCIE_1_ext = try(h2o.glm(training_frame = data_SSCIE_ext$Train, 
                         validation_frame = data_SSCIE_ext$Valid, 
                         x = x, y = y, 
                         family='binomial',
                         solver='L_BFGS',lambda_search = TRUE,
                         keep_cross_validation_predictions = TRUE))
h2o.confusionMatrix(m_SSCIE_1_ext,valid=FALSE)
h2o.confusionMatrix(m_SSCIE_1_ext,valid=TRUE)
h2o.auc(m_SSCIE_1_ext,train=TRUE, valid=TRUE)
h2o.auc(m_SSCIE_1_ext,xval = TRUE)


# save the model
h2o.saveModel(m_SSCIE_1_ext, path="./best_model_SSCIE", force=TRUE)

# load the previously saved mdoel
model_loaded_SSCIE <- h2o.loadModel("best_model_SSCIE\\GLM_model_R_1534683372852_35")
# summary(model_loaded_SSCIE)
#predict(model_loaded_SSCIE, as.h2o(train))
h2o.auc(model_loaded_SSCIE, train=TRUE, valid=TRUE)


tag_SSCIE = as.vector(data_SSCIE$Valid$SSCIEFL)
score_SSCIE = as.vector(h2o.predict(model_loaded_SSCIE, data_SSCIE_ext$Valid)[,3])

# cv_auc <- model_loaded_SSCIE@model$cross_validation_metrics@metrics$AUC
# subtitle <- paste(model_loaded_SSCIE@algorithm, "- AUC:", round(100 * cv_auc, 2))
# subdir <- paste0("Models/", round(100*cv_auc, 2), "-", model_loaded_SSCIE@algorithm)

# density plots--------
mplot_density(tag = tag_SSCIE, 
              score = score_SSCIE, 
              subtitle = "Distribution by SSCIE group", 
              save = TRUE, 
              file_name = "SSCIE_glm_distribution.png")

# ROC curve------------
mplot_roc(tag =tag_SSCIE, 
          score = score_SSCIE, 
          subtitle = "Area Under ROC Curve", 
          interval = 0.2, 
          plotly = FALSE,
          save = TRUE, 
          file_name = "SSCIE_glm_roc.png")


mplot_cuts(score = score_SSCIE, 
           splits = 10, 
           subtitle = NA, 
           model_name = NA, 
           save = TRUE, 
           file_name = "SSCIE_glm_ncuts.png")

mplot_splits(tag = tag_SSCIE, 
             score = score_SSCIE, 
             splits = 5, 
             subtitle = NA, 
             model_name = NA, 
             facet = NA, 
             save = TRUE, 
             subdir = NA, 
             file_name = "SSCIE_glm_splits.png")


mplot_full(tag = tag_SSCIE, 
           score = score_SSCIE,
           subtitle = "Area Under ROC Curve",
           save = TRUE,
           file_name = "SSCIE_glm_full.png")



#shutdown H2O
h2o.shutdown(prompt=FALSE)












































