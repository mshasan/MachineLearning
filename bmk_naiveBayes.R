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
library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(caret)    # implementing with caret
library(corrplot) # correlation plot
library(cowplot)




######################################################################################################
## Model for AE==========================================================================


h2o.init(nthreads = -1, max_mem_size = "20G")
h2o.removeAll() 

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
#bmk <- bmk[, c(-2,-3)]

#split the data
splits <- h2o.splitFrame(bmk, 0.8, seed = 1234)
train  <- h2o.assign(splits[[1]], "train.hex")
valid  <- h2o.assign(splits[[2]], "valid.hex") 


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors

# distribution of Attrition rates across train & test set
table(as.vector(train$AEFL)) %>% prop.table()

table(as.vector(valid$AEFL)) %>% prop.table()


as.data.frame(train) %>%
    select_if(is.numeric) %>%
  cor() %>%
  corrplot::corrplot(method = "ellipse")

#heatmap(1 * !is.na(as.data.frame(bmk)), Rowv = NA, Colv = NA)

as.data.frame(train) %>% 
  select("LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA23") %>% 
  gather(key = "metric", value = "value") %>% 
  ggplot(aes_string("value", fill = "metric")) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")


# # do a little preprocessing
# preprocess <- preProcess(as.data.frame(train), method = c("BoxCox", "center", "scale", "pca"))
# train_pp   <- predict(preprocess, as.data.frame(train))
# valid_pp    <- predict(preprocess, as.data.frame(valid))
# 
# # convert to h2o objects
# train_pp.h2o <- train_pp %>%
#   mutate_if(is.factor, factor, ordered = FALSE) %>%
#   as.h2o()
# 
# valid_pp.h2o <- valid_pp %>%
#   mutate_if(is.factor, factor, ordered = FALSE) %>%
#   as.h2o()
# 
# 
# # get new feature names --> PCA preprocessing reduced and changed some features
# predictors <- setdiff(names(train_pp), c("AEFL", "CIEFL", "SSCIEFL"))

# preproocess data does not work

# Model for AE============================================================
# create tuning grid
hyper_params <- list(
  laplace = seq(0, 5, by = 0.5)
)

# build grid search 
grid_AE <- h2o.grid(
  algorithm = "naivebayes",
  grid_id = "nb_grid_AE",
  x = predictors, 
  y = "AEFL", 
  training_frame = train,
  validation_frame = valid,
  nfolds = 10,
  hyper_params = hyper_params,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_fold_assignment = TRUE
)

# Sort the grid models by mse
grid_AE <- h2o.getGrid("nb_grid_AE", sort_by = "auc", decreasing = TRUE)

#grid_AE@summary_table[1,]
best_model_AE <- h2o.getModel(grid_AE@model_ids[[1]]) 

h2o.auc(best_model_AE, train = TRUE, valid = TRUE, xval = TRUE)


# save the model
h2o.saveModel(best_model_AE, path = "./best_model_AE", force = TRUE)

# load the previously saved mdoel
model_loaded_AE <- h2o.loadModel("best_model_AE\\nb_grid_AE_model_3")
# summary(model_loaded_AE)
#h2o.predict(model_loaded_AE, as.h2o(valid_pp.h2o))
#best_model_AE
h2o.performance(model_loaded_AE, newdata = valid_pp.h2o)
h2o.auc(model_loaded_AE, train = TRUE, valid = TRUE, xval = TRUE)




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
dim(bmk_extra2)
bmk_extra2_omitna <- na.omit(bmk_extra2)
dim(bmk_extra2_omitna)

test_extra2_pp <- predict(preprocess, as.data.frame(bmk_extra2_omitna), na.rm = TRUE)


# convert to h2o objects
test_pp.h2o <- test_extra2_pp %>%
  mutate_if(is.factor, factor, ordered = FALSE) %>%
  as.h2o()


tag_AE3 = as.vector(test_pp.h2o$AEFL)
score_AE3 = as.vector(h2o.predict(model_loaded_AE, as.h2o(test_pp.h2o[,c(-2,-3)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci = TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_ensem_dl_full3.png")



#shutdown H2O
h2o.shutdown(prompt = FALSE)




