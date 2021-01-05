getwd()
setwd("Z:/X-Drive/Common/MHasan/ML")
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
library(haven)


######################################################################################################
## Model for AE==========================================================================

h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

bmk <- h2o.importFile(path = normalizePath("base_bmk.csv"))
dim(bmk)
#bmk <- as.data.frame(bmk)

h2o.describe(bmk)

# par(mfrow=c(2,2)) #set up the canvas for 2x2 plots
# model_bmk <- h2o.deeplearning(4:17,1,bmk,epochs=1e5)
# summary(model_bmk)
#plot(bmk[,c(4,6)],pch=19,col=bmk[,1],cex=0.5)

#split the data
#splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
splits <- h2o.splitFrame(bmk, 0.7, seed = 0)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex") 
#test   <- h2o.assign(splits[[3]], "test.hex")  


predictors <- setdiff(names(bmk), c("AEFL", "CIEFL", "SSCIEFL"))
#predictors


aml_AE <- h2o.automl(
            y = "AEFL", 
            x = predictors,
            training_frame = train,
            leaderboard_frame = valid,
            max_models = 25,
            max_runtime_secs = 5*60,
            #exclude_algos = c("StackedEnsemble","DeepLearning"),
            nfolds = 20,
            seed = 0
          )


# Get model ids for all models in the AutoML Leaderboard
model_ids <- as.data.frame(aml_AE@leaderboard$model_id)[,1]
# Get the "All Models" Stacked Ensemble model
se <- h2o.getModel(grep("StackedEnsemble_AllModels", model_ids, value = TRUE)[1])
# Get the Stacked Ensemble metalearner model
metalearner <- h2o.getModel(se@model$metalearner$name)


h2o.varimp(metalearner)
h2o.varimp_plot(metalearner)


h2o.saveModel(aml_AE@leader, path = "./autoML_AE", force = TRUE)
model_loaded_AE <- h2o.loadModel("autoML_AE\\GBM_grid_0_AutoML_20180925_142441_model_8")
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


tag_AE3 = as.vector(bmk_extra2$AEFL)
score_AE3 = as.vector(h2o.predict(model_loaded_AE, as.h2o(bmk_extra2[,c(-2,-3)]))$Y)
pROC::roc(tag_AE3, score_AE3, ci = TRUE)
mplot_full(tag = tag_AE3, 
           score = score_AE3,
           subtitle = "Distribution by AE group - extra data",
           save = TRUE,
           file_name = "AE_ensem_dl_full3.png")




#shutdown H2O
h2o.shutdown(prompt = FALSE)






































