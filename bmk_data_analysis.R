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
library(mvnfast)
library(rsample)  # data splitting 
library(cowplot)
library(dplyr)
library(rtf)



######################################################################################################
## Model for AE==========================================================================

bmk <- read_csv("base_bmk.csv")

# str(bmk)
# dim(bmk)
# 
# summary(bmk)

# distribution of Attrition rates across train & test set
table(as.vector(bmk$AEFL)) %>% prop.table()
table(as.vector(bmk$CIEFL)) %>% prop.table()
table(as.vector(bmk$SSCIEFL)) %>% prop.table()


as.data.frame(bmk) %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot::corrplot(method = "ellipse")

# as.data.frame(bmk) %>%
#   select_if(is.numeric) %>%
#   cor() %>% write.table(file = "correlation.txt", sep = ",", quote = FALSE, row.names = F)

as.data.frame(bmk) %>%
  dplyr::select("LYSOZYME","LIPOCALIN","sIgA","LACTOFERRIN","ALBUMIN","KDA23") %>% 
  gather(key = "metric", value = "value") %>% 
  ggplot(aes_string("value", fill = "metric")) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")


# simulation
names(bmk)


beta <- 0.5
mu <- as.data.frame(bmk[,4:15]) %>%  
  select_if(is.numeric) %>% 
  scale() %>%
  colMeans(na.rm = TRUE) %>%
  as.numeric() %>% round(1)

corr <- as.data.frame(bmk[,4:15]) %>%  
  select_if(is.numeric) %>% 
  cor() %>% round(1) %>% as.matrix


output <- "rtf_vignette.doc" 
rtf <- RTF(output,width = 8.5,height = 11,font.size = 10,omi = c(1,1,1,1))
#addHeader(rtf, title = "Section Header", subtitle = "This is the subheading or section text.")
addTable(rtf,corr[1:7,1:7],font.size = 10, row.names = TRUE, NA.string = "-")
done(rtf)



# corr2 <- matrix(c(
#  1.0,      -0.1, -0.6,         0.2,    -0.5,   0.1,  -0.2,       0.6,        0.6,        0.2,     -0.5,
# -0.1,       1.0, -0.2,        -0.3,     0.0,   0.0,  -0.1,      -0.7,        0.2,       -0.8,     -0.1,
# -0.6,      -0.2,  1.0,        -0.4,     0.2,   0.0,   0.1,      -0.2,       -0.2,        0.0,      0.2,
#  0.2,      -0.3, -0.4,         1.0,    -0.5,  -0.3,  -0.2,       0.3,       -0.6,        0.6,     -0.4,
# -0.5,       0.0,  0.2,        -0.5,     1.0,   0.2,   0.5,      -0.2,        0.1,       -0.2,      0.9,
#  0.1,       0.0,  0.0,        -0.3,     0.2,   1.0,   0.1,       0.0,        0.3,       -0.1,      0.3,
# -0.2,      -0.1,  0.1,        -0.2,     0.5,   0.1,   1.0,      -0.1,        0.0,        0.0,      0.5,
#  0.6,      -0.7, -0.2,         0.3,    -0.2,   0.0,  -0.1,       1.0,        0.1,        0.8,     -0.2,
#  0.6,       0.2, -0.2,        -0.6,     0.1,   0.3,   0.0,       0.1,        1.0,       -0.3,      0.0,
#  0.2,      -0.8,  0.0,         0.6,    -0.2,  -0.1,   0.0,       0.8,       -0.3,        1.0,     -0.1,
# -0.5,      -0.1,  0.2,        -0.4,     0.9,   0.3,   0.5,      -0.2,        0.0,       -0.1,      1.0), 11,11)


corr2 <- nearPD(corr, corr = TRUE)$mat


# function for simulation
simu_dl <- function(iter, intercept, beta, n = 100, p = 11)
{
  X <- rmvn(n = n, mu = rep(0, p), sigma = corr2)
  linpred <- intercept + X %*% beta
  prob <- as.vector(exp(linpred)/(1 + exp(linpred)))
  runis <- runif(n, 0, 1)
  Y <- ifelse(runis < prob, 1, 0)
  data <- data.frame(Y = as.factor(Y), X)
  
  split <- initial_split(data, prop = .8, strata = "Y")
  train <- training(split)
  test  <- testing(split)
  
  # deepllearning model
  simu_dl <- h2o.deeplearning(
    training_frame = as.h2o(train), 
    y = "Y", 
    hidden = c(30, 30),            
    epochs = 5,                    
    stopping_metric = "logloss",     
    stopping_tolerance = 1e-2,         
    stopping_rounds = 2,
    score_duty_cycle = 0.025,          
    adaptive_rate = FALSE,            
    rate = 0.02, 
    rate_annealing = 2e-6,            
    momentum_start = 0.2,            
    momentum_stable = 0.4, 
    momentum_ramp = 1e7,
    l1 = 1e-5,                       
    l2 = 1e-5,
    activation = c("MaxoutWithDropout"),
    max_w2 = 10,
    nfolds = 10,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE,
    seed = 1
  )

auc_train <- h2o.auc(h2o.performance(simu_dl, newdata = as.h2o(train)))
auc_test  <- h2o.auc(h2o.performance(simu_dl, newdata = as.h2o(test)))

return(list(auc_train = auc_train, auc_test = auc_test))    
}


beta_AE <- c(.04, .05, -.1, -.2, -.06, -.07, -1.1, -.002, -.6, .53, .66)
beta_CIE <- c(.14, -.005, .02, -.11, .08, .11, -.78, -.54, -1.04, .41, -1.38)
beta_SSCIE <- c(.16, .03, .04, -.09, .07, .01, -.7, -.6, -.9, .7, -.96)

simu_dl_AE_results <- map(1:100, simu_dl, intercept = 1.8, beta = beta_AE)
simu_dl_CIE_results <- map(1:100, simu_dl, intercept = -1.5, beta = beta_CIE)
simu_dl_SSCIE_results <- map(1:100, simu_dl, intercept = -1.7, beta = beta_SSCIE)


auc_train_AE <- map_dbl(simu_dl_AE_results, "auc_train")
auc_test_AE  <- map_dbl(simu_dl_AE_results, "auc_test")

auc_train_CIE <- map_dbl(simu_dl_CIE_results, "auc_train")
auc_test_CIE  <- map_dbl(simu_dl_CIE_results, "auc_test")

auc_train_SSCIE <- map_dbl(simu_dl_SSCIE_results, "auc_train")
auc_test_SSCIE  <- map_dbl(simu_dl_SSCIE_results, "auc_test")

Data <- data.frame(auc_train_AE, auc_test_AE, 
                   auc_train_CIE, auc_test_CIE, 
                   auc_train_SSCIE, auc_test_SSCIE)

apply(Data, 2, mean)*100
apply(Data, 2, sd)*100

Data %>% 
  gather(key = "metric", value = "value") %>% 
  ggplot(aes_string("value", fill = "metric")) + 
  geom_histogram(aes(y = ..density..), colour = "#1F3552", fill = "#4271AE") + 
  facet_wrap(~ metric, scales = "free")







