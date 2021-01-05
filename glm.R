## R installation instructions are at http://h2o.ai/download
library(h2o)
source("feature_selection.R")

h2o.init(nthreads=-1, max_mem_size="2G")
h2o.removeAll() ## clean slate - just in case the cluster was already running


D = h2o.importFile(path = normalizePath("covtype.full.csv"))
h2o.summary(D)


# multinomial model

data = h2o.splitFrame(D,ratios=c(.7,.15),destination_frames = c("train","test","valid"))
names(data) <- c("Train","Test","Valid")
y = "Cover_Type"
x = names(data$Train)
x = x[-which(x==y)]


m1 = h2o.glm(training_frame = data$Train, 
             validation_frame = data$Valid, 
             x = x, y = y,family='multinomial',solver='L_BFGS')
h2o.confusionMatrix(m1, valid=TRUE)


m2 = h2o.glm(training_frame = data$Train, 
             validation_frame = data$Valid, 
             x = x, y = y,family='multinomial',solver='L_BFGS', lambda = 0)
h2o.confusionMatrix(m2, valid=FALSE) # get confusion matrix in the training data
h2o.confusionMatrix(m2, valid=TRUE)  # get confusion matrix in the validation data









#===========================================================================================================

# binomial model

D_binomial = D[D$Cover_Type %in% c("class_1","class_2"),]
h2o.setLevels(D_binomial$Cover_Type,c("class_1","class_2"))
# split to train/test/validation again
data_binomial = h2o.splitFrame(D_binomial,ratios=c(.7,.15),destination_frames = c("train_b","test_b","valid_b"))
names(data_binomial) <- c("Train","Test","Valid")


m_binomial = h2o.glm(training_frame = data_binomial$Train, 
                     validation_frame = data_binomial$Valid, 
                     x = x, y = y, family='binomial',lambda=0)
h2o.confusionMatrix(m_binomial, valid = TRUE)
h2o.confusionMatrix(m_binomial, valid = TRUE)


fpr = m_binomial@model$training_metrics@metrics$thresholds_and_metric_scores$fpr
tpr = m_binomial@model$training_metrics@metrics$thresholds_and_metric_scores$tpr
fpr_val = m_binomial@model$validation_metrics@metrics$thresholds_and_metric_scores$fpr
tpr_val = m_binomial@model$validation_metrics@metrics$thresholds_and_metric_scores$tpr
plot(fpr,tpr, type='l')
title('AUC')
lines(fpr_val,tpr_val,type='l',col='red')
legend("bottomright",c("Train", "Validation"),col=c("black","red"),lty=c(1,1),lwd=c(3,3))  


h2o.auc(m_binomial,valid=FALSE) # on train                   
h2o.auc(m_binomial,valid=TRUE)  # on test

m_binomial@model$training_metrics@metrics$max_criteria_and_metric_scores 





cut_column <- function(data, col) {
  # need lower/upper bound due to h2o.cut behavior (points < the first break or > the last break are replaced with missing value) 
  min_val = min(data$Train[,col])-1
  max_val = max(data$Train[,col])+1
  x = h2o.hist(data$Train[, col])
  # use only the breaks with enough support
  breaks = x$breaks[which(x$counts > 1000)]
  # assign level names 
  lvls = c("min",paste("i_",breaks[2:length(breaks)-1],sep=""),"max")
  col_cut <- paste(col,"_cut",sep="")
  data$Train[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Train[,col],breaks=c(min_val,breaks,max_val)),lvls)
  # now do the same for test and validation, but using the breaks computed on the training!
  if(!is.null(data$Test)) {
    min_val = min(data$Test[,col])-1
    max_val = max(data$Test[,col])+1
    data$Test[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Test[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  if(!is.null(data$Valid)) {
    min_val = min(data$Valid[,col])-1
    max_val = max(data$Valid[,col])+1
    data$Valid[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Valid[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  data
}


interactions <- function(data, cols, pairwise = TRUE) {
  iii = h2o.interaction(data = data$Train, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
  data$Train <- h2o.cbind(data$Train,iii)
  if(!is.null(data$Test)) {
    iii = h2o.interaction(data = data$Test, destination_frame = "itest",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Test <- h2o.cbind(data$Test,iii)
  }
  if(!is.null(data$Valid)) {
    iii = h2o.interaction(data = data$Valid, destination_frame = "ivalid",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Valid <- h2o.cbind(data$Valid,iii)
  }
  data
}


# add features to our cover type example
# let's cut all the numerical columns into intervals and add interactions between categorical terms
add_features <- function(data) {
  names(data) <- c("Train","Test","Valid")
  data = cut_column(data,'Elevation')
  data = cut_column(data,'Hillshade_Noon')
  data = cut_column(data,'Hillshade_9am')
  data = cut_column(data,'Hillshade_3pm')
  data = cut_column(data,'Horizontal_Distance_To_Hydrology')
  data = cut_column(data,'Slope')
  data = cut_column(data,'Horizontal_Distance_To_Roadways')
  data = cut_column(data,'Aspect')
  # pairwise interactions between all categorical columns
  interaction_cols = c("Elevation_cut","Wilderness_Area","Soil_Type","Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut","Horizontal_Distance_To_Hydrology_cut","Slope_cut","Horizontal_Distance_To_Roadways_cut","Aspect_cut")
  data = interactions(data, interaction_cols)
  # interactions between Hillshade columns
  interaction_cols2 = c("Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut")
  data = interactions(data, interaction_cols2,pairwise = FALSE)
  data
}



# Add Features
data_binomial_ext <- add_features(data_binomial)
data_binomial_ext$Train <- h2o.assign(data_binomial_ext$Train,"train_b_ext")
data_binomial_ext$Valid <- h2o.assign(data_binomial_ext$Valid,"valid_b_ext")
data_binomial_ext$Test <- h2o.assign(data_binomial_ext$Test,"test_b_ext")
y = "Cover_Type"
x = names(data_binomial_ext$Train)
x = x[-which(x==y)]


m_binomial_1_ext = try(h2o.glm(training_frame = data_binomial_ext$Train, 
                               validation_frame = data_binomial_ext$Valid, 
                               x = x, y = y, family='binomial'))
h2o.confusionMatrix(m_binomial_1_ext)
h2o.auc(m_binomial_1_ext,valid=TRUE)

m_binomial_1_ext = h2o.glm(training_frame = data_binomial_ext$Train, 
                           validation_frame = data_binomial_ext$Valid, x = x, y = y, 
                           family='binomial', solver='L_BFGS')
h2o.confusionMatrix(m_binomial_1_ext)
h2o.auc(m_binomial_1_ext,valid=TRUE)


m_binomial_2_ext = h2o.glm(training_frame = data_binomial_ext$Train, 
                           validation_frame = data_binomial_ext$Valid, 
                           x = x, y = y, family='binomial', solver='L_BFGS', lambda=1e-4)
h2o.confusionMatrix(m_binomial_2_ext, valid=TRUE)
h2o.auc(m_binomial_2_ext,valid=TRUE)


m_binomial_3_ext = h2o.glm(training_frame = data_binomial_ext$Train, 
                           validation_frame = data_binomial_ext$Valid, x = x, y = y, 
                           family='binomial', lambda_search=TRUE)
h2o.confusionMatrix(m_binomial_3_ext, valid=TRUE)
h2o.auc(m_binomial_3_ext,valid=TRUE)












# Multinomial Model 2
# let's revisit the multinomial case with our new features
data_ext <- add_features(data)
data_ext$Train <- h2o.assign(data_ext$Train,"train_m_ext")
data_ext$Valid <- h2o.assign(data_ext$Valid,"valid_m_ext")
data_ext$Test <- h2o.assign(data_ext$Test,"test_m_ext")
y = "Cover_Type"
x = names(data_ext$Train)
x = x[-which(x==y)]
m2 = h2o.glm(training_frame = data_ext$Train, 
             validation_frame = data_ext$Valid, 
             x = x, y = y,family='multinomial',solver='L_BFGS',lambda=1e-4)
# 21% err down from 28%
h2o.confusionMatrix(m2, valid=TRUE)



