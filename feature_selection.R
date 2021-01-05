# a convenience function to cut the column into intervals 
# working on all three of our datasets (Train/Validation/Test)

cut_column <- function(data, col) {
  # need lower/upper bound due to h2o.cut behavior (points < the first break or > the last break are replaced with missing value) 
  min_val = min(data$Train[,col])-1
  max_val = max(data$Train[,col])+1
  x = h2o.hist(data$Train[, col])
  # use only the breaks with enough support
  breaks = x$breaks #[which(x$counts > 1000)]
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

#cut_column(data=data_AE, col="LYSOZYME")


# Now let's make a convenience function generating interaction terms on 
# all three of our datasets. We'll use h2o.interaction:
  
interactions <- function(data, cols, pairwise = TRUE) {
    iii = h2o.interaction(data = data$Train, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=10)
    data$Train <- h2o.cbind(data$Train,iii)
    if(!is.null(data$Test)) {
      iii = h2o.interaction(data = data$Test, destination_frame = "itest",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=10)
      data$Test <- h2o.cbind(data$Test,iii)
    }
    if(!is.null(data$Valid)) {
      iii = h2o.interaction(data = data$Valid, destination_frame = "ivalid",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=10)
      data$Valid <- h2o.cbind(data$Valid,iii)
    }
    data
  }


#interactions(data=data_AE, cols=interaction_cols)


# Finally, let's wrap addition of the features into a separate function call, 
# as we will use it again later. We'll add intervals for each numeric 
# column and interactions between each pair of binary columns.

# add features to our cover type example
# let's cut all the numerical columns into intervals and add interactions between categorical terms
add_features <- function(data) {
  names(data) <- c("Train","Valid")
  data = cut_column(data,'LYSOZYME')
  data = cut_column(data,'LIPOCALIN')
  data = cut_column(data,'sIgA')
  data = cut_column(data,'LACTOFERRIN')
  data = cut_column(data,'ALBUMIN')
  data = cut_column(data,'KDA88')
  data = cut_column(data,'KDA23')
  # pairwise interactions between all catego"rical columns
  interaction_cols = c("LYSOZYME_cut", "LIPOCALIN_cut", "sIgA_cut", "LACTOFERRIN_cut", "ALBUMIN_cut", "KDA88_cut", "KDA23_cut")
  data = interactions(data, interaction_cols)
  # interactions between Hillshade columns
  interaction_cols2 = c("LYSOZYME_cut","sIgA_cut","LACTOFERRIN_cut")
  data = interactions(data, interaction_cols2,pairwise = FALSE)
  data
}

