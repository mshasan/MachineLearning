library(lares)
library(mlbench)
library(magrittr)   # for %>% operator
library(plyr)       # rename function
library(pROC)
library(beepr)
#data("PimaIndiansDiabetes")




mplot_density <- function(tag, score, model_name = NA, subtitle = NA, 
                          save = FALSE, file_name = "viz_distribution.png") {
  require(ggplot2)
  require(gridExtra)
  
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  if (length(unique(tag)) != 2) {
    stop("This function is for binary models. You should only have 2 unique values for the tag value!")
  }
  
  out <- data.frame(tag = as.character(tag),
                    score = as.numeric(score))
  
  p1 <- ggplot(out) + theme_minimal() +
    geom_density(aes(x = 100 * score, group = tag, fill = as.character(tag)), 
                 alpha = 0.6, adjust = 0.25) + 
    guides(fill = guide_legend(title="Tag")) + 
    xlim(0, 100) + 
    labs(title = "Distribution of predicted probability",
         y = "Density by tag", x = "Score")
  
  p2 <- ggplot(out) + theme_minimal() + 
    geom_density(aes(x = 100 * score), 
                 alpha = 0.9, adjust = 0.25, fill = "deepskyblue") + 
    labs(title = "Overall distribution", x = "", y = "Density")
  
  p3 <- ggplot(out) + theme_minimal() + 
    geom_line(aes(x = score * 100, y = 100 * (1 - ..y..), color = as.character(tag)), 
              stat = 'ecdf', size = 1) +
    geom_line(aes(x = score * 100, y = 100 * (1 - ..y..)), 
              stat = 'ecdf', size = 0.5, colour = "black", linetype="dotted") +
    ylab('Cumulative') + xlab('') + guides(color=FALSE)+
    labs(title = "Cummulative distribution by group")
  
  if(!is.na(subtitle)) {
    p1 <- p1 + labs(subtitle = subtitle)
  }
  
  if(!is.na(model_name)) {
    p1 <- p1 + labs(caption = model_name)
  }
  
  if(save == TRUE) {
    png(file_name, height = 1800, width = 2100, res = 300)
    grid.arrange(
      p1, p2, p3, 
      ncol = 2, nrow = 2, heights = 2:1,
      layout_matrix = rbind(c(1,1), c(2,3)))
    dev.off()
  }
  
  return(
    grid.arrange(
      p1, p2, p3, 
      ncol = 2, nrow = 2, heights = 2:1,
      layout_matrix = rbind(c(1,1), c(2,3))))
  
}


# only probability density plots
mplot_density2 <- function(tag, score, xlower = 0, title = "Training")
{
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  if (length(unique(tag)) != 2) {
    stop("This function is for binary models. You should only have 2 unique values for the tag value!")
  }

  out <- data.frame(tag = as.character(tag),
                    score = as.numeric(score))
  
  p <- ggplot(out) + 
    theme_minimal() +
    geom_density(aes(x = 100 * score, group = tag, fill = as.character(tag)), 
                 alpha = 0.8, adjust = 0.5) + 
    guides(fill = guide_legend(title = "Event")) + 
    xlim(xlower, 100) + 
    labs(title = title, y = "Density", x = "Predicted probability") + 
    scale_fill_manual(values = c("cyan2", "coral1"))
    theme(legend.title = element_blank(), legend.position = "bottom") 
  
  return(p)
}






# ROC Curve
mplot_roc <- function(tag, score, model_name = NA, subtitle = NA, interval = 0.2, plotly = FALSE,
                      save = FALSE, file_name = "viz_roc.png") {
  require(pROC)
  require(ggplot2)
  
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  roc <- pROC::roc(tag, score, ci=T)
  coords <- data.frame(
    x = rev(roc$specificities),
    y = rev(roc$sensitivities))
  ci <- data.frame(roc$ci, row.names = c("min","AUC","max"))
  
  p <- ggplot(coords, aes(x = x, y = y)) +
    geom_line(colour = "deepskyblue", size = 1) +
    geom_point(colour = "blue3", size = 0.9, alpha = 0.4) +
    geom_segment(aes(x = 0, y = 1, xend = 1, yend = 0), alpha = 0.2, linetype = "dotted") + 
    scale_x_reverse(name = "1 - Specificity [False Positive Rate]", limits = c(1,0), 
                    breaks = seq(0, 1, interval), expand = c(0.001,0.001)) + 
    scale_y_continuous(name = "Sensitivity [True Positive Rate]", limits = c(0,1), 
                       breaks = seq(0, 1, interval), expand = c(0.001, 0.001)) +
    theme_minimal() + 
    theme(axis.ticks = element_line(color = "grey80")) +
    coord_equal() + 
    ggtitle("ROC Curve: AUC") +
    annotate("text", x = 0.25, y = 0.10, vjust = 0, size = 4.2, 
             label = paste("AUC =", round(100*ci[c("AUC"),],2))) +
    annotate("text", x = 0.25, y = 0.05, vjust = 0, size = 2.8, 
             label = paste0("95% CI: ", 
                            round(100*ci[c("min"),],2),"-", 
                            round(100*ci[c("max"),],2)))
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  }  
  
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  
  if (plotly == TRUE) {
    require(plotly)
    p <- ggplotly(p)
  }
  
  if (save == TRUE) {
    p <- p + ggsave(file_name, width = 6, height = 6)
  }
  return(p)
}



mplot_cuts <- function(score, splits = 10, subtitle = NA, model_name = NA, 
                       save = FALSE, file_name = "viz_ncuts.png") {
  
  require(ggplot2)
  
  if (splits > 25) {
    stop("You should try with less splits!")
  }
  
  deciles <- quantile(score, 
                      probs = seq((1/splits), 1, length = splits), 
                      names = TRUE)
  deciles <- data.frame(cbind(Deciles=row.names(as.data.frame(deciles)),
                              Threshold=as.data.frame(deciles)))
  
  p <- ggplot(deciles, 
              aes(x = reorder(Deciles, deciles), y = deciles * 100, 
                  label = round(100 * deciles, 1))) + 
    geom_col(fill="deepskyblue") + 
    xlab('') + theme_minimal() + ylab('Score') + 
    geom_text(vjust = 1.5, size = 3, inherit.aes = TRUE, colour = "white", check_overlap = TRUE) +
    labs(title = paste("Cuts by score: using", splits, "equal-sized buckets"))
  
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  } 
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  if (save == TRUE) {
    p <- p + ggsave(file_name, width = 6, height = 6)
  }
  return(p)
}


mplot_splits <- function(tag, score, splits = 5, subtitle = NA, model_name = NA, facet = NA, 
                         save = FALSE, subdir = NA, file_name = "viz_splits.png") {
  
  require(ggplot2)
  require(dplyr)
  require(RColorBrewer)
  
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  if (splits > 10) {
    stop("You should try with less splits!")
  }
  
  df <- data.frame(tag, score, facet)
  npersplit <- round(nrow(df)/splits)
  names <- df %>% 
    mutate(quantile = ntile(score, splits)) %>% group_by(quantile) %>%
    summarise(n = n(), 
              max_score = round(100 * max(score), 1), 
              min_score = round(100 * min(score), 1)) %>%
    mutate(quantile_tag = paste0(quantile," (",min_score,"-",max_score,")"))
  
  p <- df %>% 
    mutate(quantile = ntile(score, splits)) %>% 
    group_by(quantile, facet, tag) %>% tally() %>%
    ungroup() %>% group_by(facet, tag) %>% 
    arrange(desc(quantile)) %>%
    mutate(p = round(100*n/sum(n),2),
           cum = cumsum(100*n/sum(n))) %>%
    left_join(names, by = c("quantile")) %>%
    ggplot(aes(x = as.character(tag), y = p, label = as.character(p),
               fill = as.character(quantile_tag))) + theme_minimal() +
    geom_col(position = "stack") +
    geom_text(size = 3, position = position_stack(vjust = 0.5), check_overlap = TRUE) +
    xlab("Tag") + ylab("Total Percentage by Tag") +
    guides(fill = guide_legend(title=paste0("~",npersplit," p/split"))) +
    labs(title = "Tag vs Score Splits Comparison") +
    scale_fill_brewer(palette = "Spectral")
  
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  }  
  
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  
  if(!is.na(facet)) {
    p <- p + facet_grid(. ~ facet, scales = "free")
  }  
  
  if (!is.na(subdir)) {
    dir.create(file.path(getwd(), subdir))
    file_name <- paste(subdir, file_name, sep="/")
  }
  
  if (save == TRUE) {
    p <- p + ggsave(file_name, width = 6, height = 6)
  }
  
  return(p)
  
}




mplot_full <- function(tag, score, splits = 8, subtitle = NA, model_name = NA, 
                       save = FALSE, file_name = "viz_full.png", subdir = NA) {
  require(ggplot2)
  require(gridExtra)
  options(warn=-1)
  
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  p1 <- mplot_density(tag = tag, score = score, subtitle = subtitle, model_name = model_name)
  p2 <- mplot_splits(tag = tag, score = score, splits = splits)
  p3 <- mplot_roc(tag = tag, score = score)
  p4 <- mplot_cuts(score = score)
  
  if(save == TRUE) {
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir))
      file_name <- paste(subdir, file_name, sep="/")
    }
    png(file_name, height = 2000, width = 3200, res = 300)
    grid.arrange(
      p1, p2, p3, p4,
      widths = c(1.3,1),
      layout_matrix = rbind(c(1,2), c(1,2), c(1,3), c(4,3)))
    dev.off()
  }
  return(
    grid.arrange(
      p1, p2, p3, p4,
      widths = c(1.3,1),
      layout_matrix = rbind(c(1,2), c(1,2), c(1,3), c(4,3)))
  ) 
}


mplot_importance <- function(var, imp, colours = NA, limit = 15, model_name = NA, subtitle = NA,
                             save = FALSE, file_name = "viz_importance.png", subdir = NA) {
  
  require(ggplot2)
  require(gridExtra)
  options(warn=-1)
  
  if (length(var) != length(imp)) {
    message("The variables and importance values vectors should be the same length.")
    stop(message(paste("Currently, there are",length(var),"variables and",length(imp),"importance values!")))
  }
  if (is.na(colours)) {
    colours <- "deepskyblue" 
  }
  out <- data.frame(var = var, imp = imp, Type = colours)
  if (length(var) < limit) {
    limit <- length(var)
  }
  
  output <- out[1:limit,]
  
  p <- ggplot(output, 
              aes(x = reorder(var, imp), y = imp * 100, 
                  label = round(100 * imp, 1))) + 
    geom_col(aes(fill = Type), width = 0.1) +
    geom_point(aes(colour = Type), size = 6) + 
    coord_flip() + xlab('') + theme_minimal() +
    ylab('Importance') + 
    geom_text(hjust = 0.5, size = 2, inherit.aes = TRUE, colour = "white") +
    labs(title = paste0("Variables Importances. (", limit, " / ", length(var), " plotted)"))
  
  if (length(unique(output$Type)) == 1) {
    p <- p + geom_col(fill = colours, width = 0.2) +
      geom_point(colour = colours, size = 6) + 
      guides(fill = FALSE, colour = FALSE) + 
      geom_text(hjust = 0.5, size = 2, inherit.aes = TRUE, colour = "white")
  }
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  }  
  if(save == TRUE) {
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir))
      file_name <- paste(subdir, file_name, sep="/")
    }
    p <- p + ggsave(file_name, width=7, height=6)
  }
  
  return(p)
  
}

# subtitle <- paste(results$project, "- AUC:", round(100 * results$auc_test, 2))
# subdir <- paste0("Models/", round(100*results$auc_test, 2), "-", results$project)
# 
# 
# library(lares)
# library(mlbench)
# library(magrittr)   # for %>% operator
# library(plyr)       # rename function
# data("PimaIndiansDiabetes")
# 
# df <- PimaIndiansDiabetes %>% dplyr::rename("tag" = "diabetes")
# results <- lares::h2o_automl(df, seed = 1)
# mplot_full(tag = results$scores$tag,
#            score = results$scores$score,
#            subtitle = "Example using Pima Indians dataset",
#            model_name = results$model_name,
#            save = TRUE)



