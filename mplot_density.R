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
                    score = as.numeric(score),
                    norm_score = lares::normalize(as.numeric(score)))
  
  p1 <- ggplot(out) + theme_minimal() +
    geom_density(aes(x = 100 * score, group = tag, fill = as.character(tag)), 
                 alpha = 0.6, adjust = 0.25) + 
    guides(fill = guide_legend(title="Tag")) + 
    xlim(0, 100) + 
    labs(title = "Score distribution for binary model",
         y = "Density by tag", x = "Score")
  
  p2 <- ggplot(out) + theme_minimal() + 
    geom_density(aes(x = 100 * score), 
                 alpha = 0.9, adjust = 0.25, fill = "deepskyblue") + 
    labs(x = "", y = "Density")
  
  p3 <- ggplot(out) + theme_minimal() + 
    geom_line(aes(x = score * 100, y = 100 * (1 - ..y..), color = as.character(tag)), 
              stat = 'ecdf', size = 1) +
    geom_line(aes(x = score * 100, y = 100 * (1 - ..y..)), 
              stat = 'ecdf', size = 0.5, colour = "black", linetype="dotted") +
    ylab('Cumulative') + xlab('') + guides(color=FALSE)
  
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


mplot_density(tag=as.vector(bmk$AEFL), score=as.vector(pred_AE), model_name = model_loaded_AE, subtitle = NA, 
                          save = FALSE, file_name = "viz_distribution.png")

