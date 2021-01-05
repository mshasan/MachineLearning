# modified h2o_automl function

h2o_automl2 <- function (df, train_test = NA, split = 0.7, seed = 0, thresh = 6, 
          max_time = 5 * 60, max_models = 25, alarm = TRUE, export = FALSE, 
          project = "Machine Learning Model") 
{
  require(dplyr)
  require(h2o)
  require(beepr)
  options(warn = -1)
  start <- Sys.time()
  message(paste(start, "| Started process..."))
  df <- data.frame(df) %>% filter(!is.na(tag))
  type <- ifelse(length(unique(df$tag)) <= as.integer(thresh), 
                 "Classifier", "Regression")
  if (!"tag" %in% colnames(df)) {
    stop("You should have a 'tag' column in your data.frame!")
  }
  if (is.na(train_test)) {
    splits <- lares::msplit(df, size = split, seed = seed)
    train <- splits$train
    test <- splits$test
    if (type == "Classifier") {
      print(table(train$tag))
      train$tag <- as.factor(as.character(train$tag))
    }
    if (type == "Regression") {
      print(summary(train$tag))
    }
  }
  else {
    if ((!unique(train_test) %in% c("train", "test")) & (length(unique(train_test)) != 
                                                         2)) {
      stop("Your train_test column should have 'train' and 'test' values only!")
    }
    train <- df %>% filter(train_test == "train")
    test <- df %>% filter(train_test == "test")
    test$tag <- NULL
    print(table(train_test))
  }
  h2o.init(nthreads = -1, port = 54321, min_mem_size = "8g")
  h2o.removeAll()
  aml <- h2o.automl(x = setdiff(names(df), "tag"), y = "tag", 
                    training_frame = as.h2o(train), leaderboard_frame = as.h2o(test), 
                    max_runtime_secs = max_time, max_models = max_models, 
                    exclude_algos = c("StackedEnsemble", "DeepLearning"), 
                    nfolds = 30, seed = seed)
  print(aml@leaderboard[, 1:3])
  m <- h2o.getModel(as.vector(aml@leaderboard$model_id[1]))
  scores <- predict(m, as.h2o(test))
  if (type == "Classifier") {
    require(pROC)
    results <- list(project = project, model = m, scores = data.frame(index = c(1:nrow(test)), 
                                                                      tag = as.vector(test$tag), score = as.vector(scores[, 
                                                                                                                          3]), norm_score = lares::normalize(as.vector(scores[, 
                                                                                                                                                                              3]))), scoring_history = data.frame(m@model$scoring_history), 
                    datasets = list(test = test, train = train), parameters = m@parameters, 
                    importance = data.frame(h2o.varimp(m)), auc_test = NA, 
                    logloss_test = NA, model_name = as.vector(m@model_id), 
                    algorithm = m@algorithm, leaderboard = aml@leaderboard, 
                    seed = seed)
    roc <- pROC::roc(results$scores$tag, results$scores$score, 
                     ci = T)
    results$auc_test <- roc$auc
    if (length(unique(test$tag)) == 2) {
      results$logloss_test <- lares::loglossBinary(tag = results$scores$tag, 
                                                   score = results$scores$score)
    }
  }
  if (type == "Regression") {
    results <- list(project = project, model = m, scores = data.frame(index = c(1:nrow(test)), 
                                                                      tag = as.vector(test$tag), score = as.vector(scores$predict)), 
                    scoring_history = data.frame(m@model$scoring_history), 
                    datasets = list(test = test, train = train), parameters = m@parameters, 
                    importance = data.frame(h2o.varimp(m)), model_name = as.vector(m@model_id), 
                    algorithm = m@algorithm, leaderboard = aml@leaderboard)
  }
  message(paste0("Training duration: ", round(difftime(Sys.time(), 
                                                       start, units = "secs"), 2), "s"))
  if (export == TRUE) {
    lares::export_results(results)
  }
  if (alarm == TRUE) {
    beepr::beep()
  }
  return(results)
}
