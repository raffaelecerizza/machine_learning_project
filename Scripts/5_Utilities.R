####################################
### Funzioni di utilità generale ###
####################################

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

## Import librerie

library(caret)

## Funzione per lo split del dataset in training set e test set
split.data = function(data, proportion = 0.7, seed = 1) {
  set.seed(seed)
  index = sample(1:nrow(data))
  train = data[index[1:floor(nrow(data) * proportion)], ]
  test = data[index[(ceiling(nrow(data) * proportion) + 1):nrow(data)], ] 
  return(list(train=train, test=test))
}

## Funzione per creare un dataframe per multiROC
create_multiROC_dataframe = function(dt_name, data, probs) {
  
  # Crea nome colonna "No.Use_pred_albero_decisionale"
  names.no_use = paste("No.Use_pred_", dt_name, sep = "") 
  
  # Crea nome colonna "Long.Term_pred_albero_decisionale"
  names.long_term = paste("Long.Term_pred_", dt_name, sep = "")
  
  # Crea nome colonna "Short.Term_pred_albero_decisionale"
  names.short_term = paste("Short.Term_pred_", dt_name, sep = "")
  
  # Crea un dataframe con 6 colonne contenenti solo zeri
  # e un'ulteriore colonna contenente i metodi contraccettivi veri di ogni istanza
  df = cbind(data.frame(matrix(0, ncol = 6, nrow = nrow(data)), 
                        data$Contraceptive_Method_Used))
  
  # Assegna i nomi alle colonne del dataframe
  colnames(df) = c("No.Use_true", "Long.Term_true", "Short.Term_true", 
                   names.no_use, names.long_term, names.short_term, 
                   "Contraceptive_Method_Used")
  
  # Quando il metodo contraccettivo vero dell'istanza è No.Use segna 1 nella colonna "No.Use_true"
  # I valori delle colonne "Long.Term_true" e "Short.Term_true" saranno 0
  df[df$Contraceptive_Method_Used == "No.Use", "No.Use_true"] = 1
  df[df$Contraceptive_Method_Used == "Long.Term", "Long.Term_true"] = 1
  df[df$Contraceptive_Method_Used == "Short.Term", "Short.Term_true"] = 1
  
  # Nella colonna "No.Use" inserisci le probabilità di No.Use
  df[, names.no_use] = probs[, "No.Use"] 
  
  # Nella colonna "Long.Term" inserisci le probabilità di Long.Term
  df[, names.long_term] = probs[, "Long.Term"] 
  
  # Nella colonna "Short.Term" inserisci le probabilità di Short.Term
  df[, names.short_term] = probs[, "Short.Term"] 
  
  # Considera solo le prime 6 colonne del dataframe
  # (quindi senza l'indicazione del metodo contraccettivo vero dell'istanza)
  df[1:6]
}

## Funzione per creare un dataframe per multiROC per problema binario
create_multiROC_dataframe_binario = function(dt_name, data, probs) {
  
  # Crea nome colonna "No_pred_rpart"
  names.no = paste("No_pred_", dt_name, sep = "") 
  
  # Crea nome colonna "Yes_pred_rpart"
  names.yes = paste("Yes_pred_", dt_name, sep = "")
  
  # Crea un dataframe con 4 colonne contenenti solo zeri
  # e un'ulteriore colonna contenente i metodi contraccettivi veri di ogni istanza
  df = cbind(data.frame(matrix(0, ncol = 4, nrow = nrow(data)), 
                        data$Contraceptive_Is_Used))
  
  # Assegna i nomi alle colonne del dataframe
  colnames(df) = c("No_true", "Yes_true", 
                   names.no, names.yes, 
                   "Contraceptive_Is_Used")
  
  # Quando il metodo contraccettivo vero dell'istanza è No segna 1 nella colonna "No_true"
  # Il valore della colonna "Yes_true" sarà 0
  df[df$Contraceptive_Is_Used == "No", "No_true"] = 1
  df[df$Contraceptive_Is_Used == "Yes", "Yes_true"] = 1
  
  # Nella colonna "No" inserisci le probabilità di No
  df[, names.no] = probs[, "No"] 
  
  # Nella colonna "Yes" inserisci le probabilità di Yes
  df[, names.yes] = probs[, "Yes"] 
  
  # Considera solo le prime 4 colonne del dataframe
  # (quindi senza l'indicazione dell'uso contraccettivo vero dell'istanza)
  df[1:4]
}

## Funzione per realizzare il plot della curva ROC
# Ref: https://mran.microsoft.com/snapshot/2018-02-12/web/packages/multiROC/vignettes/my-vignette.html
plot_ROC <- function(result, name){
  res <- result
  n_method <- length(unique(res$Methods))
  n_group <- length(unique(res$Groups))
  res_df <- data.frame(Specificity= numeric(0), Sensitivity= numeric(0), 
                       Group = character(0), AUC = numeric(0), Method = character(0))
  for (i in 1:n_method) {
    for (j in 1:n_group) {
      temp_data_1 <- data.frame(Specificity=res$Specificity[[i]][j],
                                Sensitivity=res$Sensitivity[[i]][j],
                                Group=unique(res$Groups)[j],
                                AUC=res$AUC[[i]][j],
                                Method = unique(res$Methods)[i])
      colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
      res_df <- rbind(res_df, temp_data_1)
      
    }
    temp_data_2 <- data.frame(Specificity=res$Specificity[[i]][n_group+1],
                              Sensitivity=res$Sensitivity[[i]][n_group+1],
                              Group= "Macro",
                              AUC=res$AUC[[i]][n_group+1],
                              Method = unique(res$Methods)[i])
    temp_data_3 <- data.frame(Specificity=res$Specificity[[i]][n_group+2],
                              Sensitivity=res$Sensitivity[[i]][n_group+2],
                              Group= "Micro",
                              AUC=res$AUC[[i]][n_group+2],
                              Method = unique(res$Methods)[i])
    colnames(temp_data_2) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    colnames(temp_data_3) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    res_df <- rbind(res_df, temp_data_2)
    res_df <- rbind(res_df, temp_data_3)
  }
  ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) +
    ggplot2::geom_path(ggplot2::aes(color = Group, linetype=Method)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), colour='grey', linetype = 'dotdash') +
    ggplot2::theme_bw() +
    ggplot2::ggtitle(name) +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), 
                   legend.justification=c(1, 0), legend.position=c(.95, .05), 
                   legend.title=ggplot2::element_blank(), 
                   legend.background = ggplot2::element_rect(fill=NULL, size=0.5, linetype="solid", colour ="black"))
}

## Funzione per effettuare il confronto di curve ROC per problemi multi-classe (senza Macro e Micro)
# Ref: https://mran.microsoft.com/snapshot/2018-02-12/web/packages/multiROC/vignettes/my-vignette.html
plot_ROC_comparison <- function(result1, result2, name){
  res1 <- result1
  res2 <- result2
  n_method <- length(2)
  n_group <- length(unique(res1$Groups))
  res_df <- data.frame(Specificity = numeric(0), Sensitivity = numeric(0), 
                       Group = character(0), AUC = numeric(0), Method = character(0))
  for (i in 1:n_method) {
    for (j in 1:n_group) {
      temp_data_1 <- data.frame(Specificity=res1$Specificity[[i]][j],
                                Sensitivity=res1$Sensitivity[[i]][j],
                                Group=unique(res1$Groups)[j],
                                AUC=res1$AUC[[i]][j],
                                Method = unique(res1$Methods)[i])
      colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
      res_df <- rbind(res_df, temp_data_1)
    }
  }
  
  for (i in 1:n_method) {
    for (j in 1:n_group) {
      temp_data_2 <- data.frame(Specificity=res2$Specificity[[i]][j],
                                Sensitivity=res2$Sensitivity[[i]][j],
                                Group=unique(res2$Groups)[j],
                                AUC=res2$AUC[[i]][j],
                                Method = unique(res2$Methods)[i])
      colnames(temp_data_2) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
      res_df <- rbind(res_df, temp_data_2)
    }
  }
  
  #print(res)  
  #print(res_df)
  
  ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) +
    ggplot2::geom_path(ggplot2::aes(color = Group, linetype = Method)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), colour='grey', linetype = 'dotdash') +
    ggplot2::theme_bw() +
    ggplot2::ggtitle(name) +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), 
                   legend.justification=c(1, 0), legend.position=c(.95, .05), 
                   legend.title=ggplot2::element_blank(), 
                   legend.background = ggplot2::element_rect(fill=NULL, size=0.5, linetype="solid", colour ="black"))
}

## Funzione per effettuare il confronto di curve ROC per problemi multi-classe (solo Macro e Micro)
# Ref: https://mran.microsoft.com/snapshot/2018-02-12/web/packages/multiROC/vignettes/my-vignette.html
plot_ROC_comparison_macro_micro <- function(result1, result2, name){
  res1 <- result1
  res2 <- result2
  n_method <- length(2)
  n_group <- length(unique(res1$Groups))
  res_df <- data.frame(Specificity = numeric(0), Sensitivity = numeric(0), 
                       Group = character(0), AUC = numeric(0), Method = character(0))
  for (i in 1:n_method) {
    for (j in 1:n_group) {
      temp_data_1 <- data.frame(Specificity=res1$Specificity[[i]][j],
                                Sensitivity=res1$Sensitivity[[i]][j],
                                Group=unique(res1$Groups)[j],
                                AUC=res1$AUC[[i]][j],
                                Method = unique(res1$Methods)[i])
      colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    }
    temp_data_2 <- data.frame(Specificity=res1$Specificity[[i]][n_group+1],
                              Sensitivity=res1$Sensitivity[[i]][n_group+1],
                              Group= "Macro",
                              AUC=res1$AUC[[i]][n_group+1],
                              Method = unique(res1$Methods)[i])
    temp_data_3 <- data.frame(Specificity=res1$Specificity[[i]][n_group+2],
                              Sensitivity=res1$Sensitivity[[i]][n_group+2],
                              Group= "Micro",
                              AUC=res1$AUC[[i]][n_group+2],
                              Method = unique(res1$Methods)[i])
    colnames(temp_data_2) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    colnames(temp_data_3) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    res_df <- rbind(res_df, temp_data_2)
    res_df <- rbind(res_df, temp_data_3)
  }
  
  for (i in 1:n_method) {
    for (j in 1:n_group) {
      temp_data_1 <- data.frame(Specificity=res2$Specificity[[i]][j],
                                Sensitivity=res2$Sensitivity[[i]][j],
                                Group=unique(res2$Groups)[j],
                                AUC=res2$AUC[[i]][j],
                                Method = unique(res2$Methods)[i])
      colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    }
    temp_data_2 <- data.frame(Specificity=res2$Specificity[[i]][n_group+1],
                              Sensitivity=res2$Sensitivity[[i]][n_group+1],
                              Group= "Macro",
                              AUC=res2$AUC[[i]][n_group+1],
                              Method = unique(res2$Methods)[i])
    temp_data_3 <- data.frame(Specificity=res2$Specificity[[i]][n_group+2],
                              Sensitivity=res2$Sensitivity[[i]][n_group+2],
                              Group= "Micro",
                              AUC=res2$AUC[[i]][n_group+2],
                              Method = unique(res2$Methods)[i])
    colnames(temp_data_2) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    colnames(temp_data_3) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    res_df <- rbind(res_df, temp_data_2)
    res_df <- rbind(res_df, temp_data_3)
  }
  ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) +
    ggplot2::geom_path(ggplot2::aes(color = Group, linetype = Method)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), colour='grey', linetype = 'dotdash') +
    ggplot2::theme_bw() +
    ggplot2::ggtitle(name) +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), 
                   legend.justification=c(1, 0), legend.position=c(.95, .05), 
                   legend.title=ggplot2::element_blank(), 
                   legend.background = ggplot2::element_rect(fill=NULL, size=0.5, linetype="solid", colour ="black"))
}

## Impostazione della 10-Fold CV per problema binario
train_control_bin = trainControl(method = "cv", 
                                 number = 10,
                                 classProbs = TRUE, 
                                 savePredictions = "final",
                                 summaryFunction = twoClassSummary,
                                 verboseIter = TRUE)

## Impostazione della 10-Fold CV per problema multiclasse
train_control_multi = trainControl(method = "cv", 
                                   number = 10,
                                   classProbs = TRUE, 
                                   savePredictions = "final",
                                   summaryFunction = multiClassSummary, 
                                   verboseIter = TRUE)

## Impostazione della 10-Fold CV ripetuta per problema binario
train_control_bin_repeated = trainControl(method = "repeatedcv", 
                                          repeats = 3,
                                          number = 10,
                                          classProbs = TRUE, 
                                          savePredictions = "final",
                                          summaryFunction = twoClassSummary,
                                          verboseIter = TRUE)

## Impostazione della 10-Fold CV ripetuta per problema multiclasse
train_control_multi_repeated = trainControl(method = "repeatedcv", 
                                            repeats = 3,
                                            number = 10,
                                            classProbs = TRUE, 
                                            savePredictions = "final",
                                            summaryFunction = multiClassSummary, 
                                            verboseIter = TRUE)

## Funzione per calcolo del cut-off ottimale
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

## Funzione per eliminare tutti i package istallati (tranne quelli base)
package.removeAll = function() {
  # create a list of all installed packages
  ip <- as.data.frame(installed.packages())
  # if you use MRO, make sure that no packages in this library will be removed
  ip <- subset(ip, !grepl("MRO", ip$LibPath))
  # we don't want to remove base or recommended packages either\
  ip <- ip[!(ip[,"Priority"] %in% c("base", "recommended")),]
  # determine the library where the packages are installed
  path.lib <- unique(ip$LibPath)
  # create a vector with all the names of the packages you want to remove
  pkgs.to.remove <- ip[,1]
  # remove the packages
  sapply(pkgs.to.remove, remove.packages, lib = path.lib)
}