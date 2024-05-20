#############################
## 4 - Models Comparison   ##
#############################

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

#### Caricamento modelli

source(paste(getwd(), "/Scripts/3_1_1_DecisionTree_NoPca.R",sep = ""))

source(paste(getwd(), "/Scripts/3_2_1_NeuralNet_NoPca.R",sep = ""))

#### Caricamento funzione di utilita'

source(paste(getwd(), "/Scripts/5_Utilities.R",sep = ""))

#### Import librerie necessarie

library(caret) 

library(pROC) 

library(multiROC) 

#### Confronto tramite curve ROC
## Plot delle curve ROC dei modelli per la classe Yes sul problema binario
plot(dt_total_bin_no_pca_yes.ROC, 
     type = "S", 
     col = "blue", 
     print.auc = TRUE,
     print.auc.x = 0.4,
     print.auc.y = 0.3)
plot(nn_total_bin_no_pca_yes.ROC, 
     add = TRUE, 
     col = "green", 
     print.auc = TRUE,
     print.auc.x = 0.9,
     print.auc.y = 0.8)
legend("bottomright", 
       legend = c("Decision Tree", "Neural Network"), 
       col = c("blue", "green"),
       lty = c(1, 1),
       lwd = c(1, 1))

## Plot delle curve ROC dei modelli per la classe No sul problema binario
plot(dt_total_bin_no_pca_no.ROC, 
     type = "S", 
     col = "blue", 
     print.auc = TRUE,
     print.auc.x = 0.4,
     print.auc.y = 0.3)
plot(nn_total_bin_no_pca_no.ROC, 
     add = TRUE, 
     col = "green", 
     print.auc = TRUE,
     print.auc.x = 0.9,
     print.auc.y = 0.8)
legend("bottomright", 
       legend = c("Decision Tree", "Neural Network"), 
       col = c("blue", "green"),
       lty = c(1, 1),
       lwd = c(1, 1))

## Plot delle curve ROC dei modelli per le classi sul problema multi-classe
plot_ROC_comparison(dt_total_multi_no_pca.res_multi_roc,
                    nn_total_multi_no_pca.res_multi_roc,
                    "Confronto curve ROC classi")

## Plot delle curve ROC dei modelli per le medie Macro e Micro 
## sul problema multi-classe
plot_ROC_comparison_macro_micro(dt_total_multi_no_pca.res_multi_roc,
                                nn_total_multi_no_pca.res_multi_roc,
                                "Confronto curve ROC per medie Macro e Micro")

#### Confronto tra modelli sul problema binario
cv.values = resamples(list(DT = dt_total_bin_no_pca, 
                           NN = nn_total_bin_no_pca))
summary(cv.values)

dotplot(cv.values, metric = "ROC")

bwplot(cv.values, layout = c(3, 1))

splom(cv.values, metric = "ROC")

#### Confronto tra modelli sul problema multi-classe
cv.values = resamples(list(DT = dt_total_multi_no_pca, 
                           NN = nn_total_multi_no_pca))
summary(cv.values)

## Errore
#dotplot(cv.values, metric = "ROC")

bwplot(cv.values, layout = c(2, 1))

## Errore
#splom(cv.values, metric = "ROC")