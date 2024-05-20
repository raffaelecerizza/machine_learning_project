########################################
## 3.1.1 - Decision Tree senza PCA    ##
########################################

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

#### Caricamento dataset (e istallazione package)

source(paste(getwd(), "/Scripts/2_2_Pca.R",sep = ""))

#### Caricamento funzione di utilita'

source(paste(getwd(), "/Scripts/5_Utilities.R",sep = ""))

#### Import librerie necessarie

library(caret) # necessario
library(rpart) # necessario
library(rattle) # necessario
library(rpart.plot) # necessario
library(RColorBrewer)
library(ROCR) # necessario
library(pROC) # necessario
library(multiROC) # necessario
library(dplyr)
library(MLeval)

## Seed per la riproducibilita'
set.seed(100)

#### 3.1.1.1 - Creazione del modello di albero decisionale CART
#### con 10-fold CV per problema binario (senza PCA)

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_bin$Contraceptive_Is_Used) =
  make.names(levels(factor(cmc_bin$Contraceptive_Is_Used)))

## Creazione dell'albero decisionale
## La 10-fold CV viene eseguita su tutto il dataset
dt_total_bin_no_pca = train(Contraceptive_Is_Used~.,
                            data = cmc_bin,
                            trControl = train_control_bin,
                            method ="rpart",
                            metric = "ROC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_total_bin_no_pca)

## Importanza delle variabili
varImp(dt_total_bin_no_pca)

## Plot dell'albero decisionale
plot(dt_total_bin_no_pca)
plot(dt_total_bin_no_pca$finalModel)
text(dt_total_bin_no_pca$finalModel)
fancyRpartPlot(dt_total_bin_no_pca$finalModel)

## Calcolo matrice di confusione
confusionMatrix(dt_total_bin_no_pca)

## Matrice di confusione per No
confmat_no = confusionMatrix(dt_total_bin_no_pca$pred$pred,
                             as.factor(dt_total_bin_no_pca$pred$obs),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
## Confusion Matrix and Statistics
##
##           Reference
## Prediction  No Yes
##        No  316 118
##        Yes 313 726
##
## Accuracy (average) : 0.7074 
## 'Positive' Class : No

## Matrice di confusione per Yes
confmat_yes = confusionMatrix(dt_total_bin_no_pca$pred$pred,
                              as.factor(dt_total_bin_no_pca$pred$obs),
                              mode = "prec_recall",
                              positive = "Yes")
confmat_yes
## Cambiano solo i valori di Precision, Recall, F1 ecc.

## Stampa matrice di confusione
# Ref: https://stackoverflow.com/questions/37897252/plot-confusion-matrix-in-r-using-ggplot
plt = as.data.frame(confmat_no$table)
plt$Prediction = factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(position = "top", labels=c("No","Yes")) +
  scale_y_discrete(labels=c("Yes","No"))
rm(plt)

precision_no = confmat_no[["byClass"]][["Precision"]]
precision_no
## Precision No: 0.7281106
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.6987488
values_vector = c(precision_no, precision_yes)
precision_macro_average = mean(values_vector)
precision_macro_average
## Precision Macro Average: 0.7134297

## Rimozione dei dati
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.5023847
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.8601896
values_vector = c(recall_no, recall_yes)
recall_macro_average = mean(values_vector)
recall_macro_average
## Recall Macro Average: 0.6812872

## Rimozione dei dati
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.5945437
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7711099
values_vector = c(f1_no, f1_yes)
f1_macro_average = mean(values_vector)
f1_macro_average
## F1-Measure Macro Average: 0.6828268

## Rimozione dei dati
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = ROCR::prediction(dt_total_bin_no_pca$pred$Yes, 
                             dt_total_bin_no_pca$pred$obs)

## Usa la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

## Visualizza la curva ROC per il Yes
plot(perf.tpr.rocr, colorize = T, main = paste("AUC:", (perf.rocr@y.values)))
abline(a = 0, b = 1)

## Calcolo del cut-off ottimale
print(opt.cut(perf.tpr.rocr, pred.rocr))

## Creazione variabili per le curve ROC da confrontare con altri modelli
dt_total_bin_no_pca_yes.ROC = roc(response = dt_total_bin_no_pca$pred$obs,
                                  predictor = dt_total_bin_no_pca$pred$Yes,
                                  levels = levels(cmc_bin[,c("Contraceptive_Is_Used")]))
dt_total_bin_no_pca_no.ROC = roc(response = dt_total_bin_no_pca$pred$obs,
                                 predictor = dt_total_bin_no_pca$pred$No,
                                 levels = levels(cmc_bin[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
dt_total_bin_no_pca$times
## Everything:
##    utente   sistema trascorso
##      0.72      0.00      0.76 
## Final:
##    utente   sistema trascorso
##      0.02      0.00      0.01
## Prediction:
##      NA        NA        NA

## Rimozione dei dati non più necessari
rm(confmat_no)
rm(confmat_yes)
rm(values_vector)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)



#### 3.1.1.2 - Creazione di albero decisionale CART con 10-fold CV ripetuta 
#### 3 volte con split 70/30 per problema binario (senza PCA)

## Split del dataset
allset_no_pca = split.data(cmc_bin, proportion = 0.7, seed = 1)
trainset_no_pca = allset_no_pca$train
testset_no_pca = allset_no_pca$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_no_pca$Contraceptive_Is_Used) =
  make.names(levels(factor(trainset_no_pca$Contraceptive_Is_Used)))
levels(testset_no_pca$Contraceptive_Is_Used) =
  make.names(levels(factor(testset_no_pca$Contraceptive_Is_Used)))

## Creazione dell'albero decisionale
dt_split_bin_no_pca = train(Contraceptive_Is_Used~.,
                            data = trainset_no_pca,
                            trControl = train_control_bin_repeated,
                            method ="rpart",
                            metric = "ROC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_split_bin_no_pca)

## Importanza delle variabili
varImp(dt_split_bin_no_pca)

## Plot dell'albero decisionale
plot(dt_split_bin_no_pca)
plot(dt_split_bin_no_pca$finalModel)
text(dt_split_bin_no_pca$finalModel)
fancyRpartPlot(dt_split_bin_no_pca$finalModel)

## Predizione sul testset
dt_split_bin_no_pca.pred = predict(dt_split_bin_no_pca,
                                   testset_no_pca[, !names(testset_no_pca) %in% 
                                                    c("Contraceptive_Is_Used")])

## Predizione sul testset (con probabilita')
dt_split_bin_no_pca.probs = predict(dt_split_bin_no_pca,
                                 testset_no_pca[, !names(testset_no_pca) %in% 
                                                  c("Contraceptive_Is_Used")],
                                 type = "prob")

## Matrice di confusione per No
confmat_no = confusionMatrix(dt_split_bin_no_pca.pred,
                             as.factor(testset_no_pca$Contraceptive_Is_Used),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
## La matrice di confusione e' calcolata solo sul testset
## Confusion Matrix and Statistics
##
##           Reference
## Prediction  No Yes
##        No   73  65
##        Yes  88 215
##
## Accuracy : 0.6531
## 'Positive' Class : No

## Matrice di confusione per Yes
confmat_yes = confusionMatrix(dt_split_bin_no_pca.pred,
                              as.factor(testset_no_pca$Contraceptive_Is_Used),
                              mode = "prec_recall",
                              positive = "Yes")
confmat_yes
## Cambiano solo i valori di Precision, Recall, F1 ecc.

## Stampa matrice di confusione
# Ref: https://stackoverflow.com/questions/37897252/plot-confusion-matrix-in-r-using-ggplot
plt = as.data.frame(confmat_no$table)
colnames(plt) = c("Prediction", "Reference", "Freq")
plt$Prediction = factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(position = "top", labels=c("No","Yes")) +
  scale_y_discrete(labels=c("Yes","No"))

precision_no = confmat_no[["byClass"]][["Precision"]]
precision_no
## Precision No: 0.5289855
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.709571
values_vector = c(precision_no, precision_yes)
precision_macro_average = mean(values_vector)
precision_macro_average
## Precision Macro Average: 0.6192782

## Rimozione dei dati
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.4534161
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.7678571
values_vector = c(recall_no, recall_yes)
recall_macro_average = mean(values_vector)
recall_macro_average
## Recall Macro Average: 0.6106366

## Rimozione dei dati
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.4882943
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7375643
values_vector = c(f1_no, f1_yes)
f1_macro_average = mean(values_vector)
f1_macro_average
## F1-Measure Macro Average: 0.6129293

## Rimozione dei dati
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Recupera le probabilita' di Yes
pred.to.roc = dt_split_bin_no_pca.probs[, 2]

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = prediction(pred.to.roc, testset_no_pca$Contraceptive_Is_Used)

## Usa la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

## Visualizza la curva ROC per il Yes
plot(perf.tpr.rocr, colorize = T, main=paste("AUC:", (perf.rocr@y.values)))
abline(a = 0, b = 1)

## Creazione variabili per le curve ROC da confrontare con altri modelli
dt_split_bin_no_pca_yes.ROC = roc(response = testset_no_pca[,c("Contraceptive_Is_Used")],
                                  predictor = dt_split_bin_no_pca.probs$Yes,
                                  levels = levels(testset_no_pca[,c("Contraceptive_Is_Used")]))
dt_split_bin_no_pca_no.ROC = roc(response = testset_no_pca[,c("Contraceptive_Is_Used")],
                                 predictor = dt_split_bin_no_pca.probs$No,
                                 levels = levels(testset_no_pca[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
dt_split_bin_no_pca$times
## Everything:
##    utente   sistema trascorso
##      0.96      0.00      0.99  
## Final:
##    utente   sistema trascorso
##      0.01      0.00      0.02
## Prediction:
##      NA        NA        NA

## Rimozione dei dati non piu' necessari
rm(allset_no_pca)
rm(confmat_no)
rm(confmat_yes)
rm(values_vector)
rm(pred.to.roc)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)
rm(plt)



#### 3.1.1.3 - Creazione del modello di albero decisionale CART con 10-fold CV
#### per problema multi-classe (senza PCA)

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_multi$Contraceptive_Method_Used) =
  make.names(levels(factor(cmc_multi$Contraceptive_Method_Used)))

## Creazione dell'albero decisionale
## La 10-fold CV viene eseguita su tutto il dataset
dt_total_multi_no_pca = train(Contraceptive_Method_Used~.,
                              data = cmc_multi,
                              trControl = train_control_multi,
                              method ="rpart",
                              metric = "AUC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_total_multi_no_pca)

## Importanza delle variabili
varImp(dt_total_multi_no_pca)

## Plot dell'albero decisionale
plot(dt_total_multi_no_pca)
plot(dt_total_multi_no_pca$finalModel)
text(dt_total_multi_no_pca$finalModel)
fancyRpartPlot(dt_total_multi_no_pca$finalModel)

## Calcolo matrice di confusione
confusionMatrix(dt_total_multi_no_pca)
##  Accuracy (average) : 0.5261

confmat = confusionMatrix(dt_total_multi_no_pca$pred$pred,
                          as.factor(dt_total_multi_no_pca$pred$obs),
                          mode = "prec_recall")
confmat
## Confusion Matrix and Statistics
##
##             Reference
## Prediction   No-Use Long-Term Short-Term
##   No.Use        339        68        118
##   Long.Term      44        99         56
##   Short.Term    246       166        337

## Stampa matrice di confusione
plt = as.data.frame(confmat$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(position = "top", labels=c("No.Use","Long.Term","Short.Term")) +
  scale_y_discrete(labels=c("Short.Term","Long.Term","No.Use"))

confmat_byclass = confmat$byClass
precision_no_use = confmat_byclass[c("Class: No.Use"),c("Precision")]
precision_no_use
## Precision No Use: 0.6457143
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.4974874
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4499332
precision_macro_average = 
  (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.531045

## Rimozione dei dati
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.5389507
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.2972973
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.6594912
recall_macro_average = 
  (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.4985797

## Rimozione dei dati
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5875217
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.3721805
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.5349206
f1measure_macro_average = 
  (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.4982076

## Rimozione dei dati
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Recupero le probabilita' dal modello
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0) 

dt_total_multi_no_pca.res = data.frame(No.Use, Long.Term, Short.Term) 
for (i in 1:1473) {
  dt_total_multi_no_pca.res[i, 1] = 0
  dt_total_multi_no_pca.res[i, 2] = 0
  dt_total_multi_no_pca.res[i, 3] = 0
}

for (i in 1:1473) {
  index = dt_total_multi_no_pca$pred$rowIndex[i]
  dt_total_multi_no_pca.res[index, 1] = dt_total_multi_no_pca$pred$No.Use[i]
  dt_total_multi_no_pca.res[index, 2] = dt_total_multi_no_pca$pred$Long.Term[i]
  dt_total_multi_no_pca.res[index, 3] = dt_total_multi_no_pca$pred$Short.Term[i]
}

## Creo il dataframe corretto per multiROC
dt_total_multi_no_pca.df = create_multiROC_dataframe("Albero_Decisionale",
                                                     cmc_multi,
                                                     dt_total_multi_no_pca.res)

## Invocazione di multiROC per la costruzione delle curve ROC
dt_total_multi_no_pca.res_multi_roc = multi_roc(dt_total_multi_no_pca.df, 
                                                force_diag = T)

## Plot delle curve ROC
plot_ROC(dt_total_multi_no_pca.res_multi_roc, 
         "Curva ROC per l'albero decisionale")

## Memorizzazione dei valori AUC sia per classe sia per macro e micro
No.Use = c(0)
Long.Term = c(0)
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in dt_total_multi
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
dt_total_multi_no_pca.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
dt_total_multi_no_pca.AUC[1, 1] = 
  dt_total_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$No.Use
dt_total_multi_no_pca.AUC[1, 2] = 
  dt_total_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$Long.Term
dt_total_multi_no_pca.AUC[1, 3] = 
  dt_total_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$Short.Term
dt_total_multi_no_pca.AUC[1, 4] = 
  dt_total_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$macro
dt_total_multi_no_pca.AUC[1, 5] = 
  dt_total_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$micro
dt_total_multi_no_pca.AUC
## No.Use     Long.Term   Short.Term      Macro     Micro
## 0.717548   0.6654286   0.6614156       0.6814631 0.7169151

## Tempi di calcolo
dt_total_multi_no_pca$times
## Everything:
##    utente   sistema trascorso
##      1.02      0.00      1.02 
## Final:
##    utente   sistema trascorso
##      0.02      0.00      0.02
## Prediction:
##     NA        NA        NA

## Rimozione dei dati non più necessari
rm(confmat)
rm(confmat_byclass)
rm(No.Use)
rm(Long.Term)
rm(Short.Term)
rm(Macro)
rm(Micro)
rm(plt)
rm(dt_total_multi_no_pca.df)
rm(dt_total_multi_no_pca.res)
rm(dt_total_multi_no_pca.AUC)



#### 3.1.1.4 - Creazione di albero decisionale CART con 10-fold CV ripetuta 
#### 3 volte con split 70/30 per problema multi-classe (senza PCA)

## Split del dataset
allset_multi_no_pca = split.data(cmc_multi, proportion = 0.7, seed = 1)
trainset_multi_no_pca = allset_multi_no_pca$train
testset_multi_no_pca = allset_multi_no_pca$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_multi_no_pca$Contraceptive_Method_Used) =
  make.names(levels(factor(trainset_multi_no_pca$Contraceptive_Method_Used)))
levels(testset_multi_no_pca$Contraceptive_Method_Used) =
  make.names(levels(factor(testset_multi_no_pca$Contraceptive_Method_Used)))

## Creazione dell'albero decisionale
dt_split_multi_no_pca = train(Contraceptive_Method_Used~.,
                              data = testset_multi_no_pca,
                              trControl = train_control_multi_repeated,
                              method ="rpart",
                              metric = "AUC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_split_multi_no_pca)

## Importanza delle variabili
varImp(dt_split_multi_no_pca)

## Plot dell'albero decisionale
plot(dt_split_multi_no_pca)
plot(dt_split_multi_no_pca$finalModel)
text(dt_split_multi_no_pca$finalModel)
fancyRpartPlot(dt_split_multi_no_pca$finalModel)

## Predizione sul testset
dt_split_multi_no_pca.pred = predict(dt_split_multi_no_pca,
                                     testset_multi_no_pca[, !names(testset_multi_no_pca) %in% 
                                                            c("Contraceptive_Method_Used")])

## Predizione sul testset (con probabilita')
dt_split_multi_no_pca.probs = predict(dt_split_multi_no_pca,
                                      testset_multi_no_pca[, !names(testset_multi_no_pca) %in% 
                                                             c("Contraceptive_Method_Used")],
                                   type = "prob")

## Calcolo matrice di confusione
confmat = confusionMatrix(dt_split_multi_no_pca.pred,
                          as.factor(testset_multi_no_pca$Contraceptive_Method_Used),
                          mode = "prec_recall")
confmat
## La matrice di confusione e' calcolata solo sul testset
## Confusion Matrix and Statistics
##
##             Reference
## Prediction   No-Use Long-Term Short-Term
##   No.Use         64         9         11
##   Long.Term      31        72         69
##   Short.Term     66        28         91
##
##  Accuracy : 0.5147

## Stampa matrice di confusione
plt = as.data.frame(confmat$table)
colnames(plt) = c("Prediction", "Reference", "Freq")
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(position = "top", labels=c("No.Use","Long.Term","Short.Term")) +
  scale_y_discrete(labels=c("Short.Term","Long.Term","No.Use"))

confmat_byclass = confmat$byClass
precision_no_use = confmat_byclass[c("Class: No.Use"),c("Precision")]
precision_no_use
## Precision No Use: 0.7619048
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.4186047
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4918919
precision_macro_average = 
  (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.5574671

## Rimozione dei dati
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.3975155
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.6605505
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.5321637
recall_macro_average = 
  (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.5300766

## Rimozione dei dati
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.522449
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.5124555
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.511236
f1measure_macro_average = 
  (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.5153802

## Rimozione dei dati
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Creo il dataframe corretto per multiROC
dt_split_multi_no_pca.df = create_multiROC_dataframe("Albero_Decisionale", 
                                                     testset_multi_no_pca, 
                                                     dt_split_multi_no_pca.probs)

## Invocazione di multiROC per la costruzione delle curve ROC
dt_split_multi_no_pca.res_multi_roc = multi_roc(dt_split_multi_no_pca.df, 
                                                force_diag = T)

## Plot delle curve ROC
plot_ROC(dt_split_multi_no_pca.res_multi_roc, 
         "Curva ROC per l'albero decisionale")

## Memorizzazione dei valori AUC sia per classe sia per macro e micro
No.Use = c(0)
Long.Term = c(0)
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in dt_split_multi
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
dt_split_multi_no_pca.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
dt_split_multi_no_pca.AUC[1, 1] = 
  dt_split_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$No.Use
dt_split_multi_no_pca.AUC[1, 2] = 
  dt_split_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$Long.Term
dt_split_multi_no_pca.AUC[1, 3] = 
  dt_split_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$Short.Term
dt_split_multi_no_pca.AUC[1, 4] = 
  dt_split_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$macro
dt_split_multi_no_pca.AUC[1, 5] = 
  dt_split_multi_no_pca.res_multi_roc$AUC$Albero_Decisionale$micro
dt_split_multi_no_pca.AUC
## No.Use     Long.Term   Short.Term      Macro     Micro
## 0.7457631  0.6941804   0.621789        0.6872312 0.7149336

## Tempi di calcolo
dt_split_multi_no_pca$times
## Everything:
##    utente   sistema trascorso
##      2.09      0.00      2.09 
## Final:
##    utente   sistema trascorso
##      0.04      0.00      0.04 
## Prediction:
##     NA        NA        NA

## Rimozione dei dati non piu' necessari
rm(confmat)
rm(confmat_byclass)
rm(No.Use)
rm(Long.Term)
rm(Short.Term)
rm(Macro)
rm(Micro)
rm(plt)
rm(dt_split_multi_no_pca.df)
rm(dt_split_multi_no_pca.AUC)
rm(allset_multi_no_pca)



#### 3.1.1.5 - Confronto modelli

## Plot delle curve ROC usando pROC
# Plot delle curve ROC per Yes e No per il problema totale
plot(dt_total_bin_no_pca_no.ROC, type = "S", col = "green")
plot(dt_total_bin_no_pca_yes.ROC, add = TRUE, col = "blue")
# Plot delle curve ROC per Yes e No per il problema split
plot(dt_split_bin_no_pca_no.ROC, type = "S", col = "black")
plot(dt_split_bin_no_pca_yes.ROC, add = TRUE, col = "red")

## Conferma che i valori AUC sono gli stessi per la classe positiva e negativa
## nel caso di classificatore binario
dt_total_bin_no_pca_yes.ROC$auc
dt_total_bin_no_pca_no.ROC$auc
dt_split_bin_no_pca_yes.ROC$auc

## Plot delle curve ROC usando MLeval
# Plot delle curve ROC per Yes e No (confrontando problema totale e split)
models_bin = list(dt_total_bin_no_pca = dt_total_bin_no_pca, 
                  dt_split_bin_no_pca = dt_split_bin_no_pca)
evalm(models_bin, 
      gnames = c("DT totale binario", "DT split binario"), 
      positive = "Yes")
evalm(models_bin, 
      gnames = c("DT totale binario", "DT split binario"), 
      positive = "No")

# Plot delle curve ROC per No.Use, Long.Term e Short.Term (confrontando problema totale e split)
models_multi = list(dt_total_multi_no_pca = dt_total_multi_no_pca, 
                    dt_split_multi_no_pca = dt_split_multi_no_pca)
evalm(models_multi, 
      gnames = c("DT totale multi", "DT split multi"), 
      positive = "No.Use")
evalm(models_multi, 
      gnames = c("DT totale multi", "DT split multi"), 
      positive = "Long.Term")
evalm(models_multi, 
      gnames = c("DT totale multi", "DT split multi"), 
      positive = "Short.Term")

## Plot delle curve ROC usando multiROC
# Plot delle curve ROC per Yes e No per il problema totale binario
dt_multiroc_total.probs = predict(dt_total_bin_no_pca, type = "prob")
dt_multiroc_total.df = create_multiROC_dataframe_binario("Albero_Decisionale",
                                                         cmc_bin,
                                                         dt_multiroc_total.probs)
dt_multiroc_total.res_multi_roc = multi_roc(dt_multiroc_total.df, force_diag = T)
plot_ROC(dt_multiroc_total.res_multi_roc, "Curva ROC per l'albero decisionale")

# Plot delle curve ROC per Yes e No per il problema split binario
dt_multiroc_split.probs = predict(dt_split_bin_no_pca,
                                  testset_no_pca[, !names(testset_no_pca) %in% 
                                                   c("Contraceptive_Is_Used")],
                                  type = "prob")
dt_multiroc_split.df = create_multiROC_dataframe_binario("Albero_Decisionale",
                                                         testset_no_pca,
                                                         dt_multiroc_split.probs)
dt_multiroc_split.res_multi_roc = multi_roc(dt_multiroc_split.df, force_diag = T)
plot_ROC(dt_multiroc_split.res_multi_roc, "Curva ROC per l'albero decisionale")

## I valori AUC possono essere ricavati direttamente dalla stampa degli alberi
dt_total_bin_no_pca
dt_total_multi_no_pca

## Non si puo' usare ROCR per il problema multi-classe

## Rimozione variabili non piu' necessarie
rm(models_bin)
rm(models_multi)
rm(dt_multiroc_total.probs)
rm(dt_multiroc_total.df)
rm(dt_multiroc_total.res_multi_roc)
rm(dt_multiroc_split.probs)
rm(dt_multiroc_split.df)
rm(dt_multiroc_split.res_multi_roc)

## Elimino tutti i plot
dev.off()





