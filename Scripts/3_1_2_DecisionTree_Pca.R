######################################
## 3.1.2 - Decision Tree con PCA    ##
######################################

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

#### 3.1.2.1 - Creazione del modello di albero decisionale CART
#### con 10-fold CV per problema binario (con PCA)

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_bin2$Contraceptive_Is_Used) =
  make.names(levels(factor(cmc_bin2$Contraceptive_Is_Used)))

## Creazione dell'albero decisionale
## La 10-fold CV viene eseguita su tutto il dataset
dt_total_bin = train(Contraceptive_Is_Used~.,
                     data = cmc_bin2,
                     trControl = train_control_bin,
                     method ="rpart",
                     metric = "ROC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_total_bin)

## Importanza delle variabili
varImp(dt_total_bin)

## Plot dell'albero decisionale
plot(dt_total_bin)
plot(dt_total_bin$finalModel)
text(dt_total_bin$finalModel)
fancyRpartPlot(dt_total_bin$finalModel)

## Calcolo matrice di confusione
confusionMatrix(dt_total_bin)

## Matrice di confusione per No
confmat_no = confusionMatrix(dt_total_bin$pred$pred,
                             as.factor(dt_total_bin$pred$obs),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
## Confusion Matrix and Statistics
##
##           Reference
## Prediction  No Yes
##        No  246 165
##        Yes 383 679
##
## Accuracy (average) : 0.628
## 95% CI : (0.6027, 0.6527)
## 'Positive' Class : No

## Matrice di confusione per Yes
confmat_yes = confusionMatrix(dt_total_bin$pred$pred,
                              as.factor(dt_total_bin$pred$obs),
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
## Precision No: 0.5985401
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.6393597
values_vector = c(precision_no, precision_yes)
precision_macro_average = mean(values_vector)
precision_macro_average
## Precision Macro Average: 0.6189499

## Rimozione dei dati
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.391097
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.8045024
values_vector = c(recall_no, recall_yes)
recall_macro_average = mean(values_vector)
recall_macro_average
## Recall Macro Average: 0.5977997

## Rimozione dei dati
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.4730769
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7124869
values_vector = c(f1_no, f1_yes)
f1_macro_average = mean(values_vector)
f1_macro_average
## F1-Measure Macro Average: 0.5927819

## Rimozione dei dati
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = ROCR::prediction(dt_total_bin$pred$Yes, 
                             dt_total_bin$pred$obs)

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
dt_total_bin_yes.ROC = roc(response = dt_total_bin$pred$obs,
                           predictor = dt_total_bin$pred$Yes,
                           levels = levels(cmc_bin2[,c("Contraceptive_Is_Used")]))
dt_total_bin_no.ROC = roc(response = dt_total_bin$pred$obs,
                          predictor = dt_total_bin$pred$No,
                          levels = levels(cmc_bin2[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
dt_total_bin$times
## Everything:
##    utente   sistema trascorso
##      0.57      0.00      0.58
## Final:
##    utente   sistema trascorso
##      0.01      0.00      0.01
## Prediction:
##      NA        NA        NA

## Rimozione dei dati non piu' necessari
rm(confmat_no)
rm(confmat_yes)
rm(values_vector)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)



#### 3.1.2.2 - Creazione di albero decisionale CART con 10-fold CV 
#### (ripetuta 3 volte) con split 70/30 per problema binario (con PCA)

## Split del dataset
allset = split.data(cmc_bin2, proportion = 0.7, seed = 1)
trainset = allset$train
testset = allset$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset$Contraceptive_Is_Used) =
  make.names(levels(factor(trainset$Contraceptive_Is_Used)))
levels(testset$Contraceptive_Is_Used) =
  make.names(levels(factor(testset$Contraceptive_Is_Used)))

## Creazione dell'albero decisionale
dt_split_bin = train(Contraceptive_Is_Used~.,
                     data = trainset,
                     trControl = train_control_bin_repeated,
                     method ="rpart",
                     metric = "ROC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_split_bin)

## Importanza delle variabili
varImp(dt_split_bin)

## Plot dell'albero decisionale
plot(dt_split_bin)
plot(dt_split_bin$finalModel)
text(dt_split_bin$finalModel)
fancyRpartPlot(dt_split_bin$finalModel)

## Predizione sul testset
dt_split_bin.pred = predict(dt_split_bin,
                            testset[, !names(testset) %in% 
                                      c("Contraceptive_Is_Used")])

## Predizione sul testset (con probabilita')
dt_split_bin.probs = predict(dt_split_bin,
                            testset[, !names(testset) %in% 
                                      c("Contraceptive_Is_Used")],
                            type = "prob")

## Matrice di confusione per No
confmat_no = confusionMatrix(dt_split_bin.pred,
                             as.factor(testset$Contraceptive_Is_Used),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
## La matrice di confusione e' calcolata solo sul testset
## Confusion Matrix and Statistics
##
##           Reference
## Prediction  No Yes
##        No   64  51
##        Yes  97 229
##
## Accuracy : 0.6644
## 95% CI : (0.6182, 0.7084)
## 'Positive' Class : No

## Matrice di confusione per Yes
confmat_yes = confusionMatrix(dt_split_bin.pred,
                              as.factor(testset$Contraceptive_Is_Used),
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
## Precision No: 0.5565217
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.702454
values_vector = c(precision_no, precision_yes)
precision_macro_average = mean(values_vector)
precision_macro_average
## Precision Macro Average: 0.6294879

## Rimozione dei dati
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.3975155
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.8178571
values_vector = c(recall_no, recall_yes)
recall_macro_average = mean(values_vector)
recall_macro_average
## Recall Macro Average: 0.6076863

## Rimozione dei dati
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.4637681
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7557756
values_vector = c(f1_no, f1_yes)
f1_macro_average = mean(values_vector)
f1_macro_average
## F1-Measure Macro Average: 0.6097718

## Rimozione dei dati
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Recupera le probabilita' di Yes
pred.to.roc = dt_split_bin.probs[, 2]

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = prediction(pred.to.roc, testset$Contraceptive_Is_Used)

## Usa la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

## Visualizza la curva ROC per il Yes
plot(perf.tpr.rocr, colorize = T, main=paste("AUC:", (perf.rocr@y.values)))
abline(a = 0, b = 1)

## Creazione variabili per le curve ROC da confrontare con altri modelli
dt_split_bin_yes.ROC = roc(response = testset[,c("Contraceptive_Is_Used")],
                           predictor = dt_split_bin.probs$Yes,
                           levels = levels(testset[,c("Contraceptive_Is_Used")]))
dt_split_bin_no.ROC = roc(response = testset[,c("Contraceptive_Is_Used")],
                          predictor = dt_split_bin.probs$No,
                          levels = levels(testset[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
dt_split_bin$times
## Everything:
##    utente   sistema trascorso
##    1.01      0.03      1.05
## Final:
##    utente   sistema trascorso
##      0.01      0.00      0.02
## Prediction:
##      NA        NA        NA

## Rimozione dei dati non piu' necessari
rm(allset)
#rm(trainset)
#rm(testset)
rm(confmat_no)
rm(confmat_yes)
rm(values_vector)
rm(pred.to.roc)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)
rm(plt)



#### 3.1.2.3 - Creazione del modello di albero decisionale CART con 10-fold CV
#### per problema multi-classe (con PCA)

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_multi2$Contraceptive_Method_Used) =
  make.names(levels(factor(cmc_multi2$Contraceptive_Method_Used)))

## Creazione dell'albero decisionale
## La 10-fold CV viene eseguita su tutto il dataset
dt_total_multi = train(Contraceptive_Method_Used~.,
                       data = cmc_multi2,
                       trControl = train_control_multi,
                       method ="rpart",
                       metric = "AUC")

## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_total_multi)

## Importanza delle variabili
varImp(dt_total_multi)

## Plot dell'albero decisionale
plot(dt_total_multi)
plot(dt_total_multi$finalModel)
text(dt_total_multi$finalModel)
fancyRpartPlot(dt_total_multi$finalModel)

## Calcolo matrice di confusione
confusionMatrix(dt_total_multi)
##  Accuracy (average) : 0.4718

confmat = confusionMatrix(dt_total_multi$pred$pred,
                          as.factor(dt_total_multi$pred$obs),
                          mode = "prec_recall")
confmat
## Confusion Matrix and Statistics
##
##             Reference
## Prediction   No-Use Long-Term Short-Term
##   No.Use        364        95        222
##   Long.Term      48       102         60
##   Short.Term    217       136        229
##
## Accuracy : 0.4718          
## 95% CI : (0.4461, 0.4977)

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
## Precision No Use: 0.5345081
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.4857143
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.3934708
precision_macro_average = 
  (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.4712311

## Rimozione dei dati
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.5786963
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.3063063
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.4481409
recall_macro_average = 
  (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.4443812

## Rimozione dei dati
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5557252
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.3756906
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.4190302
f1measure_macro_average = 
  (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.4501487

## Rimozione dei dati
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Recupero le probabilita' dal modello
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0) 

dt_total_multi.res = data.frame(No.Use, Long.Term, Short.Term) 
for (i in 1:1473) {
  dt_total_multi.res[i, 1] = 0
  dt_total_multi.res[i, 2] = 0
  dt_total_multi.res[i, 3] = 0
}

for (i in 1:1473) {
  index = dt_total_multi$pred$rowIndex[i]
  dt_total_multi.res[index, 1] = dt_total_multi$pred$No.Use[i]
  dt_total_multi.res[index, 2] = dt_total_multi$pred$Long.Term[i]
  dt_total_multi.res[index, 3] = dt_total_multi$pred$Short.Term[i]
}

## Creo il dataframe corretto per multiROC
dt_total_multi.df = create_multiROC_dataframe("albero_decisionale",
                                              cmc_multi2,
                                              dt_total_multi.res)

## Invocazione di multiROC per la costruzione delle curve ROC
dt_total_multi.res_multi_roc = multi_roc(dt_total_multi.df, force_diag = T)

## Plot delle curve ROC
plot_ROC(dt_total_multi.res_multi_roc, "Curva ROC per l'albero decisionale")

## Memorizzazione dei valori AUC sia per classe sia per macro e micro
No.Use = c(0)
Long.Term = c(0)
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in dt_total_multi
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
dt_total_multi.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
dt_total_multi.AUC[1, 1] = 
  dt_total_multi.res_multi_roc$AUC$albero_decisionale$No.Use
dt_total_multi.AUC[1, 2] = 
  dt_total_multi.res_multi_roc$AUC$albero_decisionale$Long.Term
dt_total_multi.AUC[1, 3] = 
  dt_total_multi.res_multi_roc$AUC$albero_decisionale$Short.Term
dt_total_multi.AUC[1, 4] = 
  dt_total_multi.res_multi_roc$AUC$albero_decisionale$macro
dt_total_multi.AUC[1, 5] = 
  dt_total_multi.res_multi_roc$AUC$albero_decisionale$micro
dt_total_multi.AUC
## No.Use     Long.Term   Short.Term      Macro     Micro
## 0.6490103  0.6730309   0.5579151       0.626652  0.6689667

## Tempi di calcolo
dt_total_multi$times
## Everything:
##    utente   sistema trascorso
##      0.97      0.00      0.99
## Final:
##    utente   sistema trascorso
##      0.01      0.00      0.02
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
rm(dt_total_multi.df)
rm(dt_total_multi.res)
#rm(dt_total_multi.res_multi_roc)
rm(dt_total_multi.AUC)



#### 3.1.2.4 - Creazione di albero decisionale CART con 10-fold CV 
#### (ripetuta 3 volte) con split 70/30 per problema multi-classe (con PCA)

## Split del dataset
allset_multi = split.data(cmc_multi2, proportion = 0.7, seed = 1)
trainset_multi = allset_multi$train
testset_multi = allset_multi$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_multi$Contraceptive_Method_Used) =
  make.names(levels(factor(trainset_multi$Contraceptive_Method_Used)))
levels(testset_multi$Contraceptive_Method_Used) =
  make.names(levels(factor(testset_multi$Contraceptive_Method_Used)))

## Creazione dell'albero decisionale
dt_split_multi = train(Contraceptive_Method_Used~.,
                       data = testset_multi,
                       trControl = train_control_multi_repeated,
                       method ="rpart",
                       metric = "AUC")


## Stampa di alcune caratteristiche dell'albero (tra cui CP)
print(dt_total_multi)

## Importanza delle variabili
varImp(dt_split_multi)

## Plot dell'albero decisionale
plot(dt_split_multi)
plot(dt_split_multi$finalModel)
text(dt_split_multi$finalModel)
fancyRpartPlot(dt_split_multi$finalModel)

## Predizione sul testset
dt_split_multi.pred = predict(dt_split_multi,
                            testset_multi[, !names(testset_multi) %in% 
                                            c("Contraceptive_Method_Used")])

## Predizione sul testset (con probabilita')
dt_split_multi.probs = predict(dt_split_multi,
                             testset_multi[, !names(testset_multi) %in% 
                                             c("Contraceptive_Method_Used")],
                             type = "prob")

## Calcolo matrice di confusione
confmat = confusionMatrix(dt_split_multi.pred,
                          as.factor(testset_multi$Contraceptive_Method_Used),
                          mode = "prec_recall")
confmat
## La matrice di confusione e' calcolata solo sul testset
## Confusion Matrix and Statistics
##
##             Reference
## Prediction   No-Use Long-Term Short-Term
##   No.Use        118        37        102
##   Long.Term      10        43         24
##   Short.Term     33        29         45
##
## Accuracy : 0.4671
## 95% CI : (0.4198, 0.5149)

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
## Precision No Use: 0.459144
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.5584416
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4205607
precision_macro_average = 
  (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.4793821

## Rimozione dei dati
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.7329193
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.3944954
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.2631579
recall_macro_average = 
  (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.4635242

## Rimozione dei dati
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5645933
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.4623656
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.323741
f1measure_macro_average = 
  (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.4502333

## Rimozione dei dati
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Creo il dataframe corretto per multiROC
dt_split_multi.df = create_multiROC_dataframe("albero_decisionale", 
                                              testset_multi, 
                                              dt_split_multi.probs)

## Invocazione di multiROC per la costruzione delle curve ROC
dt_split_multi.res_multi_roc = multi_roc(dt_split_multi.df, 
                                         force_diag = T)

## Plot delle curve ROC
plot_ROC(dt_split_multi.res_multi_roc, "Curva ROC per l'albero decisionale")

## Memorizzazione dei valori AUC sia per classe sia per macro e micro
No.Use = c(0)
Long.Term = c(0)
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in dt_split_multi
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
dt_split_multi.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
dt_split_multi.AUC[1, 1] = 
  dt_split_multi.res_multi_roc$AUC$albero_decisionale$No.Use
dt_split_multi.AUC[1, 2] = 
  dt_split_multi.res_multi_roc$AUC$albero_decisionale$Long.Term
dt_split_multi.AUC[1, 3] = 
  dt_split_multi.res_multi_roc$AUC$albero_decisionale$Short.Term
dt_split_multi.AUC[1, 4] = 
  dt_split_multi.res_multi_roc$AUC$albero_decisionale$macro
dt_split_multi.AUC[1, 5] = 
  dt_split_multi.res_multi_roc$AUC$albero_decisionale$micro
dt_split_multi.AUC
## No.Use     Long.Term   Short.Term      Macro     Micro
## 0.6545031  0.7079971   0.50483         0.6224416 0.6644171

## Tempi di calcolo
dt_split_multi$times
## Everything:
##    utente   sistema trascorso
##      0.91      0.02      0.91
## Final:
##    utente   sistema trascorso
##      0.02      0.00      0.02
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
rm(dt_split_multi.df)
#rm(dt_split_multi.res_multi_roc)
rm(dt_split_multi.AUC)
rm(allset_multi)
#rm(trainset_multi)
#rm(testset_multi)



#### 3.1.2.5 - Confronto modelli

## Plot delle curve ROC usando pROC
# Plot delle curve ROC per Yes e No per il problema totale
plot(dt_total_bin_no.ROC, type = "S", col = "green")
plot(dt_total_bin_yes.ROC, add = TRUE, col = "blue")
# Plot delle curve ROC per Yes e No per il problema split
plot(dt_split_bin_no.ROC, type = "S", col = "black")
plot(dt_split_bin_yes.ROC, add = TRUE, col = "red")

## Conferma che i valori AUC sono gli stessi per la classe positiva e negativa
## nel caso di classificatore binario
dt_total_bin_yes.ROC$auc
dt_total_bin_no.ROC$auc
dt_split_bin_yes.ROC$auc

## Plot delle curve ROC usando MLeval
# Plot delle curve ROC per Yes e No (confrontando problema totale e split)
models_bin = list(dt_total_bin = dt_total_bin, 
                  dt_split_bin = dt_split_bin)
evalm(models_bin, 
      gnames = c("DT totale binario", "DT split binario"), 
      positive = "Yes")
evalm(models_bin, 
      gnames = c("DT totale binario", "DT split binario"), 
      positive = "No")

# Plot delle curve ROC per No.Use, Long.Term e Short.Term (confrontando problema totale e split)
models_multi = list(dt_total_multi = dt_total_multi, 
                    dt_split_multi = dt_split_multi)
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
dt_multiroc_total.probs = predict(dt_total_bin, type = "prob")
dt_multiroc_total.df = create_multiROC_dataframe_binario("albero_decisionale",
                                                       cmc_bin2,
                                                       dt_multiroc_total.probs)
dt_multiroc_total.res_multi_roc = multi_roc(dt_multiroc_total.df, force_diag = T)
plot_ROC(dt_multiroc_total.res_multi_roc, "Curva ROC per l'albero decisionale")

# Plot delle curve ROC per Yes e No per il problema split binario
dt_multiroc_split.probs = predict(dt_split_bin,
                          testset[, !names(testset) %in% c("Contraceptive_Is_Used")],
                          type = "prob")
dt_multiroc_split.df = create_multiROC_dataframe_binario("albero_decisionale",
                                                         testset,
                                                         dt_multiroc_split.probs)
dt_multiroc_split.res_multi_roc = multi_roc(dt_multiroc_split.df, force_diag = T)
plot_ROC(dt_multiroc_split.res_multi_roc, "Curva ROC per l'albero decisionale")

## I valori AUC possono essere ricavati direttamente dalla stampa degli alberi
dt_total_bin
dt_total_multi

## Non si puu' usare ROCR per il problema multi-classe

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
