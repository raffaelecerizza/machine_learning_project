#############################################
## 3.2.2 - Rete Neurale su dataset con PCA ##
#############################################

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

#### Import dataset (e istallazione dei package necessari)

source(paste(getwd(), "/Scripts/2_2_Pca.R",sep = ""))

#### Caricamento funzioni di utilita'

source(paste(getwd(), "/Scripts/5_Utilities.R",sep = ""))

#### Import delle librerie necessarie

library(caret)

library(pROC)

library(ROCR)

library(nnet)

library(NeuralNetTools)

library(multiROC)

library(MLeval)

## Settaggio del seed per riproducibilita'

set.seed(100)


##----- 3.2.1 - Rete neurale con 10-fold CV per problema binario (con PCA) ---##

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_bin2$Contraceptive_Is_Used) = 
  make.names(levels(factor(cmc_bin2$Contraceptive_Is_Used)))

## Addestramento della rete neurale con 10-Fold CV su intero dataset
nn_total_bin = train(Contraceptive_Is_Used ~ ., 
                     data = cmc_bin2, 
                     method = "nnet",
                     type = 'Classification',
                     metric = "ROC",
                     trControl = train_control_bin)

## Stampa di alcune caratteristiche della rete, tra cui la dimensione
print(nn_total_bin)

## Visualizzazione importante delle variabili nella definizione della rete
varImp(nn_total_bin)

## Plot della rete
plotnet(nn_total_bin$finalModel)

## Calcolo della matrice di confusione complessiva
confusionMatrix(nn_total_bin) # Accuracy (average) : 0.6327

## Matrice di confusione per il No
confmat_no = confusionMatrix(nn_total_bin$pred$pred, 
                             as.factor(nn_total_bin$pred$obs),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no

## Matrice di confusione per il Yes
confmat_yes = confusionMatrix(nn_total_bin$pred$pred, 
                              as.factor(nn_total_bin$pred$obs),
                              mode = "prec_recall",
                              positive = "Yes")
confmat_yes

## RISULTATI per No
##     Prediction  No Yes
## No              274 186
## Yes             355 658
## Accuracy : 0.6327 
## 95% CI : (0.6075, 0.6574)

## RISULTATI per YES
##     Prediction  No Yes
## No              274 186
## Yes             355 658
## Accuracy : 0.6327          

## Variano solamento i valori di Precision, Recall etc.

## Fancy plot della matrice di confusione
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
## Precision No: 0.5956522
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.6495558
precision_macro_average = mean(c(precision_no, precision_yes))
precision_macro_average
## Precision Macro Average: 0.622604
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.4356121
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.7796209
recall_macro_average = mean(c(recall_no, recall_yes))
recall_macro_average
## Recall Macro Average: 0.6076165
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.503214
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7086699
f1_macro_average = mean(c(f1_no, f1_yes))
f1_macro_average
## F1-Measure Macro Average: 0.6059419
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Uso la funzione prediction per predire il risultato in base alla 
## probabilita' di Yes
pred.rocr = ROCR::prediction(nn_total_bin$pred$Yes, 
                             nn_total_bin$pred$obs)

## Uso la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")

# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")

## Plot della curva ROC
plot(perf.tpr.rocr, colorize=T,main=paste("AUC:",(perf.rocr@y.values)))
abline(a=0, b=1)

## Visualizzazione del cut off ottimale
print(opt.cut(perf.tpr.rocr, pred.rocr))

## Creazione variabili per le curve ROC da confrontare con altri modelli
nn_total_bin_yes.ROC = roc(response = nn_total_bin$pred$obs,
                           predictor = nn_total_bin$pred$Yes,
                           levels = levels(cmc_bin2[,c("Contraceptive_Is_Used")]))
nn_total_bin_no.ROC = roc(response = nn_total_bin$pred$obs,
                          predictor = nn_total_bin$pred$No,
                          levels = levels(cmc_bin2[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
nn_total_bin$times
## user  system elapsed 
## 10.27  0.08     10.45
## final
## user  system elapsed 
## 0.11   0.00      0.11 
## prediction
## [1] NA NA NA

## Elimino le variabili non piu' necessarie
rm(confmat_no)
rm(confmat_yes)
#rm(nn_total_bin)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)

## Elimino tutti i plot
dev.off()


##----- 3.2.2 - Rete neurale con 10-fold CV (ripetuta 3 volte) con -----------##
##----- split 70/30 per problema binario (con PCA) ---------------------------##

## Split del dataset
allset = split.data(cmc_bin2, proportion = 0.7, seed = 1)
trainset = allset$train
testset = allset$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset$Contraceptive_Is_Used) =
  make.names(levels(factor(trainset$Contraceptive_Is_Used)))
levels(testset$Contraceptive_Is_Used) =
  make.names(levels(factor(testset$Contraceptive_Is_Used)))

## Addestramento della rete neurale con 10-Fold CV su train set
nn_split_bin = train(Contraceptive_Is_Used ~ ., 
                     data = trainset, 
                     method = "nnet",
                     type = 'Classification',
                     metric = "ROC",
                     trControl = train_control_bin_repeated)

## Stampa di alcune caratteristiche della rete (tra cui size)
print(nn_split_bin)

## Vediamo importanza delle variabili
varImp(nn_split_bin)

## Plot della rete
plotnet(nn_split_bin$finalModel)

## Predizione sul testset
nn_split_bin.pred = predict(nn_split_bin,
                            testset[, !names(testset) %in% 
                                      c("Contraceptive_Is_Used")])

## Predizione sul testset (con probabilita')
nn_split_bin.probs = predict(nn_split_bin,
                             testset[, !names(testset) %in% 
                                       c("Contraceptive_Is_Used")],
                             type = "prob")

## Matrice di confusione per No
confmat_no = confusionMatrix(nn_split_bin.pred,
                             as.factor(testset$Contraceptive_Is_Used),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  No Yes
# No   75  66
# Yes  86 214
# 
# Accuracy : 0.6553          
# 95% CI : (0.6089, 0.6996)
# No Information Rate : 0.6349          
# P-Value [Acc > NIR] : 0.2006          
# 
# Kappa : 0.2364          
# 
# Mcnemar's Test P-Value : 0.1233          
#                                           
#               Precision : 0.5319          
#                  Recall : 0.4658          
#                      F1 : 0.4967          
#              Prevalence : 0.3651          
#          Detection Rate : 0.1701          
#    Detection Prevalence : 0.3197          
#       Balanced Accuracy : 0.6151          
#                                           
#        'Positive' Class : No

## Matrice di confusione per Yes
confmat_yes = confusionMatrix(nn_split_bin.pred,
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
## Precision No: 0.5319149
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.7133333
precision_macro_average = mean(c(precision_no, precision_yes))
precision_macro_average
## Precision Macro Average: 0.6226241
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.4658385
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.7642857
recall_macro_average = mean(c(recall_no, recall_yes))
recall_macro_average
## Recall Macro Average: 0.6150621
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.4966887
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.737931
f1_macro_average = mean(c(f1_no, f1_yes))
f1_macro_average
## F1-Measure Macro Average: 0.6173099
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Recupera le probabilita' di Yes
pred.to.roc = nn_split_bin.probs[, 2]

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = ROCR::prediction(pred.to.roc, 
                             testset$Contraceptive_Is_Used)

## Usa la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")

# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

## Visualizza la curva ROC per il Yes
plot(perf.tpr.rocr, colorize = T, main=paste("AUC:", (perf.rocr@y.values)))
abline(a = 0, b = 1)

## Creazione variabili per le curve ROC da confrontare con altri modelli
nn_split_bin_yes.ROC = roc(response = testset[,c("Contraceptive_Is_Used")],
                           predictor = nn_split_bin.probs$Yes,
                           levels = levels(testset[,c("Contraceptive_Is_Used")]))
nn_split_bin_no.ROC = roc(response = testset[,c("Contraceptive_Is_Used")],
                          predictor = nn_split_bin.probs$No,
                          levels = levels(testset[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
nn_split_bin$times
# $everything
# user  system elapsed 
# 20.09  0.22     20.30 
# 
# $final
# user  system elapsed 
# 0.11  0.02    0.09  
# 
# $prediction
# [1] NA NA NA

## Rimozione variabili non necessarie
rm(allset)
#rm(trainset)
#rm(testset)
rm(confmat_no)
rm(confmat_yes)
rm(pred.to.roc)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)
rm(plt)
rm(nn_split_bin.probs)
rm(nn_split_bin.pred)

## Eliminazione plot

dev.off()


##----- 3.2.3 - Rete neurale con 10-fold CV su problema multi (con PCA) ------##

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_multi2$Contraceptive_Method_Used) = 
  make.names(levels(factor(cmc_multi2$Contraceptive_Method_Used)))

## Addestramento della rete neurale con 10-Fold CV su intero dataset
nn_total_multi = train(Contraceptive_Method_Used ~ ., 
                       data = cmc_multi2, 
                       method = "nnet",
                       trControl = train_control_multi,
                       metric = "AUC")

## Stampa di alcune caratteristiche della rete
print(nn_total_multi)

## Importanza delle variabili
varImp(nn_total_multi)

## Plot della rete
plotnet(nn_total_multi$finalModel)

## Visualizzazione confusion matrix
confusionMatrix(nn_total_multi) # Accuracy (average) : 0.5037

confmat = confusionMatrix(nn_total_multi$pred$pred, 
                          as.factor(nn_total_multi$pred$obs),
                          mode = "prec_recall")
confmat
sum(confmat$table) # coincide con numero istanze del dataset
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No.Use Long.Term Short.Term
# No.Use        398       118        199
# Long.Term      82       125         93
# Short.Term    149        90        219
# 
# Overall Statistics
# 
# Accuracy : 0.5037          
# 95% CI : (0.4779, 0.5296)
# No Information Rate : 0.427           
# P-Value [Acc > NIR] : 1.884e-09       
# 
# Kappa : 0.2231          
# 
# Mcnemar's Test P-Value : 0.003323        
# 
# Statistics by Class:
# 
#                      Class: No.Use Class: Long.Term Class: Short.Term
# Precision                   0.5566          0.41667            0.4782
# Recall                      0.6328          0.37538            0.4286
# F1                          0.5923          0.39494            0.4520
# Prevalence                  0.4270          0.22607            0.3469
# Detection Rate              0.2702          0.08486            0.1487
# Detection Prevalence        0.4854          0.20367            0.3109
# Balanced Accuracy           0.6286          0.61093            0.5901

## Fancy plot della matrice di confusione
plt = as.data.frame(confmat$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(position = "top", labels=c("No.Use","Long.Term","Short.Term")) +
  scale_y_discrete(labels=c("Short.Term","Long.Term","No.Use"))
rm(plt)

## Calcolo di Precision, Recall e F1
confmat_byclass = confmat$byClass
precision_no_use = confmat_byclass[c("Class: No.Use"),c("Precision")]
precision_no_use
## Precision No Use: 0.5566434
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.4166667
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4781659
precision_macro_average = (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.4838253
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.6327504
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.3753754
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.4285714
recall_macro_average = (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.4788991
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5922619
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.3949447
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.4520124
f1measure_macro_average = (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.4797397
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

#rm(confmat)
#rm(confmat_byclass)

## Recupero le probabilita' dal modello
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0) 

nn_total_multi.res = data.frame(No.Use, Long.Term, Short.Term) 
for (i in 1:1473) {
  nn_total_multi.res[i, 1] = 0
  nn_total_multi.res[i, 2] = 0
  nn_total_multi.res[i, 3] = 0
}

for (i in 1:1473) {
  index = nn_total_multi$pred$rowIndex[i]
  nn_total_multi.res[index, 1] = nn_total_multi$pred$No.Use[i]
  nn_total_multi.res[index, 2] = nn_total_multi$pred$Long.Term[i]
  nn_total_multi.res[index, 3] = nn_total_multi$pred$Short.Term[i]
}

## Creo il dataframe corretto per multiROC
nn_total_multi.df = create_multiROC_dataframe("Rete_Neurale", 
                                               cmc_multi2, 
                                               nn_total_multi.res)

## Invocazione di multiROC per la costruzione delle curve ROC
nn_total_multi.res_multi_roc = multi_roc(nn_total_multi.df, 
                                         force_diag = T)

## Plot delle curve ROC
plot_ROC(nn_total_multi.res_multi_roc, "Curva ROC per Rete Neurale")

rm(nn_total_multi.res)
rm(nn_total_multi.df)

## Memorizzazione dei valori AUC sia per classe, sia per macro e micro
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in neuralnet_multi_pca
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
nn_total_multi.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
nn_total_multi.AUC[1, 1] = 
  nn_total_multi.res_multi_roc$AUC$Rete_Neurale$No.Use
nn_total_multi.AUC[1, 2] = 
  nn_total_multi.res_multi_roc$AUC$Rete_Neurale$Long.Term
nn_total_multi.AUC[1, 3] = 
  nn_total_multi.res_multi_roc$AUC$Rete_Neurale$Short.Term
nn_total_multi.AUC[1, 4] = 
  nn_total_multi.res_multi_roc$AUC$Rete_Neurale$macro
nn_total_multi.AUC[1, 5] = 
  nn_total_multi.res_multi_roc$AUC$Rete_Neurale$micro
nn_total_multi.AUC
# No.Use     Long.Term  Short.Term  Macro      Micro
# 0.6570819  0.7052948  0.6124187   0.6582649  0.6813222

## Tempi di calcolo
nn_total_multi$times
# $everything
# user  system elapsed 
# 14.66 0.08     14.80 
# 
# $final
# user  system elapsed 
# 0.15  0.00      0.16 
# 
# $prediction
# [1] NA NA NA

## Elimino variabili non necessarie
rm(nn_total_multi.AUC)
# rm(nn_total_multi.res_multi_roc)
rm(confmat_byclass)
rm(Long.Term)
rm(Macro)
rm(Micro)
rm(No.Use)
rm(Short.Term)
rm(confmat)

## Elimino tutti i plot
dev.off()



##----- 3.2.4 - Rete neurale con 10-fold CV (ripetuta 3 volte) con -----------##
##----- split 70/30 per problema multi-classe (con PCA) ----------------------##

## Split del dataset
allset_multi = split.data(cmc_multi2, proportion = 0.7, seed = 1)
trainset_multi = allset_multi$train
testset_multi = allset_multi$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_multi$Contraceptive_Method_Used) =
  make.names(levels(factor(trainset_multi$Contraceptive_Method_Used)))
levels(testset_multi$Contraceptive_Method_Used) =
  make.names(levels(factor(testset_multi$Contraceptive_Method_Used)))

## Addestramento della rete neurale con 10-Fold CV su train set
nn_split_multi = train(Contraceptive_Method_Used ~ ., 
                       data = trainset_multi, 
                       method = "nnet",
                       trControl = train_control_multi_repeated,
                       metric = "AUC")

## Stampa di alcune caratteristiche della rete
print(nn_split_multi)

## Importanza delle variabili
varImp(nn_split_multi)

## Plot della rete
plotnet(nn_split_multi$finalModel)

## Predizione sul testset
nn_split_multi.pred = predict(nn_split_multi,
                              testset_multi[, !names(testset_multi) %in% 
                                              c("Contraceptive_Method_Used")])

## Predizione sul testset (con probabilita')
nn_split_multi.probs = predict(nn_split_multi,
                               testset_multi[, !names(testset_multi) %in% 
                                               c("Contraceptive_Method_Used")],
                               type = "prob")

## Calcolo matrice di confusione
confmat = confusionMatrix(nn_split_multi.pred,
                          as.factor(testset_multi$Contraceptive_Method_Used),
                          mode = "prec_recall")
confmat
## La matrice di confusione e' calcolata solo sul testset
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No.Use Long.Term Short.Term
# No.Use        104        47         77
# Long.Term       7        36         24
# Short.Term     50        26         70
# 
# Overall Statistics
# 
# Accuracy : 0.4762         
# 95% CI : (0.4287, 0.524)
# No Information Rate : 0.3878         
# P-Value [Acc > NIR] : 9.768e-05      
# 
# Kappa : 0.1883         
# 
# Mcnemar's Test P-Value : 9.788e-08      
# 
# Statistics by Class:
# 
#                      Class: No.Use Class: Long.Term Class: Short.Term
# Precision                   0.4561          0.53731            0.4795
# Recall                      0.6460          0.33028            0.4094
# F1                          0.5347          0.40909            0.4416
# Prevalence                  0.3651          0.24717            0.3878
# Detection Rate              0.2358          0.08163            0.1587
# Detection Prevalence        0.5170          0.15193            0.3311
# Balanced Accuracy           0.6016          0.61845            0.5639

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
rm(plt)

## Calcolo di Precision, Recall e F1
confmat_byclass = confmat$byClass
precision_no_use = confmat_byclass[c("Class: No.Use"),c("Precision")]
precision_no_use
## Precision No Use: 0.4561404
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.5373134
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4794521
precision_macro_average = (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.4909686
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.6459627
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.3302752
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.4093567
recall_macro_average = (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.4618649
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5347044
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.4090909
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.4416404
f1measure_macro_average = (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.4797397
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Creo il dataframe corretto per multiROC
nn_split_multi.df = create_multiROC_dataframe("Rete_Neurale", 
                                              testset_multi, 
                                              nn_split_multi.probs)

## Invocazione di multiROC per la costruzione delle curve ROC
nn_split_multi.res_multi_roc = multi_roc(nn_split_multi.df, 
                                         force_diag = T)

## Plot delle curve ROC
plot_ROC(nn_split_multi.res_multi_roc, "Curva ROC per Rete Neurale")

## Memorizzazione dei valori AUC sia per classe, sia per macro e micro
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in neuralnet_multi_pca
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
nn_split_multi.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
nn_split_multi.AUC[1, 1] = 
  nn_split_multi.res_multi_roc$AUC$Rete_Neurale$No.Use
nn_split_multi.AUC[1, 2] = 
  nn_split_multi.res_multi_roc$AUC$Rete_Neurale$Long.Term
nn_split_multi.AUC[1, 3] = 
  nn_split_multi.res_multi_roc$AUC$Rete_Neurale$Short.Term
nn_split_multi.AUC[1, 4] = 
  nn_split_multi.res_multi_roc$AUC$Rete_Neurale$macro
nn_split_multi.AUC[1, 5] = 
  nn_split_multi.res_multi_roc$AUC$Rete_Neurale$micro
nn_split_multi.AUC
# No.Use    Long.Term  Short.Term Macro     Micro
# 0.6631988 0.7354924  0.5916613  0.6634527 0.6641163

## Tempi di calcolo
nn_split_multi$times
# $everything
# user  system elapsed 
# 32.67 0.24     32.83 
# 
# $final
# user  system elapsed 
# 0.16  0.00      0.16 
# 
# $prediction
# [1] NA NA NA

## Rimozione dei dati non piu' necessari
rm(confmat)
rm(confmat_byclass)
rm(No.Use)
rm(Long.Term)
rm(Short.Term)
rm(Macro)
rm(Micro)
rm(nn_split_multi.df)
# rm(nn_split_multi.res_multi_roc)
rm(nn_split_multi.AUC)
rm(allset_multi)
#rm(trainset_multi)
#rm(testset_multi)
rm(nn_split_multi.pred)
rm(nn_split_multi.probs)

## Elimino i plot
dev.off()



##----- 3.2.5 - Confronto dei modelli ottenuti -------------------------------##

## Plot delle curve ROC usando pROC
# Plot delle curve ROC per Yes e No per il problema totale
plot(nn_total_bin_no.ROC, type = "S", col = "green")
plot(nn_total_bin_yes.ROC, add = TRUE, col = "blue")
# Plot delle curve ROC per Yes e No per il problema split
plot(nn_split_bin_no.ROC, type = "S", col = "black")
plot(nn_split_bin_yes.ROC, add = TRUE, col = "red")

## Conferma che i valori AUC sono gli stessi per la classe positiva e negativa
## nel caso di classificatore binario
nn_total_bin_yes.ROC$auc
nn_total_bin_no.ROC$auc
nn_split_bin_yes.ROC$auc
nn_split_bin_no.ROC$auc

## Plot delle curve ROC usando MLeval
# Plot delle curve ROC per Yes e No (confrontando problema totale e split)
models_bin = list(nn_total_bin = nn_total_bin, 
                  nn_split_bin = nn_split_bin)
evalm(models_bin, gnames = c("NN totale binario", "NN split binario"), 
      positive = "Yes")
evalm(models_bin, gnames = c("NN totale binario", "NN split binario"), 
      positive = "No")

## Plot delle curve ROC per No.Use, Long.Term e Short.Term 
## (confrontando problema totale e split)
models_multi = list(nn_total_multi = nn_total_multi, 
                    nn_split_multi = nn_split_multi)
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "No.Use")
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "Long.Term")
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "Short.Term")

## Plot delle curve ROC usando multiROC
# Plot delle curve ROC per Yes e No per il problema totale binario
nn_multiroc_total.probs = predict(nn_total_bin, type = "prob")
nn_multiroc_total.df = create_multiROC_dataframe_binario("rete_neurale",
                                                         cmc_bin,
                                                         nn_multiroc_total.probs)
nn_multiroc_total.res_multi_roc = multi_roc(nn_multiroc_total.df, 
                                            force_diag = T)
plot_ROC(nn_multiroc_total.res_multi_roc, "Curva ROC per rete neurale")

# Plot delle curve ROC per Yes e No per il problema split binario
nn_multiroc_split.probs = predict(nn_split_bin,
                                  testset[, !names(testset) %in% 
                                            c("Contraceptive_Is_Used")],
                                  type = "prob")
nn_multiroc_split.df = create_multiROC_dataframe_binario("rete_neurale",
                                                         testset,
                                                         nn_multiroc_split.probs)
nn_multiroc_split.res_multi_roc = multi_roc(nn_multiroc_split.df, 
                                            force_diag = T)
plot_ROC(nn_multiroc_split.res_multi_roc, "Curva ROC per rete neurale")

## I valori AUC possono essere ricavati direttamente dalla stampa delle reti
nn_total_bin
nn_total_multi

## Non si puo' usare ROCR per il problema multi-classe

## Rimozione variabili non piu' necessarie
rm(models_bin)
rm(models_multi)
rm(nn_multiroc_total.probs)
rm(nn_multiroc_total.df)
rm(nn_multiroc_total.res_multi_roc)
rm(nn_multiroc_split.probs)
rm(nn_multiroc_split.df)
rm(nn_multiroc_split.res_multi_roc)

## Elimino tutti i plot
dev.off()