###########################################################
## 3.2.1 - Rete Neurale su dataset originale (senza pca) ##
###########################################################

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


##----- 3.2.1 - Rete neurale con 10-fold CV per problema binario (senza PCA) -##

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_bin$Contraceptive_Is_Used) = 
  make.names(levels(factor(cmc_bin$Contraceptive_Is_Used)))

## Addestramento della rete neurale con 10-Fold CV su intero dataset
nn_total_bin_no_pca = train(Contraceptive_Is_Used ~ ., 
                            data = cmc_bin, 
                            method = "nnet",
                            type = 'Classification',
                            metric = "ROC",
                            trControl = train_control_bin)

## Stampa di alcune caratteristiche della rete, tra cui la dimensione
print(nn_total_bin_no_pca)

## Visualizzazione importante delle variabili nella definizione della rete
varImp(nn_total_bin_no_pca)

## Plot della rete
plotnet(nn_total_bin_no_pca$finalModel)

## Calcolo della matrice di confusione complessiva
confusionMatrix(nn_total_bin_no_pca) # Accuracy (average) : 0.7054

## Matrice di confusione per il No
confmat_no = confusionMatrix(nn_total_bin_no_pca$pred$pred, 
                             as.factor(nn_total_bin_no_pca$pred$obs),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  No Yes
# No  332 130
# Yes 297 714
# 
# Accuracy : 0.7101          
# 95% CI : (0.6862, 0.7332)
# No Information Rate : 0.573           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3869          
# 
# Mcnemar's Test P-Value : 9.488e-16       
#                                           
#               Precision : 0.7186          
#                  Recall : 0.5278          
#                      F1 : 0.6086          
#              Prevalence : 0.4270          
#          Detection Rate : 0.2254          
#    Detection Prevalence : 0.3136          
#       Balanced Accuracy : 0.6869          
#                                           
#        'Positive' Class : No 

## Matrice di confusione per il Yes
confmat_yes = confusionMatrix(nn_total_bin_no_pca$pred$pred, 
                              as.factor(nn_total_bin_no_pca$pred$obs),
                              mode = "prec_recall",
                              positive = "Yes")
confmat_yes
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  No Yes
# No  332 130
# Yes 297 714
# 
# Accuracy : 0.7101          
# 95% CI : (0.6862, 0.7332)
# No Information Rate : 0.573           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3869          
# 
# Mcnemar's Test P-Value : 9.488e-16       
#                                           
#               Precision : 0.7062          
#                  Recall : 0.8460          
#                      F1 : 0.7698          
#              Prevalence : 0.5730          
#          Detection Rate : 0.4847          
#    Detection Prevalence : 0.6864          
#       Balanced Accuracy : 0.6869          
#                                           
#        'Positive' Class : Yes 
## Variano solamento i valori di Precision, Recall etc. rispetto a No

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
## Precision No: 0.7186147
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.7062315
precision_macro_average = mean(c(precision_no, precision_yes))
precision_macro_average
## Precision Macro Average: 0.7124231
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.5278219
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.8459716
recall_macro_average = mean(c(recall_no, recall_yes))
recall_macro_average
## Recall Macro Average: 0.6868968
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.6086159
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7698113
f1_macro_average = mean(c(f1_no, f1_yes))
f1_macro_average
## F1-Measure Macro Average: 0.6892136
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Uso la funzione prediction per predire il risultato in base alla 
## probabilita' di Yes
pred.rocr = ROCR::prediction(nn_total_bin_no_pca$pred$Yes, 
                             nn_total_bin_no_pca$pred$obs)

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
nn_total_bin_no_pca_yes.ROC = roc(response = nn_total_bin_no_pca$pred$obs,
                                  predictor = nn_total_bin_no_pca$pred$Yes,
                                  levels = levels(cmc_bin[,c("Contraceptive_Is_Used")]))
nn_total_bin_no_pca_no.ROC = roc(response = nn_total_bin_no_pca$pred$obs,
                                 predictor = nn_total_bin_no_pca$pred$No,
                                 levels = levels(cmc_bin[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
nn_total_bin_no_pca$times
# $everything
# user  system elapsed 
# 8.72  0.09    8.76 
# 
# $final
# user  system elapsed 
# 0.17  0.00    0.17  
# 
# $prediction
# [1] NA NA NA

## Elimino le variabili non piu' necessarie
rm(confmat_no)
rm(confmat_yes)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)

## Elimino tutti i plot
dev.off()



##----- 3.2.2 - Rete neurale con 10-fold CV (ripetuta 3 volte) con -----------##
##----- split 70/30 per problema binario (senza PCA) -------------------------##

## Split del dataset
allset_no_pca = split.data(cmc_bin, proportion = 0.7, seed = 1)
trainset_no_pca = allset_no_pca$train
testset_no_pca = allset_no_pca$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_no_pca$Contraceptive_Is_Used) =
  make.names(levels(factor(trainset_no_pca$Contraceptive_Is_Used)))
levels(testset_no_pca$Contraceptive_Is_Used) =
  make.names(levels(factor(testset_no_pca$Contraceptive_Is_Used)))

## Addestramento della rete neurale con 10-Fold CV su train set
nn_split_bin_no_pca = train(Contraceptive_Is_Used ~ ., 
                     data = trainset_no_pca, 
                     method = "nnet",
                     type = 'Classification',
                     metric = "ROC",
                     trControl = train_control_bin_repeated)

## Stampa di alcune caratteristiche della rete (tra cui size)
print(nn_split_bin_no_pca)

## Vediamo importanza delle variabili
varImp(nn_split_bin_no_pca)

## Plot della rete
plotnet(nn_split_bin_no_pca$finalModel)

## Predizione sul testset_no_pca
nn_split_bin_no_pca.pred = predict(nn_split_bin_no_pca,
                            testset_no_pca[, !names(testset_no_pca) %in% 
                                      c("Contraceptive_Is_Used")])

## Predizione sul testset_no_pca (con probabilita')
nn_split_bin_no_pca.probs = predict(nn_split_bin_no_pca,
                             testset_no_pca[, !names(testset_no_pca) %in% 
                                       c("Contraceptive_Is_Used")],
                             type = "prob")

## Matrice di confusione per No
confmat_no = confusionMatrix(nn_split_bin_no_pca.pred,
                             as.factor(testset_no_pca$Contraceptive_Is_Used),
                             mode = "prec_recall", # Calcola direttamente Precision, Recall ecc.
                             positive = "No")
confmat_no
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  No Yes
# No   83  51
# Yes  78 229
# 
# Accuracy : 0.7075          
# 95% CI : (0.6626, 0.7496)
# No Information Rate : 0.6349          
# P-Value [Acc > NIR] : 0.0007848       
# 
# Kappa : 0.3457          
# 
# Mcnemar's Test P-Value : 0.0220693       
#                                           
#               Precision : 0.6194          
#                  Recall : 0.5155          
#                      F1 : 0.5627          
#              Prevalence : 0.3651          
#          Detection Rate : 0.1882          
#    Detection Prevalence : 0.3039          
#       Balanced Accuracy : 0.6667          
#                                           
#        'Positive' Class : No 

confmat_yes = confusionMatrix(nn_split_bin_no_pca.pred,
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
## Precision No: 0.619403
precision_yes = confmat_yes[["byClass"]][["Precision"]]
precision_yes
## Precision Yes: 0.7459283
precision_macro_average = mean(c(precision_no, precision_yes))
precision_macro_average
## Precision Macro Average: 0.6826657
rm(precision_no)
rm(precision_yes)
rm(precision_macro_average)

recall_no = confmat_no[["byClass"]][["Recall"]]
recall_no
## Recall No: 0.515528
recall_yes = confmat_yes[["byClass"]][["Recall"]]
recall_yes
## Recall Yes: 0.8178571
recall_macro_average = mean(c(recall_no, recall_yes))
recall_macro_average
## Recall Macro Average: 0.6666925
rm(recall_no)
rm(recall_yes)
rm(recall_macro_average)

f1_no = confmat_no[["byClass"]][["F1"]]
f1_no
## F1-Measure No: 0.5627119
f1_yes = confmat_yes[["byClass"]][["F1"]]
f1_yes
## F1-Measure Yes: 0.7802385
f1_macro_average = mean(c(f1_no, f1_yes))
f1_macro_average
## F1-Measure Macro Average: 0.6714752
rm(f1_no)
rm(f1_yes)
rm(f1_macro_average)

## Recupera le probabilita' di Yes
pred.to.roc = nn_split_bin_no_pca.probs[, 2]

## Usa la funzione prediction per predire il risultato in base alla probabilita' di Yes
pred.rocr = ROCR::prediction(pred.to.roc, 
                             testset_no_pca$Contraceptive_Is_Used)

## Usa la funzione performance per ottenere le misure di performance
perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")

# TPR = True Positive Rate
# FPR = False Positive Rate
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

## Visualizza la curva ROC per il Yes
plot(perf.tpr.rocr, colorize = T, main=paste("AUC:", (perf.rocr@y.values)))
abline(a = 0, b = 1)

## Creazione variabili per le curve ROC da confrontare con altri modelli
nn_split_bin_no_pca_yes.ROC = roc(response = testset_no_pca[,c("Contraceptive_Is_Used")],
                           predictor = nn_split_bin_no_pca.probs$Yes,
                           levels = levels(testset_no_pca[,c("Contraceptive_Is_Used")]))
nn_split_bin_no_pca_no.ROC = roc(response = testset_no_pca[,c("Contraceptive_Is_Used")],
                          predictor = nn_split_bin_no_pca.probs$No,
                          levels = levels(testset_no_pca[,c("Contraceptive_Is_Used")]))

## Tempi di calcolo
nn_split_bin_no_pca$times
# $everything
# user  system elapsed 
# 18.63 0.19     18.69 
# 
# $final
# user  system elapsed 
# 0.11  0.00      0.11 
# 
# $prediction
# [1] NA NA NA

## Rimozione variabili non necessarie
rm(allset_no_pca)
#rm(trainset_no_pca)
#rm(testset_no_pca)
rm(confmat_no)
rm(confmat_yes)
rm(pred.to.roc)
rm(perf.rocr)
rm(perf.tpr.rocr)
rm(pred.rocr)
rm(plt)
rm(nn_split_bin_no_pca.probs)
rm(nn_split_bin_no_pca.pred)

## Eliminazione plot

dev.off()



##----- 3.2.3 - Rete neurale con 10-fold CV su problema multi (senza PCA) ----##

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(cmc_multi$Contraceptive_Method_Used) = 
  make.names(levels(factor(cmc_multi$Contraceptive_Method_Used)))

## Addestramento della rete neurale con 10-Fold CV su intero dataset
nn_total_multi_no_pca = train(Contraceptive_Method_Used ~ ., 
                              data = cmc_multi, 
                              method = "nnet",
                              trControl = train_control_multi,
                              metric = "AUC")

## Stampa di alcune caratteristiche della rete
print(nn_total_multi_no_pca)

## Importanza delle variabili
varImp(nn_total_multi_no_pca)

## Plot della rete
plotnet(nn_total_multi_no_pca$finalModel)

## Visualizzazione confusion matrix
confusionMatrix(nn_total_multi_no_pca) # Accuracy (average) : 0.5628

confmat = confusionMatrix(nn_total_multi_no_pca$pred$pred, 
                          as.factor(nn_total_multi_no_pca$pred$obs),
                          mode = "prec_recall")
confmat
sum(confmat$table) # coincide con numero istanze del dataset
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No.Use Long.Term Short.Term
# No.Use        396        78        119
# Long.Term      78       153        112
# Short.Term    155       102        280
# 
# Overall Statistics
# 
# Accuracy : 0.5628         
# 95% CI : (0.537, 0.5883)
# No Information Rate : 0.427          
# P-Value [Acc > NIR] : <2e-16         
# 
# Kappa : 0.3263         
# 
# Mcnemar's Test P-Value : 0.1579         
# 
# Statistics by Class:
# 
#                      Class: No.Use Class: Long.Term Class: Short.Term
# Precision                   0.6678           0.4461            0.5214
# Recall                      0.6296           0.4595            0.5479
# F1                          0.6481           0.4527            0.5344
# Prevalence                  0.4270           0.2261            0.3469
# Detection Rate              0.2688           0.1039            0.1901
# Detection Prevalence        0.4026           0.2329            0.3646
# Balanced Accuracy           0.6981           0.6464            0.6404

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
## Precision No Use: 0.6677909
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.4460641
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.5214153
precision_macro_average = 
  (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.5450901
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.6295707
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.4594595
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.5479452
recall_macro_average = 
  (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.5456585
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.6481178
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.4526627
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.5343511
f1measure_macro_average = 
  (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.5450439
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

nn_total_multi_no_pca.res = data.frame(No.Use, Long.Term, Short.Term) 
for (i in 1:1473) {
  nn_total_multi_no_pca.res[i, 1] = 0
  nn_total_multi_no_pca.res[i, 2] = 0
  nn_total_multi_no_pca.res[i, 3] = 0
}

for (i in 1:1473) {
  index = nn_total_multi_no_pca$pred$rowIndex[i]
  nn_total_multi_no_pca.res[index, 1] = nn_total_multi_no_pca$pred$No.Use[i]
  nn_total_multi_no_pca.res[index, 2] = nn_total_multi_no_pca$pred$Long.Term[i]
  nn_total_multi_no_pca.res[index, 3] = nn_total_multi_no_pca$pred$Short.Term[i]
}

## Creo il dataframe corretto per multiROC
nn_total_multi_no_pca.df = create_multiROC_dataframe("Rete_Neurale", 
                                                     cmc_multi, 
                                                     nn_total_multi_no_pca.res)

## Invocazione di multiROC per la costruzione delle curve ROC
nn_total_multi_no_pca.res_multi_roc = multi_roc(nn_total_multi_no_pca.df, 
                                                force_diag = T)

## Plot delle curve ROC
plot_ROC(nn_total_multi_no_pca.res_multi_roc, "Curva ROC per Rete Neurale")

rm(nn_total_multi_no_pca.res)
rm(nn_total_multi_no_pca.df)

## Memorizzazione dei valori AUC sia per classe, sia per macro e micro
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in neuralnet_multi_no_pca
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
nn_total_multi_no_pca.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
nn_total_multi_no_pca.AUC[1, 1] = 
  nn_total_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$No.Use
nn_total_multi_no_pca.AUC[1, 2] = 
  nn_total_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$Long.Term
nn_total_multi_no_pca.AUC[1, 3] = 
  nn_total_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$Short.Term
nn_total_multi_no_pca.AUC[1, 4] = 
  nn_total_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$macro
nn_total_multi_no_pca.AUC[1, 5] = 
  nn_total_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$micro
nn_total_multi_no_pca.AUC
# No.Use     Long.Term  Short.Term  Macro      Micro
# 0.7549032  0.7409646  0.7004345   0.7320958  0.7480347

## Tempi di calcolo
nn_total_multi_no_pca$times
# $everything
# user  system elapsed 
# 14.03 0.06     14.89 
# 
# $final
# user  system elapsed 
# 0.36  0.00      0.59  
# 
# $prediction
# [1] NA NA NA

## Elimino variabili non necessarie
rm(nn_total_multi_no_pca.AUC)
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
##----- split 70/30 per problema multi-classe (senza PCA) --------------------##

## Split del dataset
allset_multi_no_pca = split.data(cmc_multi, proportion = 0.7, seed = 1)
trainset_multi_no_pca = allset_multi_no_pca$train
testset_multi_no_pca = allset_multi_no_pca$test

## Formattazione dei nomi delle label per renderli compatibili con caret
levels(trainset_multi_no_pca$Contraceptive_Method_Used) =
  make.names(levels(factor(trainset_multi_no_pca$Contraceptive_Method_Used)))
levels(testset_multi_no_pca$Contraceptive_Method_Used) =
  make.names(levels(factor(testset_multi_no_pca$Contraceptive_Method_Used)))

## Addestramento della rete neurale con 10-Fold CV su train set
nn_split_multi_no_pca = train(Contraceptive_Method_Used ~ ., 
                       data = trainset_multi_no_pca, 
                       method = "nnet",
                       trControl = train_control_multi_repeated,
                       metric = "AUC")

## Stampa di alcune caratteristiche della rete
print(nn_split_multi_no_pca)

## Importanza delle variabili
varImp(nn_split_multi_no_pca)

## Plot della rete
plotnet(nn_split_multi_no_pca$finalModel)

## Predizione sul testset
nn_split_multi_no_pca.pred = predict(nn_split_multi_no_pca,
                              testset_multi_no_pca[, !names(testset_multi_no_pca) %in% 
                                              c("Contraceptive_Method_Used")])

## Predizione sul testset (con probabilita')
nn_split_multi_no_pca.probs = predict(nn_split_multi_no_pca,
                               testset_multi_no_pca[, !names(testset_multi_no_pca) %in% 
                                               c("Contraceptive_Method_Used")],
                               type = "prob")

## Calcolo matrice di confusione
confmat = confusionMatrix(nn_split_multi_no_pca.pred,
                          as.factor(testset_multi_no_pca$Contraceptive_Method_Used),
                          mode = "prec_recall")
confmat
## La matrice di confusione e' calcolata solo sul testset
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No.Use Long.Term Short.Term
# No.Use         99        26         51
# Long.Term       9        46         35
# Short.Term     53        37         85
# 
# Overall Statistics
# 
# Accuracy : 0.5215         
# 95% CI : (0.4738, 0.569)
# No Information Rate : 0.3878         
# P-Value [Acc > NIR] : 8.591e-09      
# 
# Kappa : 0.2639         
# 
# Mcnemar's Test P-Value : 0.03929        
# 
# Statistics by Class:
# 
#                      Class: No.Use Class: Long.Term Class: Short.Term
# Precision                   0.5625           0.5111            0.4857
# Recall                      0.6149           0.4220            0.4971
# F1                          0.5875           0.4623            0.4913
# Prevalence                  0.3651           0.2472            0.3878
# Detection Rate              0.2245           0.1043            0.1927
# Detection Prevalence        0.3991           0.2041            0.3968
# Balanced Accuracy           0.6700           0.6447            0.5819

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
## Precision No Use: 0.5625
precision_long_term = confmat_byclass[c("Class: Long.Term"),c("Precision")]
precision_long_term
## Precision Long Term: 0.5111111
precision_short_term = confmat_byclass[c("Class: Short.Term"),c("Precision")]
precision_short_term
## Precision Short Term: 0.4857143
precision_macro_average = (precision_no_use + precision_long_term + precision_short_term) / 3
precision_macro_average
## Precision Macro Average: 0.5197751
rm(precision_no_use)
rm(precision_long_term)
rm(precision_short_term)
rm(precision_macro_average)

recall_no_use = confmat_byclass[c("Class: No.Use"),c("Recall")]
recall_no_use
## Recall No Use: 0.6149068
recall_long_term = confmat_byclass[c("Class: Long.Term"),c("Recall")]
recall_long_term
## Recall Long Term: 0.4220183
recall_short_term = confmat_byclass[c("Class: Short.Term"),c("Recall")]
recall_short_term
## Recall Short Term: 0.497076
recall_macro_average = (recall_no_use + recall_long_term + recall_short_term) / 3
recall_macro_average
## Recall Macro Average: 0.5113337
rm(recall_no_use)
rm(recall_long_term)
rm(recall_short_term)
rm(recall_macro_average)

f1measure_no_use = confmat_byclass[c("Class: No.Use"),c("F1")]
f1measure_no_use
## F1-Measure No Use: 0.5875371
f1measure_long_term = confmat_byclass[c("Class: Long.Term"),c("F1")]
f1measure_long_term
## F1-Measure Long Term: 0.4623116
f1measure_short_term = confmat_byclass[c("Class: Short.Term"),c("F1")]
f1measure_short_term
## F1-Measure Short Term: 0.4913295
f1measure_macro_average = (f1measure_no_use + f1measure_long_term + f1measure_short_term) / 3
f1measure_macro_average
## F1-Measure Macro Average: 0.513726
rm(f1measure_no_use)
rm(f1measure_long_term)
rm(f1measure_short_term)
rm(f1measure_macro_average)

## Creo il dataframe corretto per multiROC
nn_split_multi_no_pca.df = create_multiROC_dataframe("Rete_Neurale", 
                                              testset_multi_no_pca, 
                                              nn_split_multi_no_pca.probs)

## Invocazione di multiROC per la costruzione delle curve ROC
nn_split_multi_no_pca.res_multi_roc = multi_roc(nn_split_multi_no_pca.df, 
                                         force_diag = T)

## Plot delle curve ROC
plot_ROC(nn_split_multi_no_pca.res_multi_roc, "Curva ROC per Rete Neurale")

## Memorizzazione dei valori AUC sia per classe, sia per macro e micro
No.Use = c(0)
Long.Term = c(0) 
Short.Term = c(0)
Macro = c(0)
Micro = c(0)

# Diversita' di valori rispetto a quelli riportati in neuralnet_multi_no_pca
# Spiegazione: https://stackoverflow.com/questions/31138751/roc-curve-from-training-data-in-caret
nn_split_multi_no_pca.AUC = data.frame(No.Use, Long.Term, Short.Term, Macro, Micro)
nn_split_multi_no_pca.AUC[1, 1] = 
  nn_split_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$No.Use
nn_split_multi_no_pca.AUC[1, 2] = 
  nn_split_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$Long.Term
nn_split_multi_no_pca.AUC[1, 3] = 
  nn_split_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$Short.Term
nn_split_multi_no_pca.AUC[1, 4] = 
  nn_split_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$macro
nn_split_multi_no_pca.AUC[1, 5] = 
  nn_split_multi_no_pca.res_multi_roc$AUC$Rete_Neurale$micro
nn_split_multi_no_pca.AUC

# 0.755457 0.7622416  0.6358241   0.717804  0.7232429

## Tempi di calcolo
nn_split_multi_no_pca$times
# $everything
# user  system elapsed 
# 28.45 0.25     28.71 
# 
# $final
# user  system elapsed 
# 0.20  0.00      0.22 
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
rm(nn_split_multi_no_pca.df)
# rm(nn_split_multi_no_pca.res_multi_roc)
rm(nn_split_multi_no_pca.AUC)
rm(allset_multi_no_pca)
#rm(trainset_multi_no_pca)
#rm(testset_multi_no_pca)
rm(nn_split_multi_no_pca.pred)
rm(nn_split_multi_no_pca.probs)

## Elimino i plot
dev.off()



##----- 3.2.5 - Confronto dei modelli ottenuti -------------------------------##

## Plot delle curve ROC usando pROC
# Plot delle curve ROC per Yes e No per il problema totale
plot(nn_total_bin_no_pca_no.ROC, type = "S", col = "green")
plot(nn_total_bin_no_pca_yes.ROC, add = TRUE, col = "blue")
# Plot delle curve ROC per Yes e No per il problema split
plot(nn_split_bin_no_pca_no.ROC, type = "S", col = "black")
plot(nn_split_bin_no_pca_yes.ROC, add = TRUE, col = "red")

## Conferma che i valori AUC sono gli stessi per la classe positiva e negativa
## nel caso di classificatore binario
nn_total_bin_no_pca_yes.ROC$auc
nn_total_bin_no_pca_no.ROC$auc
nn_split_bin_no_pca_yes.ROC$auc
nn_split_bin_no_pca_no.ROC$auc

## Plot delle curve ROC usando MLeval
# Plot delle curve ROC per Yes e No (confrontando problema totale e split)
models_bin = list(nn_total_bin = nn_total_bin_no_pca, 
                  nn_split_bin = nn_split_bin_no_pca)
evalm(models_bin, gnames = c("NN totale binario", "NN split binario"), 
      positive = "Yes")
evalm(models_bin, gnames = c("NN totale binario", "NN split binario"), 
      positive = "No")

## Plot delle curve ROC per No.Use, Long.Term e Short.Term 
## (confrontando problema totale e split)
models_multi = list(nn_total_multi = nn_total_multi_no_pca, 
                    nn_split_multi = nn_split_multi_no_pca)
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "No.Use")
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "Long.Term")
evalm(models_multi, gnames = c("NN totale multi", "NN split multi"), 
      positive = "Short.Term")

## Plot delle curve ROC usando multiROC
# Plot delle curve ROC per Yes e No per il problema totale binario
nn_multiroc_total.probs = predict(nn_total_bin_no_pca, type = "prob")
nn_multiroc_total.df = create_multiROC_dataframe_binario("rete_neurale",
                                                         cmc_bin,
                                                         nn_multiroc_total.probs)
nn_multiroc_total.res_multi_roc = multi_roc(nn_multiroc_total.df, 
                                            force_diag = T)
plot_ROC(nn_multiroc_total.res_multi_roc, "Curva ROC per rete neurale")

# Plot delle curve ROC per Yes e No per il problema split binario
nn_multiroc_split.probs = predict(nn_split_bin_no_pca,
                                  testset_no_pca[, !names(testset_no_pca) %in% 
                                            c("Contraceptive_Is_Used")],
                                  type = "prob")
nn_multiroc_split.df = create_multiROC_dataframe_binario("rete_neurale",
                                                         testset_no_pca,
                                                         nn_multiroc_split.probs)
nn_multiroc_split.res_multi_roc = multi_roc(nn_multiroc_split.df, 
                                            force_diag = T)
plot_ROC(nn_multiroc_split.res_multi_roc, "Curva ROC per rete neurale")

## I valori AUC possono essere ricavati direttamente dalla stampa delle reti
nn_total_bin_no_pca
nn_total_multi_no_pca

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