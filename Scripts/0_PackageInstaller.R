##############################
## 0 - Istallazione Package ##
##############################

## Ottenimento di tutti i package al momento istallati nell'ambiente corrente
allPackageInstalled = installed.packages()[, 1]

## rstudioapi
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}

## caret
if (!("caret" %in% allPackageInstalled)) {
  install.packages("caret")
}

## FactoMineR
if (!("FactoMineR" %in% allPackageInstalled)) {
  install.packages("FactoMineR")
}

## factoextra
if (!("factoextra" %in% allPackageInstalled)) {
  install.packages("factoextra")
}

## rpart
if (!("rpart" %in% allPackageInstalled)) {
  install.packages("rpart")
}

## rattle
if (!("rattle" %in% allPackageInstalled)) {
  install.packages("rattle")
}

## rpart.plot
if (!("rpart.plot" %in% allPackageInstalled)) {
  install.packages("rpart.plot")
}

## RColorBrewer
if (!("RColorBrewer" %in% allPackageInstalled)) {
  install.packages("RColorBrewer")
}

## ROCR
if (!("ROCR" %in% allPackageInstalled)) {
  install.packages("ROCR")
}

## pROC
if (!("pROC" %in% allPackageInstalled)) {
  install.packages("pROC")
}

## multiROC
if (!("multiROC" %in% allPackageInstalled)) {
  install.packages("multiROC")
}

## dplyr
if (!("dplyr" %in% allPackageInstalled)) {
  install.packages("dplyr")
}

## MLmetrics 
if (!("MLmetrics" %in% allPackageInstalled)) {
  install.packages("MLmetrics")
}

## nnet 
if (!("neuralnet" %in% allPackageInstalled)) {
  install.packages("neuralnet")
}

## NeuralNetTools
if (!("NeuralNetTools" %in% allPackageInstalled)) {
  install.packages("NeuralNetTools")
}

## MLeval 
if (!("MLeval" %in% allPackageInstalled)) {
  install.packages("MLeval")
}

## Settaggio wd
library(rstudioapi)
filePath = getSourceEditorContext()$path
setwd(dirname(dirname(filePath)))

## Eliminazione dell variabile non più utile
rm(allPackageInstalled)
rm(filePath)