############################
## 1- Caricamento Dataset ##
############################

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

## Istallazione di tutte le librerie necessarie

source(paste(getwd(), "/Scripts/0_PackageInstaller.R",sep = ""))

## Lettura raw del dataset

cmc_multi = read.csv(file = "cmc.data", header = FALSE, 
                     col.names = c("Wife_Age",
                                   "Wife_Education",
                                   "Husband_Education",
                                   "Number_Children",
                                   "Wife_Religion",
                                   "Wife_Is_Working",
                                   "Husband_Occupation",
                                   "Living_Index",
                                   "Media_Exposure",
                                   "Contraceptive_Method_Used"))

## Analisi della tipologia di attributi nella versione raw

sapply(cmc_multi, class)

## Passaggio da intero a factor per tutti quegli attributi per cui e' possibile,
## in base alle informazioni sul dataset raccolte (presenti in cmc_multi.names)

cmc_multi$Wife_Education = factor(cmc_multi$Wife_Education, levels = 1:4, 
                                labels = c("Low", 
                                           "Mid-Low", 
                                           "Mid-High", 
                                           "High"))

cmc_multi$Husband_Education = factor(cmc_multi$Husband_Education, levels = 1:4, 
                                labels = c("Low", 
                                           "Mid-Low", 
                                           "Mid-High", 
                                           "High"))

cmc_multi$Wife_Religion = factor(cmc_multi$Wife_Religion, levels = 0:1, 
                                labels = c("Non-Islam", 
                                           "Islam"))

cmc_multi$Wife_Is_Working = factor(cmc_multi$Wife_Is_Working, levels = 0:1, 
                           labels = c("Yes", 
                                      "No"))

cmc_multi$Husband_Occupation = factor(cmc_multi$Husband_Occupation, levels = 1:4, 
                                labels = c("Low", 
                                           "Mid-Low", 
                                           "Mid-High", 
                                           "High"))

cmc_multi$Living_Index = factor(cmc_multi$Living_Index, levels = 1:4, 
                           labels = c("Low", 
                                      "Mid-Low", 
                                      "Mid-High", 
                                      "High"))

cmc_multi$Media_Exposure = factor(cmc_multi$Media_Exposure, levels = 0:1, 
                             labels = c("Good", 
                                        "Not-Good"))

## Vengono definiti due dataset, uno che considera il problema multi classe, e
## l'altro che consideta il problema binario. Utili per capire se i modelli
## funziona meglio su uno oppure su un altro

cmc_bin = cmc_multi
cmc_bin$Contraceptive_Method_Used[cmc_bin$Contraceptive_Method_Used == 1] = "No"
cmc_bin$Contraceptive_Method_Used[cmc_bin$Contraceptive_Method_Used == 2 |
                                  cmc_bin$Contraceptive_Method_Used == 3] = "Yes"
names(cmc_bin)[names(cmc_bin) == "Contraceptive_Method_Used"] = 
  "Contraceptive_Is_Used"
cmc_bin$Contraceptive_Is_Used = factor(cmc_bin$Contraceptive_Is_Used)

cmc_multi$Contraceptive_Method_Used = factor(cmc_multi$Contraceptive_Method_Used, 
                                       levels = 1:3, labels = c("No-Use", 
                                                                "Long-Term",
                                                                "Short-Term"))

## Analisi della tipologia di attributi dopo il refactoring

sapply(cmc_multi, class)
sapply(cmc_bin, class)