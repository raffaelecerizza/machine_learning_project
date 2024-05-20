###############
## 2_2 - PCA ##
###############

## Settaggio wd
allPackageInstalled = installed.packages()[, 1]
if (!("rstudioapi" %in% allPackageInstalled)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
setwd(dirname(dirname(getSourceEditorContext()$path)))
rm(allPackageInstalled)

#### Caricamento dataset (con istallazione librerie)

source(paste(getwd(), "/Scripts/1_LoadDataset.R",sep = ""))

#### Import delle librerie

library(FactoMineR)

library(factoextra)

#### Calcolo della PCA

## Selezione delle colonne numeriche ed esecuzione su di esse della PCA

cmc.pca = PCA(cmc_bin[, c(1, 4)], graph = FALSE)

summary(cmc.pca)

get_eigenvalue(cmc.pca)
##        eigenvalue   variance.percent  cumulative.variance.percent
## Dim.1  1.5401259    77.00629          77.00629
## Dim.2  0.4598741    22.99371          100.00000

## Tramite estrazione degli autovalori notiamo che la prima dimensione 
## ha autovalore molto maggiore di uno, a differenza della seconda, 
## ed inoltre descrive da sola il 77% della varianza. Puo' allora avere
## senso non considerare la seconda dimensione

## Visualizzazione grafica degli autovalori
fviz_eig(cmc.pca, addlabels = TRUE, ylim = c(0,50))

## Visualizzazione del contributo delle variabili alle dimensioni
fviz_contrib(cmc.pca, choice = "var")

## Vediamo la relazione tra le variabili Wife Age e Number of Children e le
## due dimensione della PCA
cmc.pca$var$cor
##                 Dim.1      Dim.2
## Wife_Age        0.8775323  0.4795175
## Number_Children 0.8775323 -0.4795175
## Si nota una alta correlazione tra Dim1 e le due variabili, mentre piu' bassa
## con la dim2

## Da questo grafico capiamo che le variabili piu' lontane dall'origine sono
## quelle meglio rappresentate dalle dimensioni. Notiamo che la prima dimensione
## descrive bene le due variabili
fviz_pca_var(cmc.pca, col.var = "contrib", repel = TRUE)

## Notiamo che praticamente tutti gli individui sono ben desritti dai PC
fviz_pca_ind(cmc.pca, col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                                repel = TRUE # Avoid text overlapping (slow if many points)
             )

## Visto che la prima dimensione PCA descrive da sola il 77% della varianza, si
## e' deciso di eliminare le colonne Wife Age e Num Children, per sostituirle con
## Dim1
cmc_bin2 = cmc_bin[, -4]
cmc_multi2 = cmc_multi[, -4]

cmc_bin2$Wife_Age = cmc.pca$ind$coord[, 1]
names(cmc_bin2)[names(cmc_bin2) == "Wife_Age"] = "Pca_Dim1"

cmc_multi2$Wife_Age = cmc.pca$ind$coord[, 1]
names(cmc_multi2)[names(cmc_multi2) == "Wife_Age"] = "Pca_Dim1"

## Elimino le variabili non piu' necessarie
rm(cmc.pca)