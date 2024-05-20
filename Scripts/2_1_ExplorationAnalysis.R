###############################
## 2_1 - Analisi Esplorativa ##
###############################

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

#### Import librerie

library(caret) # necessaria per feature plot

#### 2.0 - Analisi dell'intero dataset

## Creo una copia del dataset, necessaria nella fasi successive
cmc.copy = cmc_bin

## Funzione per analizzare la struttura del dataset, vedere di che attributi si
## compone ed il loro tipo
str(cmc_bin)

## Funzione per avere una visione ad alto livello dei diversi valori per ogni
## attributo. In particolare, per ogni attributo vediamo misure importanti come
## il valore minimo, il massimo e la media
summary(cmc_bin)

## Controllo su tutte le variabili se sono presenti valori mancanti
apply(cmc_bin, 2, function (cmc_bin) sum(is.na(cmc_bin))) # nessun valore mancante

## Eseguo un boxplot per ogni attributo numerico, e tramite par li unisco tutti 
## in un solo plot
par(mfrow = c(1,2))
for (i in c(1, 4)) {
  boxplot(cmc_bin[, i], main = names(cmc_bin)[i])
}
rm(i)
par(mfrow = c(1,1))

pie(table(cmc_multi$Contraceptive_Method_Used))

#### 2.1 - Analisi Esplorativa Univariata

### 2.1.1 - Wife Age (Eta' della moglie) - tipo: intero

summary(cmc_bin$Wife_Age)

## Calcolo della deviazione standard per capire quanto i dati sono dispersi
## rispetto alla media
sd(cmc_bin$Wife_Age)

## IStogramma per capire come sono distribuite le eta'. Notiamo una prevalenza di
## mogli di 25 anni, ma non di molto. Si nota che i dati sono abbastanza
## uniformemente distribuite rispetto all'eta' della moglie
hist(cmc_bin$Wife_Age, main="Wife Age", xlab="Years", ylab = "Frequency")

## Non troviamo outliers
boxplot(cmc_bin$Wife_Age, main="Wife Age", xlab="Years", ylab = "Frequency")

## Plot che ci mostra la distribuzione della variabile target rispetto al
## variare del valore assunto da wife age. Le curve sono abbastanza sovrapposte
## tra loro.
featurePlot(cmc_bin[, 1], cmc_bin[, 10], plot="density", 
            scales=list(x=list(relation="free"), 
                        y=list(relation="free")), 
            auto.key=list(columns=3))

## Creazione di una variabile categoria per dividere le istanze del dataset in
## al range di eta'. Grazie a questa variabile e' stato realizzato un grafico
## a torta per capire visivamente come si distribuiscono le istanze rispetto
## all'eta' della moglie
cmc.copy$Age_Range = cut(cmc_bin$Wife_Age, breaks = c(16, 19, 29, 39, 50), 
                         include.lowest = T, 
                         ordered_result = T, 
                         labels = c("Teen", "Twenties", "Thirties", "Forty"))
pie(table(cmc.copy$Age_Range))

### 2.1.2 - Wife Education (Livello di educazione della moglie) - tipo: factor

## Contiamo il numero di istanza per ogni valore
table(cmc_bin$Wife_Education)

## Pie plot per vedere la numerosita' di ogni categoria graficamente
pie(table(cmc_bin$Wife_Education))

### 2.1.3 - Husband Education (Livello di educazione del marito) - tipo: factor

## Contiamo il numero di istanze per ogni valore
table(cmc_bin$Husband_Education)

## Percentuale di ogni valore
table(cmc_bin$Husband_Education)/length(cmc_bin$Husband_Education)

## Bar plot. Notiamo che piu' della meta' delle istanze hanno valore High su
## questo attributo
barplot(table(cmc_bin$Husband_Education))

### 2.1.4 - Number of Children (Numero di bambini della coppia) - tipo: intero

summary(cmc_bin$Number_Children)

## Calcolo della deviazione standard, per capire come i valori sono distributi
## rispetto alla media
sd(cmc_bin$Number_Children)

## Istogramma. Notiamo che, all'aumentare del numero di figli, il numero di
## istanze con tal numero di figli e' sempre meno
hist(cmc_bin$Number_Children, main="Number of Children", 
     xlab="Count", ylab = "Frequency")

## Box plot. Notiamo esserci degli outliers verso l'alto
boxplot(cmc_bin$Number_Children, main="Number of Children", 
        xlab="Count", ylab = "Frequency")

## Plot che ci mostra la distribuzione della variabile target rispetto al
## variare del valore assunto da number children. Le curve sono abbastanza 
## sovrapposte tra loro.
featurePlot(cmc_bin[, 4], cmc_bin[, 10], plot="density", 
            scales=list(x=list(relation="free"), 
                        y=list(relation="free")), 
            auto.key=list(columns=3))

cmc.copy$Children_Range = cut(cmc_bin$Number_Children, breaks = c(0, 3, 6, 9, 16), 
                              include.lowest = T, ordered_result = T, 
                              labels = c("Low", "Mid-Low", "Mid-High", "High"))
pie(table(cmc.copy$Children_Range))

## Controlliamo le istanze con valore massimo su Number_Children.
## Notiamo essercene una sola, con eta' della moglie alta, quindi la
## consideriamo una istanza consistente
cmc_bin[which(cmc_bin$Number_Children == max(cmc_bin$Number_Children)),1:10]

### 2.1.5 - Wife Religion (Religione della moglie) - tipo: factor

## Contiamo il numero di istanze per ogni valore dell'attributo
table(cmc_bin$Wife_Religion)

## Bar plot e Pie. Notiamo che la maggior parte delle istanze ha valore Islam.
## Questo potrebbe essere un buon attributo per discriminare la tipologia di
## contraccettivo utilizzato
barplot(table(cmc_bin$Wife_Religion))
pie(table(cmc_bin$Wife_Religion))

### 2.1.6 - Wife Working (Indica se la moglie lavora oppure no) - tipo: factor

## Conto il numero di istanze per ogni categoria
table(cmc_bin$Wife_Is_Working)

## Bar plot e Pie. Notiamo che la maggior parte delle mogli non lavora. Questo
## attributo potrebbe distriminare bene sul tipo di contraccettivo, perche' non
## uniformente distribuito
barplot(table(cmc_bin$Wife_Is_Working))
pie(table(cmc_bin$Wife_Is_Working))

### 2.1.7 - Husband Occupation (Livello di importanza del lavoro del marito) -
### tipo: factor

## Conto il numero di istanze per ogni categoria
table(cmc_bin$Husband_Occupation)

## Bar plot. Notiamo che pochi mariti hanno un lavoro di alto livello, mentre
## le altre categorie sono piu' uniformente distribite
barplot(table(cmc_bin$Husband_Occupation))

## 2.1.8 - Living Index (Qualita' della vita) - tipo: factor

## Conto il numero di istanze per categoria
table(cmc_bin$Living_Index)

# Bar plot
barplot(table(cmc_bin$Living_Index))

### 2.1.9 - Media Exposure (Esposizione Mediatica) - tipo: factor

## Conto il numero di istanze per categoria
table(cmc_bin$Media_Exposure)

## Bar plot. Notiamo che la maggior parte dei valore e' Good, quindi questo
## attributo potrebbe essere utile per discriminare il tipo di contraccettivo
## usato
barplot(table(cmc_bin$Media_Exposure))
pie(table(cmc_bin$Media_Exposure))

### 2.1.10 - Contraceptive Method Use (Tipologia di contraccettivo usato) -
### tipo: factor

## Conto il numero di istanze per ogni categoria (problema binario)
table(cmc_bin$Contraceptive_Is_Used)

## Conto il numero di istanze per ogni categoria (problema multi-classe)
table(cmc_multi$Contraceptive_Method_Used)

## Vediamo la distribuzione della variabile target
plot(cmc_bin[, 10], col=c(1,2,3))

#### 2.2 - Analisi Esplorativa Multivariata

### 2.2.1 - Relazioni tra Wife'Age e Number of Children

## Analizziamo il numero di figli nelle coppie con moglie di eta' tra 16 e 19
hist(cmc.copy[which(cmc.copy$Age_Range == "Teen"), 4], xlab = "Age")

## Analizziamo il numero di figli nelle con moglie di eta' tra 20 e 29
hist(cmc.copy[which(cmc.copy$Age_Range == "Twenties"), 4], xlab = "Age")

## Analizziamo il numero di figli nelle con moglie di eta' tra 20 e 39
hist(cmc.copy[which(cmc.copy$Age_Range == "Thirties"), 4], xlab = "Age")

## Analizziamo il numero di figli nelle con moglie di eta' tra 40 e 49
hist(cmc.copy[which(cmc.copy$Age_Range == "Forty"), 4], xlab = "Age")

### 2.2.2 - Relazioni tra Wife's Education e Husband's Education

## Grafico a torta del livello di educazione dei mariti di donne con basso o
## medio basso livello di educazione
pie(table(cmc_bin[which(cmc_bin$Wife_Education == "Low" | 
                    cmc_bin$Wife_Education == "Mid-Low"), 3]))

## Grafico a torta del livello di educazione dei mariti di donne con medio
## alto o alto livello di educazione
pie(table(cmc_bin[which(cmc_bin$Wife_Education == "Mid-High" | 
                    cmc_bin$Wife_Education == "High"), 3]))

### 2.2.3 - Relazioni tra Husband's Occupation e Husband's Education

## Grafico a torta del livello di occupazione dei mariti con basso o
## medio basso livello di educazione
pie(table(cmc_bin[which(cmc_bin$Husband_Education == "Low" | 
                    cmc_bin$Husband_Education == "Mid-Low"), 7]))

## Grafico a torta del livello di occupazione dei mariti con alto o
## medio alto livello di educazione
pie(table(cmc_bin[which(cmc_bin$Husband_Education == "High" | 
                    cmc_bin$Husband_Education == "Mid-High"), 7]))

### 2.2.4 - Relazione tra Wife's Working e Number of Children

## Grafico a barre relativo alle categorie di Children_Range per quanto riguarda
## le famiglie in cui la moglie non lavora
barplot(table(cmc.copy[which(cmc.copy$Wife_Is_Working == "Yes"), 
                       "Children_Range"]))

## Grafico a barre relativo alle categorie di Children_Range per quanto riguarda
## le famiglie in cui la moglie lavora
barplot(table(cmc.copy[which(cmc.copy$Wife_Is_Working == "No"), 
                       "Children_Range"]))

### 2.2.5 - Relazioni tra Wife's Working e Wife's Education

## Grafico a barre relativo alla distribuzione dell'occupazione delle moglie
## con basso o medio basso livello di educazione
barplot(table(cmc.copy[which(cmc.copy$Wife_Education == "Low" | 
                             cmc.copy$Wife_Education == "Mid-Low"), 
                       "Wife_Is_Working"]))

## Grafico a barre relativo alla distribuzione dell'occupazione delle moglie
## con alto o medio alto livello di educazione
barplot(table(cmc.copy[which(cmc.copy$Wife_Education == "High" | 
                               cmc.copy$Wife_Education == "Mid-High"), 
                       "Wife_Is_Working"]))

### 2.2.6 - Relazioni tra Wife's Religione Wife's Age (poco interessante)

## Grafico a barre relativo alla distribuzione della religione delle moglie
## di eta' tra 16 e 29
barplot(table(cmc.copy[which(cmc.copy$Age_Range == "Teen" | 
                            cmc.copy$Age_Range == "Twenties"), 
                       "Wife_Religion"]))

## Grafico a barre relativo alla distribuzione della religione delle moglie
## di eta' tra 30 e 49
barplot(table(cmc.copy[which(cmc.copy$Age_Range == "Thirties" | 
                               cmc.copy$Age_Range == "Forty"), 
                       "Wife_Religion"]))

### 2.2.7 - Relazioni tra Wife's Age e Contraceptive is Used (poco interessante)

pie(table(cmc.copy[which(cmc.copy$Age_Range == "Teen"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Age_Range == "Twenties"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Age_Range == "Thirties"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Age_Range == "Forty"), 
                   "Contraceptive_Is_Used"]))

### 2.2.8 - Relazioni tra Number of Childrene e Contraceptive is Used

pie(table(cmc.copy[which(cmc.copy$Children_Range == "Low"), 
                   "Contraceptive_Is_Used"]))
pie(table(cmc.copy[which(cmc.copy$Children_Range == "Mid-Low"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Children_Range == "Mid-High"), 
                   "Contraceptive_Is_Used"]))
pie(table(cmc.copy[which(cmc.copy$Children_Range == "High"), 
                   "Contraceptive_Is_Used"]))

### 2.2.9 - Relazione tra Wife's Educatione e Contraceptive is Used (importante)

pie(table(cmc.copy[which(cmc.copy$Wife_Education == "Low" | 
                         cmc.copy$Wife_Education == "Mid-Low"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Wife_Education == "High" | 
                         cmc.copy$Wife_Education == "Mid-High"), 
                   "Contraceptive_Is_Used"]))

### 2.2.10 - Relazione tra Wife's Religion e Contraceptive is Used

pie(table(cmc.copy[which(cmc.copy$Wife_Religion == "Islam"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Wife_Religion == "Non-Islam"), 
                   "Contraceptive_Is_Used"]))

### 2.2.11 - Relazione tra Living Index e Contraceptive is Used (importante)

pie(table(cmc.copy[which(cmc.copy$Living_Index == "High" | 
                         cmc.copy$Living_Index == "Mid-High"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Living_Index == "Low" | 
                         cmc.copy$Living_Index == "Mid-Low"), 
                   "Contraceptive_Is_Used"]))

### 2.2.12 - Relazione tra Media Exposure e Contraceptive is Used

pie(table(cmc.copy[which(cmc.copy$Media_Exposure == "Good"), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which(cmc.copy$Media_Exposure == "Not-Good"), 
                   "Contraceptive_Is_Used"]))

### 2.2.13 - Relazione tra Living Index, Wife's Education e 
### Contraceptive is Used

pie(table(cmc.copy[which((cmc.copy$Living_Index == "High" | 
                            cmc.copy$Living_Index == "Mid-High") & 
                           (cmc.copy$Wife_Education == "High" | 
                              cmc.copy$Wife_Education == "Mid-High")), 
                   "Contraceptive_Is_Used"]))

pie(table(cmc.copy[which((cmc.copy$Living_Index == "Low" | 
                            cmc.copy$Living_Index == "Mid-Low") & 
                           (cmc.copy$Wife_Education == "Low" | 
                              cmc.copy$Wife_Education == "Mid-Low")), 
                   "Contraceptive_Is_Used"]))

sum(table(cmc.copy[which((cmc.copy$Living_Index == "Low" | 
                        cmc.copy$Living_Index == "Mid-Low") & 
                       (cmc.copy$Wife_Education == "Low" | 
                          cmc.copy$Wife_Education == "Mid-Low")), 
               "Contraceptive_Is_Used"])) + sum(table(cmc.copy[which((cmc.copy$Living_Index == "High" | 
                              cmc.copy$Living_Index == "Mid-High") & 
                             (cmc.copy$Wife_Education == "High" | 
                                cmc.copy$Wife_Education == "Mid-High")), 
                     "Contraceptive_Is_Used"]))

### 2.2.14 - Feature plot multi variabile

## Costruiamo dei fetaure plot per capire come la relazione tra i valori di
## due variabili per volta descrivono la label. Se troviamo tante
## sovrapposizioni vuol dire che i dati non presenta molte relazioni, ed e'
## un problema, perche' non si capisce quali attributi sono meglio di altri per
## discriminare il tipo di contraccettivo.

## Completo. Notiamo che la relazione tra Wife Age e Number of Children e' molto
## sovrapposta, mentre le altre relazioni sono meglio distinte. In particolare,
## la classe Short-Term e' quella meglio discriminata
featurePlot(x=cmc_bin[, 1:9], y=cmc_bin[, 10], plot="pairs", 
            auto.key=list(columns=3))
featurePlot(x=cmc_bin[, c(2,8)], y=cmc_bin[, 10], plot="pairs", 
            auto.key=list(columns=3))

## Vediamo il grado di correlazione tra le due variabili numeriche. Il valore
## ottenuto indica una correlazione positiva piu' vicina (anche se di poco) ad 1
## anziche a zero, quindi, capiamo che in generale all'aumentare del valore di 
## eta' aumenta con una certa probabilita' anche il numero di figli
cor(cmc_bin[, c(1,4)]) # 0.5401259

## Elimino la copia del dataset, ormai non serve piu
rm(cmc.copy)
rm(cmc_bin)
rm(cmc_multi)

## Elimino tutti i plot
dev.off()