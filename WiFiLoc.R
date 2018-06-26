####Objective####
#Evaluate the application of machine learning techniques to the problem of indoor locationing via wifi fingerprinting


####Upload and Load Libraries####
library(corrplot)
library(caret)

####Import Data####
WiFiLoc <- read.csv("C:/Users/Saad/Desktop/Data Analytics - CPE/Course 5/Task 3/WiFi Positioning/trainingdata.csv")

#Prints out up to 530 rows of data
str(WiFiLoc, list.len = 530) #Add list.len to see all columns in database
is.na(WiFiLoc) #Returns TRUE/FALSE for each individual data point
sum(is.na(WiFiLoc)) #Calculates totalnumber of missing values


####Visualization####
non_dBm <- WiFiLoc[(521:529)]
View(non_dBm)

#Pull minimum from each column into comma-delimited list
MindBm <- apply(WiFiLoc, 2, min)
View(MindBm)
min_dBm <- as.matrix(MindBm)
View(min_dBm)
min_dBm <- min_dBm[c(1:520)]
View(min_dBm)
min_dBm <- as.matrix(min_dBm)
View(min_dBm)
colnames(min_dBm) <- "Minimum_Value"
summary(min_dBm)
plot(min_dBm)


####Correlation####
corrdata <- cor(WiFiLoc)
View(corrdata)
is.na(corrdata)
sum(is.na(corrdata))



#Removing empty columns so that I can run function below to delete columns and rows in corrdata with NA values
View(summary(corrdata)) #Manual reporting of which columns had NA values. Need to figure out an easier way to print out row numbers (or names) so that I can copy/paste into the functions below
corrdata.clean <- corrdata[, -c(3,4,92,93,94,95,152,158,159,160,215,217,226,227,238,239,240,241,242,243,244,245,246,247,254,293,296,301,303,304,307,333,349,353,360,365,416,419,423,429,433,438,441,442,444,445,451,458,482,485,487,488,491,497,520)] 

#Remove empty rows
corrdata.clean <- corrdata.clean[-c(3,4,92,93,94,95,152,158,159,160,215,217,226,227,238,239,240,241,242,243,244,245,246,247,254,293,296,301,303,304,307,333,349,353,360,365,416,419,423,429,433,438,441,442,444,445,451,458,482,485,487,488,491,497,520),]


####WiFiLoc Cleaning####
#Remove empty columns from original dataset based on emptiness in correlation table
WiFiLoc.clean <- WiFiLoc[, -c(3,4,92,93,94,95,152,158,159,160,215,217,226,227,238,239,240,241,242,243,244,245,246,247,254,293,296,301,303,304,307,333,349,353,360,365,416,419,423,429,433,438,441,442,444,445,451,458,482,485,487,488,491,497,520)]

#Remove empty rows from original dataset based on emptiness in correlation table
WiFiLoc.clean <- WiFiLoc.clean[-c(3,4,92,93,94,95,152,158,159,160,215,217,226,227,238,239,240,241,242,243,244,245,246,247,254,293,296,301,303,304,307,333,349,353,360,365,416,419,423,429,433,438,441,442,444,445,451,458,482,485,487,488,491,497,520),]


#Find attributes that are correlated with eachother by at least .80
highlyCorr <- findCorrelation(corrdata.clean, cutoff = .80)


#Remove highly correlated columns from original dataset and create new clean dataset
WiFiLoc.clean2 <- WiFiLoc.clean[,-highlyCorr] 
ncol(WiFiLoc.clean2)
str(WiFiLoc.clean2, list.len = 384)
View(WiFiLoc.clean2)


#Find out column numbers associated with column names (can also find total column count and count backwards from end since columns to remove are at the back end)
which(colnames(WiFiLoc.clean2)=="USERID")
which(colnames(WiFiLoc.clean2)=="PHONEID")
which(colnames(WiFiLoc.clean2)=="TIMESTAMP")


#Remove unnecessary columns in WiFiLoc.clean2
WiFiLoc.clean3 <- WiFiLoc.clean2[,-c(382:384)]
ncol(WiFiLoc.clean3) #Calculate total number of columns in WiFiLoc.clean3
str(WiFiLoc.clean3,list.len=381)


#Change column data types from integer to numeric
WiFiLoc.clean3$FLOOR <- as.factor(WiFiLoc.clean3$FLOOR)
WiFiLoc.clean3$BUILDINGID <- as.factor(WiFiLoc.clean3$BUILDINGID)
WiFiLoc.clean3$SPACEID <- as.factor(WiFiLoc.clean3$SPACEID)
WiFiLoc.clean3$RELATIVEPOSITION <- as.factor(WiFiLoc.clean3$RELATIVEPOSITION)
View(WiFiLoc.clean3$BUILDINGID)


#Filter using dplyr#
library(dplyr)
BUILDING0 <- filter(WiFiLoc.clean3, BUILDINGID==0)
BUILDING1 <- filter(WiFiLoc.clean3, BUILDINGID==1)
BUILDING2 <- filter(WiFiLoc.clean3, BUILDINGID==2)
write.csv(BUILDING0, file="BUILDING0.csv")
getwd()



#Combine location variables into one concatenated variable
BUILDING0$Location <- paste(BUILDING0$BUILDINGID, BUILDING0$FLOOR, BUILDING0$SPACEID, BUILDING0$RELATIVEPOSITION, sep='-')
ncol(BUILDING0)
str(BUILDING0,list.len=382)


#Remove unnecessary attributes not needed in BUILDING0 datasets (FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION)
BUILDING0$FLOOR <- NULL
BUILDING0$BUILDINGID <- NULL
BUILDING0$SPACEID <- NULL
BUILDING0$RELATIVEPOSITION <- NULL


#Change newly created Location attribute to factor (models will be run off this attribute)
View(BUILDING0)
sum(is.na(BUILDING0))
BUILDING0$Location <- as.factor(BUILDING0$Location)
str(BUILDING0, list.len=378)

####Training Modeling####
inTrain <- createDataPartition(y=BUILDING0$Location, p=.75, list=FALSE) #Creates training set equal to random 75% of original dataset


training <- BUILDING0[inTrain,] #Creates training set
testing <- BUILDING0[-inTrain,] #Creates testing set
nrow(training)#Counts number of rows for training
nrow(testing) #Counts number of rows for testing
set.seed(123)#Set psuedo-random number generator


#trainControl function for RandomForest, kNN, SVM, and C5.0
fitControl <- trainControl(method = "repeatedcv", number = 10) #Controls computational nuances of training set through repeated cross validation (repeatedcv)

#Random Forest
rffit <- train(Location~., data=training, method="rf", trControl=fitControl) #Creates Random Forest model 
rffit
predictors(rffit) #Returns all predictors
rfPredict <- predict(rffit, testing) #Predicts, using random forest model, the trained model using the testing dataset
rfPredict
postResample(rfPredict, testing$Location) #Returns accuracy and kappa values


#kNN
knnFit <- train(Location~., data=training, method="knn", trControl=fitControl) #Creates kNN model; tuneLength value provides that many k-values (minimum value is 2)
knnFit #Returns kNN model
predictors(knnFit) #Returns all predictors
knnPredict <- predict(knnFit, testing)#Predicts kNN model to testing set
postResample(knnPredict, testing$Location) #Returns accuracy and kappa values


#C5.0 Method
C5.0Method <- train(y=training$Location,x=training, method="C5.0", trControl=fitControl)
C5.0Method
predictors(C5.0Method)
C5Predict <- predict(C5.0Method, testing)
C5Predict
postResample(C5Predict, testing$Location) #Returns accuracy and kappa values


#Support Vector Machine
svmMethod <- train(Location~., data = training, method = "svmLinear", scale=FALSE, trControl=fitControl)
svmMethod
predictors(svmMethod)
svmPredict <- predict(svmMethod, testing)
svmPredict
postResample(svmPredict, testing$Location) #Returns accuracy and kappa values

####Compare Model Performance####
ModelData <- resamples(list(SVM=svmMethod, C50=C5.0Method, kNN=knnFit, RandomForest=rffit2))
summary(ModelData)
write.csv(ModelData, file="Model Data.csv")
