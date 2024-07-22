#import of the data set

library(readxl)
train <- read_excel("C:/Users/Admin/Desktop/Kaggle competition/Titanic/titanic/train.xlsx")
test <- read_excel("C:/Users/Admin/Desktop/Kaggle competition/Titanic/titanic/test.xlsx")
View(train)
View(test)
attach(train)
attach(test)

#Statistics

table(Survived)
summary(Survived)

#management of missing values and character value for train data set

mean_age <- mean(train$Age, na.rm = TRUE)
train$Age[is.na(train$Age)] <- mean_age
Sex1 <- factor(train$Sex, levels = unique(train$Sex))
Sex_num <- as.integer(Sex1)
train$Sex <- Sex_num
Embarked1 <- factor(train$Embarked, levels = unique(train$Embarked))
Embarked_num <- as.integer(Embarked1)
train$Embarked <- Embarked_num

library(dplyr)
train <- select(train, -Name)
#train <- select(train, -Fare)
train <- select(train, -Cabin)
train <- select(train, -Ticket)

any(is.na(train))
sum(is.na(train))
which(is.na(train), arr.ind = TRUE)

train <- na.omit(train)

View(train)

#management of missing values and character value for test data set

median_fare <- median(test$Fare, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- median_fare
mean_age2 <- mean(test$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- mean_age2
Sex2 <- factor(test$Sex, levels = unique(test$Sex))
Sex_num2 <- as.integer(Sex2)
test$Sex <- Sex_num2
Embarked2 <- factor(test$Embarked, levels = unique(test$Embarked))
Embarked_num2 <- as.integer(Embarked2)
test$Embarked <- Embarked_num2

library(dplyr)
test <- select(test, -Name)
test <- select(test, -Cabin)
test <- select(test, -Ticket)

any(is.na(test))
sum(is.na(test))
which(is.na(test), arr.ind = TRUE)

#test <- na.omit(test)

View(test)

#logit regression model

logit <- glm(formula = Survived ~ . , data = train , family = binomial(logit) )
logit$aic
probit <- glm(formula = Survived ~ . , data = train , family = binomial(probit) )
probit$aic

summary(logit)
step(logit , direction = "forward" , K=2)
step(logit , direction = "backward" , K=2)
step(logit , direction = "both" , K=2) #The backward and both stepwise regression give the same results. Let's use this model

f_logit <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Embarked , data = train , family = binomial(logit))
summary(f_logit)
f_logit$aic

f_probit <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Embarked , data = train , family = binomial(probit))
f_probit$aic

library(questionr)
odds.ratio(f_logit)

table(Survived,Sex)

attach(train)

#Relevance of the model

library(pROC)
library(ROCR)
prediction <- predict(f_logit, data = train , type="response")

length(train$Survived)
length(prediction)

roc_object <- roc(train$Survived , prediction) 
plot(roc_object)
auc(roc_object) #0.855 near 1 then correct

#HOSMER-LEMESHOW's test

library(performance)
performance_hosmer (f_logit) #not correct for this relevance criterion

#thumb rule

library(stats)
deviance(f_logit)/df.residual(f_logit) #0.8902421 near 1 then correct

#deviance residues

pvaleur1=1-pchisq(deviance(f_logit),df.residual(f_logit))
pvaleur1 #0.9913539 then OK!

#pearson's residues 

s2=sum(residuals(f_logit,type="pearson")^2)
dd1=df.residual(f_logit)
pvaleur=1-pchisq(s2,dd1)
pvaleur #0.212209 not good

#i have 3 out of 5 relevance criterion are goods. I decide to continue with this model. Maybe should i try later another method 

#Forcasting-1

library(caret)
library(InformationValue)
library(ISLR)

attach(test)

predicted <- predict(f_logit, test , type="response")

summary(predicted)

df_predicted <- data.frame (test$PassengerId, predicted)
View(df_predicted)

df_predicted$predicted <- ifelse(predicted > 0.5,1,0) 

#comparison

library(readxl)
gender_submission <- read_excel("C:/Users/Admin/Desktop/Kaggle competition/Titanic/titanic/gender_submission.xlsx")
View(gender_submission)

any(is.na(gender_submission))

#gender_submission <- gender_submission[-153,]

optimal <- optimalCutoff(gender_submission$Survived, predicted)[1]

length(gender_submission$PassengerId)
length(predicted)

confusionMatrix(gender_submission$Survived, predicted)


#calculate sensitivity

sensitivity(gender_submission$Survived, predicted)

#calculate specificity

specificity(gender_submission$Survived, predicted)


#calculate total misclassification error rate

misClassError(gender_submission$Survived, predicted, threshold=optimal)

#rename data set headers

df_predicted <- df_predicted %>%
  rename(PassengerId = test.PassengerId, Survived = predicted)

View(df_predicted)

write.csv(df_predicted, file = "df_predicted.csv", row.names = FALSE)

#Forcasting-2

library(randomForest)
rfBase.train <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Embarked , data = train) 
attach(test)

predictedrf <- predict(rfBase.train, test , type="response")

df_rf <- data.frame (test$PassengerId, predictedrf)
View(df_rf)

df_rf$predictedrf <- ifelse(predictedrf > 0.5,1,0) 

confusionMatrix(gender_submission$Survived, predictedrf)


#Forcasting-3 Méthode non adaptée

library(e1071)
library(caTools)
library(class)

length(train$Survived)
length(test$PassengerId)

knn1 <- knn(train = train,
                      test = test,
                      cl = train$Survived,
                      k = 20)
knn1

cm <- table(gender_submission$Survived , knn1)
cm

misClassError <- mean(knn1 != gender_submission$Survived)
print(paste('Accuracy =', 1-misClassError))




#---------To be continued ----------     ;) 



