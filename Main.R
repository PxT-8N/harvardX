###############################
# Set seed for reproductivity #
###############################
set.seed(1488)

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(pbapply)) install.packages("pbapply")
if(!require(pbapply)) install.packages("gridExtra")
if(!require(pbapply)) install.packages("ggcorrplot")
if(!require(glmnet)) install.packages("glmnet")
if(!require(glmnet)) install.packages("xgboost")

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(pbapply)
library(caret)
library(gridExtra)
library(ggcorrplot)
library(nnet)
library(randomForest)
library(glmnet)
library(xgboost)

################
# Load dataset #
################

ds <- read.csv("cardio.csv", sep = ';')


#################
# Preprocessing #
#################

# Change age from days to years
ds <- ds %>% mutate(age = round(age/365, digits = 0)) %>% select(-id)

#Unify data because it contains values like 150 and 15000
ds <- ds %>%
  mutate(ap_hi = if_else(nchar(ap_hi) > 3,as.integer(ap_hi/100),ap_hi),
         ap_lo = if_else(nchar(ap_lo) > 3,as.integer(ap_lo/100),ap_lo))


#After next examination there are still some values that are impossible, so let's preprocess it
ds <- ds %>%
  mutate(ap_hi = if_else(ap_hi > 250,as.integer(ap_hi/10),ap_hi),
         ap_lo = if_else(ap_lo > 250,as.integer(ap_lo/10),ap_lo))

#Remove all variables that are incorrect
ds <- ds[ds$ap_hi>70 & ds$ap_lo>60,]


#############################################
# Split data to learning and validation set #
#############################################
# in ration 8/2
tI <- dim(ds)[1] * 0.8

training_set <- ds[1:tI, ]
validation_set <- ds[(tI+1):dim(ds)[1], ]



#Let's start from define our evaluate function, which is Matthews correlation coefficient
#https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

ev_matt <- function (act, pred){
  TP <- sum(act == 1 & pred == 1)
  TN <- sum(act == 0 & pred == 0)
  FP <- sum(act == 0 & pred == 1)
  FN <- sum(act == 1 & pred == 0)

  btm <- as.double(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
  if (any((TP+FP) == 0, (TP+FN) == 0, (TN+FP) == 0, (TN+FN) == 0)) btm <- 1
  ((TP*TN)-(FP*FN)) / sqrt(btm)
}


##################
# Neural network #
##################

nn <- nnet(cardio~.,data=training_set,size=20,maxit=1000)
preds = predict(nn,newdata=validation_set)
nn_score <- ev_matt(validation_set$cardio,preds>0.5)

#################
# Random Forest #
#################

training_setR <- training_set %>% select(-cardio)
rf <- randomForest(training_setR,as.factor(training_set$cardio))
validation_setR <- validation_set %>% select(-cardio)
rf_score <- ev_matt(predict(rf,newdata=validation_setR),validation_set$cardio)

#######################
# logistic regression #
#######################

logi <- glm(cardio~.,data=training_set,family=binomial(link="logit"))
logi_score <- ev_matt(predict(logi,type="response",newdata=validation_set)>0.5,validation_set$cardio)

###########################################
# logistic regression with regularization #
###########################################

train.vars <- as.matrix(select(training_set,-cardio))
valid.vars <- as.matrix(select(validation_set,-cardio))
lgwr <- cv.glmnet(train.vars,training_set$cardio,family="binomial")
lgwr_score <- ev_matt(predict(lgwr,valid.vars,s="lambda.min",type="response")>0.5,validation_set$cardio)


###########
# xgboost #
###########

xgbo <- xgboost(data=train.vars,label=training_set$cardio,nrounds=100)
xgbo_score <- ev_matt(predict(xgbo,newdata=valid.vars)>0.5,validation_set$cardio)

nn_score
rf_score
lgwr_score
logi_score
xgbo_score


