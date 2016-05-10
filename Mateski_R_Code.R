# Predict 498 - Capstone - Winter 2016
# Team Illinois - Nancy Mateski
# Prudential Kaggle
#
#
setwd("~/Grad School/Predict 498/Prud")

library("caret", lib.loc="~/R/win-library/3.2")
library("dplyr", lib.loc="~/R/win-library/3.2")
library("dummies", lib.loc="~/R/win-library/3.2")
library("Metrics", lib.loc="~/R/win-library/3.2")
library("randomForest", lib.loc="~/R/win-library/3.2")
library("rpart", lib.loc="~/R/win-library/3.2")
library("glmnet", lib.loc="~/R/win-library/3.2")
library("stats", lib.loc="C:/Program Files/R/R-3.2.3/library")
#
#
# THis train.sas data set was modified in SAS by Sasha
# There were 13 numeric variables that had missing values  
# Flag variables were created and set for missing values  
# Family_Hist_2 and Family_Hist_4 were imputed by a simple linear regression 
# The other 11 variables were imputed with zero  
# The original variables were dropped 
# The data set was then exported and trasferred to R 
#
#
train.sas <- read.table("~/Grad School/Predict 498/Prud/TRAINING_SAS.csv", 
                        sep=",", header=TRUE)
my.data.2.df <- as.data.frame(train.sas)
dim(my.data.2.df)
names(my.data.2.df)
#
#
# Create new variables and transformations
my.data.2.df$bmi_age <- my.data.2.df$BMI * my.data.2.df$Ins_Age
my.data.2.df$med_hist_15_st10 <- ifelse(my.data.2.df$new_med_hist_15 < 10, 1, 0)
head(my.data.2.df$new_med_hist_15, n=30)
head(my.data.2.df$med_hist_15_st10, n=30)
my.data.2.df$prod_info_4_st0075 <- ifelse(my.data.2.df$Product_Info_4 < 0.0075, 1, 0)
my.data.2.df$prod_info_4_e1 <- ifelse(my.data.2.df$Product_Info_4 == 1, 1, 0)
my.data.2.df$bmi.p1sq2 <- (my.data.2.df$BMI + 1.0) ** 2
my.data.2.df$bmi.sq08 <- (my.data.2.df$BMI) ** 0.8
my.data.2.df$ins_age.sq85 <- my.data.2.df$Ins_Age ** 8.5
my.data.2.df$ins_age.sq25 <- my.data.2.df$Ins_Age ** 2.5
my.data.2.df$bmi_age.sq25 <- (my.data.2.df$BMI * my.data.2.df$Ins_Age) ** 2.5
my.data.2.df$bmi_prod4.sq09 <- (my.data.2.df$BMI * my.data.2.df$Product_Info_4) ** 0.9
my.data.2.df$bmi_med_key_3.p05sq3 <- (my.data.2.df$BMI * my.data.2.df$Medical_Keyword_3 + 0.5) ** 3.0
# Create Keyword_Sum
my.data.2.df.med_key <- my.data.2.df %>% select( contains("Keyword"))
my.data.2.df$med_key_sum <- apply( as.matrix(my.data.2.df.med_key), 1, sum)
head(my.data.2.df$med_key_sum, n=50)
head(my.data.2.df.med_key, n=16)
#
#
# Pull out categorical variables, make sure they are character variables
# Create dummies, rejoin with numeric variables
#
my.data.2.char.df <- my.data.2.df %>% select(Product_Info_1, Product_Info_2,  
                      Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, 
                      Employment_Info_2, Employment_Info_3, Employment_Info_5, 
                      InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, 
                      InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, 
                      Insurance_History_2, Insurance_History_3, Insurance_History_4, 
                      Insurance_History_7, Insurance_History_8, Insurance_History_9, 
                      Family_Hist_1, Medical_History_2, Medical_History_3, 
                      Medical_History_4, Medical_History_5, Medical_History_6, 
                      Medical_History_7, Medical_History_8, Medical_History_9, 
                      Medical_History_11, Medical_History_12, Medical_History_13, 
                      Medical_History_14, Medical_History_16, Medical_History_17, 
                      Medical_History_18, Medical_History_19, Medical_History_20, 
                      Medical_History_21, Medical_History_22, Medical_History_23, 
                      Medical_History_25, Medical_History_26, Medical_History_27, 
                      Medical_History_28, Medical_History_29, Medical_History_30, 
                      Medical_History_31, Medical_History_33, Medical_History_34, 
                      Medical_History_35, Medical_History_36, Medical_History_37, 
                      Medical_History_38, Medical_History_39, Medical_History_40, 
                      Medical_History_41)
my.data.2.char.df <- as.data.frame(lapply(my.data.2.char.df, as.character))
my.data.dummy.df <- dummy.data.frame(my.data.2.char.df, sep=".")
char.names <- names(my.data.2.char.df)
nonchar.names <- setdiff(names(my.data.2.df), names(my.data.2.char.df))
my.data.2.nonchar.df <- my.data.2.df[ , nonchar.names]
my.data.2.dum.df <- cbind(my.data.2.nonchar.df, my.data.dummy.df)
#
#



# Create decision trees to find simple interactive relationships
tree.data.2 <- rpart(formula = Response ~ ., data = my.data.2.dum.df, method = "class", control=rpart.control(maxdepth=4) )
summary(tree.data.2)
path.rpart(tree.data.2, nodes=8)
path.rpart(tree.data.2, nodes=29)
path.rpart(tree.data.2, nodes=15)
#
# Create binary variables to represent decision tree interactions
my.data.2.dum.df$tree.node.1.8 <- ifelse( my.data.2.dum.df$BMI < 0.494 & 
                                    my.data.2.dum.df$Medical_History_23.3 == 0, 1,0)
my.data.2.dum.df$tree.node.2.6 <- ifelse( my.data.2.dum.df$BMI < 0.609 & 
                                    my.data.2.dum.df$Medical_Keyword_3 == 0, 1,0)
my.data.2.dum.df$tree.node.3.8 <- ifelse( my.data.2.dum.df$Medical_History_23.1 == 0 & 
                                    my.data.2.dum.df$Medical_History_23.3 == 0 & 
                                    my.data.2.dum.df$Medical_History_4.1 == 0 & 
                                    my.data.2.dum.df$Medical_History_4.2 == 0 & 
                                    my.data.2.dum.df$Medical_Keyword_15 == 0, 1,0)
my.data.2.dum.df$tree.node.4.5 <- ifelse( my.data.2.dum.df$BMI < 0.494 & 
                                    my.data.2.dum.df$Medical_History_23.1 == 0 & 
                                    my.data.2.dum.df$Medical_History_4.2 == 0 & 
                                    my.data.2.dum.df$new_med_hist_15 >= 56.5, 1,0)
my.data.2.dum.df$tree.node.7.8 <- ifelse( my.data.2.dum.df$Medical_History_15_flag == 0 & 
                                    my.data.2.dum.df$Product_Info_4 < 0.074 & 
                                    my.data.2.dum.df$Medical_History_4.1 == 0 & 
                                    my.data.2.dum.df$Medical_History_4.2 == 0 & 
                                    my.data.2.dum.df$bmi_prod4.sq09 < 0.025, 1,0)
my.data.2.dum.df$tree.node.15.8 <- ifelse( my.data.2.dum.df$BMI < 0.494 & 
                                    my.data.2.dum.df$Medical_History_23.1 == 0 & 
                                    my.data.2.dum.df$Medical_History_4.2 == 1, 1,0)
my.data.2.dum.df$tree.node.29.8 <- ifelse( my.data.2.dum.df$BMI < 0.494 & 
                                    my.data.2.dum.df$Medical_History_23_1 == 0 & 
                                    my.data.2.dum.df$Medical_History_4_2 == 0 & 
                                    my.data.2.dum.df$new_med_hist_15 >= 56.5, 1,0)
#
#
# Save file as CSv and send back to Sasha for clustering
write.csv(my.data.2.dum.df, file="~/Grad School/Predict 498/Prud/my.data.2.dum.csv")
#
#
# Import training set back in data set from SAS.
# It now has variables for clustering interactions
# Create factor vector of Response, take Response out of df
full.train <- read.table("~/Grad School/Predict 498/Prud/Sasha_final_train.csv", 
                         sep=",", header=TRUE)
resp.obs <- as.factor(full.train$Response)
names(full.train)
pred.all.df <- full.train[ , -55]
# Import test set, create factor vector of Response, take Response out of df
full.test <- read.table("~/Grad School/Predict 498/Prud/sasha_final_test.csv", sep=",", header=TRUE)
resp.test.obs <- full.test$Response
names(full.test)
pred.all.test.df <- full.test[ , -55]
#
#
# Use Random Forest on Training set
# Get initial Quadratic Weighted Kappa on Test set
rf.data.3.initial <- randomForest(x=pred.all.df, y=resp.obs, 
                                  ntree=300, mtry = 50, importance = TRUE)
predictions.full.test <- predict(rf.data.3.initial, pred.all.test.df, type = "response")
qwk.full.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.full.test, 1, 8)
#
#
# Use Random Forest results to get rankings of variable importance
importance.pred.all <- as.data.frame( rf.data.3.initial$importance)
head(importance.pred.all)
# Sort on Mean Decrease in Accuracy
importance.pred.all.sorted.on9 <- sort( abs(rf.data.3.initial$importance[ ,9]), decreasing = TRUE)
head(importance.pred.all.sorted.on9)
# Pick out top 850 predictors from train and test sets
names.pred.top850 <- names(importance.pred.all.sorted.on9[1:850])
pred.top850.df <- my.data.2.dum.df[ , c(names.pred.top850)]
pred.top850.test.df <- pred.all.test.df[ , c(names.pred.top850)]
# Run RF on top 850 predictors / make predictions on test set / QWK
rf.data.3.top850 <- randomForest(x=pred.top850.df, y=resp.obs, ntree=300, mtry = 50, importance = TRUE)
predictions.top850.test <- predict(rf.data.3.top850, pred.top850.test.df, type = "response")
qwk.top850.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.top850.test, 1, 8)
#
#
# Sort predictor varialbes by importance again
# Pull out top 750
importance.pred.top850.sorted.on9 <- sort( abs(rf.data.3.top850$importance[ ,9]), decreasing = TRUE)
head(importance.pred.top850.sorted.on9)
names.pred.top750 <- names(importance.pred.top850.sorted.on9[1:750])
pred.top750.df <- my.data.2.dum.df[ , c(names.pred.top750)]
pred.top750.test.df <- pred.all.test.df[ , c(names.pred.top750)]
# Run RF on top 750 / make predictions on test set / calculate QWK
rf.data.3.top750 <- randomForest(x=pred.top750.df, y=resp.obs, ntree=300, mtry = 50, importance = TRUE)
predictions.top750.test <- predict(rf.data.3.top750, pred.top750.test.df, type = "response")
qwk.top750.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.top750.test, 1, 8)
#
#
# Sort / Pull out top 650
importance.pred.top750.sorted.on9 <- sort( abs(rf.data.3.top750$importance[ ,9]), decreasing = TRUE)
head(importance.pred.top750.sorted.on9)
names.pred.top650 <- names(importance.pred.top750.sorted.on9[1:650])
pred.top650.df <- my.data.2.dum.df[ , c(names.pred.top650)]
pred.top650.test.df <- pred.all.test.df[ , c(names.pred.top650)]
# Run RF on top 650 / make predictions on test set / calculate QWK
rf.data.3.top650 <- randomForest(x=pred.top650.df, y=resp.obs, ntree=300, mtry = 50, importance = TRUE)
predictions.top650.test <- predict(rf.data.3.top650, pred.top650.test.df, type = "response")
qwk.top650.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.top650.test, 1, 8)
#
#
# Since QWK is not dropping much with these steps of 100
# make bigger step down in number of variables
# Go to top 150
importance.pred.top650.sorted.on9 <- sort( abs(rf.data.3.top650$importance[ ,9]), decreasing = TRUE)
names.pred.top150 <- names(importance.pred.top650.sorted.on9[1:150])
pred.top150.df <- pred.top650.df[ , c(names.pred.top150)]
rf.data.3.top150 <- randomForest(x=pred.top150.df, y=resp.obs, ntree=300, mtry = 50, importance = TRUE)
predictions.top150.test <- predict(rf.data.3.top150, pred.top150.test.df, type = "response")
qwk.top150.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.top150.test, 1, 8)
#
#
# Top 100
names.pred.top100 <- names(importance.pred.top650.sorted.on9[1:100])
pred.top100.df <- my.data.2.dum.df[ , c(names.pred.top100)]
pred.top100.test.df <- pred.all.test.df[ , c(names.pred.top100)]
rf.data.3.top100 <- randomForest(x=pred.top100.df, y=resp.obs, ntree=300, mtry = 50, importance = TRUE)
predictions.top100.test <- predict(rf.data.3.top100, pred.top100.test.df, type = "response")
qwk.top100.test <- ScoreQuadraticWeightedKappa(resp.test.obs, predictions.top100.test, 1, 8)
#
#
#
#
##################### Try linear models ####################################
# Run on full set of predictors to get baseline
# Have to join resp.obs bank into df
full.train.withresp <- cbind(my.data.2.dum.df, resp.obs)
# Run linear model / get predictions
linearmodel.full <- lm(resp.obs ~ ., data = full.train.withresp)
pred.linear.full <- predict(linearmodel.full, newdata = full.test)
# Have to create cutoffs to turn continuous predictions to discrete
pred.linear.full.adj <- pred.linear.full
pred.linear.full.adj[pred.linear.full < 1] <- 1
pred.linear.full.adj[pred.linear.full > (7 + hardcoded_cutoffs[7])] <- 8
for(i in 1:6)
{
  lowcut <- hardcoded_cutoffs[i]
  hicut <- hardcoded_cutoffs[(i + 1)]
  condi <- (pred.linear.full.adj > (i + lowcut)) & (pred.linear.full.adj < (i + 1 + hicut))
  pred.linear.full.adj[condi] <- (i + 1)
}
pred.linear.full.adj.rnd <- round(pred.linear.full.adj)
qwk.linear.full <- ScoreQuadraticWeightedKappa(resp.test.obs, pred.linear.full.adj.rnd, 1, 8)
#
#
# Try top 200 predictors
pred.resp.top200.df <- cbind(pred.top200.df, resp.obs)
linearmodel.top200 <- lm(resp.obs ~ ., data = pred.resp.top200.df)
pred.linear.top200 <- predict(linearmodel.top200, newdata = pred.top200.test.df)
pred.linear.top200.adj <- pred.linear.top200
pred.linear.top200.adj[pred.linear.top200 < 1] <- 1
pred.linear.top200.adj[pred.linear.top200 > (7 + hardcoded_cutoffs[7])] <- 8
for(i in 1:6)
{
  lowcut <- hardcoded_cutoffs[i]
  hicut <- hardcoded_cutoffs[(i + 1)]
  condi <- (pred.linear.top200.adj > (i + lowcut)) & (pred.linear.top200.adj < (i + 1 + hicut))
  pred.linear.top200.adj[condi] <- (i + 1)
}
pred.linear.top200.adj.rnd <- round(pred.linear.top200.adj)
qwk.linear.top200 <- ScoreQuadraticWeightedKappa(resp.test.obs, pred.linear.top200.adj.rnd, 1, 8)
#
#
#
# Try with top 150 predictors
pred.resp.top150.df <- cbind(pred.top150.df, resp.obs)
linearmodel.top150 <- lm(resp.obs ~ ., data = pred.resp.top150.df)
pred.linear.top150 <- predict(linearmodel.top150, newdata = pred.top150.test.df)
pred.linear.top150.adj <- pred.linear.top150
pred.linear.top150.adj[pred.linear.top150 < 1] <- 1
pred.linear.top150.adj[pred.linear.top150 > (7 + hardcoded_cutoffs[7])] <- 8
for(i in 1:6)
{
  lowcut <- hardcoded_cutoffs[i]
  hicut <- hardcoded_cutoffs[(i + 1)]
  condi <- (pred.linear.top150.adj > (i + lowcut)) & (pred.linear.top150.adj < (i + 1 + hicut))
  pred.linear.top150.adj[condi] <- (i + 1)
}
pred.linear.top150.adj.rnd <- round(pred.linear.top150.adj)
qwk.linear.top150 <- ScoreQuadraticWeightedKappa(resp.test.obs, pred.linear.top150.adj.rnd, 1, 8)
#
#
#
# Try with top 100 predictors
pred.resp.top100.df <- cbind(pred.top100.df, resp.obs)
linearmodel <- lm(resp.obs ~ ., data = pred.resp.top100.df)
pred.linear <- predict(linearmodel, newdata = pred.top100.test.df)
hardcoded_cutoffs <- c(0.8717, 0.9034, 0.8119, 0.7567, 0.6588, 0.2360, 0.0490)
pred.linear.adj <- pred.linear
pred.linear.adj[pred.linear < 1] <- 1
pred.linear.adj[pred.linear > (7 + hardcoded_cutoffs[7])] <- 8
for(i in 1:6)
{
  lowcut <- hardcoded_cutoffs[i]
  hicut <- hardcoded_cutoffs[(i + 1)]
  condi <- (pred.linear.adj > (i + lowcut)) & (pred.linear.adj < (i + 1 + hicut))
  pred.linear.adj[condi] <- (i + 1)
}
pred.linear.adj.rnd <- round(pred.linear.adj)
qwk.linear <- ScoreQuadraticWeightedKappa(resp.test.obs, pred.linear.adj.rnd, 1, 8)
#
#
#
# Get Confusuion Matrix for Linera Model with 150 predictors
conf.mat.linear <- confusionMatrix(resp.test.linear.top150, resp.test.obs)