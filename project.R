library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(glmnet)

dat <- read_csv('https://raw.githubusercontent.com/selva86/datasets/master/economics.csv', show_col_types = FALSE)
glimpse(dat)
head(dat)
#checking for nulls
lapply(dat,function(x) { length(which(is.na(x)))})
str(dat)

# employment related data plots
ggplot(data = dat, aes(x = date, y = unemploy)) +
  geom_point(aes(color =uempmed)) +
  labs(x = "Date",
       y = "Unemployment",
       title = "Number of Unemployed Over Time") +
  theme_bw()
ggplot(data = dat) +
  geom_bar(aes(x = date, y = psavert, fill = pop), stat = 'identity') +
  labs(x = "Date",
       y = "Personal Savings Rate",
       fill = 'population',
       title = "How much an individual is Saving Per Year") +
  theme_bw()
#Plot on personal consumption expenditure
ggplot(data = dat, aes(x = date, y = pce)) + 
  geom_point() +
  labs(
    x = 'Date',
    y = 'personal consumption expenditure (pce)',
    title = 'personal consumption expenditure against date'
  )

# Remove first and last columns
dat <- dat[,c(-1)]
set.seed(1234) 

# Creating test and train data sets
index = sample(1:nrow(dat), 0.7*nrow(dat)) 
train = dat[index,] 
test = dat[-index,] 
  
# Dimensions of train and test data sets
dim(train)
dim(test)

# Pre processing required columns
cols = c('pce', 'pop', 'psavert', 'uempmed','unemploy')
pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))

train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

summary(train)
head(train)
#Linear regression

train_cont <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats =5,
                           search = "random",verboseIter = TRUE)
set.seed(1234)
lm<-train(unemploy~.,train,method='lm',trControl=train_cont)
lm$results
summary(lm)
predictions =predict(lm,newdata=test)
Linear_results = c(RMSE = RMSE(predictions,test$unemploy),R2 =R2(predictions,test$unemploy))
plot(lm$finalModel)
Linear_results
cols_reg = c('pce', 'pop', 'psavert', 'uempmed', 'unemploy')

dummies <- dummyVars(unemploy ~ ., data = dat[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))

# RIDGE

x_train = as.matrix(train_dummies)
y_train = train$unemploy

x_test = as.matrix(test_dummies)
y_test = test$unemploy

set.seed(1234)
ridge_reg <- train(y =y_train,x = x_train,method = "glmnet",
                   tuneGrid = expand.grid(alpha=0,lambda=10^seq(5, -3,length =100)),
                   trControl = train_cont)

ridge_reg
plot(ridge_reg$finalModel,xvar="lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(ridge_reg$finalModel,xvar='dev')
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(varImp(ridge_reg,scale=F))
predictions_ridge <- ridge_reg %>% predict(x_test)
#Test predicted results by unscaling.
head(predictions_ridge)
Unscale<-predictions_ridge*sd(dat$unemploy)+mean(dat$unemploy)
head(Unscale)
Ridge_results =c(RMSE = RMSE(predictions,y_test),R2 =R2(predictions,y_test))
plot(ridge_reg$finalModel)
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
summary(ridge_reg)
ridge_reg$bestTune
Ridge_results
ridge_reg


# LASSO

#Setting alpha = 1 implements lasso regression
set.seed(1234)
lasso_reg <- train(y =y_train,
                   x = x_train,
                   method = "glmnet",
                   tuneGrid =expand.grid(alpha=1,lambda=10^seq(5, -4, length =100)),
                   trControl = train_cont)
#plot(lasso_reg)
lasso_reg
plot(lasso_reg$finalModel,xvar="lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(lasso_reg$finalModel,xvar='dev')
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(varImp(lasso_reg,scale=F))
predictions_lasso <- lasso_reg %>% predict(x_test)
Lasso_results =c(RMSE = RMSE(predictions_lasso,y_test),R2 =R2(predictions_lasso,y_test))
plot(lasso_reg$finalModel)
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
summary(lasso_reg)
lasso_reg$bestTune
Lasso_results



#Elasticnet regression
set.seed(1234)

# Train the model
elastic_reg <- train(y =y_train,x = x_train,method = "glmnet",
                   tuneGrid =expand.grid(alpha=seq(0,1,length=10),
                  lambda=10^seq(6,-4,length =100)),trControl = train_cont)


plot(elastic_reg$finalModel,xvar="lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(elastic_reg$finalModel,xvar='dev')
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x_train), cex = .7)
plot(varImp(elastic_reg,scale=F))
predictions_elastic <- elastic_reg %>% predict(x_test)
elastic_results =c(RMSE = RMSE(predictions_elastic,y_test),R2 =R2(predictions_elastic,y_test))
summary(elastic_reg)
elastic_reg$bestTune
---------------------------------

#compare models
data.frame(ridge = as.data.frame.matrix(coef(ridge_reg$finalModel, ridge_reg$finalModel$lambdaOpt)),
             lasso = as.data.frame.matrix(coef(lasso_reg$finalModel, lasso_reg$finalModel$lambdaOpt)), 
             linear = (lm$finalModel$coefficients)
)%>%   rename(lasso = s1, ridge = s1.1)
data.frame(
  Linear_results,Ridge_results,Lasso_results,elastic_results
)


