#install R and RStudio from here: https://cran.rstudio.com/
# Do these installations beforehand
#install.packages("RSQLite")
#install.packages("ISLR")
#find.package('ISLR') # if you are curious where the ISLR library is stored.
rm(list = ls())
library(ISLR)
library(MASS)
getwd();
setwd("/Users/amirmanzour/Documents/SUNY/Courses/CS\ 548/programs")
getwd(); # verify working directory
#download Advertising data from here: https://www.statlearning.com/resources-python

#Boston["rm"] number of rooms
#Boston["age"] age of house
#Boston["lstat"] low socioecomomic status

# Create a model:
lm.fit =lm(medv ~ lstat, data=Boston)

lm.fit

summary(lm.fit)

names(lm.fit)

# Confidence interval of parameters 
confint(lm.fit)

# Confidence interval of parameters
predict (lm.fit ,data.frame(lstat =(c(5 ,10 ,15) )),
         interval =" confidence ")
predict (lm.fit ,data.frame(lstat =(c(5 ,10 ,15) )),
         interval =" prediction ")

# plotting the 'istat' prdictor and 'medv' response
plot(Boston$lstat ,Boston$medv)
abline (lm.fit)

# plotting the residuals:
predict (lm.fit) # predictions for our data
residuals (lm.fit) # residuals of our data
plot(predict (lm.fit), residuals (lm.fit))     
plot(predict (lm.fit), rstudent (lm.fit)) # normalized residuals

#Multiple linear regression (2 predictors)
lm.fit =lm(medv ~ lstat+age ,data=Boston )

#Multiple linear regression (all predictors)
lm.fit =lm(medv ~ .,data=Boston )

# regression incorporating non-additive terms. eg:
lm.fit=lm(medv ~ lstat*age ,data=Boston )
summary(lm.fit)

# Non-linear Transformations of the Predictors
lm.fit=lm(medv ~ lstat +I(lstat^2))

# Non-linear (polynomial) Transformations of the Predictors
lm.fit=lm(medv ~ poly(lstat ,5))

# Non-linear (log) Transformations of the Predictors
lm.fit = lm(medv ~ log(rm),data=Boston)
