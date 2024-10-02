#install R and RStudio from here: https://cran.rstudio.com/
#install.packages("ISLR")
#find.package('ISLR') # if you are curious where the ISLR library is stored.
#library(ISLR)
getwd();
setwd("/Users/amirmanzour/Documents/SUNY/Courses/CS\ 548/programs")
getwd(); # verify working directory
#download Advertising data from here: https://www.statlearning.com/resources-python
rm(list = ls())
#PROGRAM
Advertising=read.csv("Advertising.csv", header =T,row.names='X',na.strings ="?")

colnames(Advertising)

hist(Advertising$TV, breaks = 50, c = 'blue')
hist(Advertising$radio, breaks = 50, c = 'blue')
hist(Advertising$newspaper, breaks = 50, c = 'blue')
hist(Advertising$sales, breaks = 50, c = 'blue')

summary(Advertising)

plot(Advertising$TV,Advertising$sales)

lm.fit =lm(Advertising$sales ~ Advertising$TV)
abline(lm.fit)
summary(lm.fit, lwd=3)

pred = predict(lm.fit, Advertising, interval = "confidence")

pred2 = predict(lm.fit, Advertising, interval = "prediction")

pred = as.data.frame(pred)
MSE = sum((pred$fit - Advertising$sales)^2)/length(Advertising$sales)

lmall.fit =lm(Advertising$sales ~ Advertising$TV + Advertising$radio + Advertising$newspaper)

summary(lmall.fit)

lm1.fit =lm(Advertising$sales ~ Advertising$TV); summary(lm1.fit)
lm2.fit =lm(Advertising$sales ~ Advertising$radio); summary(lm2.fit)
lm3.fit =lm(Advertising$sales ~ Advertising$newspaper); summary(lm3.fit)
lm4.fit =lm(Advertising$sales ~ Advertising$TV + Advertising$radio); summary(lm4.fit)
lm5.fit =lm(Advertising$sales ~ Advertising$TV + Advertising$newspaper); summary(lm5.fit)
lm6.fit =lm(Advertising$sales ~ Advertising$radio + Advertising$newspaper); summary(lm6.fit)
lm7.fit =lm(Advertising$sales ~ Advertising$TV + Advertising$radio + Advertising$newspaper); summary(lm7.fit)


