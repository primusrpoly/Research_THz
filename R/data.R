# setwd("C:\\Users\\ryanj\\Code\\Research_THz\\full_Results.csv")
# getwd(); # verify working directory

MyData = read.csv("full_Results.csv")

colnames(MyData)

summary(MyData)

plot(MyData$PhaseNoise, MyData$CBER)

lm1.fit =lm(MyData$CBER ~ MyData$PhaseNoise + MyData$SymbolRate); summary(lm1.fit)
lm2.fit =lm(MyData$CBER ~ MyData$PhaseNoise + MyData$SNR); summary(lm2.fit)
lm3.fit =lm(MyData$CBER ~ MyData$SNR + MyData$SymbolRate); summary(lm3.fit)
lm4.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData); summary(lm4.fit)

dim(MyData)
colnames(MyData)

#low CBER
MyData_low_CBER = MyData[MyData$CBER < 0.005, ]

dim(MyData_low_CBER)
head(MyData_low_CBER)
summary(MyData_low_CBER)

lm9.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData_low_CBER); summary(lm9.fit)
plot(x= MyData_low_CBER$CBER, y = lm9.fit$residuals)

lm10.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SNR, data = MyData_low_CBER); summary(lm10.fit)
plot(x= MyData_low_CBER$CBER, y = lm10.fit$residuals)

lm11.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SNR, data = MyData_low_CBER); summary(lm11.fit)
plot(x= MyData_low_CBER$CBER, y = lm11.fit$residuals)

lm12.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate, data = MyData_low_CBER); summary(lm12.fit)
plot(x= MyData_low_CBER$CBER, y = lm12.fit$residuals)

lm13.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SNR*SNR, data = MyData_low_CBER); summary(lm13.fit)
plot(x= MyData_low_CBER$CBER, y = lm13.fit$residuals)

lm14.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*PhaseNoise, data = MyData_low_CBER); summary(lm14.fit)
plot(x= MyData_low_CBER$CBER, y = lm14.fit$residuals)

lm15.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SymbolRate, data = MyData_low_CBER); summary(lm15.fit)
plot(x= MyData_low_CBER$CBER, y = lm15.fit$residuals)

lm16.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate*SNR, data = MyData_low_CBER); summary(lm16.fit)
plot(x= MyData_low_CBER$CBER, y = lm16.fit$residuals)


#high CBER
MyData_high_CBER = MyData[MyData$CBER > 0.02, ]
dim(MyData_high_CBER)


lm9.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData_high_CBER); summary(lm9.fit)
plot(x= MyData_high_CBER$CBER, y = lm9.fit$residuals)

lm10.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SNR, data = MyData_high_CBER); summary(lm10.fit)
plot(x= MyData_high_CBER$CBER, y = lm10.fit$residuals)

lm11.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SNR, data = MyData_high_CBER); summary(lm11.fit)
plot(x= MyData_high_CBER$CBER, y = lm11.fit$residuals)

lm12.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate, data = MyData_high_CBER); summary(lm12.fit)
plot(x= MyData_high_CBER$CBER, y = lm12.fit$residuals)

lm13.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SNR*SNR, data = MyData_high_CBER); summary(lm13.fit)
plot(x= MyData_high_CBER$CBER, y = lm13.fit$residuals)

lm14.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*PhaseNoise, data = MyData_high_CBER); summary(lm14.fit)
plot(x= MyData_high_CBER$CBER, y = lm14.fit$residuals)

lm15.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SymbolRate, data = MyData_high_CBER); summary(lm15.fit)
plot(x= MyData_high_CBER$CBER, y = lm15.fit$residuals)

lm16.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate*SNR, data = MyData_high_CBER); summary(lm16.fit)
plot(x= MyData_high_CBER$CBER, y = lm16.fit$residuals)


install.packages("caret")

# Install the caret package if not already installed
# install.packages("caret")

library(caret)


preprocess_params <- preProcess(MyData_filtered, method = c("range"))

MyData_normalized <- predict(preprocess_params, MyData_filtered)
head(MyData_normalized)

summary(MyData_normalized)

#Normalized low CBER
MyData_normlow_CBER = MyData_normalized[MyData_normalized$CBER < 0.15, ]

dim(MyData_normlow_CBER)
head(MyData_normlow_CBER)
summary(MyData_normlow_CBER)

lm9.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData_normlow_CBER); summary(lm9.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm9.fit$residuals)

lm10.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SNR, data = MyData_normlow_CBER); summary(lm10.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm10.fit$residuals)

lm11.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SNR, data = MyData_normlow_CBER); summary(lm11.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm11.fit$residuals)

lm12.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate, data = MyData_normlow_CBER); summary(lm12.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm12.fit$residuals)

lm13.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SNR*SNR, data = MyData_normlow_CBER); summary(lm13.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm13.fit$residuals)

lm14.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*PhaseNoise, data = MyData_normlow_CBER); summary(lm14.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm14.fit$residuals)

lm15.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SymbolRate, data = MyData_normlow_CBER); summary(lm15.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm15.fit$residuals)

lm16.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate*SNR, data = MyData_normlow_CBER); summary(lm16.fit)
plot(x= MyData_normlow_CBER$CBER, y = lm16.fit$residuals)

#Normalized high CBER
MyData_normh_CBER = MyData_normalized[MyData_normalized$CBER > 0.65, ]

dim(MyData_normh_CBER)
head(MyData_normh_CBER)
summary(MyData_normh_CBER)

lm9.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData_normh_CBER); summary(lm9.fit)
plot(x= MyData_normh_CBER$CBER, y = lm9.fit$residuals)

lm10.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SNR, data = MyData_normh_CBER); summary(lm10.fit)
plot(x= MyData_normh_CBER$CBER, y = lm10.fit$residuals)

lm11.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SNR, data = MyData_normh_CBER); summary(lm11.fit)
plot(x= MyData_normh_CBER$CBER, y = lm11.fit$residuals)

lm12.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate, data = MyData_normh_CBER); summary(lm12.fit)
plot(x= MyData_normh_CBER$CBER, y = lm12.fit$residuals)

lm13.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SNR*SNR, data = MyData_normh_CBER); summary(lm13.fit)
plot(x= MyData_normh_CBER$CBER, y = lm13.fit$residuals)

lm14.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*PhaseNoise, data = MyData_normh_CBER); summary(lm14.fit)
plot(x= MyData_normh_CBER$CBER, y = lm14.fit$residuals)

lm15.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SymbolRate, data = MyData_normh_CBER); summary(lm15.fit)
plot(x= MyData_normh_CBER$CBER, y = lm15.fit$residuals)

lm16.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate*SNR, data = MyData_normh_CBER); summary(lm16.fit)
plot(x= MyData_normh_CBER$CBER, y = lm16.fit$residuals)





all_indices_zero= which(MyData['CBER'] <= 0.005)
total_rows = nrow(MyData)

num_zeros = length(all_indices_zero)
percent_zeros = (num_zeros / total_rows) * 100

num_non_zeros = total_rows - num_zeros
percent_non_zeros = (num_non_zeros / total_rows) * 100

cat("Percentage of data with CBER equal to 0:", percent_zeros, "%\n")
cat("Percentage of data with CBER greater than 0:", percent_non_zeros, "%\n")

hist(unlist(MyData['CBER']), breaks = 1000)
hist(unlist(MyData['BER']), breaks = 50)

# plot every 2 variables vs CBER
# plot all three against each other but color is CBER

plot(MyData$SNR, MyData$PhaseNoise)
plot(MyData$SNR, MyData$SymbolRate)
plot(MyData$SymbolRate, MyData$PhaseNoise)
plot(MyData$SNR, MyData$SymbolRate, MyData$PhaseNoise)

MyData$SymbolRate
# plotting residuals
plot(x= MyData$CBER, y = lm4.fit$residuals)
abline(lm4.fit)

#  before all of this  balance your data
# 1. downsample zeros
# 2. change weights of data*

# Try non-linear situations Path 1
lm5.fit =lm(CBER ~ PhaseNoise + SymbolRate^3 + SNR, data = MyData); summary(lm5.fit)
lm6.fit =lm(CBER ~ log(PhaseNoise) + log(SymbolRate) + exp(SNR+0.0001), data = MyData); summary(lm5.fit)

lm5.fit =glm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData); summary(lm5.fit)

plot(x= MyData$CBER, y = lm5.fit$residuals)

# Try classification situations Path 2

# next step, randomly eliminate 80% of your zeros 
all_indices_zero= which(MyData['CBER'] <= 0.0)
length(all_indices_zero)
# randomselection of allindicieas
num_zeros_to_remove = floor(0.80 * length(all_indices_zero))

set.seed(123)  
indices_to_remove = sample(all_indices_zero, num_zeros_to_remove)

MyData_filtered = MyData[-indices_to_remove, ]

dim(MyData_filtered)
new_indices_zero= which(MyData_filtered['CBER'] <= 0.0)
length(new_indices_zero)

total_rows = nrow(MyData_filtered)

num_zeros = length(new_indices_zero)
percent_zeros = (num_zeros / total_rows) * 100

num_non_zeros = total_rows - num_zeros
percent_non_zeros = (num_non_zeros / total_rows) * 100

cat("Percentage of data with CBER equal to 0:", percent_zeros, "%\n")
cat("Percentage of data with CBER greater than 0:", percent_non_zeros, "%\n")


summary(MyData_filtered)
hist(unlist(MyData_filtered['CBER']), breaks = 100)
lm_filter.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR, data = MyData_filtered); summary(lm_filter.fit)
plot(x= MyData_filtered$CBER, y = lm_filter.fit$residuals)
abline(lm_filter.fit)

hist(unlist(MyData_filtered['CBER']), breaks = 10000)




lm10.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SNR, data = MyData); summary(lm10.fit)
plot(x= MyData$CBER, y = lm10.fit$residuals)

lm11.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SNR, data = MyData); summary(lm11.fit)
plot(x= MyData$CBER, y = lm11.fit$residuals)

lm12.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate, data = MyData); summary(lm12.fit)
plot(x= MyData$CBER, y = lm12.fit$residuals)

lm13.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + SNR*SNR, data = MyData); summary(lm13.fit)
plot(x= MyData$CBER, y = lm13.fit$residuals)

lm14.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*PhaseNoise, data = MyData); summary(lm14.fit)
plot(x= MyData$CBER, y = lm14.fit$residuals)

lm15.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + SymbolRate*SymbolRate, data = MyData); summary(lm15.fit)
plot(x= MyData$CBER, y = lm15.fit$residuals)

lm16.fit =lm(CBER ~ PhaseNoise + SymbolRate + SNR + PhaseNoise*SymbolRate*SNR, data = MyData); summary(lm16.fit)
plot(x= MyData$CBER, y = lm16.fit$residuals)
