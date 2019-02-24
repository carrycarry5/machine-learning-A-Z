# importing the dataset
dataset = read.csv('../Data.csv')

# taking care of missing data
dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T)
dataset$Salary[is.na(dataset$Salary)] = mean(dataset$Salary, na.rm = T)

# encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France',"Spain",'Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No',"Yes"),
                         labels = c(0,1))


# splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(42)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)

# Feature Scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
