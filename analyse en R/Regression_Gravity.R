library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(glmnet)
library(car)
library(dplyr)
library(estimatr)

data <- read.csv("ProjetStat/data/FINAL_GRAVITY_TRAINING_MATRIX.csv")
head(data)
data.frame(
  Variable = names(data),
  NA_Count = colSums(is.na(data)),
  NA_Percent = colSums(is.na(data)) / nrow(data) * 100
)
