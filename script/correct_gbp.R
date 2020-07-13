library(tidyverse)
library(data.table)
library(stringr)

google_train <- fread("neural_entity_matching/data/amazon_google/Google_train.csv",header = TRUE)
google_test <- fread("neural_entity_matching/data/amazon_google/Google_test.csv",header = TRUE)

google_train[str_detect(google_train$price,"gbp"),"price"] <- as.numeric(str_replace_all(google_train$price[str_detect(google_train$price,"gbp")],"gbp",""))* 1.5

google_test[str_detect(google_test$price,"gbp"),"price"] <- as.numeric(str_replace_all(google_test$price[str_detect(google_test$price,"gbp")],"gbp",""))* 1.5


google_train %>% 
  fwrite("neural_entity_matching/data/amazon_google/Google_train_clean.csv")


google_test %>% 
  fwrite("neural_entity_matching/data/amazon_google/Google_test_clean.csv")
