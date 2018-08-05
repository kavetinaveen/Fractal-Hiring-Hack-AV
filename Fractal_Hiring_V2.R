# ======================== Title: Fractal Hiring Hackathon ======================== #

# Setting working directory
# ===========================
filepath <- c("/Users/nkaveti/Documents/Kaggle/Fractal_Hiring_Hack")
setwd(filepath)

# Loading required packages
# ===========================
library(data.table)
library(dplyr)
library(Matrix)
library(xgboost)
library(Metrics)

# Reading data
# =============
train_main <- fread("train.csv")
test_main <- fread("test.csv")
test_main <- test_main[, c(ncol(test_main), 1:(ncol(test_main)-1)), with = F]
# Arranging columns in an order
cols <- c("ID", "Item_ID", "Datetime", "Category_1", "Category_2", "Category_3", "Price", "Number_Of_Sales")
train_main <- train_main[, cols, with = F]
sample_submission <- fread("Sample_Submission_Zxs5Ys1.csv")

# Filling missing observarions
# ==============================
train_main$Category_2[is.na(train_main$Category_2)] <- as.numeric(names(sort(table(train_main$Category_2), decreasing = TRUE))[1])

test_main$Category_2[is.na(test_main$Category_2)] <- as.numeric(names(sort(table(test_main$Category_2), decreasing = TRUE))[1])

# Variable conversion
# ======================
train_main$Datetime <- as.Date(train_main$Datetime)
test_main$Datetime <- as.Date(test_main$Datetime)

train_main$Category_1 <- as.factor(train_main$Category_1)
train_main$Category_2 <- as.factor(train_main$Category_2)
train_main$Category_3 <- as.factor(train_main$Category_3)
train_main$Item_ID <- as.factor(train_main$Item_ID)

test_main$Category_1 <- as.factor(test_main$Category_1)
test_main$Category_2 <- as.factor(test_main$Category_2)
test_main$Category_3 <- as.factor(test_main$Category_3)
test_main$Item_ID <- as.factor(test_main$Item_ID)

# Feature Generation
# ====================
train_main$Month <- month(train_main$Datetime)
train_main$DOW <- weekdays(train_main$Datetime)
train_main$WeekNum <- week(train_main$Datetime)

test_main$Month <- month(test_main$Datetime)
test_main$DOW <- weekdays(test_main$Datetime)
test_main$WeekNum <- week(test_main$Datetime)

# train_main <- train_main[order(train_main$Item_ID), ]
# temp <- as.data.frame(table(train_main$Item_ID))
# temp <- sapply(temp$Freq, function(x){1:x}, simplify = TRUE)
# train_main$trend <- do.call(c, temp)

# Average Features
# =============================
month_dow_mean_price <- train_main[, mean(Price), by = .(Month, WeekNum, DOW)]
month_dow_mean_nos <- train_main[, mean(Number_Of_Sales), by = .(Month, WeekNum, DOW)]
colnames(month_dow_mean_price)[ncol(month_dow_mean_price)] <- "Price_mon_dow"
colnames(month_dow_mean_nos)[ncol(month_dow_mean_price)] <- "Number_Of_Sales_mon_dow"

setkeyv(train_main, c("Month", "DOW"))
setkeyv(test_main, c("WeekNum", "Month", "DOW"))
setkeyv(month_dow_mean_price, c("WeekNum", "Month", "DOW"))
setkeyv(month_dow_mean_nos, c("WeekNum", "Month", "DOW"))

train_main <- merge(train_main, month_dow_mean_price, all.x = TRUE)
train_main <- merge(train_main, month_dow_mean_nos, all.x = TRUE)
test_main <- merge(test_main, month_dow_mean_price, all.x = TRUE)
test_main <- merge(test_main, month_dow_mean_nos, all.x = TRUE)

cat1_mean_price <- train_main[, mean(Price), by = .(Category_1)]
colnames(cat1_mean_price)[2] <- "Price_Cat1"
cat2_mean_price <- train_main[, mean(Price), by = .(Category_2)]
colnames(cat2_mean_price)[2] <- "Price_Cat2"
cat3_mean_price <- train_main[, mean(Price), by = .(Category_3)]
colnames(cat3_mean_price)[2] <- "Price_Cat3"

cat1_mean_nos <- train_main[, mean(Number_Of_Sales), by = .(Category_1)]
colnames(cat1_mean_nos)[2] <- "Number_Of_Sales_Cat1"
cat2_mean_nos <- train_main[, mean(Number_Of_Sales), by = .(Category_2)]
colnames(cat2_mean_nos)[2] <- "Number_Of_Sales_Cat2"
cat3_mean_nos <- train_main[, mean(Number_Of_Sales), by = .(Category_3)]
colnames(cat3_mean_nos)[2] <- "Number_Of_Sales_Cat3"

setkey(train_main, Category_1)
setkey(test_main, Category_1)
setkey(cat1_mean_price, Category_1)
setkey(cat1_mean_nos, Category_1)
train_main <- merge(train_main, cat1_mean_price, all.x = TRUE)
train_main <- merge(train_main, cat1_mean_nos, all.x = TRUE)
test_main <- merge(test_main, cat1_mean_price, all.x = TRUE)
test_main <- merge(test_main, cat1_mean_nos, all.x = TRUE)

setkey(train_main, Category_2)
setkey(test_main, Category_2)
setkey(cat2_mean_price, Category_2)
setkey(cat2_mean_nos, Category_2)
train_main <- merge(train_main, cat2_mean_price, all.x = TRUE)
train_main <- merge(train_main, cat2_mean_nos, all.x = TRUE)
test_main <- merge(test_main, cat2_mean_price, all.x = TRUE)
test_main <- merge(test_main, cat2_mean_nos, all.x = TRUE)

setkey(train_main, Category_3)
setkey(test_main, Category_3)
setkey(cat3_mean_price, Category_3)
setkey(cat3_mean_nos, Category_3)
train_main <- merge(train_main, cat3_mean_price, all.x = TRUE)
train_main <- merge(train_main, cat3_mean_nos, all.x = TRUE)
test_main <- merge(test_main, cat3_mean_price, all.x = TRUE)
test_main <- merge(test_main, cat3_mean_nos, all.x = TRUE)

# # Combining these mean values to test data
# # ==========================================
# setkeyv(test_main, c("Month", "DOW"))
# setkeyv(month_dow_mean_price, c("Month", "DOW"))
# setkeyv(month_dow_mean_nos, c("Month", "DOW"))
# 
# test_main <- merge(test_main, month_dow_mean_price, all.x = TRUE)
# test_main <- merge(test_main, month_dow_mean_nos, all.x = TRUE)

# Splitting data into train and test
# ======================================
train_main$DOW <- as.factor(train_main$DOW)
train_main$Month <- as.factor(train_main$Month)
train_main$WeekNum <- as.factor(train_main$WeekNum)
test_main$DOW <- as.factor(test_main$DOW)
test_main$Month <- as.factor(test_main$Month)
test_main$WeekNum <- as.factor(test_main$WeekNum)

train <- train_main[year(train_main$Datetime) < 2016, ]
test <- train_main[year(train_main$Datetime) >= 2016, ]

features <- c("Month", "Item_ID", "DOW", "Price_mon_dow", "Number_Of_Sales_mon_dow")

# Category_3, "Price_Cat1", "Number_Of_Sales_Cat1", "Price_Cat2", "Number_Of_Sales_Cat2", "Price_Cat3" ,"Number_Of_Sales_Cat3"

train_feat <- train[, features, with = F]
test_feat <- test[, features, with = F]

test_main_feat <- test_main[, features, with = F]

# Using xgboost
# ================
# Converting train/test data into sparse matrices
train_sparse <- sparse.model.matrix(~., data = train_feat)
test_sparse <- sparse.model.matrix(~., data = test_feat)

test_main_sparse <- sparse.model.matrix(~., data = test_main_feat)

# Converting sparse matrices into xgb matrices
train_xgb_price <- xgb.DMatrix(train_sparse, label = train$Price)
test_xgb_price <- xgb.DMatrix(test_sparse, label = test$Price)

train_xgb_nos <- xgb.DMatrix(train_sparse, label = train$Number_Of_Sales)
test_xgb_nos <- xgb.DMatrix(test_sparse, label = test$Number_Of_Sales)

# Parameter Setting
params_price <- list(booster = "gbtree", eta = 0.01, gamma = 0.1, max_depth = 6, min_child_weight = 3, subsample = 0.6, colsample_bytree = 0.8, nthread = 8)

# Watchlist for watching validate set performace along train set performace
watchlist <- list(eval = test_xgb_price, train = train_xgb_price)

# Training xgb model
xgb_model <- xgb.train(data = train_xgb_price, params = params_price, nrounds = 500, watchlist = watchlist, early_stopping_rounds = 30, objective = "reg:linear")

# Parameter Setting
params_nos <- list(booster = "gbtree", eta = 0.001, gamma = 0.1, max_depth = 4, min_child_weight = 3, subsample = 0.6, colsample_bytree = 0.8, nthread = 8)

# Watchlist for watching validate set performace along train set performace
watchlist <- list(eval = test_xgb_nos, train = train_xgb_nos)

# Training xgb model
xgb_model2 <- xgb.train(data = train_xgb_nos, params = params_nos, nrounds = 250, watchlist = watchlist, early_stopping_rounds = 30, objective = "reg:linear")

# Predicting for Price
pred_price_main <- predict(xgb_model, test_main_sparse)
pred_nos_main <- predict(xgb_model2, test_main_sparse)

result <- data.frame(ID = test_main$ID, Number_Of_Sales = pred_nos_main, Price =  pred_price_main)
write.csv(result, file = "sol_V1.csv", row.names = FALSE)


