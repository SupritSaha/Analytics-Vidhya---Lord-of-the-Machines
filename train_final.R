# Loading required libraries
# ======================================================================
library(data.table) 
library(lubridate)
library(xgboost)
library(stringr)
library(dplyr)
library(NLP)
library(tm)
library(SparseM)
library(SnowballC)

# Reading data
# =======================================================================
dt_train <- fread("train_HFxi8kT/train.csv")
dt_test <- fread("test_BDIfz5B.csv")
dt_campaign <- fread("train_HFxi8kT/campaign_data_processed.csv")

# Combining train and test
# ========================================================================
dt_train[,is_train:= 1] 
dt_test[,is_train:= 0]
dt_test[,is_click:= NA]
dt_test[,is_open:= NA]
dt_test <- dt_test[,names(dt_train),with=FALSE]
dt_full <- rbindlist(list(dt_train,dt_test))

# Date Time features
# ========================================================================
dt_full[,send_date_parsed :=  as.Date(lubridate::parse_date_time(send_date,orders="dmy HM"))]
dt_full[,day_of_week := wday(send_date_parsed)]
dt_full[,day_of_month := lubridate::day(send_date_parsed)]
dt_full[,hour_of_day := as.integer(substr(send_date,start = 12,stop = 13))]

# Add campaign info
# ========================================================================
dt_full <- dt_full[dt_campaign,on = 'campaign_id']

# Campaign body and link features
# ========================================================================
dt_full[,internal_link_ratio := no_of_internal_links/total_links]
dt_full[,image_link_ratio := no_of_images/total_links]
dt_full[,section_image_ratio := no_of_sections/no_of_images]
dt_full[,section_link_ratio := no_of_sections/total_links]
dt_full[,body_length := nchar(email_body)]
dt_full[,image_body_ratio := no_of_images/body_length]
dt_full[,link_body_ratio := total_links/body_length]
dt_full[,header_body_ratio := nchar(subject)/body_length]
dt_full[,capital_subject := str_count(subject, "\\b[A-Z]{2,}\\b")]
dt_full[,capital_body := str_count(email_body, "\\b[A-Z]{2,}\\b")]
dt_full[,communication_type := as.factor(communication_type)]
dt_full[,topic := as.factor(topic)]
dt_full[,user_campaign_number := seq_along(sort(send_date_parsed)),by = "user_id"]


communication_type_date <- dt_full[,.(last_date = (max(send_date_parsed))),by = c("campaign_id","communication_type")][order(communication_type,last_date)]
communication_type_rank <- communication_type_date[,.(communication_rank = seq_along(campaign_id)),by = c("communication_type")]
communication_type_rank[,campaign_id := communication_type_date$campaign_id]

dt_full <- dt_full[communication_type_rank, on = c("campaign_id","communication_type")]

Campaign_Mails_Total <- data.table(dt_full %>% arrange(campaign_id, send_date_parsed) %>%  group_by(campaign_id) %>% mutate(Camp_Tot_Mails = dense_rank(send_date_parsed)))
dt_full <- dt_full[unique(Campaign_Mails_Total[,c("campaign_id","send_date_parsed","Camp_Tot_Mails"),with=FALSE]),on = c("campaign_id","send_date_parsed")]

dt_full <- data.table(dt_full %>% arrange(campaign_id, send_date) %>%  group_by(send_date) %>% mutate(Camp_Tot_Mails_timestamp = dense_rank(user_id)))
dt_full <- dt_full[order(user_id,send_date),]

# Get train and test back
# ==========================================================================
dt_train_processed <- dt_full[dt_full$is_train==1,]
dt_test_processed <- dt_full[dt_full$is_train==0,]

# Train summary
# ==========================================================================
n_unique_train_users <- length(unique(dt_train_processed$user_id))
last_campaign_send_date <- max(dt_train_processed$send_date_parsed) 
first_campaign_send_date <- min(dt_train_processed$send_date_parsed)
total_train_campaign_tenure <- as.numeric(last_campaign_send_date -first_campaign_send_date) # 153 days
n_unique_train_campaign <- length(unique(dt_train_processed$campaign_id)) # 26 campaigns

# User features
# ===========================================================================
dt_train_processed[,user_total_campaigns := length(campaign_id),by = "user_id"]
max_campaigns <- max(dt_train_processed$user_total_campaigns)
dt_train_processed[,user_prop_campaigns := user_total_campaigns/max_campaigns,by = "user_id"]

# Click and open features
# ===========================================================================
dt_train_processed[,c("total_clicks","total_open","avg_click_rate","avg_open_rate") := list(sum(is_click,na.rm = TRUE),sum(is_open,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_open,na.rm = TRUE)),by = 'user_id']
dt_train_processed[,c("hour_clicks","hour_open","hour_click_rate","hour_open_rate") := list(sum(is_click,na.rm = TRUE),sum(is_open,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_open,na.rm = TRUE)),by = 'hour_of_day']
dt_train_processed[,c("communication_clicks","communication_open","communication_click_rate","communication_open_rate") := list(sum(is_click,na.rm = TRUE),sum(is_open,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_open,na.rm = TRUE)),by = 'communication_type']
dt_train_processed[,c("dow_clicks","dow_open","dow_click_rate","dow_open_rate") := list(sum(is_click,na.rm = TRUE),sum(is_open,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_open,na.rm = TRUE)),by = 'day_of_week']
dt_train_processed[,c("day_clicks","day_open","day_click_rate","day_open_rate") := list(sum(is_click,na.rm = TRUE),sum(is_open,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_click,na.rm = TRUE),mean(is_open,na.rm = TRUE)),by = 'day_of_month']


dt_train_processed <- dt_train_processed[order(user_id,send_date_parsed),]

dt_train_processed[,c("max_run_click","max_run_open") := list(max(rle(is_click)$lengths,na.rm = TRUE),max(rle(is_open)$lengths,na.rm = TRUE)),by = 'user_id']
dt_train_processed[,c("max_run_click_prop","max_run_open_prop") := list(max_run_click/user_total_campaigns,max_run_open/user_total_campaigns),by = 'user_id']

# Contact features
# ================================================================================
interval_contact <- dt_train_processed %>% arrange(user_id, send_date_parsed) %>%  group_by(user_id) %>% mutate(diff = as.numeric(send_date_parsed - lag(send_date_parsed, default=first(send_date_parsed))))
interval_contact_user <- data.table(interval_contact)[,.(user_contact_interval_rate = mean(diff,na.rm = TRUE)),by="user_id"]
dt_train_processed <- dt_train_processed[interval_contact_user,on = "user_id"]

dt_train_processed[,user_contact_interval_rate := ifelse(is.na(user_contact_interval_rate),-1,user_contact_interval_rate)]

dt_train_processed[,user_last_contact_rate := mean(as.numeric(last_campaign_send_date - send_date_parsed),na.rm = TRUE),by = 'user_id']
dt_train_processed[,user_first_contact_rate := mean(as.numeric(send_date_parsed - first_campaign_send_date),na.rm = TRUE),by = 'user_id']

# Communication features
# ================================================================================
dt_train_processed[,unique_communication_rate := length(unique(communication_type))/user_total_campaigns,by = 'user_id']
communication_interval <- dt_train_processed[,.(max(send_date_parsed)),by = c("campaign_id","communication_type")][order(communication_type,V1),]
communication_interval <- communication_interval[,.(mean(as.numeric(V1 - lag(V1)),na.rm = TRUE)),by = "communication_type"]
communication_interval$V1[is.na(communication_interval$V1)] <- -999
names(communication_interval)[2] <- "avg_communication_interval"

dt_train_processed <- dt_train_processed[communication_interval,on="communication_type"]
  
dt_train_processed <- dt_train_processed[order(user_id,send_date_parsed),]
# Transition from 0 
# ==================================================================================
transitionZero <- function(x)
{
  x <- as.character(x)
  return(markovchainFit(x)$estimate[1][1])
}

dt_train_processed[,trans_zero := transitionZero(is_click),by = "user_id"]

# Mapping features to test data
# ==================================================================================
dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                            c("user_id","user_total_campaigns","user_prop_campaigns",
                            "total_clicks","total_open","avg_click_rate","avg_open_rate",
                            "max_run_click","max_run_open","max_run_click_prop","max_run_open_prop",
                            "user_contact_interval_rate","user_last_contact_rate","user_first_contact_rate",
                            "unique_communication_rate","trans_zero"),with = FALSE]),
                            by = "user_id",all.x =TRUE)

dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                            c("hour_of_day","hour_clicks","hour_open","hour_click_rate","hour_open_rate"),with = FALSE]),
                            by = "hour_of_day",all.x =TRUE)

dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                          c("communication_type","communication_clicks","communication_open","communication_click_rate",
                            "communication_open_rate","avg_communication_interval"),with = FALSE]),
                            by = "communication_type",all.x =TRUE)

dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                            c("day_of_week","dow_clicks","dow_open","dow_click_rate","dow_open_rate"),with = FALSE]),
                            by = "day_of_week",all.x =TRUE)

dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                          c("day_of_month","day_clicks","day_open","day_click_rate","day_open_rate"),with = FALSE]),
                          by = "day_of_month",all.x =TRUE)

dt_test_processed <- dt_test_processed[,names(dt_train_processed),with=FALSE]


# Smoothing
# ==============================================================================
avgRateSmoothing <- function(target,count,raw_average,smoothing = 1,min_samples_leaf = 1)
{
  smoothing <- 1/(1+ exp(-(count-min_samples_leaf)/smoothing))
  prior <- mean(as.numeric(target))
  smoothed_avg <- prior * (1 - smoothing) + raw_average*smoothing
  return(smoothed_avg)
}
dt_train_processed[,avg_click_rate_smoothed := avgRateSmoothing(target = dt_train_processed$is_click,count = dt_train_processed$total_clicks,raw_average = dt_train_processed$avg_click_rate)]
dt_train_processed[,avg_open_rate_smoothed := avgRateSmoothing(target = dt_train_processed$is_open,count = dt_train_processed$total_open,raw_average = dt_train_processed$avg_open_rate)]

dt_test_processed <- merge(dt_test_processed,unique(dt_train_processed[,
                            c("user_id","avg_click_rate_smoothed","avg_open_rate_smoothed"),with = FALSE]),
                            by = "user_id",all.x =TRUE)


# Chaging data type of target variable
# ===============================================================================
dt_train_processed[,is_click :=  as.factor(is_click)]
dt_test_processed[,is_click := as.factor(is_click)]


dt_test_processed[is.na(dt_test_processed)] <- -999

# Model development
# ===============================================================================
library(h2o)

h2o.init(nthreads = 5)

features <- names(dt_train_processed)[c(6,9:16,24:71)]

trainHex <- as.h2o(dt_train_processed[,features,with =FALSE])
testHex <- as.h2o(dt_test_processed[,features,with =FALSE])

names(trainHex)

# Dropping not so important variables
# ==============================================================================
rm_variables <- c("day_of_week","day_of_month","avg_communication_interval","capital_subject",
                  "dow_open","dow_clicks","dow_click_rate","dow_open_rate","day_clicks","day_open", #0.69095
                   "hour_open","hour_of_day", # 0.69106
                   # "body_length","capital_body",#0.69096
                   #  "section_image_ratio","no_of_images","section_link_ratio",
                    #"campaign_body","no_of_images",
                  "Camp_Tot_Mails_timestamp",
                  "avg_click_rate","avg_open_rate","no_of_sections","trans_zero")

trainHex <- trainHex[,!features %in% rm_variables]

# Random forest
# =======================================================================================
model_rf <- h2o.randomForest(
                              training_frame = trainHex,       
                              x=c(2:ncol(trainHex)),                       
                              y= 1,
                              ntrees = 1200,
                              max_depth = 4,
                              sample_rate = 0.7,
                              stopping_rounds = 5,
                              col_sample_rate_per_tree = 0.7,
                              col_sample_rate = 0.7,
                              seed = 123,
                              balance_classes = TRUE,
                              min_rows = 500,
                              stopping_metric = "AUC"
                              )

summary(model_rf)
var_imp_rf <- h2o.varimp(model_rf)
predictions_rf <- as.data.frame(h2o.predict(object = model_rf, newdata = testHex[,-1]))

base_submission <- ifelse(dt_test_processed$avg_click_rate_smoothed < 0,mean(dt_train_processed$avg_click_rate_smoothed),dt_test_processed$avg_click_rate_smoothed)
results_rf <- as.data.table(cbind(dt_test_processed[,c("id")], (0.6*as.numeric(predictions_rf$p1) +  0.4*base_submission)))
colnames(results_rf) <- c("id", "is_click")
summary(results_rf)
fwrite(results_rf,"pred_rf.csv", row.names = FALSE)
# This scores 7th on Private LB and 9 th on Public LB


# GBM 
# =======================================================================================
model_gbm <- h2o.gbm(training_frame = trainHex,       
                      x=c(2:ncol(trainHex)),                       
                      y= 1,
                      ntrees = 200,
                      stopping_rounds = 5,
                      min_rows = 500,
                      max_depth = 4,
                      learn_rate = 0.005,
                      balance_classes = TRUE,
                      sample_rate = 0.65,
                      col_sample_rate = 0.65,
                      seed = 123, 
                      stopping_metric = "AUC")

summary(model_gbm)
var_imp_gbm <- h2o.varimp(model_gbm)

predictions_gbm <- as.data.frame(h2o.predict(object = model_gbm, newdata = testHex[,-1] ))

results_gbm <- as.data.table(cbind(dt_test_processed[,c("id")], (0.5*as.numeric(predictions_gbm$p1) +  0.5*base_submission)))
colnames(results_gbm) <- c("id", "is_click")

summary(results_gbm)
fwrite(results_gbm,"pred_gbm.csv", row.names = FALSE)

results <- as.data.table(cbind(dt_test_processed[,c("id")], (0.1*as.numeric(predictions_gbm$p1) + 0.7*as.numeric(predictions_rf$p1)) + 0.2*base_submission) )
colnames(results) <- c("id", "is_click")
summary(results)
fwrite(results,"pred_ensemble.csv", row.names = FALSE)

h2o.shutdown(prompt = FALSE)

