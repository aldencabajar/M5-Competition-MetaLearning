library(data.table)
library(forecast)
library(parallel)
library(xgboost)
source("functions.R")

sales <- fread('data/sales_train_validation.csv')
pre_proc_ts  <- readRDS("data/pre_proc_ts.rds")
tsfeat  <- readRDS("data/meta_learner_features.rds")
model_preds  <- readRDS("data/model_preds_full.rds")
col_ind_days <- grep('d_', colnames(data), value = T)

### import in additional model predictions
# theta method
theta_preds_val <-fread("data/theta_predictions_validation.Rds")
theta_preds_val[,id:=factor(id, levels = tsfeat$id)]
model_preds <- add_model_preds(model_preds, theta_preds_val)

# auto_arima
auto_arima_valid <- readRDS("data/auto_arima_forecast.Rds")
model_preds <- add_model_preds(model_preds,auto_arima_valid)

# tbats
tbats_valid <- readRDS("data/tbats_prediction.Rds")
model_preds <- add_model_preds(model_preds, tbats_valid)

### calculate aggregated time series for evaluation period from 1914-1941 days
evaluation  <- fread("data/sales_train_evaluation.csv")
evaluation_d  <- paste0("d_", 1914:1941)
evaluation  <- evaluation[,c(evaluation_d, "id", "item_id", "dept_id", "cat_id",
                              "store_id", "state_id"), with = F]
levels  <- list(
  total = NULL,
  state = c('state_id'),
  store = c('store_id'),
  category = 'cat_id',
  department = 'dept_id',
  state_cat = c('state_id', 'cat_id'),
  state_dept = c('state_id', 'dept_id'),
  store_cat = c('store_id', 'cat_id'),
  store_dept = c('store_id', 'dept_id'),
  item = 'item_id',
  item_state = c('state_id', 'item_id'),
  item_store = c('item_id', 'store_id')
)

agg_ts_evaluation  <- rbindlist(lapply(levels, agg_time_series, 
                                       df_items = evaluation), 
                                use.names = T)
agg_ts_evaluation[,id := apply(.SD, 1, paste, collapse = '_'), 
                  .SDcols = c("id1", "id2")]
agg_ts_evaluation[,c("id1", "id2") := NULL]

#### fit the xgboost meta learner ####
## The models being used to form ensemble predictions
model_list  <-  c("naive", "snaive", "ets", "rwf", "stlm", 
                  "thetaf", "auto_arima", "tbats")

## parameters to train the xgboost model
param <- list(max_depth=4, eta=0.019*6, silent=0,
              objective = combi_softmax_loss,
              num_class = length(model_list),
              subsample=0.9,
              colsample_bytree=0.6)

library(tictoc)
for (i in 1:9) {    
# loop across the 9 quantiles
  tic()
  bst  <- train_meta_learner(model_preds, 
                             quantile = i, 
                             train_data = pre_proc_ts, 
                             test_data = agg_ts_evaluation, 
                             nseries = length(pre_proc_ts),
                             model_list = model_list, 
                             features = tsfeat, params = param, ncores= 5)
  
  xgb.save(bst, paste0("bst/bst_quantile", i, ".model"))
  rm(bst); gc()
  toc()
}



