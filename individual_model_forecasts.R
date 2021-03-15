library(data.table)
library(ggplot2)
library(forecast)
library(parallel)

# These series of baseline model fitting will be a modification of what was done in the m5 baseline models repository, 
# that is, it will try to predict first from the most granular level (i.e. item-level), and work its way to the the top most level by 
# a bottom-up approach

data <- fread('../input/m5-forecasting-uncertainty/sales_train_evaluation.csv')
col_ind_days <- grep('d_', colnames(data), value = T)

benchmarks_f  = function(data, model, h) {
    
    
    model_str <- rlang::quo_text(enquo(model))
    
    # current available benchmarks: Naive, sNaive, auto.arima, ets
    
    levels  <- list(total = NULL, state = c('state_id'), store = c('store_id'), category = 'cat_id', department = 'dept_id', 
               state_cat = c('state_id', 'cat_id'), state_dept = c('state_id', 'dept_id'), store_cat = c('store_id', 'cat_id'),
               store_dept = c('store_id', 'dept_id'), item = 'item_id', item_state = c('state_id', 'item_id'), item_store = c('item_id', 'store_id'))
    
    preProc = function(v) {
    # remove periods when the item wasn't active
    start_period  <-  min(which(v  > 0))
    return(v[start_period:length(v)])
        
    }
    
    agg_time_series = function(df_items, levels = NULL) {
    
        cols = grep('\\d+', colnames(df_items), value = T)

        if (is.null(levels)) {
            df2  <- df_items[,lapply(.SD, sum), .SDcols = cols]
            df2[,id1 := 'Total']
            df2[,id2 := 'X']
        } else {
            df2  <-  df_items[,lapply(.SD, sum), .SDcols = cols, by = levels]
            if (length(levels) == 1) {
                setnames(df2, levels, 'id1')
                df2[,id2 := 'X']
            } else {
                setnames(df2, levels, c('id1', 'id2'))
            }
            

        }
    }
    
    # PRE-PROCESSING DATA
    ## aggregating time series
    agg_ts  <- rbindlist(lapply(levels, agg_time_series, df_items = data), use.names = T)
    
    ## remove trailing zero sales (i.e drop periods where there are consecutive zero sales) and convert to time series objects
    pre_proc_ts <- apply(agg_ts[,1:1941], 1, function(x) ts(preProc(x), frequency = 7))
    names(pre_proc_ts) <- paste(agg_ts[['id1']], agg_ts[['id2']], sep = '_')
    rm(agg_ts)

    # calculate index boundaries for processing in batches  
    start_end_batch <- data.table(start = seq(1, 42840, by = 42840/10))
    start_end_batch[,end := shift(start, type = "lead") - 1]
    start_end_batch[is.na(end), end := 42840]
    
    # indicate the batch num
    batch_num = 3
    start_end <- start_end_batch[batch_num,]
                         
    if (model_str %in% c('naive', 'snaive', 'theta')) {
        preds_list <- mclapply(start_end$start:start_end$end, 
                           function(i, ts, h, model) {

                               preds <- do.call(model, list(ts[[i]],  h = h, level = c(50, 67, 95, 99)))
                               df_tmp <- data.table::data.table(cbind(preds$lower[,4], preds$lower[,3], preds$lower[,2], preds$lower[,1],
                                                    preds$mean,
                                                    preds$upper[,1], preds$upper[,2], preds$upper[,3], preds$upper[,4]))
                               #df_tmp[,(1:9) := lapply(.SD, function(x) sapply(x, max, 0))]
                               df_tmp[,id := names(ts[i])]
                               df_tmp[,day := 1:h]
                               return(df_tmp[])

                        },ts = pre_proc_ts, h = h, model = model, mc.cores = 4)
     

    } else if (model_str %in% c('auto.arima', 'ets')) {
       preds_list <- mclapply(start_end$start:start_end$end, 
                           function(i, ts, h, model) {

                               preds <- forecast::forecast(do.call(model, list(ts[[i]])),  h = h, level = c(50, 67, 95, 99))
                               df_tmp <- data.table::data.table(cbind(preds$lower[,4], preds$lower[,3], preds$lower[,2], preds$lower[,1],
                                                    preds$mean,
                                                    preds$upper[,1], preds$upper[,2], preds$upper[,3], preds$upper[,4]))
                               
                               #df_tmp[,(1:9) := lapply(.SD, function(x) sapply(x, max, 0))]
                               df_tmp[,id := names(ts[i])]
                               df_tmp[,day := 1:h]
                               return(df_tmp[])

                        },ts = pre_proc_ts, h = h, model = model, mc.cores = 4) 
    }
    
    pred_list_bind <- rbindlist(preds_list)
    rm(preds_list)
                                      
    return(pred_list_bind)
    
        
        
}


output <- benchmarks_f(data, auto.arima, 28)
saveRDS(output, 'auto_arima_predictions3.Rds')                                      

    
    





                         
                         