# 

preProc = function(v) {
  "remove periods when the item wasn't active"
  start_period  <-  min(which(v  > 0))
  return(v[start_period:length(v)])
  
}

agg_time_series = function(df_items, levels = NULL) {
	"an aggregating function at different levels as indicated by the levels list"
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

spl_loss = function(q_preds, quantile, train, test) {
  "
  calculates the spl function, with the following arguments:
  q_preds = the prediction of a quantile q
  train = data of time series used to train a model to output prediction q_preds
  test = test data for time series
  "
  diff  <- q_preds - test
  mt_q  <- replicate(28, quantile)
  
  # obtain the denominator of the spl function
  S  <- sum(abs(train[2:length(train)] - 
                  train[1:(length(train) - 1)]))/(length(train) - 1) 
  
  # calculate the whole equation
  abs(diff) * ifelse(diff <= 0, mt_q, 1-mt_q)/S
  
}

train_meta_learner = function(model_preds, quantile, train_data, test_data, 
                              nseries = length(train_data), model_list, 
                              features,
                              params, ncores = 8) {
  
  "trains an xgboost meta-learner for a specific quantile 
  Arguments:
  model_preds = a list of model predictions with each being a 
  list also of data tables with corresponding quantile predictions
  quantile = integer, quantile of interest
  train_data = a list of time series used for training models
  test_data = a data table of test data
  model_list = list of models that were used for prediction
  features = additional static features to include, 
   should have a column named 'id'
  params = xgboost parameters"
  
  quantiles = c(0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995)
  
  # for each model, obtain spl across different time series
  loss_per_model  <- vector(mode = "list", length = length(model_list))
  
  for (i in 1:length(model_list)) {
    
    modelListSPL  <- mclapply(
      1:nseries,
      function(j) {
        tmp  <- spl_loss(q_preds = model_preds[[i]][[j]][,..quantile], 
                         quantile = quantiles[quantile],
                         train = train_data[[j]], 
                         test = as.matrix(
                           test_data[j,evaluation_d, with = F]
                           )[1,]
                         )
        tmp[,id := names(train_data)[j]]
        tmp[,day := model_preds[[i]][[j]]$day]
      }, mc.cores = ncores
    )
    
    loss_per_model[[i]]  <- modelListSPL
    
  }
  
  # reshape list into a h x m matrix (stacked vertically by time series) where h is the number of
  # prediction days and m is the number of models
  
  qt_loss_list  <- vector(mode = "list", length = length(model_list))
  
  for (i in 1:length(loss_per_model)) {
    df_bind  <- rbindlist(loss_per_model[[i]])
    setnames(df_bind, 1, model_list[i])
    qt_loss_list[[i]]  <- df_bind
    
  }
  rm(loss_per_model); gc()
  quantile_dt  <- Reduce(function(x,y) merge(x, y, by = c("day", "id")), 
                         qt_loss_list)[order(id)]                           
  #print(paste('dataframe input to xgboost model has', nrow(quantile_dt), "rows"))
  
  
  # adding features for meta-learner
  var_update  <- names(features)[which(names(features) != "id")]
  quantile_dt[features, (var_update) := mget(paste0('i.', var_update)), on = 'id']
  
  # prepare dataset for xgboost train and train the model
  feature_idx  <- which(!(colnames(quantile_dt) %in% c("id", model_list)))
  xgb_train  <- xgboost::xgb.DMatrix(as.matrix(quantile_dt[,..feature_idx]))                      
  attr(xgb_train, "col")  <- length(model_list)
  attr(xgb_train, "ff")  <- as.matrix(quantile_dt[,..model_list, which = F])
  rm(quantile_dt); gc()
  bst <- xgboost::xgb.train(params, xgb_train, 150)
  
  return(bst)
  
}

combi_softmax_loss <- function(preds, dtrain) {
  "loss function to be used to train xgb model"
  
  ff <- attr(dtrain, "ff")
  col  <- attr(dtrain, "col")
  preds  <- matrix(preds, ncol = col, byrow = T)
  preds <- exp(preds)
  sp <- rowSums(preds)
  preds <- preds / replicate(ncol(preds), sp)
  S <- rowSums(preds * ff)
  GradSxx <- preds*(ff - S)
  grad <- GradSxx 
  hess <- preds*(ff*(1-preds) - grad)
  
  return(list(grad = t(grad), hess = t(hess)))
}

add_model_preds = function(model_preds, new_model_preds) {
	# split by id 
	frcst_list <- split(new_model_preds, new_model_preds$id)
	current_n <- length(model_preds)
	model_preds[[current_n + 1]] <- frcst_list 
	return(model_preds)
}

qt_preds_aggregate = function(preds_mod_bind, quantiles, model_list, 
                              type = "validation") {
  "aggregated quantile predictions from a set of model predictions"
  qt_mod  <- vector(mode = "list", length = length(quantiles)) 
  for(q in 1:length(quantiles)) {
    
    preds_mod_list  <- 
      lapply(
        1:length(model_list), 
        
        function(x) {
          tmp  <- preds_mod_bind[[x]][,c(q, 10, 11), with = F]
          setnames(tmp, 1, model_list[x])  
          
        })
    quantile_dt  <- Reduce(function(x,y) merge(x, y, by = c("day", "id")), 
                           preds_mod_list)
    quantile_dt[,id:=factor(id, levels = tsfeat$id)]
    setorder(quantile_dt, id)
    
    
    # using xgboost model to predict the weights for each prediction                       
    prds  <- predict(bst_list[[q]], test_data, outputmargin = T, reshape = T)
    prds  <- exp(prds)
    sp <- rowSums(prds)
    prds <- prds / replicate(ncol(prds), sp)
    
    # get the final prediction using the weights
    quantile_dt[,prds := rowSums(quantile_dt[,-c("day", "id"), with = F]*prds)]
    fnl  <- dcast(quantile_dt, id ~ day, value.var = "prds")
    
    #change the id to the quantile                       
    fnl[,id_f := paste(id, format(quantiles[q], nsmall = 3), type, sep = "_")]
    qt_mod[[q]]  <- copy(fnl)                       
  }
  
  qt_mod  <- rbindlist(qt_mod)
  setorder(qt_mod, id)
  qt_mod[,id := NULL]
  setnames(qt_mod, c("id_f", as.character(1:28)), 
           c("id", paste0("F", as.character(1:28))))
  setcolorder(qt_mod, c("id", paste0("F", as.character(1:28))))
  
  return(copy(qt_mod))                               
}

