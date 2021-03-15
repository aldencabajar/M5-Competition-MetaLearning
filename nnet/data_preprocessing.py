import pandas
import numpy as numpy
import gc
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import argparse

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

def ts_agg(df, grp, i):
    days_col = df.columns[df.columns.str.contains(r'd_\d*')] 
    if 'X' in grp and 'Total' not in grp:
        agg_col = grp[np.where(np.array(grp) != 'X')[0][0]]
        fn = df[np.append(agg_col, days_col)].set_index(agg_col).groupby(agg_col).apply(sum).reset_index()
        fn['id2'] = 'X'
        fn['id'] = fn[np.array([agg_col, 'id2'])].apply(lambda x: '_'.join(x), axis = 1).values
        #fn.drop([agg_col, 'id2'], inplace = True, axis = 1)
        
    elif 'Total' in grp:
        df['id'] = 'Total_X'
        fn = df[np.append('id', days_col)].set_index('id').groupby('id').apply(sum).reset_index()
        
        
    else:
        fn = df[np.append(grp, days_col)].set_index(grp).groupby(grp).apply(sum).reset_index()
        fn['id'] = fn[grp].apply(lambda x: '_'.join(x), axis = 1).values
        #fn.drop(grp, inplace = True, axis = 1)
    
    fn['lvl'] = i 

    return(fn)


def create_dataset(agg_sales_all, calendar_file, start_day, lags):
    days_col = agg_sales_all.columns[
    agg_sales_all.columns.str.contains(r'd_\d*')]

    # determine the minimum day when sales was non-zero  
    agg_sales_all['min_day_sales'] = agg_sales_all.loc[:,days_col] \
    .apply(lambda x: np.min(np.where(x > 0)[0] + 1), axis = 1)

    print("melting the dataset...")
    agg_melt = pd.melt(agg_sales_all.loc[:,list(days_col) + ['id', 'min_day_sales', 'wt']], 
                    var_name = "day", value_name = "sales", 
                    id_vars = ['id', 'min_day_sales', 'wt'])

    start_mem = agg_melt.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    agg_melt['day'] = agg_melt['day'].str.replace('d_', '').astype(np.int16)
    agg_melt['min_day_sales'] = agg_melt['min_day_sales'].astype(np.int16)
    agg_melt['sales'] = agg_melt['sales'].astype(np.float32)
    agg_melt['wt'] = agg_melt['wt'].astype(np.float32)
    end_mem = agg_melt.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(end_mem))

    agg_melt.sort_values(by = ["id", "day"], inplace = True)
    agg_melt.reset_index(inplace = True, drop = True)

    print( 'maximum weight:', np.nanmax(agg_melt['wt']), '\nminimum weight:', 
        np.nanmin(agg_melt['wt']))

    agg_melt = agg_melt.loc[
    (agg_melt.min_day_sales <= agg_melt.day) & (agg_melt.day >= start_day)]

    agg_melt['scaled'] = agg_melt['sales'] / agg_melt['wt']

    # create lagged variables
    print('creating lagged variables...')
    if np.any(np.array(lags) < 28):
    return(print("please pick a lagged variable greater than 28 days")) 
    else:
    for lag in lags:
        lag_name = "lag" + str(lag)
        agg_melt[lag_name] = agg_melt.groupby("id")['scaled'].shift(lag)
    # drop lagged variables with null values 
    lag_vars = agg_melt.columns[agg_melt.columns.str.contains('lag')]
    agg_melt = agg_melt.dropna(subset = lag_vars)

    # remmove unnecessary vars
    print('removing unnecessary variables...')
    agg_melt.drop(columns = ['min_day_sales'], inplace = True)
    gc.collect()


    # prepare calendar variables
    print('preparing calendar variables...')
    calendar = pd.read_csv(calendar_file)
    reduce_mem_usage(calendar)
    calendar['date1'] = calendar.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    calendar['start_date_year'] = calendar.year.apply(lambda x: datetime.strptime(str(x) + '-01-01', '%Y-%m-%d'))
    calendar['day_of_year'] = calendar['date1'] - calendar['start_date_year']
    calendar['day_of_year'] = calendar['day_of_year'].apply(lambda x: x.days + 1).astype(np.int16)
    calendar['d'] = calendar['d'].str.replace("d_", "").astype(np.int16)
    var_list = ['wday', 'month', 'year', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI', 'day_of_year']

    # convert events to integers, can be thought of as ordinal encoding, although
    # this might not be a good way to represent it 
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    calendar[event_cols] = calendar[event_cols].apply(lambda x: x.astype("category").cat.codes)

    print('merging with sales...')
    sales_calendar_merge = pd.merge(calendar[var_list], 
                                    agg_melt, right_on = "day", 
                                    left_on = "d", how = "right") \
                                    .sort_values(by =['id', 'day'], ignore_index  = True)
    sales_calendar_merge.drop(columns = ['d'], inplace=True)                                
    del agg_melt; gc.collect()
    print('dataset shape:', sales_calendar_merge.shape)


  return(sales_calendar_merge)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sales_file', type=str, help = 'path of sales data')
    parser.add_argument('calendar_file', type=str, help = 'path of calendar data')
    parser.add_argument('-s','--start' type=int, help = 'day to start for truncation')
    parser.add_argument('-l','--lag' nargs ='+', help = 'day to start for truncation')

    args = parser.parse_args()
    calendar = pd.read_csv(args.calendar_file)
    sales = pd.read_csv(args.sales_file)

    # reduce memory
    reduce_mem_usage(calendar)

    ### DETERMINE AGGREGATED SALES ####
    levels = (['Total','X'], ['state_id', 'X'], ['store_id', 'X'], ['cat_id', 'X'], ['dept_id', 'X'], 
            ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'], ['store_id', 'dept_id'], 
            ['item_id', 'X'], ['state_id', 'item_id'], ['item_id', 'store_id'])

    # create a dataframe with all aggregations from all levels
    agg_sales_list = []
    for i, lv in tqdm(enumerate(levels)):
        agg_sales_list.append(ts_agg(sales, lv, i +s 1))
    agg_sales_all = pd.concat(agg_sales_list)
    agg_sales_all.reset_index(drop = True, inplace = True)

    # calculate weights for standardization later             
    agg_sales_all['wt'] = agg_sales_all.loc[:,days_col] \
    .apply(lambda x: np.abs(x- x.shift(1)).mean(), axis = 1)
    agg_sales_all['wt'] 

    # Add dummy sales columns (from day 1942 to 1969) 
    # to facilitate creation of evaluation days features
    d_cols_eval = ['d_' + str(i) for i in range(1942, 1970)] 
    for d in d_cols_eval:
    agg_sales_all[d] = np.nan 
