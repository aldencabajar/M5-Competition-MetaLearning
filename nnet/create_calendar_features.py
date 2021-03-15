import pandas as pd
import numpy as np
import pickle

agg_sales_all_lvl = pickle.load(open("agg_sales_all_lvl.pkl", "rb")) \
.reset_index().drop('index', axis = 1) 
calendar_w_price = pickle.load(open("calendar_processed.pkl", "rb")) 

days_col = ['d_'+ str(i+1) for i in range(1913)]

arr_min_sales = agg_sales_all_lvl[days_col] \
.apply(lambda x: x.values[np.min(np.where(x > 0)):], 
axis = 1)

lag = 4 # time lag to be considered
lg_sales_arr, lbl = [], []
for arr in arr_min_sales:
    lg_sales_arr.append(np.array([arr[i:i + lag] for i in range(len(arr) - lag)]))
    lbl.append(np.array([arr[i + lag] for i in range(len(arr) - lag)]))
lg_sales_arr = np.array(lg_sales_arr)
lbl = np.array(lbl)

del arr_min_sales

### PREPARING RELEVANT ARRAYS FOR TRAINING
# get relevant calendar vars for features in 1913 and onwards
var_list = ['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI',
	'year_frm', 'day_of_year', 'day_num', 'snap_all', 'Chanukah End',
	'Christmas', 'Cinco De Mayo', 'ColumbusDay', 'Easter', 'Eid al-Fitr',
	'EidAlAdha', "Father's day", 'Halloween', 'IndependenceDay', 'LaborDay',
	'LentStart', 'LentWeek2', 'MartinLutherKingDay', 'MemorialDay',
	"Mother's day", 'NBAFinalsEnd', 'NBAFinalsStart', 'NewYear',
	'OrthodoxChristmas', 'OrthodoxEaster', 'Pesach End', 'PresidentsDay',
	'Purim End', 'Ramadan starts', 'StPatricksDay', 'SuperBowl',
	'Thanksgiving', 'ValentinesDay', 'VeteransDay', 'num_events']
nitem = len(lg_sales_arr) 

h_feat_mt = []
feat_train_mt = []

for i, (arr, lvl) in enumerate(zip(lg_sales_arr[0:nitem], agg_sales_all_lvl['lvl'].values[0:nitem])):
    start_day = 1913 - len(arr) 
    agg_id = agg_sales_all_lvl.loc[i, 'id'] 
    if lvl in (2, 6, 7, 11):
        if agg_id.find('CA') == 0:
            snap = 'snap_CA'
        elif agg_id.find('WI') == 0:
            snap = 'snap_WI'
        elif agg_id.find('TX') == 0:
            snap = 'snap_TX'
    else:
        # for levels other than 2,6, 7 and 11, adapt the snap_all var
        snap = 'snap_all'
    
    if lvl >= 11:
        new_var_list = var_list +[snap] + [agg_id]
    else:
        new_var_list = var_list + [snap]
    
    feat_train = np.append(calendar_w_price.loc[start_day:1912, new_var_list].values,
                np.reshape(np.arange(start_day + 1, 1914), (len(range(start_day, 1912)) + 1, 1)), axis = 1)
    h_feat = np.append(calendar_w_price.loc[1913:,new_var_list].values,
                       np.reshape(np.arange(1914, 1970), (len(range(1914, 1969)) + 1, 1)), axis = 1)
    feat_train_mt.append(feat_train)
    h_feat_mt.append(h_feat)
    
feat_train_mt = np.array(feat_train_mt)
final_feature_mt = np.array([np.concatenate((feat_train_mt[i], lg_sales_arr[i]), axis = 1) for i in range(len(feat_train_mt)) ])

pickle.dump(final_feature_mt, open('features_train.pkl', 'wb'))
pickle.dump(lbl, open('label.pkl', 'wb'))
pickle.dump(h_feat_mt, open('x_predict_features.pkl', 'wb'))
