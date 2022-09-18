import pandas
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib
from statsmodels.formula.api import ols
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

USA = pandas.read_csv('/Users/carstenjuliansavage/Documents/BRG/ACS Microdata Code Interview/usa_00003.csv')
CPI = pandas.read_excel('/Users/carstenjuliansavage/Documents/BRG/ACS Microdata Code Interview/cpi yearly.xlsx')

# Display max columns
pandas.set_option('display.max_columns', None)

# Let's convert all variables to strings to see unique values.
USA_Object = USA.astype(str)
USA_Object.describe()

USA['inctot'] = np.where(USA['inctot']==9999999,np.nan,USA['inctot'])
USA['incwelfr'] = np.where(USA['incwelfr']==99999,np.nan,USA['incwelfr'])

USA.describe()

USA_AdjCPI = pandas.merge(USA,CPI,on='year',how='left')

# Adjusting the Total Income so that the numbers are adjusted for inflation using CPI data.

USA_AdjCPI = (USA_AdjCPI
 .assign(Total_Income=lambda a: a.incwelfr+a.inctot)
 .assign(Total_Income_Adjusted=lambda a: a.Total_Income*(237/a.cpi))
)

USAA_Slim = USA_AdjCPI.drop(['sample','serial','cbserial','strata','raced','cpi','Total_Income'],axis=1)

USA_With_Dummies = pandas.get_dummies(USAA_Slim, columns=['year','statefip','gq','bedrooms','ssmc','race','hcovany',
                                                         'empstat','empstatd','disabwrk','vetstat','vetstatd',
                                                          'cluster','ind','yrsusa1'])

# uhrswork should be numeric
USA_With_Dummies['uhrswork'] = pandas.to_numeric(USA_With_Dummies['uhrswork'], errors='coerce')

USA_With_Dummies[['hhwt', 'pernum', 'perwt', 'uhrswork', 'inctot', 'incwelfr','Total_Income_Adjusted']] = USA_With_Dummies[['hhwt', 'pernum', 'perwt', 'uhrswork', 'inctot', 'incwelfr','Total_Income_Adjusted']].astype('float64')

USA_With_Dummies.dtypes

USA_With_Dummies = (USA_With_Dummies
.query('Total_Income_Adjusted.notnull()')
)

from sklearn.utils import shuffle
Shuffled = shuffle(USA_With_Dummies)

USA_Shuffled = Shuffled.iloc[0:20000]
USA_Shuffled = (USA_Shuffled.query('Total_Income_Adjusted>=0'))

## Version 1

X = USA_Shuffled.drop(['Total_Income_Adjusted','bedrooms_5+ (1970-2000, 2000-2007 ACS/PRCS)',#'ind','yrsusa1',
                       'inctot','incwelfr'],axis=1)
y = pandas.DataFrame(USA_Shuffled['Total_Income_Adjusted'])

y = y.astype(int)

y = pandas.DataFrame.to_numpy(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

REG_xgb = xgb.XGBRegressor(objective='reg:linear',seed=47,booster='gblinear')
REG_xgb.fit(X,
            y)#,
            #verbose=False,
            #early_stopping_rounds=10,
            #eval_metric = 'aucpr',
            #eval_set=[(X_test,y_test)])

print(REG_xgb.coef_)

Coefficients = pandas.DataFrame(REG_xgb.coef_)
Columns_X = pandas.DataFrame(X.columns)

Coef_Col = pandas.concat([Columns_X, Coefficients], axis=1)