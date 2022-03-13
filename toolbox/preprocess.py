#IMPORTS

import numpy as np
import pandas as pd 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import os
import warnings
warnings.filterwarnings('ignore')


# Version Complète
# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.
# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns   

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv(r'C:/OPCR/PROJET7OPCR/data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# -------------------------------------------------
# Fin du Features engeneering
# --------------------------------------------------

def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    
    with timer("Fin calcul df"):
        cc = credit_card_balance(num_rows)      
        print("df shape:",df.shape)
    
    DF_FE = df.copy()
    
    # Vérification des valeurs manquantes dans le fichier Df_application_train
    Calc_Taux_NanDF_FE = DF_FE.isnull().sum() * 100 / len(DF_FE) 
    Df_Taux_NanDF_FE = pd.DataFrame({'column_name': DF_FE.columns,'percent_missing': Calc_Taux_NanDF_FE}).sort_values('percent_missing')
    Df_Taux_NanDF_FE
    
    # Conservation des colonnes ayant moins de 20% de valeurs manquantes
    List_Columns_20_Nan=list(Df_Taux_NanDF_FE[Df_Taux_NanDF_FE["percent_missing"] < 20].index)
    
    # Filtre du Dataframe sur ces colonnes
    DF_FE_20_NAN = DF_FE[List_Columns_20_Nan]
    
    # Imputation des valeurs manquantes
    DF_FE_20_NAN = DF_FE_20_NAN.replace(-np.inf, np.nan)
    DF_FE_20_NAN = DF_FE_20_NAN.replace(np.inf, np.nan)
    
    List_Colonnes_DF_FE_20_NAN = list(DF_FE_20_NAN.columns)
    
    # Liste des variables explicatives retenues
    Liste_Var_Explicatives = ['SK_ID_CURR','TARGET','OCCUPATION_TYPE_Medicine staff',
     'OCCUPATION_TYPE_Private service staff',
     'OCCUPATION_TYPE_Realty agents',
     'OCCUPATION_TYPE_Sales staff',
     'OCCUPATION_TYPE_Secretaries',
     'OCCUPATION_TYPE_Security staff',
     'OCCUPATION_TYPE_Waiters/barmen staff',
     'OCCUPATION_TYPE_Managers',
     'OCCUPATION_TYPE_Low-skill Laborers',
     'OCCUPATION_TYPE_Laborers',
     'OCCUPATION_TYPE_IT staff',
     'OCCUPATION_TYPE_Accountants',
     'OCCUPATION_TYPE_Cleaning staff',
     'OCCUPATION_TYPE_Cooking staff',
     'OCCUPATION_TYPE_Core staff',
     'OCCUPATION_TYPE_Drivers',
     'OCCUPATION_TYPE_HR staff',
     'OCCUPATION_TYPE_High skill tech staff',
     'ORGANIZATION_TYPE_Advertising',
     'ORGANIZATION_TYPE_Agriculture',
     'ORGANIZATION_TYPE_Bank',
     'ORGANIZATION_TYPE_Business Entity Type 1',
     'ORGANIZATION_TYPE_Business Entity Type 2',
     'ORGANIZATION_TYPE_Business Entity Type 3',
     'ORGANIZATION_TYPE_Cleaning',
     'ORGANIZATION_TYPE_Construction',
     'ORGANIZATION_TYPE_Culture',
     'ORGANIZATION_TYPE_Electricity',
     'ORGANIZATION_TYPE_Emergency',
     'ORGANIZATION_TYPE_Government',
     'ORGANIZATION_TYPE_Hotel',
     'ORGANIZATION_TYPE_Housing',
     'ORGANIZATION_TYPE_Industry: type 1',
     'ORGANIZATION_TYPE_Industry: type 10',
     'ORGANIZATION_TYPE_Trade: type 4',
     'ORGANIZATION_TYPE_Trade: type 5',
     'ORGANIZATION_TYPE_Trade: type 6',
     'ORGANIZATION_TYPE_Trade: type 7',
     'ORGANIZATION_TYPE_Transport: type 1',
     'ORGANIZATION_TYPE_Transport: type 2',
     'ORGANIZATION_TYPE_Transport: type 3',
     'ORGANIZATION_TYPE_Transport: type 4',
     'ORGANIZATION_TYPE_University',
     'ORGANIZATION_TYPE_XNA',
     'ORGANIZATION_TYPE_Trade: type 2',
     'ORGANIZATION_TYPE_Telecom',
     'ORGANIZATION_TYPE_Industry: type 11',
     'ORGANIZATION_TYPE_Industry: type 12',
     'ORGANIZATION_TYPE_Industry: type 13',
     'ORGANIZATION_TYPE_Industry: type 2',
     'ORGANIZATION_TYPE_Industry: type 3',
     'ORGANIZATION_TYPE_Industry: type 5',
     'ORGANIZATION_TYPE_Industry: type 6',
     'ORGANIZATION_TYPE_Industry: type 7',
     'ORGANIZATION_TYPE_Industry: type 8',
     'ORGANIZATION_TYPE_Industry: type 9',
     'ORGANIZATION_TYPE_Insurance',
     'ORGANIZATION_TYPE_Kindergarten',
     'ORGANIZATION_TYPE_Legal Services',
     'ORGANIZATION_TYPE_Medicine',
     'ORGANIZATION_TYPE_Military',
     'ORGANIZATION_TYPE_Mobile',
     'ORGANIZATION_TYPE_Other',
     'ORGANIZATION_TYPE_Police',
     'ORGANIZATION_TYPE_Postal',
     'ORGANIZATION_TYPE_Realtor',
     'ORGANIZATION_TYPE_Religion',
     'ORGANIZATION_TYPE_Restaurant',
     'ORGANIZATION_TYPE_School',
     'ORGANIZATION_TYPE_Security',
     'ORGANIZATION_TYPE_Security Ministries',
     'ORGANIZATION_TYPE_Self-employed',
     'ORGANIZATION_TYPE_Services',
     'ORGANIZATION_TYPE_Trade: type 1',
     'NAME_INCOME_TYPE_Student',
     'NAME_INCOME_TYPE_Unemployed',
     'NAME_INCOME_TYPE_Working',
     'NAME_EDUCATION_TYPE_Academic degree',
     'NAME_EDUCATION_TYPE_Higher education',
     'NAME_EDUCATION_TYPE_Incomplete higher',
     'NAME_EDUCATION_TYPE_Lower secondary',
     'NAME_EDUCATION_TYPE_Secondary / secondary special',
     'NAME_FAMILY_STATUS_Civil marriage',
     'NAME_FAMILY_STATUS_Married',
     'NAME_FAMILY_STATUS_Separated',
     'NAME_FAMILY_STATUS_Single / not married',
     'NAME_FAMILY_STATUS_Unknown',
     'NAME_FAMILY_STATUS_Widow',
     'NAME_HOUSING_TYPE_Co-op apartment',
     'NAME_HOUSING_TYPE_House / apartment',
     'NAME_HOUSING_TYPE_Municipal apartment',
     'NAME_HOUSING_TYPE_Office apartment',
     'NAME_HOUSING_TYPE_Rented apartment',
     'NAME_HOUSING_TYPE_With parents',
     'HOUSETYPE_MODE_block of flats',
     'HOUSETYPE_MODE_specific housing',
     'HOUSETYPE_MODE_terraced house',
     'INCOME_CREDIT_PERC',
     'ORGANIZATION_TYPE_Trade: type 3',
     'NAME_INCOME_TYPE_State servant',
     'NAME_INCOME_TYPE_Pensioner',
     'ORGANIZATION_TYPE_Industry: type 4',
     'NAME_INCOME_TYPE_Commercial associate',
     'FLAG_DOCUMENT_5',
     'FLAG_DOCUMENT_4',
     'FLAG_DOCUMENT_3',
     'FLAG_DOCUMENT_2',
     'FLAG_OWN_CAR',
     'FLAG_OWN_REALTY',
     'CNT_CHILDREN',
     'AMT_INCOME_TOTAL',
     'AMT_CREDIT',
     'DAYS_BIRTH',
     'FLAG_MOBIL',
     'FLAG_EMP_PHONE',
     'FLAG_WORK_PHONE',
     'FLAG_CONT_MOBILE',
     'FLAG_PHONE',
     'FLAG_EMAIL',
     'REGION_RATING_CLIENT',
     'REGION_RATING_CLIENT_W_CITY',
     'HOUR_APPR_PROCESS_START',
     'REG_REGION_NOT_LIVE_REGION',
     'REG_REGION_NOT_WORK_REGION',
     'LIVE_REGION_NOT_WORK_REGION',
     'REG_CITY_NOT_LIVE_CITY',
     'FLAG_DOCUMENT_7',
     'FLAG_DOCUMENT_8',
     'FLAG_DOCUMENT_6',
     'FLAG_DOCUMENT_10',
     'NAME_INCOME_TYPE_Businessman',
     'NAME_TYPE_SUITE_Unaccompanied',
     'NAME_TYPE_SUITE_Spouse, partner',
     'NAME_TYPE_SUITE_Other_B',
     'NAME_TYPE_SUITE_Other_A',
     'NAME_TYPE_SUITE_Group of people',
     'NAME_TYPE_SUITE_Family',
     'NAME_TYPE_SUITE_Children',
     'NAME_CONTRACT_TYPE_Revolving loans',
     'NAME_CONTRACT_TYPE_Cash loans',
     'NAME_INCOME_TYPE_Maternity leave',
     'FLAG_DOCUMENT_9',
     'CODE_GENDER',
     'LIVE_CITY_NOT_WORK_CITY',
     'FLAG_DOCUMENT_11',
     'FLAG_DOCUMENT_12',
     'FLAG_DOCUMENT_13',
     'FLAG_DOCUMENT_14',
     'FLAG_DOCUMENT_20',
     'FLAG_DOCUMENT_19',
     'FLAG_DOCUMENT_18',
     'FLAG_DOCUMENT_17',
     'FLAG_DOCUMENT_21',
     'FLAG_DOCUMENT_16',
     'FLAG_DOCUMENT_15',
     'CNT_FAM_MEMBERS',
     'INCOME_PER_PERSON',
     'ANNUITY_INCOME_PERC',
     'AMT_ANNUITY',
     'PAYMENT_RATE',
     'AMT_GOODS_PRICE',
     'EXT_SOURCE_2',
     'DEF_60_CNT_SOCIAL_CIRCLE',
     'OBS_60_CNT_SOCIAL_CIRCLE',
     'DEF_30_CNT_SOCIAL_CIRCLE',
     'OBS_30_CNT_SOCIAL_CIRCLE',
     'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',
     'INSTAL_DAYS_ENTRY_PAYMENT_SUM',
     'INSTAL_AMT_PAYMENT_SUM',
     'INSTAL_AMT_INSTALMENT_SUM',
     'INSTAL_AMT_INSTALMENT_MAX',
     'INSTAL_PAYMENT_DIFF_SUM',
     'INSTAL_AMT_INSTALMENT_MEAN',
     'INSTAL_DBD_MAX',
     'INSTAL_DPD_MAX',
     'INSTAL_DPD_MEAN',
     'INSTAL_DPD_SUM',
     'INSTAL_DBD_MEAN',
     'INSTAL_COUNT',
     'INSTAL_DBD_SUM',
     'INSTAL_AMT_PAYMENT_MIN',
     'INSTAL_AMT_PAYMENT_MAX',
     'INSTAL_AMT_PAYMENT_MEAN',
     'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
     'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
     'INSTAL_PAYMENT_PERC_MAX',
     'INSTAL_PAYMENT_DIFF_MEAN',
     'INSTAL_PAYMENT_DIFF_MAX',
     'INSTAL_PAYMENT_PERC_SUM',
     'INSTAL_PAYMENT_PERC_MEAN',
     'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
     'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
     'PREV_NAME_PAYMENT_TYPE_XNA_MEAN',
     'PREV_CODE_REJECT_REASON_CLIENT_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN',
     'PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN',
     'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
     'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
     'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
     'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',
     'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN',
     'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
     'PREV_CODE_REJECT_REASON_HC_MEAN',
     'PREV_NAME_TYPE_SUITE_Children_MEAN',
     'PREV_CODE_REJECT_REASON_SCO_MEAN',
     'PREV_NAME_CLIENT_TYPE_nan_MEAN',
     'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
     'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
     'PREV_NAME_CLIENT_TYPE_Refreshed_MEAN',
     'PREV_NAME_CLIENT_TYPE_New_MEAN',
     'PREV_NAME_TYPE_SUITE_nan_MEAN',
     'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',
     'PREV_NAME_TYPE_SUITE_Spouse, partner_MEAN',
     'PREV_NAME_TYPE_SUITE_Other_B_MEAN',
     'PREV_NAME_TYPE_SUITE_Other_A_MEAN',
     'PREV_NAME_TYPE_SUITE_Group of people_MEAN',
     'PREV_NAME_TYPE_SUITE_Family_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
     'PREV_CODE_REJECT_REASON_nan_MEAN',
     'PREV_CODE_REJECT_REASON_XNA_MEAN',
     'PREV_CODE_REJECT_REASON_XAP_MEAN',
     'PREV_CODE_REJECT_REASON_VERIF_MEAN',
     'PREV_CODE_REJECT_REASON_SYSTEM_MEAN',
     'PREV_CODE_REJECT_REASON_SCOFR_MEAN',
     'PREV_CODE_REJECT_REASON_LIMIT_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
     'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN',
     'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN',
     'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
     'PREV_HOUR_APPR_PROCESS_START_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
     'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN',
     'PREV_DAYS_DECISION_MIN',
     'PREV_DAYS_DECISION_MAX',
     'PREV_DAYS_DECISION_MEAN',
     'PREV_CNT_PAYMENT_SUM',
     'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN',
     'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
     'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
     'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN',
     'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN',
     'PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
     'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
     'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN',
     'PREV_PRODUCT_COMBINATION_nan_MEAN',
     'PREV_HOUR_APPR_PROCESS_START_MAX',
     'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Construction_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN',
     'PREV_CHANNEL_TYPE_nan_MEAN',
     'PREV_CHANNEL_TYPE_Stone_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN',
     'PREV_CHANNEL_TYPE_Regional / Local_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Industry_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN',
     'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
     'PREV_NAME_YIELD_GROUP_XNA_MEAN',
     'PREV_NAME_YIELD_GROUP_high_MEAN',
     'PREV_NAME_YIELD_GROUP_low_action_MEAN',
     'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
     'PREV_NAME_YIELD_GROUP_middle_MEAN',
     'PREV_NAME_YIELD_GROUP_nan_MEAN',
     'PREV_PRODUCT_COMBINATION_Card Street_MEAN',
     'PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash Street: low_MEAN',
     'PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN',
     'PREV_CHANNEL_TYPE_Credit and cash offices_MEAN',
     'PREV_CHANNEL_TYPE_Contact center_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN',
     'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
     'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
     'BURO_AMT_CREDIT_SUM_MAX',
     'BURO_AMT_CREDIT_SUM_MEAN',
     'BURO_DAYS_CREDIT_ENDDATE_MIN',
     'BURO_DAYS_CREDIT_ENDDATE_MAX',
     'BURO_DAYS_CREDIT_ENDDATE_MEAN',
     'BURO_AMT_CREDIT_SUM_DEBT_MAX',
     'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
     'DAYS_EMPLOYED_PERC',
     'DAYS_EMPLOYED',
     'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
     'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
     'BURO_CREDIT_TYPE_Real estate loan_MEAN',
     'BURO_CREDIT_TYPE_Car loan_MEAN',
     'BURO_CREDIT_TYPE_nan_MEAN',
     'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
     'BURO_CREDIT_TYPE_Microloan_MEAN',
     'BURO_CREDIT_TYPE_Mortgage_MEAN',
     'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
     'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
     'BURO_CREDIT_TYPE_Loan for business development_MEAN',
     'BURO_CREDIT_TYPE_Interbank credit_MEAN',
     'BURO_CREDIT_TYPE_Credit card_MEAN',
     'BURO_CREDIT_TYPE_Consumer credit_MEAN',
     'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN']
     
    # Sélection des variables retenues
    DF_FE_20_NAN = DF_FE_20_NAN[Liste_Var_Explicatives]
     
    # Sélection d'un échantillon (10% des données d'entrainement)
    DF_FE_20_NAN = DF_FE_20_NAN.sample(frac = 0.1) 
     
    # Imputation des valeurs manquantes (par la médiane)
    median_imputer = SimpleImputer(strategy='median')

    result_median_imputer = median_imputer.fit_transform(DF_FE_20_NAN)

    DF_FE_20_NAN = pd.DataFrame(result_median_imputer, columns=list(DF_FE_20_NAN.columns))
    
    DF_FE_20_NAN = DF_FE_20_NAN.astype(np.int64)
    
    # Export des données après preprocessing
    DF_FE_20_NAN.to_csv("C:/OPCR/PROJET7OPCR/data/traindata.csv", index=False)
    
    with timer("Fin calcul DF_FE_20_NAN"):     
        print("DF_FE_20_NAN shape:",DF_FE_20_NAN.shape)
    # return df
       
if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
        
        
        
