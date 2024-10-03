import numpy as np
import pandas as pd
# import seaborn as sns
from datetime import datetime
import pickle
import logging

# pd.set_option('display.max_columns', None)

#Create and configure logger
logging.basicConfig(filename='randomforest_forecast_model.log', format='%(asctime)s %(levelname)s:%(message)s', filemode='w')

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)



## import Data  
def import_data():
    profile_data = pd.read_csv(r'D:\KBSL\Data_analytics_projects_kbsl\Seylan\Seylan_Final_Chulax\Final_Model\Data\june\CREDIT_CX_202407221635.csv', 
                            usecols=['CARD_ACCOUNT_NUMBER','CITIZEN_ID','CREDIT_LIMIT_ADJUSTMENT_AMOUNT', 'ACCOUNT_OPEN_DATE',
                                        'DATE_OF_BIRTH', 'CUSTOMER_GENDER'])
    trx_his_data = pd.read_csv(r'D:\KBSL\Data_analytics_projects_kbsl\Seylan\Seylan_Final_Chulax\Final_Model\Data\june\_WITH_TRAX_TABLE_AS_SELECT_CCI_CARD_ACCOUNT_NUMBER_ST_BILLING_DA_202407221638.csv')
    status_data = pd.read_csv(r"D:\KBSL\Data_analytics_projects_kbsl\Seylan\Seylan_Final_Chulax\Final_Model\Data\june\_SELECT_CCI_CARD_ACCOUNT_NUMBER_ST_BILLING_DATE_ST_CARD_ACCOUNT__202407221635.csv")

    # Remove leading and trailing whitespaces from all values in all columns
    profile_data = profile_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print('Profile_table shape',profile_data.shape)

    trx_his_data = trx_his_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print('Trx_table shape',trx_his_data.shape)

    status_data = status_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print('Status_table shape',status_data.shape)

    return profile_data, trx_his_data, status_data



##### Prepare Profile Data
## Derived age from ID number if not proper ID number, get it from 'DATE_OF_BIRTH' column
def calculate_age_from_citizen_id(df):
    current_year = pd.to_datetime('today').year
    
    def calculate_age(row):
        citizen_id = row['CITIZEN_ID']
        
        # Check if 'CITIZEN_ID' length is 10 and matches the specified pattern
        if len(citizen_id) == 10 and (citizen_id.upper().endswith('V') or citizen_id.upper().endswith('X')) and citizen_id[:-1].isdigit():
            birth_year = int(citizen_id[:2]) + 1900
            return current_year - birth_year
        
        # Check if 'CITIZEN_ID' length is 12 and matches the specified pattern
        elif len(citizen_id) == 12 and citizen_id.isdigit():
            birth_year = int(citizen_id[:4])
            return current_year - birth_year
        
        # If 'DATE_OF_BIRTH' is a valid datetime, calculate age from it
        elif pd.notnull(row['DATE_OF_BIRTH']) and pd.to_datetime(row['DATE_OF_BIRTH'], errors='coerce') is not pd.NaT:
            return current_year - pd.to_datetime(row['DATE_OF_BIRTH']).year
        
        # If none of the conditions are met, return NaN for age
        else:
            return np.nan
    
    # Apply the calculate_age function to the DataFrame
    df['AGE'] = df.apply(calculate_age, axis=1)
    
    # Drop rows where 'AGE' is negative
    df = df[df['AGE'] >= 18]
    return df


## Derived gender from ID number if not proper ID number, get it from CUSTOMER_GENDER column
def get_gender_from_citizen_id(df):
    def get_gender(row):
        citizen_id = row['CITIZEN_ID']
        
        # Check if 'CITIZEN_ID' length is 10 and matches the specified pattern
        if len(citizen_id) == 10 and (citizen_id.upper().endswith('V') or citizen_id.upper().endswith('X')) and citizen_id[:-1].isdigit():
            gender = 'Female' if int(citizen_id[2:5]) >= 500 else 'Male'
            return gender
        
        # Check if 'CITIZEN_ID' length is 12 and matches the specified pattern
        elif len(citizen_id) == 12 and citizen_id.isdigit():
            gender = 'Female' if int(citizen_id[4:7]) >= 500 else 'Male'
            return gender
        
        # If none of the conditions are met, get it from CUSTOMER_GENDER
        else:            
            return row['CUSTOMER_GENDER']
    # Apply the calculate_age function to the DataFrame
    df['CUSTOMER_GENDER'] = df.apply(get_gender, axis=1)
    
    #gender_mapping = {'F': 'Female', 'M': 'Male','Female': 'Female','Male':'Male','':None}
    gender_mapping = {'F': 0, 'M':1,'Female':0,'Male':1}
    df2 = df
    #df2['CUSTOMER_GENDER'] = df2['CUSTOMER_GENDER'].map(gender_mapping).fillna('')
    df2['CUSTOMER_GENDER'] = df2['CUSTOMER_GENDER'].map(gender_mapping)
    
    # Drop rows where 'CUSTOMER_GENDER' is null
    df2 = df2.dropna(subset=['CUSTOMER_GENDER'])
    
    return df2


# Genarate number of year stay with bank
def year_of_stay(df):
    # Convert to Datatiem
    df['ACCOUNT_OPEN_DATE'] = pd.to_datetime(df['ACCOUNT_OPEN_DATE'],format='%Y-%m-%d')
    
    # Filterout unrealitics date
    df2 = df[df['ACCOUNT_OPEN_DATE'] != '0001-01-01']
    
    # Get current year
    #current_year = pd.to_datetime('today').year
    today = datetime.today()
    
    # Substract current year - Account open date
    #df2['YEAR_OF_STAY'] = current_year - df2['ACCOUNT_OPEN_DATE'].dt.year
    df2['YEAR_OF_STAY'] = ((today - df2['ACCOUNT_OPEN_DATE']).dt.days) / 365.25
    
    # Filterout who has joined recently (less than 5 months)
    df2 = df2[df2['YEAR_OF_STAY'] > 0.5]
    return df2


def credit_score(df):
    weights = {
    "01": 10,
    "OV": 9,
    "P1": 4,
    "P2": 2,
    "P3": 1,
    "P4": 0.5,
    "PC": 0}

    df['WEIGHT'] = df['CARD_ACCOUNT_STATUS'].map(weights)
    credit_scores = df.groupby('CARD_ACCOUNT_NUMBER')['WEIGHT'].sum().reset_index()
    credit_scores.rename(columns={'WEIGHT':'CREDIT_SCORE'}, inplace=True)

    return credit_scores
    

def profile_table(profile_data,status_data):
    try:
        # try:
        #     _active_cx = active_cx(df)
        #     logger.debug("Sucessfully filtered 01 and OV customers only from profile table. Total records : %d",_active_cx.shape[0])
        # except OSError as e:
        #     logging.error("Failed to filter active customers from profile table.")

        try:
            _age = calculate_age_from_citizen_id(profile_data)
            logger.debug("Sucessfully created age column from profile table. Total records : %d",_age.shape[0])
        except OSError as e:
            logging.error("Failed to create the age column.")
        
        try:
            _gender = get_gender_from_citizen_id(_age)
            logger.debug("Sucessfully created gender column from profile table. Total records : %d",_gender.shape[0])
        except OSError as e:
            logging.error("Failed to create the gender column.")

        try:
            _stay = year_of_stay(_gender)
            logger.debug("Sucessfully created 'year of stay' column from profile table. Total records : %d",_stay.shape[0])
        except OSError as e:
            logging.error("Failed to create the 'year of stay' column.")

        try: 
            _credit_score = credit_score(status_data)
            logger.debug("Sucessfully created credit score column from profile table. Total records : %d",_credit_score.shape[0])
        except OSError as e:
            logging.error("Failed to create the credit score column.")
            
        
        _profile_data = pd.merge(_stay,_credit_score, on='CARD_ACCOUNT_NUMBER', how='inner')

        usecols = ['CARD_ACCOUNT_NUMBER','AGE','YEAR_OF_STAY','CUSTOMER_GENDER','CREDIT_SCORE'] #,'CREDIT_LIMIT_ADJUSTMENT_AMOUNT'
        final_profile_df = _profile_data[usecols]
        logger.debug("Sucessfully created cleaned profile table. Total records : %d",final_profile_df.shape[0])
        
    except OSError as e:
        logging.error("Failed to create the profile data.")
    
    return final_profile_df   

################################### *********************** ########################

####################### Prepare Transcation Table for 1st month ####################
####################################################################################
def _fill_zero_to_numarical_col(df,column):
    df_filled = df.copy()
    df_filled[column] = df_filled[column].fillna(0.0)
    return df_filled


### Select 10 months from transcation table
def select_columns(df, start_month=0, end_month=9):
    # Select numerical columns based on the given range of months
    selected_cols = []
    
    for category in ['TOT_OUTSTAND_BAL', 'TOTAL_PURCHASE', 'TOTAL_PAYMENTS']:
        for month in range(end_month, start_month - 1, -1):
            col_name = f"{category}_{month}"
            selected_cols.append(col_name)
    
    # Include 'CARD_ACCOUNT_NUMBER' in the selected columns
    selected_cols = ['CARD_ACCOUNT_NUMBER'] + selected_cols
    
    # Create a new DataFrame with the selected columns
    selected_df = df[selected_cols]
    return selected_df


def trx_data_prep(df):
    try:
        ### select only last 10 months transcations columns
        selected_df = select_columns(df)
        logger.debug("Sucessfully selected training column from transcation history table. Total records : %d",selected_df.shape[0])
    except OSError as e:
        logging.error("Failed to select transcational columns from  the transcation history table.")
    
    try:
        first_column_name = 'CARD_ACCOUNT_NUMBER'
        rest_columns = selected_df.columns.difference([first_column_name])
        logger.debug("Selected transcations column expect card account number from transcation history table.")

        ### Drop all rows if have null values in all columns
        selected_df = selected_df.dropna(subset=rest_columns, how='all')
        logger.debug("Sucessfully droped if all columns have null values from transcation history table.")
    
        ### Replace zero if have null values in specific columns
        new_trx_his_data_zero = _fill_zero_to_numarical_col(selected_df,rest_columns)
        logger.debug("Sucessfully filled zero if columns have null values from transcation history table.")
    
        ### get only float data type columns
        float_columns = new_trx_his_data_zero.select_dtypes(include=['float64']).columns
        #### make all float column into one decimal number boz some numbers have both one decimal and two decimal value
        ###  Example 0.0 and 0.00
        new_trx_his_data_zero[float_columns] = new_trx_his_data_zero[float_columns].round(1)
        logger.debug("Sucessfully round all numerical values to one decimal from transcation history table.")

    
        ### Drop rows if all columns has zero
        # Find indices of rows to drop
        rows_to_drop = new_trx_his_data_zero[new_trx_his_data_zero[float_columns].eq(0.0).all(axis=1)].index
        # Drop rows from the original dataset
        new_trx_his_data_zero.drop(rows_to_drop, inplace=True)
        
        new_trx_his_data_zero_ = new_trx_his_data_zero.reset_index(drop=True)
        logger.debug("Sucessfully dropped if all numerical values have zero from transcation history table.")
    
    except OSError as e:
        logging.error("Failed to preprocess the transcation history data.")
    
    return new_trx_his_data_zero_


### 1st Month predictions
### split dataset
def batch_1_split_trx_data(df):
    dff = df.copy().set_index('CARD_ACCOUNT_NUMBER')
    
    tot_outstand = dff.iloc[:,0:10]*0.05  ### if you select 11 
    tot_purchase = dff.iloc[:,10:20]
    tot_payment = dff.iloc[:,20:]

    return tot_outstand, tot_purchase, tot_payment


def batch_1_add_std_avg_each_split_tables(df):
    tot_out, tot_pur, tot_pay = batch_1_split_trx_data(df)

    Tot_outstand_table = tot_out.reset_index()
    Tot_purchase_table = tot_pur.reset_index()
    Tot_payment_table = tot_pay.reset_index()
    
    return Tot_outstand_table, Tot_purchase_table, Tot_payment_table


def trans_table_merage(df1,df2,df3):
    merge1 = pd.merge(df1, df2, on='CARD_ACCOUNT_NUMBER', how='inner')
    merge2 = pd.merge(merge1, df3, on='CARD_ACCOUNT_NUMBER', how='inner')
    return merge2


def merge_trx_and_profile_table(df1,df2,df3):
    merge1 = pd.merge(df1, df2, on='CARD_ACCOUNT_NUMBER', how='inner')
    merge2 = pd.merge(merge1, df3, on='CARD_ACCOUNT_NUMBER', how='inner')
    return merge2


#### 2nd Month Transcation data preparaing
### split dataset
def batch_2_split_trx_data(df):
    dff = df.copy().set_index('CARD_ACCOUNT_NUMBER')
    
    tot_outstand = dff.iloc[:,1:10]*0.05  ### if you select 11 
    tot_purchase = dff.iloc[:,11:20]
    tot_payment = dff.iloc[:,21:]

    return tot_outstand, tot_purchase, tot_payment


def batch_2_calcuate_moving_average(table):
    category_cols = [col for col in table.columns]
    table[f"{category_cols[-1]}_MV"] = table[category_cols[-3:]].mean(axis=1)
    return table


def batch_2_add_std_avg_each_split_tables(df):
    tot_out, tot_pur, tot_pay = batch_2_split_trx_data(df)
    tables = [tot_out, tot_pur, tot_pay]

    modified_tables = []

    for i, table in enumerate(tables):
        moving_average_table = batch_2_calcuate_moving_average(table.copy())
        modified_tables.append(moving_average_table)

    Tot_outstand_table = modified_tables[0].reset_index()
    Tot_purchase_table = modified_tables[1].reset_index()
    Tot_payment_table = modified_tables[2].reset_index()

    pd.set_option('display.float_format', '{:.6f}'.format)
    return Tot_outstand_table, Tot_purchase_table, Tot_payment_table


#### 3rd Month Transcation data preparaing
### split dataset
def batch_3_split_trx_data(df):
    dff = df.copy().set_index('CARD_ACCOUNT_NUMBER')
    
    tot_outstand = dff.iloc[:,2:10]*0.05  ### if you select 11 
    tot_purchase = dff.iloc[:,12:20]
    tot_payment = dff.iloc[:,22:]

    return tot_outstand, tot_purchase, tot_payment


def batch_3_calcuate_moving_average(table):
    category_cols = [col for col in table.columns]
    table[f"{category_cols[-1]}_MV"] = table[category_cols[-3:]].mean(axis=1)
    return table


def batch_3_add_std_avg_each_split_tables(df):
    tot_out, tot_pur, tot_pay = batch_3_split_trx_data(df)
    tables = [tot_out, tot_pur, tot_pay]

    modified_tables = []

    for i, table in enumerate(tables):
        moving_average_table_1 = batch_3_calcuate_moving_average(table.copy())
        moving_average_table_2 = batch_3_calcuate_moving_average(moving_average_table_1.copy())
        modified_tables.append(moving_average_table_2)

    Tot_outstand_table = modified_tables[0].reset_index()
    Tot_purchase_table = modified_tables[1].reset_index()
    Tot_payment_table = modified_tables[2].reset_index()

    pd.set_option('display.float_format', '{:.6f}'.format)
    
    return Tot_outstand_table, Tot_purchase_table, Tot_payment_table


def rename_col(df):
    new_columns = {}
    numeric_order = 1
    for column in df.columns:
        new_columns[column] = numeric_order
        numeric_order += 1   
    df.rename(columns=new_columns, inplace=True)
    
    return df


#### Get Active customers
def active_cx(df):
    try:
    ## get only below mentioned customer status
        status_val =['OV','01']
        df = df[df['CARD_ACCOUNT_STATUS'].isin(status_val)].reset_index(drop=True)
        logger.debug("Sucessfully filtered OV and 01 status from status table.")
    except OSError as e:
        logging.error("Failed to filtered OV and 01 status from status table.")
    return df



def fit_model_forecast(df):
    with open('model_and_scaler_aug_looped.pkl', 'rb') as file:
        model_and_scaler = pickle.load(file)

    loaded_model = model_and_scaler['model']
    loaded_scaler = model_and_scaler['scaler']

    X = df.drop(columns=[1])

    X_dataset_scaled = loaded_scaler.transform(X)

    pred_rf = loaded_model.predict(X_dataset_scaled)
    pred_rf_prob = loaded_model.predict_proba(X_dataset_scaled)
    rf_nagative_class_probabilities = pred_rf_prob[:, 1]  ####Get only probability of default customer(1)

    return rf_nagative_class_probabilities



def main():

    try:
        profile_data, trx_his_data, status_data = import_data()
        logger.debug("Reading profile_data data from DB completed. Total records : %d",profile_data.shape[0])
        logger.debug("Reading trx_his_data data from DB completed. Total records : %d",trx_his_data.shape[0])
        logger.debug("Reading status_data data data from DB completed. Total records : %d",status_data.shape[0])
    except OSError as e:
        logging.error("Reading data from DB failed.")

    try:
        new_profile_data = profile_table(profile_data,status_data)
        logger.debug("Creating new profile data completed. Total records : %d",new_profile_data.shape[0])
    except OSError as e:
        logging.error("Failed to create new profile data.")

    try: 
        ##### Get Active customer only
        last_month = status_data[status_data['DIFF_MONTH_1']==0]
        last_month_active = active_cx(last_month)
        active_status = pd.DataFrame(last_month_active['CARD_ACCOUNT_NUMBER'])
    except OSError as e:
        logging.error("Failed to get active customers.")
    
        
    try:
        #FIRST MONTH
        Trx_table_batch_1 = trx_data_prep(trx_his_data)
        tot_outstand_1, tot_purchase_1, tot_payment_1 = batch_1_add_std_avg_each_split_tables(Trx_table_batch_1)
        ### Join trx data with profile Data
        final_trx_data_1 = trans_table_merage(tot_outstand_1, tot_purchase_1, tot_payment_1)
        final_table_batch_1 = merge_trx_and_profile_table(final_trx_data_1,new_profile_data,active_status)
        #final_table_batch_1.to_csv('final_table_batch_1.csv', index=False)

        final_table_batch_1 = rename_col(final_table_batch_1)
        print('final_table_batch_1', final_table_batch_1.shape)
        first_month_prediction = fit_model_forecast(final_table_batch_1)
        logger.debug("Sucessfully predicted for first month.")
    except OSError as e:
        logging.error("Failed predicted for first month.")

    try:
        #SECOND MONTH
        tot_outstand_2, tot_purchase_2, tot_payment_2 = batch_2_add_std_avg_each_split_tables(Trx_table_batch_1)
        ### Join trx data with profile Data
        final_trx_data_2 = trans_table_merage(tot_outstand_2, tot_purchase_2, tot_payment_2)
        final_table_batch_2 = merge_trx_and_profile_table(final_trx_data_2,new_profile_data,active_status)

        final_table_batch_2 = rename_col(final_table_batch_2)
        print('final_table_batch_2', final_table_batch_2.shape)
        second_month_prediction = fit_model_forecast(final_table_batch_2)
        logger.debug("Sucessfully predicted for second month.")
    except OSError as e:
        logging.error("Failed to predicted for second month.")

    try:
        #THIRD MONTH
        tot_outstand_3, tot_purchase_3, tot_payment_3 = batch_3_add_std_avg_each_split_tables(Trx_table_batch_1)
        ### Join trx data with profile Data
        final_trx_data_3 = trans_table_merage(tot_outstand_3, tot_purchase_3, tot_payment_3)
        final_table_batch_3 = merge_trx_and_profile_table(final_trx_data_3,new_profile_data,active_status)

        final_table_batch_3 = rename_col(final_table_batch_3)
        print('final_table_batch_3', final_table_batch_3.shape)
        third_month_prediction = fit_model_forecast(final_table_batch_3)
        logger.debug("Sucessfully predicted for third month.")
    except OSError as e:
        logging.error("Failed to predicted for third month.")

    try:
        #MERGING RESULTS
        df_final = pd.DataFrame(final_table_batch_1[1])
        df_final['PROBABILITY OF DEFAULT FOR FIRST MONTH'] = pd.DataFrame(first_month_prediction)
        df_final['PROBABILITY OF DEFAULT FOR SECOND MONTH'] = pd.DataFrame(second_month_prediction)
        df_final['PROBABILITY OF DEFAULT FOR THIRD MONTH'] = pd.DataFrame(third_month_prediction)
        print("Final table shape:",df_final.shape)
        df_final.rename(columns = {1:"CARD_ACCOUNT_NUMBER"}, inplace = True)
        df_final.to_csv("Prediction_file.csv", index=False)
        logger.debug("Sucessfully created final forecasted data for next three month.Total records : %d",df_final.shape[0])
    except OSError as e:
        logging.error("Failed to create final forecasted data.")

    return df_final




if __name__ =='__main__':
    main()