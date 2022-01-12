
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2011-01-01', end='2019-12-31').to_pydatetime()


def all_data_pre_processing(df):
    df = add_columns(df)
    dum = pd.get_dummies(df, columns=['DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime',
                                      'Year', 'Month'], drop_first=True)
    dum = dum.drop(['ArrDelay', 'Holiday', 'Distance',
                    'CRSElapsedTime', 'DelayFactor'], axis=1)
    # TODO check that all columes we expect are here in test data pre processing

    df = pd.concat([df, dum], axis=1)
    df = remove_columns(df)
    All_cols = sorted(df.columns.tolist())
    return All_cols



def pre_process(df,required_columns):
    """
    :param df: original raw data frame
    This is The main function for the pre processing of our flight data
    :return: df : processed data_frame ready to be used for regression and classification
    """
    df = add_columns(df)
    dum = pd.get_dummies(df, columns=['DayOfWeek','Reporting_Airline','Origin','Dest','CRSDepTime','CRSArrTime',
                                      'Year','Month'],drop_first=True)
    dum =dum.drop(['ArrDelay','Holiday','Distance',
             'CRSElapsedTime','DelayFactor'], axis=1)
    # TODO check that all columes we expect are here in test data pre processing

    df = pd.concat([df, dum], axis=1)
    df = remove_columns(df)
    fix_cols(df,required_columns)
    df.columns=sorted(df.columns.tolist())
    return df


def fix_cols(df,required_columns):
    """
    This function adds missing feature columns
    :param df:
    :param required_columns: all d features
    :return:
    """
    #print("number of required_columns is : ", len(required_columns))
    #print("number of columns is : ", len(df.columns))

    l=[x for x in required_columns if x not in df.columns]
    for col in l:
        #print(col)
        df[col] = 0
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    #print("number of columns is : ",len(df.columns))
    return df




def convert_to_time(x):

    x=str(x)[:-2]
    if len(x)<4:
        y=('0'*(4-len(x)))+x
        x = y
    if x=='2400':
        x='0000'

    return x

def convert_date(str):
    ret = ''
    if len(str)<8:
        j=0

        for i in str:
            if i=='-' and (j!=2 or j!=4):
                ret=ret+'0'
                j=j+1

            j=j+1
            ret = ret + i
        str=ret
    return str


def get_part_of_day(hour):
    return (
        "morning" if 5 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 17
        else
        "evening" if 18 <= hour <= 22
        else
        "night"
    )

def add_columns(df):


    df['CRSDepTime'] = df['CRSDepTime'].apply(
        lambda x: 'Departure in The ' + get_part_of_day(datetime.strptime(convert_to_time(x), '%H%M').hour))
    df['CRSArrTime'] = df['CRSArrTime'].apply(
        lambda y: 'Arrival in The ' + get_part_of_day(datetime.strptime(convert_to_time(y), '%H%M').hour))

    df['Year'] = df['FlightDate'].apply(lambda y: datetime.strptime(convert_date(str(y)), '%Y-%m-%d').year)
    df['Month'] = df['FlightDate'].apply(lambda y: datetime.strptime(convert_date(str(y)), '%Y-%m-%d').month)
    df['Holiday'] = df['FlightDate'].apply(lambda y: datetime.strptime(convert_date(str(y)), '%Y-%m-%d') in holidays)
    df['Holiday']=df['Holiday'].replace(True, 1, regex=True)
    df['Holiday']=df['Holiday'].replace(False,0, regex=True)

    df['DelayFactor'] = df['DelayFactor'].replace(np.nan,0, regex=True)
    df['DelayFactor'] = df['DelayFactor'].replace('NASDelay', 1, regex=True)
    df['DelayFactor'] = df['DelayFactor'].replace('WeatherDelay', 2, regex=True)
    df['DelayFactor'] = df['DelayFactor'].replace('LateAircraftDelay', 3, regex=True)
    df['DelayFactor'] = df['DelayFactor'].replace('CarrierDelay', 4, regex=True)

    return df

def remove_columns(df):
    """Here we remove columns t"""
    df= df.drop(['FlightDate','Tail_Number','OriginCityName','Flight_Number_Reporting_Airline',
             'OriginState','DestState','DestCityName','DayOfWeek','Reporting_Airline','Origin','Dest','CRSDepTime',
                 'Month','Year','CRSArrTime'], axis=1)
    return df


