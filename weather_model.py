import pandas as pd
import numpy as np


def weather_processing(flight_data, weather_data):
    dfs = []


    weather_data = weather_data.drop(
        columns=['climo_high_f', 'climo_low_f', 'climo_precip_in', 'snowd_in', 'min_feel', 'avg_feel', 'max_feel'])


    for column in weather_data.columns:  # filling any missing data with column mean
        if column in ["station","day"]:
            continue
        if column in ["snow_in","snowd_in"]:
            weather_data[column] = weather_data[column].replace("-99", 0)

        top_lim = int(np.ceil(weather_data[column].shape[0] / 10))
        sample = weather_data[column][0:top_lim].replace("None", 0).astype(float)
        column_mean = sample.mean()

        weather_data[column] = weather_data[column].replace("None", column_mean)



    origin_airports = flight_data['Origin'].unique()

    for org in origin_airports:  # for each unique origin of a flight the logic is to find all the flight withs same
                                 # origin and same date and update all in onve swipe

        dates = flight_data.loc[flight_data['Origin'] == org]['FlightDate'].unique()

        for date in dates:

            day = date[len(date) - 2:len(date)] + "-" + date[len(date) - 5:len(date) - 2] + date[2:4]  # formating date from yyyy-mm-dd to dd-mm-yy

            origin_weather_info_index = np.where((weather_data['station'] == org) & (weather_data['day'] == day))

            indexes_to_update = np.where((flight_data['Origin'] == org) & (flight_data['FlightDate'] == date))[0].tolist()

            temp_data = arrange_data_in_dict(weather_data.loc[origin_weather_info_index],indexes_to_update)

            df = pd.DataFrame(data=temp_data,index=indexes_to_update)
            dfs.append(df)



    result = pd.concat(dfs, axis=0).sort_index

    return result


def arrange_data_in_dict(data, indexes):
    data = data.drop(columns=["station","day"])
    dict = {}

    for column in data.columns.values:
        if data[column].tolist()[0]:
            enter = [float(data[column].values)] * len(indexes)
        else:
            enter = [25] * len(indexes)
        dict[column] = enter

    return dict



# raw_weather_data = pd.read_csv("C:/IML programing/Hackaton/all_weather_data.csv")
# raw_flight_data = pd.read_csv("C:/IML programing/Hackaton/train_data.csv")
# weather_processing(raw_flight_data, raw_weather_data)
