import pandas as pd

def build_input_df(
    day, weather, road_type, speed_limit, light, junction
):
    return pd.DataFrame([{
        "Day_of_Week": day,
        "Weather_Conditions": weather,
        "Road_Type": road_type,
        "Speed_limit": speed_limit,
        "Light_Conditions": light,
        "Junction_Detail": junction
    }])

def preprocess_for_ml(input_df, preprocessor):
    return preprocessor.transform(input_df)

def preprocess_for_dl(input_df, preprocessor, svd, scaler):
    processed = preprocessor.transform(input_df)
    reduced = svd.transform(processed)
    scaled = scaler.transform(reduced)
    return scaled
