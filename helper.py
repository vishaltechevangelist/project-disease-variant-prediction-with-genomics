import numpy as np
import json, os

def df_to_json(df):
    json_str = {}
    features = df.columns
    # print(df.info())
    for column in features:
        if df[column].dtype in [np.float16, np.float32, np.float64]:
            json_str[column] = float(df[column][0])
        elif df[column].dtype in [np.int16, np.int32, np.int64]:
            json_str[column] = int(df[column][0])
        elif df[column][0] == np.True_ or df[column][0] == np.False_:
            json_str[column] = bool(df[column][0])
        elif df[column].dtype in [object]:
            json_str[column] = str(df[column][0])
    return json_str

def format_output(display_output):
    return f'<div style="background-color:#2b6cb0;color:white;border-left:6px solid #2196F3;padding:10px 16px;border-radius:8px;">{display_output}</div>'

def load_json_for_dd(path, filename):
    with open(os.path.join(path, filename), 'r') as f:
        return json.load(f)