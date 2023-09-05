from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from darts.models import NHiTSModel
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pandas.tseries.offsets import DateOffset
import pickle
import wandb

# Initialize the Flask application
app = Flask(__name__)

# Load the model
model = NHiTSModel.load_from_checkpoint(
    model_name='nh_v4', map_location='cpu', best=True)
model.to_cpu()

wandb.init(mode="disabled")


def load_scaler(meter_type = 'total'):
    if meter_type == 'phase1':
        filename = 'scaler_phase1.pkl'
    elif meter_type == 'total':
        filename = 'scaler_total.pkl'
    with open(filename, 'rb') as f:
        transformer = pickle.load(f)
        return transformer


def create_time_series(start_time, interval, values):
    freq_map = {'5min': DateOffset(minutes=5), '1H': DateOffset(
        hours=1), '1D': DateOffset(days=1)}
    start_time = pd.Timestamp(start_time)
    time_index = pd.date_range(
        start=start_time, periods=len(values), freq=freq_map[interval])
    return TimeSeries.from_times_and_values(time_index, values).astype(np.float32)

def scale(ts, desired_increase):
    
    time_index = ts.time_index
    data=ts.all_values()
    # Compute the min and max for each row
    min_per_row = np.min(data, axis=2, keepdims=True)
    max_per_row = np.max(data, axis=2, keepdims=True)

    # Compute the original bandwidth for each row
    original_bandwidth = max_per_row - min_per_row

    # Adjust the min and max values for each row
    adjusted_min = min_per_row - desired_increase / 2
    adjusted_max = max_per_row + desired_increase / 2

    # Scale the data to the adjusted bandwidth
    scaled_data = (data - min_per_row) / original_bandwidth * (adjusted_max - adjusted_min) + adjusted_min

    return TimeSeries.from_times_and_values(time_index,scaled_data)

@app.route('/')
def home():
    return "Please use the /predict endpoint to make predictions, see readme for more info"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Create Darts TimeSeries instances
    current_day_series = create_time_series(
        data['current_day']['start_time'], data['current_day']['interval'], data['current_day']['values'])
    one_week_ago_series = create_time_series(
        data['current_day']['start_time'], data['current_day']['interval'], data['one_week_ago']['values'])

    # Divede the value of the series by 3000 as the data is measured in kW and a freq on 5min
    # The model was trained on data with a freq of 15m and values in MW
    current_day_series = current_day_series.map(lambda x: x/3000)
    one_week_ago_series = one_week_ago_series.map(lambda x: x/3000)

    # Load the scaler
    transformer = load_scaler()

    # Standardize the series
    current_day_series = transformer.transform(current_day_series)
    one_week_ago_series = transformer.transform(one_week_ago_series)

    # Make prediction using model
    prediction = model.predict(len(current_day_series), series=current_day_series, past_covariates=one_week_ago_series, mc_dropout=True,
                               num_samples=50)

    # Align the mean and the std of the prediction with the current day series
    mean_prev = current_day_series.mean().mean(axis=0).values()  # Calculate the difference in means
    mean_today = prediction.mean().mean(axis=0).values()
    std_prev = current_day_series.values().std()
    std_today = prediction.values().std()

    def shift_by_mean_diff(value):
        return ((value - mean_today) / std_today) * std_prev + mean_prev
    
    # Shift the prediction by the difference in means and std
    prediction = prediction.map(shift_by_mean_diff)

    # Scale the confidence range to the desired increase
    prediction = scale(prediction, 2)
    
    # Inverse transform the prediction and multiply by 3000
    prediction = transformer.inverse_transform(prediction)
    prediction = prediction.map(lambda x: x*3000)


    # return only the upper and lower confidence interval, remove the last value as it is the prediction for the next timestep

    confidence_upper = np.percentile(prediction.all_values(),95,axis=2).reshape(-1).tolist()
    confidence_lower = np.percentile(prediction.all_values(),5,axis=2).reshape(-1).tolist()
    return jsonify({'confidence_upper': confidence_upper, 'confidence_lower': confidence_lower})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
