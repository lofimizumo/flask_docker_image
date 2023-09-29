from flask import Flask, request, jsonify
from battery_alloc_test import BatteryScheduler
from threading import Thread


# Initialize the Flask application
app = Flask(__name__)

schedulers = {}


@app.route('/')
def home():
    return "Please use the /predict endpoint to make predictions, see readme for more info"


@app.route('/start', methods=['POST'])
def basic_scheduler():
    data = request.get_json()
    sn = data['deviceSn']
    if sn in schedulers:
        return jsonify(status='error', message='Scheduler already exists for this deviceSn'), 400
    
    scheduler = BatteryScheduler(
        scheduler_type='PeakValley', battery_sn=sn)
    schedulers[sn] = scheduler
    thread = Thread(target=scheduler.start)
    thread.daemon = True  # This will make sure the thread exits when the main program exits
    thread.start()
    return jsonify(status='success', message='Scheduler started'), 200

@app.route('/stop', methods=['POST'])
def stop_scheduler():
    data = request.get_json()
    sn = data['deviceSn']
    
    # Get the scheduler instance from the dictionary and stop it
    scheduler = schedulers.get(sn)
    if scheduler:
        scheduler.stop()
        schedulers.pop(sn)
        return jsonify(status='success', message='Scheduler stopped'), 200
    else:
        return jsonify(status='error', message='Scheduler not found'), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
