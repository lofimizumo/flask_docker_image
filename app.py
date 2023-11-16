from flask import Flask, request, jsonify
from battery_alloc_test import BatteryScheduler
from threading import Thread
from amber import cost_savings


# Initialize the Flask application
app = Flask(__name__)

schedulers = {}
scheduler_shawsbay = None
thread_shawsbay = None


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
        scheduler_type='PeakValley', battery_sn=sn, api_version='redx')
    schedulers[sn] = scheduler
    thread = Thread(target=scheduler.start)
    thread.daemon = True  # This will make sure the thread exits when the main program exits
    thread.start()
    return jsonify(status='success', message='Scheduler started'), 200

@app.route('/stop', methods=['POST'])
def stop_basic_scheduler():
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

@app.route('/start_shawsbay', methods=['POST'])
def ai_scheduler():
    global scheduler_shawsbay, thread_shawsbay

    if thread_shawsbay and thread_shawsbay.is_alive():
        return jsonify(status='error', message='Scheduler already running'), 400

    shawsbay_phase2_devices = ['RX2505ACA10J0A180011', 'RX2505ACA10J0A170035', 'RX2505ACA10J0A170033', 'RX2505ACA10J0A160007', 'RX2505ACA10J0A180010'] 
    scheduler_shawsbay = BatteryScheduler(
        scheduler_type='AIScheduler', battery_sn=shawsbay_phase2_devices, test_mode=False, api_version='redx')
    
    thread_shawsbay = Thread(target=scheduler_shawsbay.start)
    thread_shawsbay.daemon = True
    thread_shawsbay.start()

    return jsonify(status='success', message='Shawsbay Scheduler started'), 200

@app.route('/stop_shawsbay', methods=['POST'])
def stop_ai_scheduler():
    global scheduler_shawsbay, thread_shawsbay

    if scheduler_shawsbay and thread_shawsbay.is_alive():
        scheduler_shawsbay.stop()  # Ensure this method stops the thread gracefully
        scheduler_shawsbay = None
        return jsonify(status='success', message='Scheduler stopped'), 200
    else:
        return jsonify(status='error', message='Shawsbay Scheduler is not running'), 404


@app.route('/cost_savings', methods=['POST'])
def route_cost_savings():
    data = request.get_json()
    start_date = data['start_date']
    end_date = data['end_date']
    amber_key = data['amber_key']
    sn = data['deviceSn']
    return jsonify(cost_savings(start_date, end_date, amber_key, sn))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

