from flask import Flask, request, jsonify
from battery_automation import BatteryScheduler
from threading import Thread
# from amber import cost_savings, get_prices


# Initialize the Flask application
app = Flask(__name__)

amber_devices = {}
scheduler_amber = None
scheduler_shawsbay = None
thread_shawsbay = None
scheduler_shawsbay_phase3 = None
thread_shawsbay_phase3 = None
scheduler_shawsbay_phase1 = None
thread_shawsbay_phase1 = None


@app.route('/')
def home():
    return "Please use the /predict endpoint to make predictions, see readme for more info"


@app.route('/start', methods=['POST'])
def basic_scheduler():
    data = request.get_json()
    sn = data['deviceSn']
    global scheduler_amber  # Add global declaration
    if sn in amber_devices:
        return jsonify(status='error', message='Scheduler already exists for this deviceSn'), 400
    if scheduler_amber is None:
        scheduler_amber = BatteryScheduler(
            scheduler_type='PeakValley', battery_sn=sn, api_version='redx')
        thread = Thread(target=scheduler_amber.start)
        thread.daemon = True  # This will make sure the thread exits when the main program exits
        thread.start()
        amber_devices[sn] = scheduler_amber
    return jsonify(status='success', message='Scheduler started'), 200

@app.route('/add_amber_device', methods=['POST'])
def add_amber_device():
    data = request.get_json()
    sn = data['deviceSn']
    if sn in amber_devices:
        return jsonify(status='error', message='Scheduler already exists for this deviceSn'), 400
    if scheduler_amber is None:
        return jsonify(status='error', message='Scheduler not running, call /start to init the Amber scheduler'), 400

    scheduler_amber.add_amber_device(sn)
    amber_devices[sn] = scheduler_amber
    return jsonify(status='success', message='Device {sn} added'), 200

@app.route('/remove_amber_device', methods=['POST'])
def remove_amber_device():
    data = request.get_json()
    sn = data['deviceSn']
    
    if sn in amber_devices:
        amber_devices.pop(sn)
        scheduler_amber.remove_amber_device(sn)
        return jsonify(status='success', message='Device {sn} removed'), 200
    else:
        return jsonify(status='error', message='Device not found'), 404

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

@app.route('/stop_shawsbay_phase3', methods=['POST'])
def stop_ai_scheduler_phase3():
    global scheduler_shawsbay_phase3, thread_shawsbay_phase3

    if scheduler_shawsbay_phase3 and thread_shawsbay_phase3.is_alive():
        scheduler_shawsbay_phase3.stop()
        scheduler_shawsbay_phase3 = None
        return jsonify(status='success', message='Scheduler stopped'), 200
    else:
        return jsonify(status='error', message='Shawsbay Scheduler is not running'), 404

@app.route('/start_shawsbay_phase3', methods=['POST'])
def ai_scheduler_phase3():
    global scheduler_shawsbay_phase3, thread_shawsbay_phase3

    if thread_shawsbay_phase3 and thread_shawsbay_phase3.is_alive():
        return jsonify(status='error', message='Scheduler_Phase3 already running'), 400

    scheduler_shawsbay_phase3 = BatteryScheduler(
        scheduler_type='AIScheduler', 
        battery_sn=['RX2505ACA10J0A170013', 'RX2505ACA10J0A150006', 'RX2505ACA10J0A180002', 'RX2505ACA10J0A170025', 'RX2505ACA10J0A170019','RX2505ACA10J0A150008'], 
        test_mode=False, 
        api_version='redx', 
        pv_sn=['RX2505ACA10J0A180002'],
        phase=3)
    thread_shawsbay_phase3 = Thread(target=scheduler_shawsbay_phase3.start)
    thread_shawsbay_phase3.daemon = True
    thread_shawsbay_phase3.start()

    return jsonify(status='success', message='Shawsbay Scheduler started'), 200

@app.route('/start_shawsbay_phase1', methods=['POST'])
def ai_scheduler_phase1():
    global scheduler_shawsbay_phase1, thread_shawsbay_phase1

    if thread_shawsbay_phase1 and thread_shawsbay_phase1.is_alive():
        return jsonify(status='error', message='Scheduler_Phase1 already running'), 400

    scheduler_shawsbay_phase1 = BatteryScheduler(
        scheduler_type='AIScheduler', 
        battery_sn=['RX2505ACA10J0A150009', 'RX2505ACA10J0A180037', 'RX2505ACA10J0A160039', 'RX2505ACA10J0A160014', 'RX2505ACA10J0A180009'], 
        test_mode=False, 
        api_version='redx', 
        pv_sn=['RX2505ACA10J0A160039'],
        phase=1)
    thread_shawsbay_phase1 = Thread(target=scheduler_shawsbay_phase1.start)
    thread_shawsbay_phase1.daemon = True
    thread_shawsbay_phase1.start()

    return jsonify(status='success', message='Shawsbay_Phase_1 Scheduler started'), 200

@app.route('/stop_shawsbay_phase1', methods=['POST'])
def stop_ai_scheduler_phase1():
    global scheduler_shawsbay_phase1, thread_shawsbay_phase1

    if scheduler_shawsbay_phase1 and thread_shawsbay_phase1.is_alive():
        scheduler_shawsbay_phase1.stop()
        scheduler_shawsbay_phase1 = None
        return jsonify(status='success', message='Scheduler stopped'), 200
    else:
        return jsonify(status='error', message='Shawsbay_Phase_1 Scheduler is not running'), 404

# @app.route('/cost_savings', methods=['POST'])
# def route_cost_savings():
#     data = request.get_json()
#     try:
#         start_date = data['start_date']
#         end_date = data['end_date']
#         amber_key = data['amber_key']
#         sn = data['deviceSn']
#     except KeyError:
#         return jsonify({"error": "Missing keys in request"})
#     return jsonify(cost_savings(start_date, end_date, amber_key, sn))

# @app.route('/prices', methods=['POST'])
# def route_get_prices():
#     try:
#         data = request.get_json()
#         start_date = data['start_date']
#         end_date = data['end_date']
#         amber_key = data['amber_key']
#         return jsonify(get_prices(start_date, end_date, amber_key))
#     except:
#         return jsonify({
#             "error": "bad request"
#         })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

