from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from battery_automation import BatteryScheduler
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse

app = FastAPI()
executor = ThreadPoolExecutor()

class SchedulerRequest(BaseModel):
    deviceSn: list = Field(..., description="List of device serial numbers")
    schedulerType: str = Field(..., description="Type of the scheduler (e.g., 'AIScheduler', 'PeakValley')")
    testMode: bool = Field(False, description="Flag to indicate if the scheduler is running in test mode")
    apiVersion: str = Field('redx', description="Version of the API")
    pvSn: list = Field(None, alias='PV_SN', description="List of PV (Photo-Voltaic) device serial numbers")
    phase: int = Field(None, description="Phase number of the scheduler")

    @validator('pvSn', 'testMode', always=True)
    def validate_pv_sn_and_test_mode(cls, v, values):
        if values.get('schedulerType', None) == 'AIScheduler':
            if v is None:
                raise ValueError(f"{v} is required when schedulerType is AIScheduler")
        return v
    
    @validator('schedulerType')
    def validate_scheduler_type(cls, v):
        allowed_schedulers = ['AIScheduler', 'PeakValley']
        if v not in allowed_schedulers:
            raise ValueError('unknown')
        return v

schedulers = {}

@app.post("/scheduler")
def start_scheduler(request: SchedulerRequest):
    scheduler_id = f"{request.schedulerType}"
    if scheduler_id in schedulers:
        raise HTTPException(status_code=400, detail=f"Scheduler {scheduler_id} is already running")

    scheduler = BatteryScheduler(
        scheduler_type=request.schedulerType,
        battery_sn=request.deviceSn,
        test_mode=request.testMode,
        api_version=request.apiVersion,
        pv_sn=request.pvSn,
        phase=request.phase
    )
    future = executor.submit(scheduler.start)
    schedulers[scheduler_id] = (scheduler, future)

    return {"status": "success", "message": f"Scheduler {scheduler_id} started"}

@app.post("/stop_scheduler")
def stop_scheduler(request: SchedulerRequest):
    scheduler_id = f"{request.schedulerType}"
    if scheduler_id not in schedulers:
        raise HTTPException(status_code=404, detail=f"Scheduler {scheduler_id} is not running")

    scheduler, future = schedulers[scheduler_id]
    scheduler.stop()
    future.cancel()
    del schedulers[scheduler_id]

    return {"status": "success", "message": f"Scheduler {scheduler_id} stopped"}

@app.post("/add_device")
def add_device(request: SchedulerRequest):
    scheduler_id = f"{request.schedulerType}"
    if scheduler_id not in schedulers:
        raise HTTPException(status_code=404, detail=f"Scheduler {scheduler_id} is not running")

    scheduler, _ = schedulers[scheduler_id]
    scheduler.add_amber_device(request.deviceSn)

    return {"status": "success", "message": f"Device {request.deviceSn} added to scheduler {scheduler_id}"}

@app.post("/remove_device")
def remove_device(request: SchedulerRequest):
    scheduler_id = f"{request.schedulerType}"
    if scheduler_id not in schedulers:
        raise HTTPException(status_code=404, detail=f"Scheduler {scheduler_id} is not running")

    scheduler, _ = schedulers[scheduler_id]
    scheduler.remove_amber_device(request.deviceSn)

    return {"status": "success", "message": f"Device {request.deviceSn} removed from scheduler {scheduler_id}"}

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"schedulerType": str(exc)})