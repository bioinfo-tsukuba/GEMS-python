
from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import polars as pl

def update_time_file(time: int, overwrite: bool = False):
    """
    time: int
    overwrite: bool

    unit is seconds
    if overwrite is True, overwrite the current time
    otherwise, add time to the current time
    """

    current_time = get_current_time_from_simulation_file()
    if overwrite:
        current_time = time
    else:
        current_time = current_time.timestamp() + time

    load_dotenv(override=True)
    path = os.environ['TIME_PATH']

    # write as datetime (astimezone), as YYYYMMDDHHMMSS
    current_time = datetime.fromtimestamp(current_time).astimezone()
    current_time = current_time.strftime("%Y%m%d%H%M%S")
    with open(path, "w") as f:
        f.write(current_time)

    return current_time


def get_current_time_from_simulation_file()->datetime:
    load_dotenv(override=True)
    path = os.environ['TIME_PATH']
    
    # YYYYMMDDHHMMSS
    with open(path, "r") as f:
        current_time = f.read()

    # read as datetime (astimezone)
    current_time = datetime.strptime(current_time, "%Y%m%d%H%M%S").astimezone()
    return current_time


def get_current_time_from_simulation_file_as_int()->int:
    load_dotenv(override=True)
    time_unit = os.environ['TIME_UNIT']

    current_time = get_current_time_from_simulation_file()
    current_time = current_time.timestamp()
    
    if time_unit == "s":
        return int(current_time)
    elif time_unit == "m":
        return int(current_time / 60)
    elif time_unit == "h":
        return int(current_time / 60 / 60)
    elif time_unit == "D":
        return int(current_time / 60 / 60 / 24)
    elif time_unit == "M":
        return int(current_time / 60 / 60 / 24 / 30)
    elif time_unit == "Y":
        return int(current_time / 60 / 60 / 24 / 365)
    else:
        raise ValueError(f"Invalid time unit: {time_unit}")

    


def send_schedule_to_simulator_and_get_result(task_path: Path) -> (int, pl.DataFrame):
    # Send schedule to simulator
    with open(task_path) as f:
        task_json = json.load(f)

    print(task_json)
        # 各要素を辞書としてパース
    parsed_data = [json.loads(item) for item in task_json]

    # scheduled_timeが最も小さい要素を取得
    min_scheduled_item = min(parsed_data, key=lambda x: x['scheduled_time'])


    scheduled_time = min_scheduled_item["scheduled_time"]
    scheduled_time_seconds = scheduled_time*60
    scheduled_time = datetime.fromtimestamp(scheduled_time_seconds).astimezone()
    processing_time = min_scheduled_item["processing_time"]
    experiment_operation = min_scheduled_item["experiment_operation"]
    experiment_uuid = min_scheduled_item["experiment_uuid"]
    task_id = min_scheduled_item["task_id"]

    print(f"Do {experiment_operation} at {scheduled_time} for {processing_time} minutes, experiment_uuid: {experiment_uuid}")

    print(f"{scheduled_time.strftime("%Y%m%d%H%M%S")=}")

    # Get result from simulator
    api = CellAPIClient("http://localhost:8000")
    density = None
    real_density = api.get_density(experiment_uuid, scheduled_time.strftime("%Y%m%d%H%M%S"))
    if experiment_operation == "GetImage":
        density = api.get_density(experiment_uuid, scheduled_time.strftime("%Y%m%d%H%M%S"))
        density = density["density"]

    if experiment_operation == "Passage":
        api.update_cell(experiment_uuid, 
                        reference_time=scheduled_time.strftime("%Y%m%d%H%M%S")
                        )
        
        
    # Update time file
    update_time_file(processing_time*60+scheduled_time_seconds, overwrite=True)

    df = pl.DataFrame({
        "time": [int(scheduled_time_seconds/60)],
        "density": [density],
        "operation": [experiment_operation],
        "real_density": [real_density["density"]]
    })

    return task_id, df




import requests
from datetime import datetime
from typing import Optional

class CellAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def create_cell(self, n0: float, k: float, r: float, reference_time: str, memo: Optional[str] = None):
        url = f"{self.base_url}/create/"
        payload = {
            "n0": n0,
            "k": k,
            "r": r,
            "reference_time": reference_time,
            "memo": memo
        }
        response = requests.post(url, json=payload)
        return self._handle_response(response)

    def get_density(self, uuid: str, t: str):
        url = f"{self.base_url}/density/"
        params = {
            "uuid": uuid,
            "t": t
        }
        response = requests.get(url, params=params)
        return self._handle_response(response)

    def update_cell(self, uuid: str, n0: Optional[float] = None, k: Optional[float] = None, r: Optional[float] = None, reference_time: Optional[str] = None):
        url = f"{self.base_url}/update/"
        payload = {
            "n0": n0,
            "k": k,
            "r": r,
            "reference_time": reference_time
        }
        response = requests.post(url, json=payload, params={"uuid": uuid})
        return self._handle_response(response)

    def delete_cell(self, uuid: str):
        url = f"{self.base_url}/delete/{uuid}"
        response = requests.delete(url)
        return self._handle_response(response)

    def get_uuids(self):
        url = f"{self.base_url}/uuids/"
        response = requests.get(url)
        return self._handle_response(response)

    def export_csv(self, path: str):
        url = f"{self.base_url}/export_csv/"
        payload = {"path": path}
        response = requests.post(url, json=payload)
        return self._handle_response(response)

    def _handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            response.raise_for_status()
