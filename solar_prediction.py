from typing import TypedDict, Optional
import requests

class WeatherData(TypedDict, total=False):
    list: 'CurrentWeatherData'

class CurrentWeatherData(TypedDict, total=False):
    main: Optional[dict]
    rain: Optional[list]
    clouds: Optional[dict]

class WeatherInfoFetcher:
    def __init__(self, location: str):
        self.api_key = '1a1a182a6a5535477288d73d2bf20a55'
        lat = self._get_lat_lon(location)[0]
        lon = self._get_lat_lon(location)[1]
        self.base_url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&appid={self.api_key}"
    
    def _get_lat_lon(self, location: str) -> (float, float):
        '''
        Returns the latitude and longitude of the given location.
        TODO: Return the actual lat and lon of the given location. Currently returning Shaws Bay.
        '''
        lat_shawsbay = 28.86714
        lon_shawsbay = 153.582824 
        return (lat_shawsbay, lon_shawsbay)
    
    def get_response(self) -> WeatherData:
        """
        Fetches weather data for the given location.
        """
        response = requests.get(self.base_url)
        response.raise_for_status()
        return response.json()  

    def get_rain_cloud_forecast_24h(self, weather_data: WeatherData) -> CurrentWeatherData: 
        """
        Safely retrieves rain information from weather data.
        """
        def _is_time_between_8_and_16(timestamp: int) -> bool:
            from datetime import datetime, time
            time_of_day = datetime.fromtimestamp(timestamp).time()
            start_time = time(8, 0)
            end_time = time(16, 0)
            return start_time <= time_of_day <= end_time

        weather_24hours = weather_data.get('list')[:24]
        if weather_24hours:
            weather_8_to_16 = [item for item in weather_24hours if _is_time_between_8_and_16(item['dt'])]
            return self.calculate_averages(weather_8_to_16)
        else:
            default_rain = {
                'rain': 0,
                'clouds': 0
            }
            return default_rain


    def calculate_averages(self, weather_data: WeatherData) -> CurrentWeatherData:
        total_rain = 0
        total_clouds = 0
        rain_count = 0
        cloud_count = 0
        
        for item in weather_data:
            if 'rain' in item and '1h' in item['rain']:
                total_rain += item['rain']['1h']
                rain_count += 1
            if 'clouds' in item and 'all' in item['clouds']:
                total_clouds += item['clouds']['all']
                cloud_count += 1
        
        # Calculate averages
        average_rain = total_rain / rain_count if rain_count > 0 else 0
        average_clouds = total_clouds / cloud_count if cloud_count > 0 else 0
        
        return {
            'rain': average_rain,
            'clouds': average_clouds
        }
