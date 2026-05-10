from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import pvlib
import numpy as np
from datetime import datetime, timezone, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ 修正：測站名稱必須精準對應氣象署（不能加市），但我們另外設定了 display 名稱給網頁顯示用
SIX_CITIES = {
    "臺北": {"display": "臺北市", "lat": 25.037, "lon": 121.514},
    "板橋": {"display": "新北市", "lat": 25.014, "lon": 121.462},
    "桃園": {"display": "桃園市", "lat": 24.993, "lon": 121.301},
    "臺中": {"display": "臺中市", "lat": 24.145, "lon": 120.683},
    "臺南": {"display": "臺南市", "lat": 22.993, "lon": 120.204},
    "高雄": {"display": "高雄市", "lat": 22.566, "lon": 120.316}
}

CWA_API_KEY = "CWA-0145ECC9-2CD1-40C0-BC42-C11F38BF7D09"

def get_nir_data():
    # 🌟 節能優化：設定台灣時區 (UTC+8) 並取得現在時間與小時
    tw_tz = timezone(timedelta(hours=8))
    now = datetime.now(tw_tz)
    current_hour = now.hour

    # 🌟 節能優化：判斷是否為夜間休眠時段 (晚上 21:00 到凌晨 04:59 不進行運算)
    if not (5 <= current_hour <= 20):
        return {
            "status": "night_mode",
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "data": [],
            "message": "目前為夜間休眠時段，API 停止運算以節省資源。"
        }

    # --- 以下為原本的日照運算邏輯 (05:00 - 20:00 才會走到這裡) ---
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
    station_names = ",".join(SIX_CITIES.keys())
    params = {"Authorization": CWA_API_KEY, "format": "JSON", "StationName": station_names}
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        stations = response.json().get("records", {}).get("Station", [])
        
        results = []
        for station in stations:
            st_name = station.get("StationName")
            if st_name not in SIX_CITIES: continue
            
            elements = station.get("WeatherElement", {})
            temp = float(elements.get("AirTemperature", 0))
            humidity = float(elements.get("RelativeHumidity", 0))
            pressure = elements.get("StationPressure")
            pressure = 1013.25 if pressure is None or float(pressure) < 0 else float(pressure)
            obs_time = station.get("ObsTime", {}).get("DateTime")
            
            lat, lon = SIX_CITIES[st_name]["lat"], SIX_CITIES[st_name]["lon"]
            pwv = pvlib.atmosphere.gueymard94_pw(temp, humidity)
            
            time_index = pd.DatetimeIndex([obs_time])
            solpos = pvlib.solarposition.get_solarposition(time_index, lat, lon)
            zenith = solpos['apparent_zenith'].iloc[0]
            doy = time_index.dayofyear[0] # 算出今天是今年的第幾天
            
            if zenith > 90:
                nir_total_w_m2 = 0.0
            else:
                airmass = pvlib.atmosphere.get_relative_airmass(zenith)
                spectra = pvlib.spectrum.spectrl2(
                    apparent_zenith=zenith, aoi=zenith, surface_tilt=0, ground_albedo=0.2,
                    surface_pressure=pressure * 100, relative_airmass=airmass, precipitable_water=pwv,
                    ozone=0.34, aerosol_turbidity_500nm=0.1,
                    dayofyear=doy # 把天數餵給光譜模型
                )
                mask = (spectra['wavelength'] >= 700) & (spectra['wavelength'] <= 2500)
                nir_total_w_m2 = np.trapezoid(spectra['dni'][mask].flatten(), spectra['wavelength'][mask])
            
            results.append({
                "city": SIX_CITIES[st_name]["display"], # 傳送給網頁時，換回漂亮的「臺北市」
                "temp": temp,
                "humidity": humidity,
                "pwv": round(pwv, 2),
                "nir": round(nir_total_w_m2, 2)
            })
            
        return {
            "status": "active",
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"), 
            "data": results
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/nir")
def read_nir():
    return get_nir_data()