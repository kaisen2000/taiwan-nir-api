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

# 🌟 優化 1：加入各地精細化的地表反射率 (albedo)
SIX_CITIES = {
    "臺北": {"display": "臺北市", "lat": 25.037, "lon": 121.514, "albedo": 0.15},
    "板橋": {"display": "新北市", "lat": 25.014, "lon": 121.462, "albedo": 0.15},
    "桃園": {"display": "桃園市", "lat": 24.993, "lon": 121.301, "albedo": 0.12}, # 埤塘、水體較多，反射率低
    "臺中": {"display": "臺中市", "lat": 24.145, "lon": 120.683, "albedo": 0.18},
    "臺南": {"display": "臺南市", "lat": 22.993, "lon": 120.204, "albedo": 0.20}, # 空曠、農業地較多，反射率高
    "高雄": {"display": "高雄市", "lat": 22.566, "lon": 120.316, "albedo": 0.18}
}

CWA_API_KEY = "CWA-0145ECC9-2CD1-40C0-BC42-C11F38BF7D09"
MOENV_API_KEY = "6eb2e439-39c7-4e22-ae2c-bd1fcff8959b" # 王sir申請的環境部金鑰

def get_nir_data():
    tw_tz = timezone(timedelta(hours=8))
    now = datetime.now(tw_tz)
    current_hour = now.hour

    # 夜間節能模式 (21:00 到凌晨 04:59 不運算)
    if not (5 <= current_hour <= 20):
        return {
            "status": "night_mode",
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "data": [],
            "message": "目前為夜間休眠時段，API 停止運算以節省資源。"
        }

    # 🌟 優化 2：向環境部索取即時 PM2.5 數據，用於計算大氣混濁度
    pm25_data = {}
    try:
        moenv_url = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=100&format=JSON"
        moenv_res = requests.get(moenv_url, verify=False, timeout=5)
        if moenv_res.status_code == 200:
            for r in moenv_res.json().get("records", []):
                county = r.get("county")
                pm25_val = r.get("pm2.5")
                if county and pm25_val:
                    try:
                        if county not in pm25_data: pm25_data[county] = []
                        pm25_data[county].append(float(pm25_val))
                    except ValueError: pass
            # 計算各縣市平均 PM2.5
            for c in pm25_data: pm25_data[c] = sum(pm25_data[c]) / len(pm25_data[c])
    except Exception as e:
        print("環境部 API 連線失敗，使用預設空氣品質。")

    # --- 向氣象署索取即時天氣 ---
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
            
            city_display = SIX_CITIES[st_name]["display"]
            elements = station.get("WeatherElement", {})
            
            # 基礎氣象數據
            temp = float(elements.get("AirTemperature", 0))
            humidity = float(elements.get("RelativeHumidity", 0))
            pressure = elements.get("StationPressure")
            pressure = 1013.25 if pressure is None or float(pressure) < 0 else float(pressure)
            obs_time = station.get("ObsTime", {}).get("DateTime")
            
            # 🌟 優化 3：萃取雲量與降雨特徵
            weather_desc = elements.get("Weather", "")
            rain_val = elements.get("Now", {}).get("Precipitation", 0.0)
            try: rain = float(rain_val)
            except: rain = 0.0

            # 建立透光率折算模型 (1.0為完全無雲)
            cloud_transmissivity = 1.0
            if rain > 0 or "雨" in weather_desc:
                cloud_transmissivity = 0.2
            elif "陰" in weather_desc:
                cloud_transmissivity = 0.4
            elif "多雲" in weather_desc:
                cloud_transmissivity = 0.7
            
            # 物理模型計算
            lat, lon, albedo = SIX_CITIES[st_name]["lat"], SIX_CITIES[st_name]["lon"], SIX_CITIES[st_name]["albedo"]
            pwv = pvlib.atmosphere.gueymard94_pw(temp, humidity)
            
            time_index = pd.DatetimeIndex([obs_time])
            solpos = pvlib.solarposition.get_solarposition(time_index, lat, lon)
            zenith = solpos['apparent_zenith'].iloc[0]
            doy = time_index.dayofyear[0]
            
            if zenith > 90:
                nir_total_w_m2 = 0.0
            else:
                # 換算 PM2.5 為大氣混濁係數 (Turbidity)
                local_pm25 = pm25_data.get(city_display, 15.0) # 預設PM2.5為15
                turbidity = 0.1 + (local_pm25 * 0.005)
                turbidity = min(max(turbidity, 0.05), 0.8) # 限制合理範圍
                
                airmass = pvlib.atmosphere.get_relative_airmass(zenith)
                spectra = pvlib.spectrum.spectrl2(
                    apparent_zenith=zenith, aoi=zenith, surface_tilt=0, 
                    ground_albedo=albedo, # 引入專屬地表反射率
                    surface_pressure=pressure * 100, relative_airmass=airmass, precipitable_water=pwv,
                    ozone=0.34, 
                    aerosol_turbidity_500nm=turbidity, # 引入真實空污數據
                    dayofyear=doy
                )
                mask = (spectra['wavelength'] >= 700) & (spectra['wavelength'] <= 2500)
                # 結合雲層與降雨的透光率衰減
                raw_nir = np.trapezoid(spectra['dni'][mask].flatten(), spectra['wavelength'][mask])
                nir_total_w_m2 = raw_nir * cloud_transmissivity
            
            results.append({
                "city": city_display,
                "temp": temp,
                "humidity": humidity,
                "pwv": round(pwv, 2),
                "nir": round(nir_total_w_m2, 2),
                "debug_info": f"PM2.5={round(pm25_data.get(city_display, 15),1)}, 天氣={weather_desc}, 透光率={cloud_transmissivity}" # 藏在背後供您檢查用的數據
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