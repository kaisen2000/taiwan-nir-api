from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse # 🌟 引入快取標頭控制器
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

ALL_REGIONS = {
    "基隆": {"display": "基隆市", "lat": 25.133, "lon": 121.741, "albedo": 0.15},
    "臺北": {"display": "臺北市", "lat": 25.037, "lon": 121.514, "albedo": 0.15},
    "板橋": {"display": "新北市", "lat": 25.014, "lon": 121.462, "albedo": 0.15},
    "桃園": {"display": "桃園市", "lat": 24.993, "lon": 121.301, "albedo": 0.12},
    "新竹": {"display": "新竹縣", "lat": 24.827, "lon": 121.012, "albedo": 0.15},
    "苗栗": {"display": "苗栗縣", "lat": 24.565, "lon": 120.820, "albedo": 0.18},
    "臺中": {"display": "臺中市", "lat": 24.145, "lon": 120.683, "albedo": 0.18},
    "彰化": {"display": "彰化縣", "lat": 24.080, "lon": 120.539, "albedo": 0.20},
    "南投": {"display": "南投縣", "lat": 23.903, "lon": 120.684, "albedo": 0.20},
    "雲林": {"display": "雲林縣", "lat": 23.709, "lon": 120.431, "albedo": 0.20},
    "嘉義": {"display": "嘉義縣", "lat": 23.451, "lon": 120.255, "albedo": 0.20},
    "臺南": {"display": "臺南市", "lat": 22.993, "lon": 120.204, "albedo": 0.20},
    "高雄": {"display": "高雄市", "lat": 22.566, "lon": 120.316, "albedo": 0.18},
    "屏東": {"display": "屏東縣", "lat": 22.669, "lon": 120.486, "albedo": 0.20},
    "宜蘭": {"display": "宜蘭縣", "lat": 24.766, "lon": 121.756, "albedo": 0.15},
    "花蓮": {"display": "花蓮縣", "lat": 23.977, "lon": 121.604, "albedo": 0.15},
    "臺東": {"display": "臺東縣", "lat": 22.752, "lon": 121.144, "albedo": 0.18},
    "澎湖": {"display": "澎湖縣", "lat": 23.565, "lon": 119.563, "albedo": 0.20},
    "金門": {"display": "金門縣", "lat": 24.432, "lon": 118.312, "albedo": 0.20},
    "連江": {"display": "連江縣", "lat": 26.151, "lon": 119.936, "albedo": 0.20}
}

CWA_API_KEY = "CWA-0145ECC9-2CD1-40C0-BC42-C11F38BF7D09"
MOENV_API_KEY = "6eb2e439-39c7-4e22-ae2c-bd1fcff8959b"

CACHE_DATA = None
CACHE_TIME = None

def get_uvi_text(val):
    if val < 3: return "低量級"
    if val < 6: return "中量級"
    if val < 8: return "高量級"
    if val < 11: return "過量級"
    return "危險級"

def get_nir_data():
    global CACHE_DATA, CACHE_TIME
    tw_tz = timezone(timedelta(hours=8))
    now = datetime.now(tw_tz)
    
    if not (5 <= now.hour <= 20):
        return JSONResponse(
            content={"status": "night_mode", "update_time": now.strftime("%Y-%m-%d %H:%M:%S"), "data": []},
            headers={"Cache-Control": "public, max-age=1800"}
        )

    # 伺服器端記憶體快取
    if CACHE_DATA and CACHE_TIME and (now - CACHE_TIME).total_seconds() < 1800:
        return JSONResponse(content=CACHE_DATA, headers={"Cache-Control": "public, max-age=1800"})

    pm25_map = {}
    try:
        res = requests.get(f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}&limit=100&format=JSON", timeout=5, verify=False)
        if res.status_code == 200:
            for r in res.json().get("records", []):
                if r.get("county") and r.get("pm2.5"):
                    c = r.get("county").replace("台", "臺")
                    if c not in pm25_map: pm25_map[c] = []
                    pm25_map[c].append(float(r.get("pm2.5")))
            pm25_map = {c: sum(v)/len(v) for c, v in pm25_map.items()}
    except: pass

    uv_map = {}
    try:
        uv_res = requests.get(f"https://data.moenv.gov.tw/api/v2/uv_s_01?api_key={MOENV_API_KEY}&limit=100&format=JSON", timeout=5, verify=False)
        if uv_res.status_code == 200:
            for r in uv_res.json().get("records", []):
                if r.get("county") and r.get("uvi"):
                    c = r.get("county").replace("台", "臺")
                    try:
                        val = float(r.get("uvi"))
                        if val >= 0:
                            if c not in uv_map: uv_map[c] = []
                            uv_map[c].append(val)
                    except: pass
            uv_map = {c: sum(v)/len(v) for c, v in uv_map.items()}
    except: pass

    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
    try:
        response = requests.get(url, params={"Authorization": CWA_API_KEY, "format": "JSON"}, verify=False)
        stations = response.json().get("records", {}).get("Station", [])
        
        results = []
        for station in stations:
            raw_name = station.get("StationName", "").replace("台", "臺")
            match_key = next((k for k in ALL_REGIONS.keys() if k in raw_name), None)
            if not match_key: continue
            
            cfg = ALL_REGIONS[match_key]
            if any(r['city'] == cfg['display'] for r in results): continue

            elements = station.get("WeatherElement", {})
            
            # 🌟 修正 -99 儀器維護錯誤
            raw_temp = float(elements.get("AirTemperature", -99))
            temp_display = "維護中" if raw_temp <= -50 else round(raw_temp, 1)
            math_temp = 25.0 if raw_temp <= -50 else raw_temp

            humidity = float(elements.get("RelativeHumidity", 80))
            if humidity < 0: humidity = 80.0
            
            weather = elements.get("Weather", "")
            rain = float(elements.get("Now", {}).get("Precipitation", 0.0))

            trans = 1.0
            if rain > 0 or "雨" in weather: trans = 0.25
            elif "陰" in weather: trans = 0.45
            elif "多雲" in weather: trans = 0.75

            pwv = pvlib.atmosphere.gueymard94_pw(math_temp, humidity)
            time_idx = pd.DatetimeIndex([station.get("ObsTime", {}).get("DateTime")])
            solpos = pvlib.solarposition.get_solarposition(time_idx, cfg["lat"], cfg["lon"])
            zenith = solpos['apparent_zenith'].iloc[0]
            
            county_for_uv = cfg["display"]
            uvi_val = uv_map.get(county_for_uv)
            if uvi_val is None:
                uvi_val = max(0, (90 - zenith) / 10 * trans) if zenith < 90 else 0
            
            if zenith > 90: nir = 0.0
            else:
                pm25 = pm25_map.get(county_for_uv, 15.0)
                turb = 0.1 + (pm25 * 0.005)
                spectra = pvlib.spectrum.spectrl2(
                    apparent_zenith=zenith, aoi=zenith, surface_tilt=0, ground_albedo=cfg["albedo"],
                    surface_pressure=101325, relative_airmass=pvlib.atmosphere.get_relative_airmass(zenith),
                    precipitable_water=pwv, ozone=0.34, aerosol_turbidity_500nm=min(max(turb,0.05),0.8),
                    dayofyear=time_idx.dayofyear[0]
                )
                mask = (spectra['wavelength'] >= 700) & (spectra['wavelength'] <= 2500)
                nir = np.trapezoid(spectra['dni'][mask].flatten(), spectra['wavelength'][mask]) * trans
            
            results.append({
                "city": cfg["display"], 
                "temp": temp_display, # 輸出字串 "維護中" 或數字
                "humidity": round(humidity),
                "pwv": round(pwv, 2), 
                "nir": round(nir, 2),
                "uvi": round(uvi_val, 1),
                "uvi_text": get_uvi_text(uvi_val) # 🌟 輸出紫外線等級文字
            })
            
        final_response = {"status": "active", "update_time": now.strftime("%Y-%m-%d %H:%M:%S"), "data": results}
        CACHE_DATA = final_response
        CACHE_TIME = now
        
        # 🌟 回傳並加上 Cache-Control，強制 CDN/瀏覽器快取 30 分鐘
        return JSONResponse(content=final_response, headers={"Cache-Control": "public, max-age=1800"})
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/nir")
def read_nir(): return get_nir_data()