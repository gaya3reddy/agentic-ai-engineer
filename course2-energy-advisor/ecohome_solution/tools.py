"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    MAX_DAYS = 5  # OpenWeatherMap free tier limit
    warning = None
    if days > MAX_DAYS:
        warning = f"Requested {days} days but live forecast is only available for up to {MAX_DAYS} days. Returning {MAX_DAYS} days of live data. To get live forecast, request days <= {MAX_DAYS}."
        days = MAX_DAYS

    def map_condition(description: str) -> str:
        description = description.lower()
        if "clear" in description:
            return "sunny"
        elif "few clouds" in description or "scattered clouds" in description:
            return "partly_cloudy"
        else:
            return "cloudy"

    def calc_solar_irradiance(condition: str, hour: int) -> float:
        if hour < 6 or hour > 20:
            return 0.0
        solar_factor = max(0.0, 1 - abs(hour - 12) / 6)
        max_by_condition = {"sunny": 900, "partly_cloudy": 500, "cloudy": 150}
        return round(max_by_condition.get(condition, 0) * solar_factor, 1)

    def mock_forecast(reason: str) -> Dict[str, Any]:
        condition = random.choice(["sunny", "partly_cloudy", "cloudy"])
        current = {"temperature_c": 20.0, "condition": condition, "humidity": 55, "wind_speed": 3.5}
        hourly = [
            {
                "hour": h,
                "temperature_c": round(20.0 + random.uniform(-3, 3), 1),
                "condition": condition,
                "solar_irradiance": calc_solar_irradiance(condition, h),
                "humidity": random.randint(45, 70),
                "wind_speed": round(random.uniform(2, 6), 1)
            }
            for h in range(24)
        ]
        result = {"location": location, "forecast_days": days, "current": current, "hourly": hourly,
                  "fallback": True, "fallback_reason": reason}
        if warning:
            result["warning"] = warning
        return result

    if not api_key:
        return mock_forecast("OPENWEATHER_API_KEY not found in environment, using mock data")

    try:
        forecast_resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"q": location, "appid": api_key, "units": "metric", "cnt": days * 8},
            timeout=10
        )
        if forecast_resp.status_code != 200:
            msg = forecast_resp.json().get("message", "Unknown error")
            return mock_forecast(f"API error ({forecast_resp.status_code}): {msg}")

        items = forecast_resp.json()["list"]
    except requests.RequestException as e:
        return mock_forecast(f"Request failed: {str(e)}")

    first = items[0]
    current_condition = map_condition(first["weather"][0]["description"])
    current = {
        "temperature_c": round(first["main"]["temp"], 1),
        "condition": current_condition,
        "humidity": first["main"]["humidity"],
        "wind_speed": round(first["wind"]["speed"], 1)
    }

    forecast_by_hour = {}
    for item in items:
        hour = datetime.fromtimestamp(item["dt"]).hour
        forecast_by_hour[hour] = {
            "temperature_c": round(item["main"]["temp"], 1),
            "condition": map_condition(item["weather"][0]["description"]),
            "humidity": item["main"]["humidity"],
            "wind_speed": round(item["wind"]["speed"], 1)
        }

    last = {k: current[k] for k in ("temperature_c", "condition", "humidity", "wind_speed")}
    hourly = []
    for hour in range(24):
        if hour in forecast_by_hour:
            last = forecast_by_hour[hour]
        hourly.append({
            "hour": hour,
            "temperature_c": last["temperature_c"],
            "condition": last["condition"],
            "solar_irradiance": calc_solar_irradiance(last["condition"], hour),
            "humidity": last["humidity"],
            "wind_speed": last["wind_speed"]
        })

    result = {"location": location, "forecast_days": days, "current": current, "hourly": hourly}
    if warning:
        result["warning"] = warning
    return result

# TODO: Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates 
        E.g: 
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": f"Invalid date format '{date}'. Expected YYYY-MM-DD (e.g. '2026-04-28')."}

    # Mock electricity pricing - in real implementation, this would call a pricing API
    # Use a base price per kWh
    # Then generate hourly rates with peak/off-peak pricing
    # Peak normally between 6 and 22...
    # demand_charge should be 0 if off-peak
    
    BASE_RATE = 0.12        # off-peak base rate ($/kWh)
    PEAK_RATE = 0.22        # peak rate ($/kWh)
    SHOULDER_RATE = 0.16    # shoulder rate ($/kWh)
    DEMAND_CHARGE = 0.05    # demand charge added during peak hours
    
    def classify_hour(hour: int):
        if 6 <= hour < 9 or 17 <= hour < 21:    # morning & evening peaks
            return "peak", PEAK_RATE, DEMAND_CHARGE
        elif 9 <= hour < 17 or 21 <= hour < 22:  # daytime / early evening shoulder
            return "shoulder", SHOULDER_RATE, 0.0
        else:                                     # overnight off-peak
            return "off_peak", BASE_RATE, 0.0

    hourly_rates = []
    for hour in range(24):
        period, rate, demand_charge = classify_hour(hour)
        hourly_rates.append({
            "hour": hour,
            "rate": rate,
            "period": period,
            "demand_charge": demand_charge
        })

    return {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        persist_directory = "data/vectorstore"
        os.makedirs(persist_directory, exist_ok=True)

        doc_dir = "data/documents"
        manifest_path = os.path.join(persist_directory, "manifest.json")
        current_files = sorted(f for f in os.listdir(doc_dir) if f.endswith(".txt"))

        indexed_files = []
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                indexed_files = json.load(f)

        needs_rebuild = (
            not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))
            or current_files != indexed_files
        )

        embeddings = OpenAIEmbeddings()
        if needs_rebuild:
            documents = []
            for filename in current_files:
                loader = TextLoader(os.path.join(doc_dir, filename))
                documents.extend(loader.load())

            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            with open(manifest_path, "w") as f:
                json.dump(current_files, f)
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
