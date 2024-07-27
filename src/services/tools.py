import requests, json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# def get_weather(city_name,):
#     api_key = "c73551a3c8ce8360ff1001fac33f3193"
#     base_url = "http://api.openweathermap.org/data/2.5/onecall?"
#     complete_url = base_url + "appid=" + api_key + "&lat=21.8537&lon=106.7615"
#     response = requests.get(complete_url)
#     ret = response.json()
#     print(ret)
#     if ret["cod"] != "404":
#         weather_des = ret["weather"][0]["description"]
#         temp = ret["main"]["temp"]
#         pressure = ret["main"]["pressure"]
#         humidity = ret["main"]["humidity"] 
#     else:
#         return "Cannot return the information of weather because of missing connection to weather tool"
    
def get_weather(location):
    weather_url = "https://www.google.com/search?q=" + "weather " + location + " capital"
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(options=chrome_options)
    driver.minimize_window()
    driver.get(weather_url)
    access_link_button = driver.find_element(By.PARTIAL_LINK_TEXT, 'https://www.accuweather.com')
    access_link_button.click()
    type_time_button = driver.find_element(By.XPATH, "//a[@data-qa='now']")
    type_time_button.click()
    
    page = BeautifulSoup(driver.page_source, features="html.parser")
    
    div_air_category = page.find_all("span", {"class": "air-quality-module__row__category"})
    div_air_state = page.find_all("p", {"class": "air-quality-module__statement"})
    if div_air_category and div_air_state:
        air_quality_des = f"The air quality is {div_air_category[0].get_text()}: {div_air_state[0].get_text().strip()}"
        # print(air_quality_des)
    
    current_weather_button = driver.find_element(By.CLASS_NAME, "cur-con-weather-card")
    current_weather_button.click()
    
    page = BeautifulSoup(driver.page_source, features="html.parser")
    div_current_temp =  page.find_all("div", {"class": "current-weather-info"})
    if div_current_temp: 
        current_temp_des = f"Current temperature: {div_current_temp[0].get_text().strip()}"
        # print(current_temp_des)
    
    div_current_status =  page.find_all("div", {"class": "current-weather"})
    if div_current_status:
        div_current_status = div_current_status[0].find("div", {"class": "phrase"})
        current_status_des = f"Current status: {div_current_status.get_text().strip()}"
        # print(current_status_des)
    
    div_current_infor =  page.find_all("div", {"class": "current-weather-details"})
    if div_current_infor:
        list_infor = []
        for infor in div_current_infor[0].get_text().split("\n\n\n")[3:-1]:
            infor = infor.split("\n")
            list_infor.append(f"{infor[0]}: {infor[1]}")
        current_infor_des = ", ".join(list_infor)
        # print(current_infor_des)
        
    #-------------------------day------------------------
    div_temp =  page.find_all("div", {"class": "weather"})
    if div_temp:
        temp_des = f"Temperature from {div_temp[1].get_text().strip()} to {div_temp[0].get_text().strip()}"
        # print(temp_des)
    
    div_infor =  page.find_all("div", {"class": "half-day-card-content"})
    if div_infor:
        des = f'Daylight: {div_infor[0].find("div", {"class": "phrase"}).get_text().strip()}, Tonight: {div_infor[1].find("div", {"class": "phrase"}).get_text().strip()}'
        # print(des)
        div_day_infor = div_infor[0].find_all("span", {"class": "value"})
        day_infor = [x.get_text() for x in div_day_infor]
        day_infor_des = f"Max UV Index: {day_infor[0]}, Wind: {day_infor[1]}, Wind Gusts: {day_infor[2]}, Probability of Precipitation: {day_infor[3]}, Probability of Thunderstorms: {day_infor[4]}, Rain: {day_infor[6]}, Hours of Rain: {day_infor[8]}, Cloud Cover: {day_infor[9]}" 
                        
        div_night_infor = div_infor[1].find_all("span", {"class": "value"})
        night_infor = [x.get_text() for x in div_night_infor]
        night_infor_des = f"Wind: {night_infor[0]}, Wind Gusts: {night_infor[1]}, Probability of Precipitation: {night_infor[2]}, Probability of Thunderstorms: {night_infor[3]}, Rain: {night_infor[5]}, Hours of Rain: {night_infor[7]}, Cloud Cover: {night_infor[8]}" 
        # print(day_infor_des)
        # print(night_infor_des)
    #/////////////////////////////////////////////////////
    weather_forecast_news = f"""
        Current weather:
            {air_quality_des}
            {current_status_des}
            {current_temp_des}
            The others weather index: {{{current_infor_des}}}
        
        General weather forecast:
            {des}
            {temp_des}
            Others weather index for daylight: {{{day_infor_des}}}
            Others weather index for tonight: {{{night_infor_des}}}
    """
    print(weather_forecast_news)
    
if __name__=="__main__":
    location = "vietnam"
    get_weather(location)