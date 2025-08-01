import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import pytz # Для работы с часовыми поясами
from sklearn.preprocessing import OneHotEncoder # Явно импортируем OneHotEncoder
from sklearn.preprocessing import StandardScaler # Явно импортируем StandardScaler для проверки


# --- Конфигурация и загрузка моделей ---
# Убедитесь, что все эти файлы находятся в одной папке с app.py
MODEL_PATH = 'weather_clothing_model.h5'
OHE_PATH = 'ohe_categories.pkl' # Теперь ожидаем, что здесь напрямую OneHotEncoder
SCALER_PATH = 'scaler.pkl'
CLOTHING_MAPPING_PATH = 'clothing_mapping.pkl'
CLOTHING_GROUPS_PATH = 'clothing_groups.pkl' 

# ТВОЙ API КЛЮЧ OPENWEATHERMAP! ОБЯЗАТЕЛЬНО ЗАМЕНИ ПЛЕЙСХОЛДЕР!
OPENWEATHER_API_KEY = "c80a654b9866303179325d953f8d0c79"
OPENWEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# ТВОЯ ПАРТНЕРСКАЯ ССЫЛКА SELA! ОБЯЗАТЕЛЬНО ЗАМЕНИ ПЛЕЙСХОЛДЕР!
SELA_AFFILIATE_LINK = "https://kpwfp.com/g/2d356747430c2ebe1cc726a738318c/?erid=5jtCeReLm1S3Xx3LfVkzjYr"

@st.cache_resource
def load_all_resources():
    try:
        model = load_model(MODEL_PATH)
        with open(OHE_PATH, 'rb') as f:
            ohe = pickle.load(f) # Загружаем OneHotEncoder напрямую
            
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f) # Загружаем Scaler
        with open(CLOTHING_MAPPING_PATH, 'rb') as f:
            clothing_mapping = pickle.load(f) # Series/словарь {encoded_id: clothing_item_name}
            # Если clothing_mapping это DataFrame, а не Series, убедимся, что он готов к использованию
            if isinstance(clothing_mapping, pd.DataFrame):
                # Предполагаем, что clothing_mapping DataFrame имеет колонку 'clothing_item'
                # и индекс (или колонку 'encoded_value') для маппинга
                # Создадим Series для удобного доступа по индексу
                clothing_mapping_series = pd.Series(clothing_mapping['clothing_item'].values, index=clothing_mapping.index)
                clothing_mapping = clothing_mapping_series
            
        with open(CLOTHING_GROUPS_PATH, 'rb') as f:
            clothing_groups = pickle.load(f) # DataFrame или другой список групп

        # --- Собираем ожидаемый порядок признаков для модели ---
        # Это критически важно, так как модель ожидала 20 признаков в определенном порядке.
        # Мы предполагаем, что scaler обучался на английских названиях колонок (как в API),
        # а OHE на русских названиях категорий (как в исходных данных).
        
        # Названия численных признаков, как они БЫЛИ в исходных данных обучения для SCALER (англ.)
        numerical_cols_order_for_scaler = ['temperature_c', 'humidity_percent', 'wind_speed_mps']
        
        # Названия категориальных признаков, как они БЫЛИ в исходных данных обучения для OHE (рус.)
        categorical_cols_for_ohe = ['Осадки', 'Облачность', 'Время суток']
        
        # Получаем названия OHE-колонок из обученного OHE
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols_for_ohe)
        
        # Полный порядок признаков, который ожидается моделью
        # Должен быть: масштабированные численные, затем OHE-кодированные категориальные.
        input_features_order = numerical_cols_order_for_scaler + list(ohe_feature_names)
        
        # Проверка, что количество признаков соответствует ожидаемому моделью (20)
        if len(input_features_order) != 20:
             st.warning(f"ВНИМАНИЕ: Ожидалось 20 признаков на входе модели, но собрано {len(input_features_order)}. Это может вызвать ошибку модели. Проверьте process_data.py, чтобы убедиться в количестве признаков после OHE.")
             # Если тут ошибка, то проблема в том, как была обучена модель/OHE.
             # Это может быть самым сложным моментом.
             # Если warnings продолжаются, возможно, нужно заглянуть в `process_data.py`
             # и убедиться, что `ohe.get_feature_names_out()` был вызван с теми же именами
             # и что `model.fit()` получил именно такой порядок колонок.
             # Для дебага, можно добавить: st.write("Ожидаемый порядок:", input_features_order)
             
        return model, ohe, scaler, clothing_mapping, clothing_groups, input_features_order
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки файлов: {e}. Убедитесь, что 'process_data.py' и 'define_clothing_groups.py' были успешно запущены и создали все необходимые файлы.")
        st.stop()
    except Exception as e:
        st.error(f"Непредвиденная ошибка при загрузке ресурсов: {e}")
        st.stop()

model, ohe, scaler, clothing_mapping, clothing_groups, input_features_order = load_all_resources()
st.success("Все модели и маппинги успешно загружены!")

# --- Функции для предсказания и маппинга ---

# Ваша функция map_precipitation
def map_precipitation(description):
    desc = description.lower()
    if 'дождь' in desc or 'ливень' in desc or 'морось' in desc:
        return 'дождь' 
    elif 'снег' in desc:
        return 'снег' 
    elif 'туман' in desc or 'дымка' in desc:
        return 'туман' 
    elif 'гроза' in desc: 
        return 'гроза' 
    return 'ясно' 

# Ваша функция map_cloudiness
def map_cloudiness(percentage):
    if percentage <= 10:
        return 'ясно'  
    elif percentage <= 40:
        return 'небольшая облачность'  
    elif percentage <= 70:
        return 'переменная облачность'  
    else:
        return 'пасмурно'  

# Ваша функция map_time_of_day
def map_time_of_day(hour):
    if 5 <= hour < 12:
        return 'утро'  
    elif 12 <= hour < 18:
        return 'день'  
    elif 18 <= hour < 23:
        return 'вечер'  
    else:
        return 'ночь'  

# Вспомогательная функция для вывода времени суток текстом (для st.info)
def map_time_of_day_to_text(encoded_string): 
    return encoded_string.capitalize()


def predict_clothing_for_app(temp, humidity, wind, precipitation_cat, cloudiness_cat, time_of_day_cat):
    # Создаем DataFrame для передачи в scaler (с английскими названиями)
    numerical_input_for_scaler = pd.DataFrame([[temp, humidity, wind]],
                              columns=['temperature_c', 'humidity_percent', 'wind_speed_mps'])
    
    # Масштабирование численных признаков
    input_data_scaled = scaler.transform(numerical_input_for_scaler)
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_input_for_scaler.columns)

    # Создаем DataFrame для передачи в OHE (с русскими названиями)
    categorical_input_for_ohe = pd.DataFrame([[precipitation_cat, cloudiness_cat, time_of_day_cat]],
                              columns=['Осадки', 'Облачность', 'Время суток'])

    # One-Hot Encoding для категориальных признаков
    input_ohe = ohe.transform(categorical_input_for_ohe)
    ohe_feature_names = ohe.get_feature_names_out(categorical_input_for_ohe.columns)
    input_ohe_df = pd.DataFrame(input_ohe.toarray(), columns=ohe_feature_names)

    # Объединение всех признаков и обеспечение правильного порядка
    final_input_df = pd.concat([input_data_scaled_df, input_ohe_df], axis=1)
    
    # Переупорядочиваем колонки в соответствии с порядком, на котором обучалась модель
    # Это должно быть `input_features_order`, загруженный в `load_all_resources`
    final_input_df = final_input_df[input_features_order]

    # Преобразуем в numpy array для подачи в модель Keras
    input_for_prediction = final_input_df.values

    # Предсказание
    predictions = model.predict(input_for_prediction)
    
    # Обработка предсказаний для получения списка рекомендованной одежды
    if predictions.ndim > 1:
        predictions = predictions[0]

    recommended_items = []
    threshold = 0.2 # Порог активации для мульти-лейбл классификации. Можно настроить.
    
    # clothing_mapping должен быть Series {encoded_id: clothing_item_name}
    # Мы предполагаем, что индексы predictions соответствуют encoded_id в clothing_mapping
    
    # clothing_mapping содержит все 71 название одежды, проиндексированные 0..70
    # Просто перебираем предсказания и добавляем, если вероятность выше порога
    
    for i, prob in enumerate(predictions):
        if prob > threshold:
            # i - это индекс предсказания, который соответствует закодированному значению
            # clothing_mapping[i] должно давать название одежды
            if i < len(clothing_mapping): # Проверка на всякий случай
                recommended_items.append(clothing_mapping[i])
            else:
                st.warning(f"Индекс {i} из предсказания вне диапазона clothing_mapping.")
        
        if len(recommended_items) >= 7: # Ограничим количество рекомендаций
            break

    # Если ничего не предсказано выше порога, возвращаем что-то дефолтное
    if not recommended_items:
        # Более умная дефолтная рекомендация на основе температуры
        if temp >= 25:
            return ["футболка", "шорты", "сандалии", "кепка"]
        elif 15 <= temp < 25:
            return ["кофта", "кроссовки", "джинсы", "легкая куртка"]
        elif 0 <= temp < 15:
            return ["свитер", "куртка", "ботинки", "шапка"]
        elif temp < 0:
            return ["зимняя куртка", "шапка", "шарф", "перчатки", "зимние ботинки", "пальто"]
        return ["одежда по сезону"] # Дефолт
    
    return recommended_items # Возвращаем список строк

# --- UI часть Streamlit ---
st.title("👕 WeatherApp_AI")
st.header("Получите детальные рекомендации по одежде в зависимости от погоды.")

# Выбор города
city_options = ["Москва", "Санкт-Петербург", "Сочи", "Новосибирск", "Екатеринбург", "Казань", "Нижний Новгород", "Самара", "Омск", "Челябинск", "Ростов-на-Дону", "Уфа", "Красноярск", "Пермь", "Волгоград", "Воронеж", "Другое"]
selected_city = st.selectbox("Выберите город:", city_options)

if selected_city == "Другое":
    custom_city = st.text_input("Введите название города:")
    if custom_city:
        selected_city = custom_city

# Кнопка для получения рекомендаций
if st.button("Получить рекомендации по одежде"):
    if not selected_city or selected_city == "Другое":
        st.warning("Пожалуйста, выберите или введите название города.")
    else:
        with st.spinner(f"Получаем погоду для {selected_city} и генерируем рекомендации..."):
            try:
                # 1. Получение погодных данных с OpenWeatherMap
                params = {
                    'q': selected_city,
                    'appid': OPENWEATHER_API_KEY,
                    'units': 'metric',
                    'lang': 'ru'
                }
                response = requests.get(OPENWEATHER_API_URL, params=params)
                response.raise_for_status() 
                weather_data = response.json()

                temp = weather_data['main']['temp']
                # Убедимся, что поля существуют, используя .get()
                humidity = weather_data['main'].get('humidity', 0)
                wind = weather_data['wind'].get('speed', 0)
                
                precipitation_text = weather_data['weather'][0]['description'] if weather_data.get('weather') and weather_data['weather'] else 'ясно'
                cloudiness_percent = weather_data['clouds']['all'] if 'clouds' in weather_data else 0
                
                # Время суток с учетом часового пояса города
                timezone_offset_seconds = weather_data.get('timezone', 0) 
                city_timezone = pytz.FixedOffset(timezone_offset_seconds / 60) 
                
                utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
                city_time = utc_now.astimezone(city_timezone)
                current_hour = city_time.hour

                # Преобразование текстовых/числовых значений в текстовые для OHE
                precipitation_cat = map_precipitation(precipitation_text)
                cloudiness_cat = map_cloudiness(cloudiness_percent)
                time_of_day_cat = map_time_of_day(current_hour)
                
                # 2. Получение рекомендаций от нейросети
                recommended_clothing = predict_clothing_for_app(
                    temp, humidity, wind, 
                    precipitation_cat, 
                    cloudiness_cat, 
                    time_of_day_cat
                )

                st.subheader("Ваши рекомендации по одежде:")

                # Вывод каждого предмета одежды в отдельном раскрывающемся блоке
                for item in recommended_clothing:
                    with st.expander(f"**{item.capitalize()}**"):
                        st.write(f"_Детальное описание для {item.capitalize()} будет добавлено позже._")
                        st.markdown(f"**[Поиск {item.capitalize()} на Sela]({SELA_AFFILIATE_LINK})**", unsafe_allow_html=True)

                # Разделитель и общая ссылка на Sela
                st.markdown(f"---")
                st.write("Ищете что-то еще или хотите больше вариантов?")
                st.markdown(f"**[Посетите весь каталог Sela]({SELA_AFFILIATE_LINK})**", unsafe_allow_html=True)
                st.markdown(f"---")

                # Вывод текущих параметров
                st.info(f"Текущие параметры для {selected_city}: {temp}°C, {humidity}% влажность, {wind} м/с ветер, {precipitation_cat.capitalize()} осадки, {cloudiness_percent}% облачность, {map_time_of_day_to_text(time_of_day_cat)}.")

            except requests.exceptions.HTTPError as http_err:
                st.error(f"Ошибка HTTP при получении погоды: {http_err}. Код статуса: {response.status_code}")
                st.warning("Убедитесь, что название города введено корректно и API-ключ OpenWeatherMap действителен.")
            except requests.exceptions.ConnectionError as conn_err:
                st.error(f"Ошибка соединения: {conn_err}. Проверьте ваше интернет-соединение.")
            except requests.exceptions.Timeout as timeout_err:
                st.error(f"Истекло время ожидания запроса: {timeout_err}. Попробуйте еще раз.")
            except requests.exceptions.RequestException as req_err:
                st.error(f"Произошла ошибка при запросе к OpenWeatherMap API: {req_err}")
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")
                st.warning("Не удалось получить погодные данные. Попробуйте еще раз или выберите другой город.")

st.markdown("---")
st.write("Проект: WeatherApp_AI")
st.write("Разработчик: Сэм (при поддержке AI)")
