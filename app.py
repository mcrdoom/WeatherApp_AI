import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import pytz # Для работы с часовыми поясами

# --- Конфигурация и загрузка моделей ---
# Убедитесь, что все эти файлы находятся в одной папке с app.py
MODEL_PATH = 'weather_clothing_model.h5'
OHE_PATH = 'ohe_categories.pkl'
SCALER_PATH = 'scaler.pkl'
CLOTHING_MAPPING_PATH = 'clothing_mapping.pkl'
CLOTHING_GROUPS_PATH = 'clothing_groups.pkl' # Добавлено для возможного использования

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
            ohe = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(CLOTHING_MAPPING_PATH, 'rb') as f:
            clothing_mapping = pickle.load(f)
        with open(CLOTHING_GROUPS_PATH, 'rb') as f:
            clothing_groups = pickle.load(f) # Загружаем clothing_groups
        return model, ohe, scaler, clothing_mapping, clothing_groups
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки файлов: {e}. Убедитесь, что 'process_data.py' и 'define_clothing_groups.py' были успешно запущены и создали все необходимые файлы.")
        st.stop()
    except Exception as e:
        st.error(f"Непредвиденная ошибка при загрузке ресурсов: {e}")
        st.stop()

model, ohe, scaler, clothing_mapping, clothing_groups = load_all_resources()
st.success("Все модели и маппинги успешно загружены!")

# --- Функции для предсказания и маппинга ---

# Ваша функция map_precipitation
def map_precipitation(description):
    desc = description.lower()
    if 'дождь' in desc or 'ливень' in desc or 'морось' in desc:
        return 1
    elif 'снег' in desc:
        return 2
    elif 'туман' in desc or 'дымка' in desc:
        return 3
    elif 'гроза' in desc: # Добавил грозу
        return 4
    return 0 # Нет осадков / Ясно

# Ваша функция map_cloudiness
def map_cloudiness(percentage):
    if percentage <= 10:
        return 0  # Ясно
    elif percentage <= 40:
        return 1  # Небольшая облачность
    elif percentage <= 70:
        return 2  # Переменная облачность
    else:
        return 3  # Пасмурно

# Ваша функция map_time_of_day
def map_time_of_day(hour):
    if 5 <= hour < 12:
        return 0  # Утро
    elif 12 <= hour < 18:
        return 1  # День
    elif 18 <= hour < 23:
        return 2  # Вечер
    else:
        return 3  # Ночь

# Вспомогательная функция для вывода времени суток текстом (для st.info)
def map_time_of_day_to_text(encoded_value):
    if encoded_value == 0: return "Утро"
    elif encoded_value == 1: return "День"
    elif encoded_value == 2: return "Вечер"
    elif encoded_value == 3: return "Ночь"
    return "Неизвестно"


def predict_clothing_for_app(temp, humidity, wind, precipitation_encoded, cloudiness_encoded, time_of_day_encoded):
    # СОЗДАЕМ DATAFRAME С ТЕМИ ЖЕ РУССКИМИ НАЗВАНИЯМИ КОЛОНОК, КАК И ПРИ ОБУЧЕНИИ
    input_data = pd.DataFrame([[temp, humidity, wind, precipitation_encoded, cloudiness_encoded, time_of_day_encoded]],
                              columns=['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)',
                                       'precipitation_encoded', 'cloudiness_encoded', 'time_of_day_encoded'])

    # Числовые признаки, которые нужно масштабировать (с русскими названиями)
    numerical_features = ['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)']
    
    # Масштабирование численных признаков
    # scaler.transform теперь получит DataFrame с ожидаемыми именами
    input_data_scaled = scaler.transform(input_data[numerical_features])
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_features)

    # Объединение всех признаков для предсказания
    # Категориальные признаки (уже закодированные числами)
    categorical_encoded_features_df = input_data[['precipitation_encoded', 'cloudiness_encoded', 'time_of_day_encoded']]
    
    # Собираем финальный DataFrame для модели.
    # Важно: порядок колонок должен соответствовать порядку, на котором модель обучалась.
    # Предполагаем, что порядок был: численные, потом категориальные.
    final_input_df = pd.concat([input_data_scaled_df, categorical_encoded_features_df], axis=1)

    # Преобразуем в numpy array для подачи в модель Keras
    input_for_prediction = final_input_df.values

    # Предсказание
    predictions = model.predict(input_for_prediction)
    
    # Обработка предсказаний:
    # Здесь мы предполагаем, что модель выдает 71 вероятность (одна для каждого clothing_item).
    # И clothing_mapping используется для сопоставления индекса с именем одежды.
    
    # Убедимся, что predictions одномерный массив, если он пришел как [[...]]
    if predictions.ndim > 1:
        predictions = predictions[0]

    # Сортируем индексы по убыванию вероятности
    sorted_indices = np.argsort(predictions)[::-1]
    
    recommended_items = []
    
    # Используем `clothing_mapping` (которое должно быть словарем или Series {индекс: название_одежды})
    # Или `clothing_groups` (если это DataFrame с колонками 'clothing_item' и 'encoded_value')
    
    # Если clothing_mapping - это Series или словарь, где ключ = индекс, значение = имя одежды
    # ИЛИ если clothing_groups - это DataFrame, где индекс = id, колонка 'clothing_item' = название
    
    # Предполагаем, что clothing_groups - это DataFrame с колонкой 'clothing_item' и индексом,
    # соответствующим ID предмета одежды, как это было в исходных данных.
    
    # Возьмем топ N рекомендаций, которые имеют высокую вероятность
    # Это важно для получения нескольких предметов, как было раньше ("кепка, майка, шорты")
    
    # Порог активации для предсказания
    threshold = 0.5 # Можно настроить. Если вероятность выше 0.5, то рекомендуем.
    
    # Собираем список рекомендованной одежды
    # Здесь мы используем clothing_mapping, который должен быть словарем {encoded_id: clothing_item_name}
    for idx in sorted_indices:
        if predictions[idx] > threshold:
            # Ищем название одежды по индексу.
            # `clothing_mapping` должен быть словарем или Series, где ключ - это индекс (encoded value)
            # а значение - это название одежды.
            
            # Если clothing_mapping - это Series с индексом 'encoded' и колонкой 'clothing_item'
            if idx in clothing_mapping.index: # Проверяем, что индекс существует в clothing_mapping
                recommended_items.append(clothing_mapping.loc[idx, 'clothing_item']) # Получаем имя
        
        # Ограничиваем количество рекомендаций, чтобы не было слишком много
        if len(recommended_items) >= 6: # Например, до 6 предметов
            break

    # Если ничего не предсказано выше порога, возвращаем что-то дефолтное
    if not recommended_items:
        # Можно использовать более умную дефолтную рекомендацию на основе температуры, как раньше
        if temp >= 25:
            return ["футболка", "шорты"]
        elif 15 <= temp < 25:
            return ["кофта", "джинсы"]
        elif 0 <= temp < 15:
            return ["свитер", "куртка"]
        elif temp < 0:
            return ["зимняя куртка", "шапка", "перчатки"]
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
                response.raise_for_status() # Вызовет исключение для ошибок HTTP (4xx или 5xx)
                weather_data = response.json()

                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                wind = weather_data['wind']['speed']
                
                # Описание осадков и облачности
                # Убедимся, что 'weather' и 'description' существуют
                precipitation_text = weather_data['weather'][0]['description'] if weather_data.get('weather') and weather_data['weather'] else 'ясно'
                cloudiness_percent = weather_data['clouds']['all'] if 'clouds' in weather_data else 0
                
                # Время суток с учетом часового пояса города
                timezone_offset_seconds = weather_data.get('timezone', 0) # По умолчанию 0, если нет
                city_timezone = pytz.FixedOffset(timezone_offset_seconds / 60) # pytz принимает минуты
                
                utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
                city_time = utc_now.astimezone(city_timezone)
                current_hour = city_time.hour

                # Преобразование текстовых/числовых значений в кодированные для модели
                precipitation_encoded = map_precipitation(precipitation_text)
                cloudiness_encoded = map_cloudiness(cloudiness_percent)
                time_of_day_encoded = map_time_of_day(current_hour)
                
                # 2. Получение рекомендаций от нейросети
                recommended_clothing = predict_clothing_for_app(
                    temp, humidity, wind, 
                    precipitation_encoded, 
                    cloudiness_encoded, 
                    time_of_day_encoded
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
                st.info(f"Текущие параметры для {selected_city}: {temp}°C, {humidity}% влажность, {wind} м/с ветер, {precipitation_text.capitalize()} осадки, {cloudiness_percent}% облачность, {map_time_of_day_to_text(time_of_day_encoded)}.")

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
