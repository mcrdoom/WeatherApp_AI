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
            # ohe_categories должен содержать обученный OneHotEncoder
            # и список order_of_features_for_model
            ohe_data = pickle.load(f)
            ohe = ohe_data['ohe_encoder']
            # Также нам нужен порядок признаков, на котором обучалась модель
            # Предполагаем, что он был сохранен в 'ohe_categories.pkl'
            # или мы можем его собрать динамически
            input_features_order = ohe_data.get('input_features_order', None)
            
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(CLOTHING_MAPPING_PATH, 'rb') as f:
            clothing_mapping = pickle.load(f) # Это, скорее всего, Series или Dict {encoded_id: clothing_item_name}
        with open(CLOTHING_GROUPS_PATH, 'rb') as f:
            clothing_groups = pickle.load(f) # Это может быть DataFrame или другой список групп

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
        return 'дождь' # Возвращаем строку для OHE
    elif 'снег' in desc:
        return 'снег' # Возвращаем строку для OHE
    elif 'туман' in desc or 'дымка' in desc:
        return 'туман' # Возвращаем строку для OHE
    elif 'гроза' in desc: 
        return 'гроза' # Возвращаем строку для OHE
    return 'ясно' # Возвращаем строку для OHE

# Ваша функция map_cloudiness
def map_cloudiness(percentage):
    if percentage <= 10:
        return 'ясно'  # Возвращаем строку для OHE
    elif percentage <= 40:
        return 'небольшая облачность'  # Возвращаем строку для OHE
    elif percentage <= 70:
        return 'переменная облачность'  # Возвращаем строку для OHE
    else:
        return 'пасмурно'  # Возвращаем строку для OHE

# Ваша функция map_time_of_day
def map_time_of_day(hour):
    if 5 <= hour < 12:
        return 'утро'  # Возвращаем строку для OHE
    elif 12 <= hour < 18:
        return 'день'  # Возвращаем строку для OHE
    elif 18 <= hour < 23:
        return 'вечер'  # Возвращаем строку для OHE
    else:
        return 'ночь'  # Возвращаем строку для OHE

# Вспомогательная функция для вывода времени суток текстом (для st.info)
def map_time_of_day_to_text(encoded_string): # Теперь принимает строку, а не число
    return encoded_string.capitalize() # Просто капитализируем для вывода


def predict_clothing_for_app(temp, humidity, wind, precipitation_cat, cloudiness_cat, time_of_day_cat):
    # СОЗДАЕМ DATAFRAME С ТЕМИ ЖЕ РУССКИМИ НАЗВАНИЯМИ КОЛОНОК, КАК И ПРИ ОБУЧЕНИИ
    # И категориальными значениями в текстовом виде, как ожидает OHE
    input_df = pd.DataFrame([[temp, humidity, wind, precipitation_cat, cloudiness_cat, time_of_day_cat]],
                              columns=['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)',
                                       'Осадки', 'Облачность', 'Время суток'])

    # Числовые признаки
    numerical_features = ['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)']
    
    # Категориальные признаки (теперь в текстовом виде)
    categorical_features = ['Осадки', 'Облачность', 'Время суток']

    # Масштабирование численных признаков
    input_data_scaled = scaler.transform(input_df[numerical_features])
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_features)

    # One-Hot Encoding для категориальных признаков
    # ohe_encoder ожидает DataFrame
    input_ohe = ohe.transform(input_df[categorical_features])
    
    # Получаем названия колонок после OHE
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    input_ohe_df = pd.DataFrame(input_ohe.toarray(), columns=ohe_feature_names)

    # Объединение всех признаков
    # Важно: колонки должны быть в том же порядке, в каком они были при обучении модели.
    # Если input_features_order был сохранен, используем его.
    # Если нет, предполагаем, что numerical_features идут первыми, затем ohe_feature_names.
    
    # Собираем финальный DataFrame для модели
    final_input_df = pd.concat([input_data_scaled_df, input_ohe_df], axis=1)

    # Если input_features_order был успешно загружен, используем его для переупорядочивания колонок
    if input_features_order is not None and len(input_features_order) == final_input_df.shape[1]:
        final_input_df = final_input_df[input_features_order]
    elif input_features_order is not None and len(input_features_order) != final_input_df.shape[1]:
        st.warning("Количество признаков в input_features_order не совпадает с количеством признаков после обработки. Могут быть ошибки.")
        # Fallback: попытаемся использовать тот порядок, который есть, но это рискованно.
        # Лучше пересохранить ohe_categories.pkl с корректным input_features_order из process_data.py
    elif input_features_order is None:
        st.warning("Порядок признаков для модели не был загружен. Убедитесь, что 'input_features_order' сохранен в 'ohe_categories.pkl'. Порядок признаков может быть неправильным.")
        # Для дебага, можно вывести final_input_df.columns

    # Преобразуем в numpy array для подачи в модель Keras
    input_for_prediction = final_input_df.values

    # Предсказание
    predictions = model.predict(input_for_prediction)
    
    # Обработка предсказаний для получения списка рекомендованной одежды
    if predictions.ndim > 1:
        predictions = predictions[0]

    recommended_items = []
    threshold = 0.2 # Порог активации, можно настроить (0.5 может быть слишком высоким для мульти-лейбл)
    
    # clothing_mapping должен быть Series/словарь {encoded_id: clothing_item_name}
    # Мы предполагаем, что индексы predictions соответствуют encoded_id в clothing_mapping
    
    # Получаем список всех возможных предметов одежды из clothing_mapping
    all_clothing_items = clothing_mapping.tolist() # Assuming clothing_mapping is a Series of names
    
    for i, prob in enumerate(predictions):
        if prob > threshold:
            # i - это индекс предсказания, который соответствует закодированному значению
            # clothing_mapping должен преобразовать этот индекс обратно в название одежды
            # Если clothing_mapping - это Series с индексом, а значения - названия одежды
            if i < len(all_clothing_items): # Проверка на всякий случай
                recommended_items.append(all_clothing_items[i])
            else:
                st.warning(f"Индекс {i} из предсказания вне диапазона clothing_mapping.")
        
        # Ограничим количество рекомендаций (можно настроить)
        if len(recommended_items) >= 7: 
            break

    # Если ничего не предсказано выше порога, возвращаем что-то дефолтное
    if not recommended_items:
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
                    precipitation_cat, # Теперь передаем текстовые категории
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
            except requests.exceptions.Timeout as timeout_err:st.error(f"Истекло время ожидания запроса: {timeout_err}. Попробуйте еще раз.")
            except requests.exceptions.RequestException as req_err:
                st.error(f"Произошла ошибка при запросе к OpenWeatherMap API: {req_err}")
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")
                st.warning("Не удалось получить погодные данные. Попробуйте еще раз или выберите другой город.")

st.markdown("---")
st.write("Проект: WeatherApp_AI")
st.write("Разработчик: Сэм (при поддержке AI)")
