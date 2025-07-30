import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import random
import requests # Новый импорт для HTTP-запросов
import datetime # Для определения времени суток по текущему времени
import tensorflow as tf
from tensorflow.keras.models import load_model

# Отключить предупреждения TensorFlow (для более чистого вывода в консоли)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- API КЛЮЧ OpenWeatherMap ---
# ВСТАВЬ СВОЙ API КЛЮЧ ЗДЕСЬ:
OPENWEATHER_API_KEY = "c80a654b9866303179325d953f8d0c79" 
OPENWEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# --- Загрузка необходимых файлов ---
# Эти объекты загружаются ОДИН РАЗ при запуске Streamlit приложения,
# чтобы не загружать их при каждом запросе пользователя.
@st.cache_resource # Кэширует загруженные объекты, чтобы не перезагружать их при каждом изменении UI
def load_all_models_and_encoders():
    try:
        model = load_model('weather_clothing_model.h5')
        ohe_encoder = pickle.load(open('ohe_categories.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        clothing_categories_map = pickle.load(open('clothing_mapping.pkl', 'rb'))
        clothing_groups_detail = pickle.load(open('clothing_groups.pkl', 'rb'))
        st.success("Все модели и маппинги успешно загружены.")
        return model, ohe_encoder, scaler, clothing_categories_map, clothing_groups_detail
    except Exception as e:
        st.error(f"Ошибка загрузки файлов: {e}. Убедитесь, что 'process_data.py' и 'define_clothing_groups.py' были успешно запущены и создали все необходимые файлы.")
        st.stop() # Останавливаем приложение, если критическая ошибка

model, ohe_encoder, scaler, clothing_categories_map, clothing_groups_detail = load_all_models_and_encoders()

# --- Функция для получения реальных погодных данных ---
def get_weather_data(city_name):
    params = {
        'q': city_name,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric', # Получаем температуру в Цельсиях
        'lang': 'ru' # Получаем описание погоды на русском
    }
    try:
        response = requests.get(OPENWEATHER_API_URL, params=params)
        response.raise_for_status() # Вызывает HTTPError для плохих ответов (4xx или 5xx)
        data = response.json()
        
        # Извлечение данных
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed'] # м/с

        # Осадки
        # OpenWeatherMap предоставляет 'weather' массив. 'main' - основное описание.
        # https://openweathermap.org/weather-conditions
        weather_main = data['weather'][0]['main'].lower()
        if weather_main in ['rain', 'snow', 'drizzle', 'thunderstorm', 'squall']:
            precipitation = 'Есть'
        else:
            precipitation = 'Нет'

        # Облачность
        clouds_percent = data['clouds']['all'] # Процент облачности
        if clouds_percent < 20:
            clouds = 'Ясно'
        elif clouds_percent < 70:
            clouds = 'Переменная облачность'
        else:
            clouds = 'Пасмурно'
        
        # Время суток (по текущему времени сервера Streamlit, можно улучшить до времени в городе)
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12:
            time_of_day = 'Утро'
        elif 12 <= current_hour < 18:
            time_of_day = 'День'
        elif 18 <= current_hour < 23:
            time_of_day = 'Вечер'
        else:
            time_of_day = 'Ночь'

        return {
            'temp': temp,
            'humidity': humidity,
            'wind': wind_speed,
            'precipitation': precipitation,
            'clouds': clouds,
            'time_of_day': time_of_day,
            'city': city_name # Возвращаем запрошенный город
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при подключении к OpenWeatherMap: {e}. Проверьте подключение к интернету или API ключ.")
        return None
    except KeyError as e:
        st.error(f"Не удалось получить данные для города '{city_name}'. Возможно, город не найден или в ответе API нет ожидаемых полей: {e}")
        return None
    except Exception as e:
        st.error(f"Произошла непредвиденная ошибка: {e}")
        return None


# --- Функция для предсказания одежды по погоде (ВСТРОЕНАЯ) ---
def predict_clothing_for_app(temp, humidity, wind, precipitation, clouds, time_of_day, city):
    # Создание DataFrame для входных данных
    input_data = pd.DataFrame([[temp, humidity, wind, precipitation, clouds, time_of_day, city]],
                              columns=['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)',
                                       'Осадки', 'Облачность', 'Время суток', 'Город'])

    # Масштабирование числовых признаков
    numerical_features_scaled = scaler.transform(input_data[['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)']])
    numerical_df = pd.DataFrame(numerical_features_scaled, columns=['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)'])

    # One-Hot Encoding категориальных признаков
    categorical_features_encoded = ohe_encoder.transform(input_data[['Осадки', 'Облачность', 'Время суток', 'Город']])
    categorical_df = pd.DataFrame(categorical_features_encoded, columns=ohe_encoder.get_feature_names_out(['Осадки', 'Облачность', 'Время суток', 'Город']))

    # Объединение всех признаков
    X_input = pd.concat([numerical_df, categorical_df], axis=1)

    # Предсказание вероятностей для каждой категории одежды
    predictions = model.predict(X_input, verbose=0)[0] # verbose=0 для скрытия вывода предсказаний
    
    # Определение рекомендованных категорий на основе порога
    # Порог можно настроить. 0.35 - хороший баланс между полнотой и точностью.
    recommended_categories_indices = np.where(predictions > 0.35)[0] 
    predicted_categories = [clothing_categories_map[i] for i in recommended_categories_indices]

    # --- Детализация рекомендаций на основе предсказанных категорий и погоды ---
    detailed_recommendations = set()

    for category in predicted_categories:
        if category in clothing_groups_detail:
            possible_items = clothing_groups_detail[category]
            
            chosen_item = None
            # Логика выбора конкретного предмета из категории
            if category == "Верхняя одежда (очень теплая)":
                if temp < -20 and "утепленный пуховик" in possible_items: chosen_item = "утепленный пуховик"
                elif temp < -10 and "пуховик" in possible_items: chosen_item = "пуховик"
                elif temp < 0 and "пальто" in possible_items: chosen_item = "пальто"
                elif temp < 5 and "плотный свитер" in possible_items: chosen_item = "плотный свитер" # как очень теплый элемент
            elif category == "Верхняя одежда (легкая)":
                if precipitation == 'Есть' and "дождевик" in possible_items: chosen_item = "дождевик"
                elif wind > 7 and "ветровка" in possible_items: chosen_item = "ветровка"
                elif temp < 15 and "легкая куртка" in possible_items: chosen_item = "легкая куртка"
                elif temp < 20 and "пиджак" in possible_items: chosen_item = "пиджак"
                elif "джинсовая куртка" in possible_items: chosen_item = "джинсовая куртка"
            elif category == "Верх (основной)":
                if temp < 10 and "свитер" in possible_items: chosen_item = "свитер"
                elif temp < 18 and "кофта" in possible_items: chosen_item = "кофта"
                elif temp < 20 and "рубашка" in possible_items: chosen_item = "рубашка"
                elif "поло" in possible_items: chosen_item = "поло"
            elif category == "Верх (легкий)":
                if temp > 25 and "майка" in possible_items: chosen_item = "майка"
                elif "футболка" in possible_items: chosen_item = "футболка"
                if "купальник" in possible_items and temp > 25 and clouds == 'Ясно': chosen_item = "купальник"
                if "плавки" in possible_items and temp > 25 and clouds == 'Ясно': chosen_item = "плавки"
            elif category == "Низ (теплый)":
                if temp < 15 and "джинсы" in possible_items: chosen_item = "джинсы"
                elif "брюки" in possible_items: chosen_item = "брюки"
                if "спортивный костюм" in possible_items and temp < 20: chosen_item = "спортивный костюм"
            elif category == "Низ (легкий)":
                if temp > 20 and "шорты" in possible_items: chosen_item = "шорты"
                elif temp > 15 and "юбка" in possible_items: chosen_item = "юбка"
                elif temp > 18 and "легкое платье" in possible_items: chosen_item = "легкое платье"
                elif "платье" in possible_items: chosen_item = "платье"
                elif "комбинезон" in possible_items: chosen_item = "комбинезон"
            elif category == "Обувь (зимняя)":
                if temp < 0 and "зимние ботинки" in possible_items: chosen_item = "зимние ботинки"
                elif temp < 5 and "ботинки" in possible_items: chosen_item = "ботинки"
                elif temp < -15 and "валенки" in possible_items: chosen_item = "валенки"
                elif "сапоги" in possible_items: chosen_item = "сапоги"
            elif category == "Обувь (демисезонная)":
                if temp < 15 and "полуботинки" in possible_items: chosen_item = "полуботинки"
                elif temp < 20 and "кроссовки" in possible_items: chosen_item = "кроссовки"
                elif "кеды" in possible_items: chosen_item = "кеды"
                elif "туфли" in possible_items: chosen_item = "туфли"
                elif "мокасины" in possible_items: chosen_item = "мокасины"
            elif category == "Обувь (летняя)":
                if temp > 20 and "сандалии" in possible_items: chosen_item = "сандалии"
            elif category == "Головные уборы (зимние)":
                if temp < 5 and "шапка" in possible_items: chosen_item = "шапка"
                elif temp < -10 and "балаклава" in possible_items: chosen_item = "балаклава"
                elif "капюшон" in possible_items: chosen_item = "капюшон"
            elif category == "Головные уборы (летние)":
                if temp > 20 and clouds == 'Ясно' and "кепка" in possible_items: chosen_item = "кепка"
                elif temp > 25 and clouds == 'Ясно' and "шляпа" in possible_items: chosen_item = "шляпа"
                elif "солнцезащитные очки" in possible_items and clouds == 'Ясно': chosen_item = "солнцезащитные очки"
            elif category == "Перчатки/Варежки":
                if temp< 5 and "перчатки" in possible_items: chosen_item = "перчатки"
                elif temp < -5 and "варежки" in possible_items: chosen_item = "варежки"
            elif category == "Шарф/Палантин":
                if temp < 10 and "шарф" in possible_items: chosen_item = "шарф"
                elif temp < 0 and "снуд" in possible_items: chosen_item = "снуд"
            elif category == "Белье/Носки":
                if temp < 0 and "термобелье" in possible_items: chosen_item = "термобелье"
                if temp < 5 and "шерстяные носки" in possible_items: chosen_item = "шерстяные носки"
                elif "носки" in possible_items: chosen_item = "носки"
            elif category == "Дополнительно (от осадков)":
                if precipitation == 'Есть' and "зонт" in possible_items: chosen_item = "зонт"
                if precipitation == 'Есть' and "дождевик" in possible_items: chosen_item = "дождевик"
                if precipitation == 'Есть' and temp < 10 and "резиновые сапоги" in possible_items: chosen_item = "резиновые сапоги"

            if chosen_item:
                detailed_recommendations.add(chosen_item)
            else: # Если ничего специфического не выбрали, берем случайный из категории (если не пустая)
                if possible_items:
                    detailed_recommendations.add(random.choice(possible_items))

    if not detailed_recommendations:
        detailed_recommendations.add("Не удалось подобрать конкретные рекомендации. Оденьтесь по погоде.")

    return ", ".join(sorted(list(detailed_recommendations)))


# --- Streamlit UI ---
st.set_page_config(page_title="WeatherApp_AI - Рекомендации по одежде", layout="centered")

st.title("👕 WeatherApp_AI")
st.markdown("### Получите детальные рекомендации по одежде в зависимости от погоды.")

st.header("Выберите город:")
# Список городов для выбора (можешь расширить его по желанию)
cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань', 'Сочи', 'Владивосток', 'Лондон', 'Париж', 'Нью-Йорк', 'Токио', 'Дубай', 'Другое']
selected_city = st.selectbox("Город", cities)

# Ручной ввод для "Другое"
if selected_city == 'Другое':
    custom_city = st.text_input("Введите название города (на английском, например, 'Berlin'):")
    if custom_city:
        selected_city = custom_city
    else:
        st.warning("Пожалуйста, введите название города.")
        st.stop()

if st.button("Получить рекомендации по одежде"):
    if selected_city:
        with st.spinner(f"Получаем погоду для {selected_city} и генерируем рекомендации..."):
            weather_data = get_weather_data(selected_city)
            
            if weather_data:
                # Извлекаем данные для передачи в predict_clothing_for_app
                temp = weather_data['temp']
                humidity = weather_data['humidity']
                wind = weather_data['wind']
                precipitation = weather_data['precipitation']
                clouds = weather_data['clouds']
                time_of_day = weather_data['time_of_day'] # Время суток будет определено по текущему времени запроса
                city_for_prediction = weather_data['city'] # Используем город, возвращенный API

                recommendations = predict_clothing_for_app(temp, humidity, wind, precipitation, clouds, time_of_day, city_for_prediction)
                
                st.subheader("Ваши рекомендации по одежде:")
                st.success(recommendations)
                
                st.markdown("---")
                st.info(f"Текущие параметры для **{selected_city}**: **{temp}°C**, **{humidity}%** влажность, **{wind} м/с** ветер, **{precipitation}** осадки, **{clouds}** облачность, **{time_of_day}**.")
            else:
                st.error("Не удалось получить погодные данные. Попробуйте еще раз или выберите другой город.")
    else:
        st.warning("Пожалуйста, выберите или введите город.")

# Дополнительная информация для пользователя
st.markdown("---")
st.markdown("Проект: WeatherApp_AI")
st.markdown("Разработчик: Сэм (при поддержке AI)")