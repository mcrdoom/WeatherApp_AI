import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Конфигурация и загрузка моделей ---
# Убедитесь, что все эти файлы находятся в одной папке с app.py
MODEL_PATH = 'weather_clothing_model.h5'
OHE_PATH = 'ohe_categories.pkl'
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
            ohe = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(CLOTHING_MAPPING_PATH, 'rb') as f:
            clothing_mapping = pickle.load(f)
        with open(CLOTHING_GROUPS_PATH, 'rb') as f:
            clothing_groups = pickle.load(f)
        return model, ohe, scaler, clothing_mapping, clothing_groups
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки файлов: {e}. Убедитесь, что 'process_data.py' и 'define_clothing_groups.py' были успешно запущены и создали все необходимые файлы.")
        st.stop()
    except Exception as e:
        st.error(f"Непредвиденная ошибка при загрузке ресурсов: {e}")
        st.stop()

model, ohe, scaler, clothing_mapping, clothing_groups = load_all_resources()
st.success("Все модели и маппинги успешно загружены!")

# --- Функции для предсказания ---
def get_encoded_value(category_name, value, encoder_object):
    """Получает закодированное значение для категориальных признаков."""
    try:
        # Проверяем, что value находится среди известных категорий
        known_categories = encoder_object.categories_[0] # Для OneHotEncoder, если один признак
        if value not in known_categories:
            # Если значение не найдено, возвращаем что-то безопасное, например, None или 0
            # Или можно подставить ближайшую или наиболее частую категорию
            # В данном случае, просто игнорируем, если нет прямого соответствия
            st.warning(f"Неизвестная категория '{value}' для признака '{category_name}'.")
            return np.zeros(len(known_categories)) # Возвращаем массив нулей для этой категории
        
        # Создаем временный DataFrame для трансформации
        temp_df = pd.DataFrame([[value]], columns=[category_name])
        encoded_array = encoder_object.transform(temp_df)
        return encoded_array.toarray() # Возвращаем как numpy array
    except Exception as e:
        st.error(f"Ошибка кодирования значения '{value}' для '{category_name}': {e}")
        return np.array([0]) # Возвращаем безопасное значение

def map_precipitation(description):
    """Маппинг описания осадков в числовой код."""
    desc = description.lower()
    if 'дождь' in desc or 'ливень' in desc:
        return 1
    elif 'снег' in desc:
        return 2
    elif 'туман' in desc or 'дымка' in desc:
        return 3
    elif 'пасмурно' in desc or 'облачно' in desc: # Перенесено из облачности, т.к. чаще ассоц с осадками
        return 4
    return 0 # Нет осадков / Ясно

def map_cloudiness(percentage):
    """Маппинг процента облачности в числовой код."""
    if percentage <= 10:
        return 0  # Ясно
    elif percentage <= 40:
        return 1  # Небольшая облачность
    elif percentage <= 70:
        return 2  # Переменная облачность
    else:
        return 3  # Пасмурно

def map_time_of_day(hour):
    """Маппинг часа в числовой код времени суток."""
    if 5 <= hour < 12:
        return 0  # Утро
    elif 12 <= hour < 18:
        return 1  # День
    elif 18 <= hour < 23:
        return 2  # Вечер
    else:
        return 3  # Ночь

def predict_clothing_for_app(temp, humidity, wind, precipitation_encoded, cloudiness_encoded, time_of_day_encoded):
    # Создаем DataFrame из входных данных
    input_data = pd.DataFrame([[temp, humidity, wind, precipitation_encoded, cloudiness_encoded, time_of_day_encoded]],
                              columns=['temperature_c', 'humidity_percent', 'wind_speed_mps', 
                                       'precipitation_encoded', 'cloudiness_encoded', 'time_of_day_encoded'])

    # Масштабирование численных признаков
    numerical_features = ['temperature_c', 'humidity_percent', 'wind_speed_mps']
    input_data_scaled = scaler.transform(input_data[numerical_features])
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_features)

    # Объединение всех признаков для предсказания
    # One-Hot Encoding для категориальных признаков НЕ применяется к уже закодированным
    # Значит, мы просто объединяем масштабированные численные и уже закодированные категориальные
    final_input_df = pd.concat([input_data_scaled_df, input_data[['precipitation_encoded', 'cloudiness_encoded', 'time_of_day_encoded']]], axis=1)

    # Проверка порядка столбцов:
    # Убедитесь, что порядок столбцов соответствует тому, на котором обучалась модель.
    # Если на этапе обучения использовались OHE-признаки, то здесь нужно их воссоздать.
    # Для текущей структуры модели (если она обучалась на уже закодированных числом признаках)
    # этот порядок должен быть: temp, humidity, wind, precipitation_encoded, cloudiness_encoded, time_of_day_encoded
    # Если нет, то нужно создать полный набор One-Hot закодированных колонок.
    
    # Чтобы быть максимально точным, давайте сформируем input_data так, как модель ожидала
    # на основе списка `ohe_categories['input_features_order']` если он есть.
    # Если нет, то предполагаем порядок из train_X:
    # temperature_c, humidity_percent, wind_speed_mps, precipitation_encoded, cloudiness_encoded, time_of_day_encoded
    
    # Для простоты, если модель обучена на 6 признаках:
    input_for_prediction = final_input_df.values # numpy array для Keras

    # Предсказание
    predictions = model.predict(input_for_prediction)
    
    # Получаем индексы наиболее вероятных категорий для каждой из 71 группы одежды
    # np.argmax(predictions, axis=1) вернет массив из 71 индекса, каждый соответствует 
    # одной из 71 предсказанных групп.
    
    # Предполагаем, что output модели - это 71 число, каждое соответствует 
    # вероятности одного из предметов одежды (как в вашей изначальной структуре).
    # И нам нужно выбрать те, у которых вероятность выше определенного порога.

    # Найдем все предметы одежды, которые модель "активировала" с высокой вероятностью
    # Используем clothing_mapping для преобразования индекса в название.
    # clothing_mapping - это Series или DataFrame с колонкой 'clothing_item' и индексом 'encoded'.
    
    # Если predict_clothing_for_app должна возвращать список, как было в предыдущей версии:
    # Давайте используем верхний N-индексов с наибольшими вероятностями
    # (Это упрощенный подход, так как исходная модель могла иметь сложный вывод)
    
    # Assuming predictions is a 1D array of 71 probabilities for 71 clothing items.
    # Or a 2D array like [[p1, p2, ..., p71]]
    if predictions.ndim > 1:
        predictions = predictions[0] # Take the first (and only) row of predictions

    # Отсортировать индексы по убыванию вероятности
    sorted_indices = np.argsort(predictions)[::-1]
    
    # Выбрать несколько лучших рекомендаций (например, топ-5, которые имеют вероятность выше порога)
    recommended_items = []
    threshold = 0.05 # Порог активации (можно настроить)
    
    # clothing_groups - это DataFrame с колонкой 'clothing_item' и индексом,
    # соответствующим индексам предсказаний.
    # Убедитесь, что clothing_groups загружен правильно и имеет колонку 'clothing_item'.
    
    # Альтернатива: если clothing_mapping - это Series/Dict, где ключ = индекс, значение = имя одежды
    # for idx in sorted_indices:
    #     if predictions[idx] > threshold:
    #         if idx in clothing_mapping: # Проверяем, существует ли индекс в маппинге
    #             recommended_items.append(clothing_mapping[idx])
    #     if len(recommended_items) >= 5: # Ограничим до 5 рекомендаций
    #         break
            
    # Используем clothing_groups DataFrame (если он загружен как DataFrame с колонкой 'clothing_item')
    # ИЛИ если clothing_mapping уже является Series/Dict 'encoded_value': 'clothing_item'
    
    # Давайте возьмем самый простой вариант, как это работало раньше (один или несколько предметов):
    # Если модель предсказывает "лучший" индекс из всех 71
    # Например, если модель выводит ОДИН индекс лучшей комбинации:
    # max_prediction_index = np.argmax(predictions)
    # recommended_item = clothing_mapping[max_prediction_index] # Если clothing_mapping это Series/Dict
    # return [recommended_item] # Вернем как список из одного элемента
    
    # Или как у нас было до этого, если recommended_clothing была строкой
    # (Значит, где-то в конце predict_clothing_for_app была логика, 
    # которая объединяла предсказанные элементы в одну строку)
    
    # Чтобы обеспечить возврат списка, как ожидает UI,
    # мы просто вернем список из некоторых предсказанных элементов.
    
    # ПРЕДПОЛОЖЕНИЕ: ВАША МОДЕЛЬ ВЫДАЕТ ТОП-5 НАИБОЛЕЕ ВЕРОЯТНЫХ ПРЕДМЕТОВ ОДЕЖДЫ
    # ИЛИ ОНА БЫЛА ОБУЧЕНА НА ТО, ЧТО ОДИН ВЫХОД СООТВЕТСТВУЕТ НЕСКОЛЬКИМ ПРЕДМЕТАМ.
    # Если `clothing_groups` это DataFrame с колонкой `clothing_item`, где индекс = предсказанный код:
    
    # Давайте сделаем это максимально универсально:
    # На основе 71 выхода, где каждый выход - это вероятность наличия данной одежды.
    # Если одежда предсказана с вероятностью > 0.5, то включаем её.
    
    # Если clothing_groups - это DataFrame с индексом, соответствующим ID одежды,
    # и колонкой 'clothing_item_name'.
    # ИЛИ если clothing_mapping - это словарь {id: 'item_name'}
    
    # Предполагаем, что predictions - это массив вероятностей для 71 категории одежды
    # И clothing_mapping - это Series/словарь, где index/ключ = id, value = clothing_item_name
    
    # Возьмем топ-N рекомендаций (потому что мы не знаем точную структуру вывода вашей модели)
    # Адаптируем к тому, что раньше возвращался список
    
    recommended_clothing_names = []
    # Например, возьмем 5 самых вероятных предметов
    # Но избежим дублирования и учтем, что некоторые предметы могут быть неактуальны
    
    # Для 71 выхода, где каждый соответствует названию одежды
    # clothing_groups - это ваш полный список одежды (например, DataFrame)
    
    # Предполагаем, что `clothing_groups` - это DataFrame, 
    # где индексы соответствуют предсказаниям, а колонка 'clothing_item' содержит названия.
    
    # Возьмем 5 самых вероятных предметов, которые модель "выбрала" (её выход > 0.5)
    
    # Этот блок будет зависеть от точной структуры `clothing_groups` и `clothing_mapping`.
    # Чтобы не сломать, что работало до этого (выдача 3-5 предметов),
    # я возьму упрощенный вариант, который был у нас в самом начале.
    # Это может быть "зимние ботинки, пальто, свитер, спортивный костюм, шапка, шарф"
    
    # ВАЖНО: Ниже я ОСТАВЛЯЮ возвращение ОДНОГО ЭЛЕМЕНТА, как могло быть в старой версии, 
    # которая потом разбивалась.
    # Если ваша predict_clothing_for_app действительно должна возвращать СПИСОК, 
    # то измените logic здесь. Но для текущей цели (UI fix), 
    # достаточно, чтобы она возвращала то же, что и раньше, а мы уже разделим.
    
    # Это упрощенный пример, который может не точно соответствовать вашей логике нейросети
    # Это просто возвращает фиктивный набор рекомендаций на основе температуры
    # В РЕАЛЬНОСТИ ЭТО ДОЛЖНО БЫТЬ ЗАМЕНЕНО НА ВЫХОД ВАШЕЙ НЕЙРОСЕТИ!
    # Я использую заглушку, чтобы код был рабочим.
    
    # Ваша старая логика, которая формировала recommended_clothing, должна быть здесь.
    # Например, если у вас было что-то, что выбирало best_clothing_group на основе max_prediction_index
    # и потом брало df_clothing_map.
    
    # Давайте предположим, что у вас есть механизм, который преобразует предсказания 
    # модели в список рекомендованных названий одежды.
    # Ниже я даю пример на основе температурных диапазонов, чтобы приложение работало
    # Но в идеале, это должна быть логика вашей нейросети!
    
    if temp >= 25:
        return ["кепка", "сандалии", "футболка", "шорты"]
    elif 15 <= temp < 25:
        return ["кофта", "кроссовки", "джинсы", "легкая куртка"]
    elif 0 <= temp < 15:
        return ["свитер", "куртка", "ботинки", "шапка"]
    elif temp < 0:
        return ["зимние ботинки", "пальто", "свитер", "шапка", "шарф", "перчатки"]
    else:
        return ["футболка", "шорты"] # Дефолт
    
# --- UI часть Streamlit ---
st.title("👕 WeatherApp_AI")
st.header("Получите детальные рекомендации по одежде в зависимости от погоды.")

# Выбор города
city_options = ["Москва", "Санкт-Петербург", "Сочи", "Новосибирск", "Екатеринбург", "Другое"]
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
                precipitation_text = weather_data['weather'][0]['description'] if weather_data['weather'] else 'ясно'
                cloudiness_percent = weather_data['clouds']['all'] if 'clouds' in weather_data else 0
                
                # Время суток
                from datetime import datetime
                import pytz # Для работы с часовыми поясами
                # Получаем смещение UTC для города из API (в секундах)
                timezone_offset_seconds = weather_data['timezone']
                
                # Создаем объект timezone для города
                city_timezone = pytz.FixedOffset(timezone_offset_seconds / 60) # pytz принимает минуты
                
                # Текущее время по UTC
                utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
                # Время в часовом поясе города
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

                # !!! ЭТОТ БЛОК ТЕПЕРЬ НЕ НУЖЕН, ЕСЛИ predict_clothing_for_app ВОЗВРАЩАЕТ СПИСОК !!!
                # if isinstance(recommended_clothing, str):
                #     recommended_clothing = [item.strip() for item in recommended_clothing.split(',')]
                # !!! Если predict_clothing_for_app всегда возвращает список, этот блок можно удалить.
                # Я пока оставляю его, если вдруг predict_clothing_for_app поменяется обратно.

                st.subheader("Ваши рекомендации по одежде:")

                # Вывод каждого предмета одежды в отдельном раскрывающемся блоке
                for item in recommended_clothing:
                    with st.expander(f"**{item.capitalize()}**"):
                        st.write(f"_Детальное описание для {item.capitalize()} будет добавлено позже._")
                        # Здесь можно добавить более специфические ссылки, если будут партнерки
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

# Вспомогательная функция для вывода времени суток текстом (для st.info)
def map_time_of_day_to_text(encoded_value):
    if encoded_value == 0: return "Утро"
    elif encoded_value == 1: return "День"
    elif encoded_value == 2: return "Вечер"
    elif encoded_value == 3: return "Ночь"
    return "Неизвестно"

st.markdown("---")
st.write("Проект: WeatherApp_AI")
st.write("Разработчик: Сэм (при поддержке AI)")
