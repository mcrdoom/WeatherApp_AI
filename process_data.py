import pandas as pd
import numpy as np
import json
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Отключить предупреждения TensorFlow (для более чистого вывода)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Отключает специфичные оптимизации
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Отключает информационные сообщения TensorFlow

print("Начинается выполнение process_data.py...")

# --- Константы для парсинга result.json ---
clothing_blacklist = [
    'вискоза', 'шерсть', 'хлопок', 'лен', 'шелк', 'синтетика', 'фланель', 'деним',
    'защита от ветра', 'защита от влаги', 'дышащая одежда', 'многослойная одежда', 'утепленная одежда',
    'комфорт', 'стиль', 'оценка комфорта', 'погода', 'время суток', 'температура', 'влажность', 'ветер', 'осадки', 'облачность',
    'няя куртка', 'одежда', 'гардероб', 'рекомендуется', 'подойдет', 'нужна', 'вам', 'чтобы',
    'одеть', 'носить', 'одеваться', 'взять', 'возьмите', 'это', 'то', 'а', 'и', 'или', 'не', 'привет', 'спасибо',
    'отлично', 'хорошо', 'ясно', 'пасмурно', 'есть', 'нет',
    'весь', 'вся', 'все', 'любой', 'любая', 'любое', 'какой', 'какая', 'какое', 'какие',
    'здесь', 'там', 'куда', 'откуда', 'когда', 'сколько', 'почему', 'как', 'зачем', 'что', 'кто', 'где',
    'просто', 'очень', 'немного', 'сильно', 'слабо', 'будет', 'было', 'есть', 'нет', 'да', 'может быть', 'пожалуй', 'конечно',
    'для', 'с', 'в', 'на', 'от', 'до', 'по', 'за', 'из', 'у', 'к', 'о', 'об', 'под', 'над', 'перед', 'около', 'через',
    'утром', 'днем', 'вечером', 'ночью', 'сегодня', 'завтра', 'вчера',
    'дождь', 'снег', 'солнце', 'холод', 'жара', 'ветер', 'влажность',
    'почему это хорошо', 'другие варианты', 'плюсы и минусы'
]

excluded_words_for_city_expanded = set([
    'привет', 'хочу', 'спросить', 'какая', 'сегодня', 'будет', 'это', 'не', 'относится', 'к',
    'погода', 'очень', 'просто', 'вот', 'моя', 'нужен', 'для', 'вас', 'мой', 'вам', 'пожалуйста',
    'здравствуйте', 'завтра', 'вчера', 'сегодня', 'сейчас', 'данные', 'информация', 'помощь',
    'идеально', 'комфортно', 'удобно', 'совет', 'рекомендации', 'найти', 'выбрать', 'главное',
    'что', 'как', 'почему', 'пример', 'например', 'холод', 'жара', 'дождь', 'снег', 'солнце',
    'ветер', 'влажность', 'облака', 'туман', 'много', 'мало', 'очень', 'немного', 'сильный',
    'слабый', 'мокрый', 'сухой', 'чистый', 'грязный', 'да', 'нет', 'давай', 'конечно', 'возможно',
    'невозможно', 'впрочем', 'кстати', 'правда', 'ложно', 'добрый', 'утро', 'день', 'вечер', 'ночь',
    'какой', 'какая', 'какое', 'какие', 'сколько', 'температура', 'влажность', 'осадки', 'облачность',
    'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург', 'нижний новгород', 'казань', 'челябинск',
    'омск', 'самара', 'ростов-на-дону', 'уфа', 'красноярск', 'пермь', 'воронеж', 'волгоград',
    'иркутск', 'сочи', 'владивосток', 'калининград', 'мурманск', 'краснодар', 'тверь', 'анапа',
    'отлично', 'хорошо', 'ясно', 'пасмурно', 'переменная', 'облачность', 'местами', 'без', 'с', 'к', 'в', 'за',
    'юг', 'север', 'запад', 'восток', 'центр', 'край', 'область', 'город',
    'чем', 'время', 'малооблачно', 'теплые', 'рекомендуемая', 'информация', 'характеристики', 'описание', 'данные'
])

my_chat_garbage_patterns = [
    r".*вот\s*ваши\s*новые\s*\d+\s*записей.*", r".*сам,\s*это\s*потрясающие\s*новости.*",
    r".*ты\s*абсолютно\s*прав,\s*сайт\s*http:\/\/localhost:8501\/\s*открылся.*",
    r".*мы\s*добились\s*того,\s*что\s*весь\s*пайплайн\s*работает.*",
    r".*теперь\s*по\s*поводу\s*[\"']странностей[\"'].*оценки\s*комфорта[\"'].*",
    r".*наш\s*web-сервис\s*функционирует.*мы\s*преодолели\s*все\s*основные\s*технические\s*барьеры.*",
    r".*пример\s*с\s*иркутском.*теплый\s*пуховик.*флисовая\s*кофта.*термобелье.*это\s*великолепно.*",
    r".*что\s*делать\s*дальше.*",
    r".*теперь,\s*когда\s*весь\s*технологический\s*стэк\s*запущен.*повышение\s*качества\s*рекомендаций.*",
    r".*нам\s*нужно\s*сделать\s*две\s*вещи.*",
    r".*хорошо,\s*жду\s*вывод\s*\.?(?:\s*а\s*пока,\s*вот\s*твоё\s*последнее\s*сообщение.*)?",
    r".*(?:processed_weather_clothing_data\.csv|ohe_categories\.pkl|scaler\.pkl|clothing_mapping\.pkl|weather_clothing_model\.h5).*не\s*появился.*",
    r".*начало\s*терминала.*", r".*конец\s*тер.*", r".*модель\s*обучена.*", r".*точность\s*на\s*тестовых\s*данных:.*",
    r".*модель\s*сохранена\s*в\s*weather_clothing_model\.h5.*", r".*шаг\s*\d+:\s*сохранение\s*обработанных\s*данных.*",
    r".*processed\s*data\s*saved\s*to\s*processed_weather_clothing_data\.csv.*",
    r".*все\s*этапы\s*process_data\.py\s*успешно\s*завершены.*", r".*теперь\s*у\s*вас\s*есть.*",
    r".*пожалуйста,\s*проверь\s*содержимое\s*processed_weather_clothing_data\.csv\s*и\s*вывод\s*из\s*терминала.*",
    r".*ps\s*c:\\users\\dumni\\onedrive\\документы\\weatherapp_ai>.*", r".*мне\s*кажется\s*или\s*всё\s*начинает\s*повторятся.*тебе\s*освежить\s*память.*",
    r".*result\.json\s*это\s*экспорт\s*из\s*телеграмма.*", r".*tensorflow.*oneapi.*",
    r".*warning:absl:you\s*are\s*saving\s*your\s*model.*", r".*epoch\s*\d+\/\d+.*",
    r".*\d+\/\d+\s*━━━━━━━━━━━━━━━━━━━━.*accuracy:\s*\d+\.\d+e\+00.*",
    r".*начинается\s*выполнение\s*process_data\.py.*", r".*debug:\s*msg.*", r".*debug:\s*clothing.*",
    r".*общая\s*статистика:.*", r".*обработано\s*исходных\s*сообщений:\s*\d+.*",
    r".*успешно\s*извлечено\s*отдельных\s*записей.*",
    r".*сообщений,\s*которые\s*были\s*полностью\s*отфильтрованы.*",
    r".*внимание:\s*не\s*удалось\s*извлечь\s*ни\s*одной\s*полной\s*записи.*",
    r".*заполнены\s*пропущенные\s*значения\s*в\s*'.+'\s*медианой.*", r".*шаг\s*\d+:\s*.+",
    r".*ошибка:\s*y_train\s*пуст.*", r".*\*\s*погода:.*",
    r".*\*\s*(?:температура|влажность|ветер|осадки|облачность|время\s*суток|одежда|рекомендуемая\s*одежда):.*"
]

clothing_keywords_base = [
    'футболка', 'шорты', 'сандалии', 'кепка', 'солнцезащитные очки', 'джинсы', 'кроссовки',
    'рубашка', 'легкая куртка', 'кофта', 'свитер', 'шапка', 'перчатки', 'пуховик',
    'зимние ботинки', 'термобелье', 'шерстяные носки', 'зонт', 'плащ', 'резиновые сапоги',
    'платье', 'юбка', 'туфли', 'пиджак', 'брюки', 'мокасины', 'балаклава', 'варежки',
    'утепленный пуховик', 'плотный свитер', 'тонкая куртка', 'ветровка', 'пальто', 'сапоги',
    'ботинки', 'кеды', 'майка', 'блузка', 'кардиган', 'шарф', 'куртка', 'спортивный костюм',
    'купальник', 'плавки', 'шляпа', 'галстук', 'костюм', 'туника', 'легинсы', 'комбинезон',
    'жилет', 'шаль', 'чулки', 'носки', 'пижама', 'халат', 'дождевик', 'бандана', 'головной убор',
    'джинсовая куртка', 'кожаная куртка', 'полуботинки', 'валенки', 'митенки', 'снуд', 'палантин',
    'купальный костюм', 'polo', 'толстовка', 'флис', 'капюшон', 'наушники', 'легкое платье', 'чиносы' # 'polo' -> 'поло'
]

clothing_keywords_full = set(clothing_keywords_base)
adjectives = ['легкая', 'теплая', 'зимняя', 'летняя', 'плотная', 'тонкая', 'утепленная', 'флисовая', 'шерстяная', 'хлопковая', 'дождевая']
for item in clothing_keywords_base:
    for adj in adjectives:
        clothing_keywords_full.add(f"{adj} {item}")
    if item.endswith('а'): clothing_keywords_full.add(item[:-1] + 'и')
    if item.endswith('ы'): clothing_keywords_full.add(item)
    if item.endswith('ок'): clothing_keywords_full.add(item[:-2] + 'ки')
    if item.endswith('и'): clothing_keywords_full.add(item)
    if item.endswith('ь'): clothing_keywords_full.add(item[:-1] + 'и')

DEBUG_MODE = False

# --- Функция для извлечения данных из текстовых блоков (используется при парсинге result.json) ---
def extract_weather_and_clothing(text_block, message_id, block_idx):
    temp = None
    humidity = None
    wind = None
    precipitation = 'Неизвестно'
    clouds = 'Неизвестно'
    time_of_day = 'Неизвестно'
    city = 'Неизвестно'

    normalized_text = text_block.lower().replace('ё', 'е')

    temp_match = re.search(r'(?:температура|темп)\s*[:\-—]?\s*([+\-]?\d+(?:[.,]\d+)?)\s*°?c?', normalized_text, re.IGNORECASE)
    if temp_match: temp = float(temp_match.group(1).replace(',', '.'))

    humidity_match = re.search(r'(?:влажность|влажн)\s*[:\-—]?\s*(\d+(?:[.,]\d+)?)\s*%', normalized_text, re.IGNORECASE)
    if humidity_match: humidity = float(humidity_match.group(1).replace(',', '.'))

    wind_match = re.search(r'(?:ветер|вет)\s*[:\-—]?\s*(\d+(?:[.,]\d+)?)\s*м/с', normalized_text, re.IGNORECASE)
    if wind_match: wind = float(wind_match.group(1).replace(',', '.'))

    if wind is None:
        if re.search(r'\b(?:слабый|легкий|небольшой)\s*ветер\b', normalized_text): wind = 2.0
        elif re.search(r'\b(?:умеренный|средний)\s*ветер\b', normalized_text): wind = 5.0
        elif re.search(r'\b(?:сильный|порывистый|штормовой)\s*ветер\b', normalized_text): wind = 10.0
        elif re.search(r'\b(?:штиль|безветренно)\b', normalized_text): wind = 0.0

    if re.search(r'\b(дождь|ливень|морось|град|снег|ледяной дождь)\b', normalized_text): precipitation = 'Есть'
    elif re.search(r'\b(без осадков|осадков нет|нет осадков)\b', normalized_text): precipitation = 'Нет'

    if re.search(r'\b(ясно|солнечно|безоблачно)\b', normalized_text): clouds = 'Ясно'
    elif re.search(r'\b(облачно с прояснениями|переменная облачность)\b', normalized_text): clouds = 'Переменная облачность'
    elif re.search(r'\b(пасмурно|облачно)\b', normalized_text): clouds = 'Пасмурно'

    if re.search(r'\b(утро|утром)\b', normalized_text): time_of_day = 'Утро'
    elif re.search(r'\b(день|днем)\b', normalized_text): time_of_day = 'День'
    elif re.search(r'\b(вечер|вечером)\b', normalized_text): time_of_day = 'Вечер'
    elif re.search(r'\b(ночь|ночью)\b', normalized_text): time_of_day = 'Ночь'

    city_match = re.search(r'\bв\s+городе\s+([А-Яа-яЁё\s\-]+)\b', text_block, re.IGNORECASE)
    if city_match:
        found_city_name = city_match.group(1).strip()
        if found_city_name.lower() not in excluded_words_for_city_expanded:
            city = found_city_name

    found_clothing_items = set()
    potential_clothing_text_segment = ""

    explicit_clothing_match = re.search(
        r'(?:рекомендуемая\s+одежда|одежда|рекомендуется|советуем|лучше\s+надеть|подойдет|можно\s+надеть|выбирайте|вам\s+понадоб(?:ятся|ится)|стоит\s+одеть|как\s+одеться)\s*[:—,\-]?\s*(.+?)(?=\n\s*(?:[*\-–]?\s*\d*\.?\s*)?(?:почему|другие|плюсы|минусы|\*?\s*[а-яё]+\s*[:—])|$)',
        normalized_text, re.IGNORECASE | re.DOTALL
    )

    if explicit_clothing_match:
        potential_clothing_text_segment = explicit_clothing_match.group(1).strip()
    else:
        if any(ck in normalized_text for ck in clothing_keywords_base) and \
           (',' in normalized_text or '*' in normalized_text):
            cleaned_segment_for_heuristic = re.sub(r'(?:почему\s+хорошо/плохо|другие\s+варианты|плюсы\s+и\s+минусы|why\s+good/bad).*', '', normalized_text, flags=re.IGNORECASE | re.DOTALL).strip()
            if cleaned_segment_for_heuristic and len(cleaned_segment_for_heuristic.split()) > 1:
                potential_clothing_text_segment = cleaned_segment_for_heuristic

    if potential_clothing_text_segment:
        for clothing_item_phrase in clothing_keywords_full:
            if re.search(r'\b' + re.escape(clothing_item_phrase) + r'\b', potential_clothing_text_segment):
                if clothing_item_phrase not in clothing_blacklist:
                    found_clothing_items.add(clothing_item_phrase)

    clothing_rec_str = ', '.join(sorted(list(found_clothing_items))) if found_clothing_items else 'Неизвестно'
    return temp, humidity, wind, precipitation, clouds, time_of_day, clothing_rec_str, city

# --- ОСНОВНАЯ ЛОГИКА ЗАГРУЗКИ И ОБРАБОТКИ ДАННЫХ ---

# --- Шаг 1: Загрузка и объединение данных ---
print("\nШаг 1: Загрузка и объединение реальных и синтетических данных.")

# 1. Загрузка реальных данных из result.json
print("  Загрузка реальных данных из result.json...")
all_real_records = []
try:
    with open('result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Ошибка: Файл 'result.json' не найден. Убедитесь, что он находится в той же директории.")
    exit()
except json.JSONDecodeError:
    print("Ошибка: Невозможно декодировать JSON из 'result.json'. Проверьте формат файла.")
    exit()

for message_idx, message in enumerate(data['messages']):
    message_id = message.get('id', f'unknown_real_{message_idx}')
    
    message_text_raw = ""
    if 'text' in message and isinstance(message['text'], (str, list)):
        if isinstance(message['text'], list):
            for item in message['text']:
                if isinstance(item, str): message_text_raw += item + " "
                elif isinstance(item, dict) and 'text' in item: message_text_raw += item['text'] + " "
        else:
            message_text_raw = message['text']
        message_text_raw = message_text_raw.strip()

        cleaned_message_text = message_text_raw
        for pattern in my_chat_garbage_patterns:
            cleaned_message_text = re.sub(pattern, '', cleaned_message_text, flags=re.IGNORECASE | re.DOTALL).strip()

        if not cleaned_message_text.strip(): continue

        potential_blocks_raw = re.split(r'\n{2,}|(?=\n\s*\d+\.\s*Погода:)', cleaned_message_text, flags=re.IGNORECASE | re.DOTALL)
        if not potential_blocks_raw: potential_blocks_raw = [cleaned_message_text]

        for block_idx, block_text in enumerate(potential_blocks_raw):
            block_text = block_text.strip()
            if not block_text: continue

            normalized_block_text = block_text.lower().replace('ё', 'е')
            if len(normalized_block_text) < 15 and \
               not re.search(r'\b(?:температура|влажность|ветер|осадки|облачность|рекомендуется|одежда|шапка|куртка|брюки|шорты|платье|свитер)\b', normalized_block_text):
                continue
            if re.search(r'^\s*(?:[*\-–]?\s*\d*\.?\s*)?(?:почему\s+это\s+хорошо|другие\s+варианты|плюсы\s+и\s+минусы)', normalized_block_text, re.IGNORECASE):
                continue

            temp, humidity, wind, precipitation, clouds, time_of_day, clothing_rec_str, city = extract_weather_and_clothing(block_text, message_id, block_idx)

            if temp is not None and clothing_rec_str != 'Неизвестно' and clothing_rec_str != '':
                all_real_records.append({
                    'ID': f"{message_id}-{block_idx}",
                    'Температура (°C)': temp, 'Влажность (%)': humidity if humidity is not None else np.nan, 'Ветер (м/с)': wind if wind is not None else np.nan,
                    'Осадки': precipitation, 'Облачность': clouds, 'Время суток': time_of_day,
                    'Рекомендации по одежде': clothing_rec_str, 'Город': city
                })

if not all_real_records:
    print("Внимание: Не удалось извлечь ни одной полной записи из result.json. Модель будет обучаться только на синтетических данных.")
    df_real = pd.DataFrame() # Пустой DataFrame
else:
    df_real = pd.DataFrame(all_real_records)
    print(f"  Успешно извлечено {len(df_real)} записей из result.json.")

# Заполнение пропущенных значений в реальных данных (если есть)
if not df_real.empty:
    for col in ['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)']: # ИСПРАВЛЕНО: с на русскую
        if df_real[col].isnull().any():
            median_val = df_real[col].median()
            df_real[col] = df_real[col].fillna(median_val)
            print(f"  Заполнены пропущенные значения в '{col}' реальных данных медианой: {median_val}")
    for col in ['Осадки', 'Облачность', 'Время суток', 'Город']:
        if df_real[col].isnull().any() or 'Неизвестно' in df_real[col].unique():
            df_real[col] = df_real[col].replace('Неизвестно', np.nan)
            mode_val = df_real[col].mode()[0] if not df_real[col].dropna().mode().empty else 'Другое'
            df_real[col] = df_real[col].fillna(mode_val)
            print(f"  Заполнены пропущенные значения (включая 'Неизвестно') в '{col}' реальных данных модой: {mode_val}")

# 2. Загрузка синтетических данных
print("  Загрузка синтетических данных из synthetic_weather_clothing_data.csv...")
try:
    df_synthetic = pd.read_csv('synthetic_weather_clothing_data.csv')
    print(f"  Успешно загружено {len(df_synthetic)} записей из synthetic_weather_clothing_data.csv.")
except FileNotFoundError:
    print("Ошибка: Файл 'synthetic_weather_clothing_data.csv' не найден. Будет использован только result.json.")
    df_synthetic = pd.DataFrame() # Пустой DataFrame
except pd.errors.EmptyDataError:
    print("Ошибка: Файл 'synthetic_weather_clothing_data.csv' пуст. Будет использован только result.json.")
    df_synthetic = pd.DataFrame()
except Exception as e:
    print(f"Ошибка при загрузке синтетических данных: {e}. Будет использован только result.json.")
    df_synthetic = pd.DataFrame()

# 3. Объединение данных
df = pd.concat([df_real, df_synthetic], ignore_index=True)

if df.empty:
    print("Ошибка: Нет данных для обработки. Проверьте result.json и synthetic_weather_clothing_data.csv.")
    exit()

print(f"  Общий объем данных после объединения: {len(df)} записей.")

# --- Шаг 2: Обработка категориальных признаков (One-Hot Encoding) ---
print("\nШаг 2: Обработка категориальных признаков (One-Hot Encoding)")

# Загрузка списка уникальных предметов одежды (clothing_mapping),
# который должен быть создан create_clothing_mapping.py.
try:
    with open('clothing_mapping.pkl', 'rb') as f:
        clothing_mapping = pickle.load(f) # Теперь это будут КАТЕГОРИИ!
    print(f"Загружено {len(clothing_mapping)} уникальных предметов одежды (категорий) из clothing_mapping.pkl для определения классов модели.")
    # Дополнительная проверка, чтобы убедиться, что маппинг не пустой и не слишком мал.
    if len(clothing_mapping) == 0:
        print("ОШИБКА: clothing_mapping.pkl пуст. Модель не сможет обучаться.")
        exit()
    if len(clothing_mapping) > 20 or len(clothing_mapping) < 10: # Ожидаем около 15 категорий
        print("ВНИМАНИЕ: Количество категорий в 'clothing_mapping.pkl' кажется необычным.")
        print("Убедитесь, что вы запустили 'define_clothing_groups.py' для генерации правильных категорий.")

except FileNotFoundError:
    print("ОШИБКА: Файл 'clothing_mapping.pkl' не найден!")
    print("Он содержит список всех возможных категорий одежды.")
    print("Пожалуйста, сначала запустите 'define_clothing_groups.py' для его создания.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке clothing_mapping.pkl: {e}. Убедитесь, что файл не поврежден.")
    exit()

ohe_columns = ['Осадки', 'Облачность', 'Время суток', 'Город']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[ohe_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(ohe_columns))

with open('ohe_categories.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("OneHotEncoder сохранен в ohe_categories.pkl")

# --- Шаг 3: Масштабирование числовых признаков ---
print("\nШаг 3: Масштабирование числовых признаков")

scaler = MinMaxScaler()
numerical_features = df[['Температура (°C)', 'Влажность (%)', 'Ветер (м/с)']]
scaled_features = scaler.fit_transform(numerical_features)
scaled_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("MinMaxScaler сохранен в scaler.pkl")

# --- Шаг 4: Подготовка данных для нейросети ---
print("\nШаг 4: Подготовка данных для нейросети")

X = pd.concat([scaled_df, encoded_df], axis=1)

# Создание Y для мультилейбл классификации (теперь для КАТЕГОРИЙ)
Y = np.zeros((len(df), len(clothing_mapping)), dtype=int)
for i, recommendations_str in enumerate(df['Рекомендации по одежде']):
    if isinstance(recommendations_str, str):
        items = [item.strip() for item in recommendations_str.split(',') if item.strip()]
        for item in items:
            # item здесь - это категория!
            if item in clothing_mapping:
                Y[i, clothing_mapping.index(item)] = 1

# Проверка, что Y_train не пуст
if Y.shape[0] == 0 or Y.shape[1] == 0:
    print("Ошибка: Матрица Y пуста. Невозможно обучить модель. Проверьте данные и clothing_mapping.")
    exit()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Размер обучающей выборки X_train: {X_train.shape}")
print(f"Размер тестовой выборки X_test: {X_test.shape}")
print(f"Размер обучающей выборки Y_train: {Y_train.shape}")
print(f"Размер тестовой выборки Y_test: {Y_test.shape}")

# --- Шаг 5: Создание и обучение нейросети ---
print("\nШаг 5: Создание и обучение нейросети")

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    # Выходной слой для мультилейбл классификации:
    # len(clothing_mapping) нейронов, по одному на каждую КАТЕГОРИЮ одежды.
    # 'sigmoid' активация, так как каждый нейрон предсказывает вероятность (0 или 1).
    Dense(len(clothing_mapping), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nМодель обучена.")
print(f"Точность на тестовых данных: {accuracy*100:.2f}%")

model.save('weather_clothing_model.h5')
print("Модель сохранена в weather_clothing_model.h5")

# --- Шаг 6: Сохранение обработанных данных ---
print("\nШаг 6: Сохранение обработанных данных в processed_weather_clothing_data.csv")
print(f"DEBUG: Attempting to save to: {os.path.abspath('processed_weather_clothing_data.csv')}")
df.to_csv('processed_weather_clothing_data.csv', index=False, encoding='utf-8')
print("Processed data saved to processed_weather_clothing_data.csv")

print("\nВсе этапы process_data.py успешно завершены!")
print("Теперь у вас есть:")
print("- processed_weather_clothing_data.csv (очищенные и заполненные данные)")
print("- ohe_categories.pkl (сохраненный OneHotEncoder)")
print("- scaler.pkl (сохраненный MinMaxScaler)")
print("- clothing_mapping.pkl (список уникальных рекомендаций для декодирования - теперь категории)")
print("- weather_clothing_model.h5 (обученная модель нейросети)")

print("\nПожалуйста, проверь содержимое processed_weather_clothing_data.csv и вывод из терминала.")