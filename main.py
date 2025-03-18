from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import logging
import logging.handlers
import sys
import os
import json
import re
import traceback
import torch
import datetime
from typing import Dict, Any
from fastapi import Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

# Создание директории для логов
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования
def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                os.path.join(LOG_DIR, 'app_debug.log'),
                maxBytes=10*1024*1024,  # 10 МБ
                backupCount=5,
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Настройка логгеров библиотек
    logging.getLogger('transformers').setLevel(logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)

# Вызов настройки логгера
setup_logger()
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Модели Pydantic
class ReviewItem(BaseModel):
    text: str = Field(..., min_length=2)
    rating: float = Field(..., gt=0, le=5)
    date: Optional[str] = None

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        return v.strip()

class BusinessReviewRequest(BaseModel):
    business_name: Optional[str] = "Unknown Place"
    extracted_reviews: List[ReviewItem]

    @field_validator('extracted_reviews')
    @classmethod
    def validate_reviews(cls, v):
        if not v:
            raise ValueError('At least one review is required')
        return v

# Инициализация FastAPI приложения
app = FastAPI(
    title="Business Reviews AI Analyzer",
    description="Advanced AI-powered review analysis service",
    version="1.0.0"
)

# Middleware для CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Параметры ML модели
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Глобальные переменные для модели
model = None
tokenizer = None

def initialize_model():
    global model, tokenizer
    try:
        logger.info(f"Attempting to initialize model: {MODEL_NAME}")

        # Авторизация в Hugging Face
        if HF_TOKEN:
            logger.info("Logging in to Hugging Face")
            login(token=HF_TOKEN)
        else:
            logger.warning("No Hugging Face token provided")

        # Загрузка токенизатора
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Конфигурация квантизации
        logger.info("Configuring quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Загрузка модели
        logger.info("Loading model")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )

        logger.info("Mistral model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        logger.error(f"Detailed traceback: {traceback.format_exc()}")
        return False

# Инициализация модели при старте
initialize_model()

def calculate_sentiment(reviews: List[ReviewItem]) -> float:
    """Вычисление среднего балла с расширенным логированием"""
    try:
        logger.debug("=" * 80)
        logger.debug("ENTERING calculate_sentiment")
        logger.debug(f"Total Reviews: {len(reviews)}")

        # Логирование каждого рейтинга
        for i, review in enumerate(reviews, 1):
            logger.debug(f"Review {i} Rating: {review.rating}")

        if not reviews:
            logger.warning("No reviews for averageScore calculation")
            return 0.0

        # Точный расчет среднего балла
        ratings = [review.rating for review in reviews]
        avg_rating = sum(ratings) / len(ratings)

        # Округление до одного знака после запятой
        result = round(avg_rating, 1)

        logger.debug(f"Calculated Sentiment: {result}")
        logger.debug("=" * 80)
        return result
    except Exception as e:
        logger.error(f"Error calculating averageScore: {e}")
        return 0.0

def trim_to_full_sentence(text, max_length=500):
    """Обрезает текст до max_length, сохраняя полные предложения."""
    if len(text) <= max_length:
        return text  # Если текст и так короче лимита, возвращаем как есть

    truncated_text = text[:max_length]  # Первоначально обрезаем по лимиту
    last_sentence_end = max(truncated_text.rfind('.'), truncated_text.rfind('!'), truncated_text.rfind('?'))

    if last_sentence_end == -1:  # Если нет знаков завершения предложения, просто возвращаем обрезанный вариант
        return truncated_text

    return truncated_text[:last_sentence_end+1]  # Возвращаем текст до последнего найденного знака


def generate_mistral_summary(reviews: list, business_name: str = "Unknown Place") -> str:
    """Генерация саммари с улучшенной обработкой ошибок, завершая его на знаке пунктуации"""
    try:
        # Проверка инициализации модели
        if model is None:
            return "Model initialization failed"

        if tokenizer is None:
            return "Tokenizer initialization failed"

        # Проверка наличия отзывов
        if not reviews:
            return "No reviews available for analysis"

        # Ограничение длины входных отзывов
        reviews = [review[:500] for review in reviews]  # Максимум 500 символов на отзыв

        # Формирование промпта
        prompt = f"""<s>[INST] Summarize the following customer reviews about {business_name} in 3 sentences: 
Provide a comprehensive summary (500 tokens max) that includes:
1. Overall sentiment
2. Main positive points
3. Main criticisms

Reviews:
"""
        for review in reviews[:50]:  # Ограничиваем до 50 отзывов
            prompt += f"- {review}\n"
        prompt += "[/INST]"

        # Токенизация и генерация
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=400,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Декодирование ответа
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлечение саммари
        if "[/INST]" in full_response:
            summary = full_response.split("[/INST]")[-1].strip()
        else:
            summary = full_response.strip()

        # Обрезаем, сохраняя завершённое предложение
        summary = trim_to_full_sentence(summary, 500)

        # Если получилось пустое саммари, даём дефолтный текст
        if not summary:
            summary = "Comprehensive analysis of customer reviews reveals mixed experiences. Specific details about service quality, customer satisfaction, and key characteristics could not be definitively extracted."

        return summary

    except Exception as e:
        return "Unable to process reviews and generate summary at this time."

def extract_themes(summary: str) -> List[str]:
    """Извлечение ключевых тем"""
    try:
        theme_indicators = [
            "positive aspects", "negative aspects", "criticized for",
            "praised for", "customers appreciate", "complaints about",
            "notable features", "key issues"
        ]

        themes = []
        for indicator in theme_indicators:
            if indicator in summary.lower():
                match = re.search(rf'{indicator}[:\s]*([^.;,]+)', summary, re.IGNORECASE)
                if match:
                    theme = match.group(1).strip()
                    if theme and theme not in themes:
                        themes.append(theme.capitalize())

        # Темы по умолчанию
        if not themes:
            themes = ["Service Quality", "Customer Experience", "Atmosphere"]

        logger.debug(f"Extracted Themes: {themes}")
        return themes[:4]
    except Exception as e:
        logger.error(f"Error extracting themes: {e}")
        return ["Service Quality", "Customer Experience", "Atmosphere"]

# НОВЫЙ ЭНДПОИНТ: Принимает JSON напрямую вместо файла
@app.post("/analyze-business-reviews-json")
async def analyze_business_reviews_json(payload: Dict[str, Any] = Body(...)):
    try:
        # Записываем в лог начало обработки
        with open(os.path.join(LOG_DIR, "request_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n\n=== NEW REQUEST {datetime.datetime.now()} ===\n")
            f.write(f"Payload keys: {list(payload.keys())}\n")

        logger.debug("=" * 100)
        logger.debug("ENTERING analyze_business_reviews_json ENDPOINT")
        logger.debug(f"Received payload with keys: {list(payload.keys())}")

        # Проверка наличия ключа с отзывами
        reviews_keys = [
            'Extracted Reviews',
            'extracted_reviews',
            'reviews',
            'Extracted_Reviews'
        ]

        reviews_key = next((key for key in reviews_keys if key in payload), None)

        if not reviews_key:
            logger.error("No reviews key found")
            logger.error(f"Available keys: {list(payload.keys())}")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "No reviews found",
                    "details": f"Expected one of {reviews_keys}",
                    "available_keys": list(payload.keys())
                }
            )

        # Извлечение названия бизнеса
        business_name = payload.get('Business Name', payload.get('business_name', 'Unknown Place'))
        logger.debug(f"Business Name: {business_name}")

        # Логирование найденного ключа
        logger.debug(f"Found reviews key: {reviews_key}")
        logger.debug(f"Number of reviews: {len(payload[reviews_key])}")

        # Гибкое извлечение отзывов
        extracted_reviews = []
        for item in payload[reviews_key]:
            try:
                review_text = str(item.get('text', '')).strip()
                review_rating = float(item.get('rating', 0))
                review_date = item.get('date')

                if review_text and review_rating:
                    extracted_reviews.append(
                        ReviewItem(
                            text=review_text,
                            rating=review_rating,
                            date=review_date
                        )
                    )
            except Exception as review_error:
                logger.warning(f"Skipping problematic review: {review_error}")

        # Проверка наличия отзывов
        if not extracted_reviews:
            logger.error("No valid reviews could be extracted")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "No valid reviews",
                    "details": "No valid reviews could be extracted from the input"
                }
            )

        # Извлечение текстов отзывов
        review_texts = [review.text for review in extracted_reviews]

        # Генерация анализа
        summary = generate_mistral_summary(review_texts, business_name)

        # Извлечение тем
        key_themes = extract_themes(summary)

        # Расчет среднего балла
        average_score = calculate_sentiment(extracted_reviews)

        # Определение диапазона дат
        dates = [review.date for review in extracted_reviews if review.date]
        first_date = min(dates) if dates else None
        last_date = max(dates) if dates else None

        # Подготовка ответа
        result = {
            "nameOfPlace": business_name,
            "summary": summary,
            "averageScore": average_score,
            "keyThemes": key_themes,
            "firstReviewDate": first_date,
            "lastReviewDate": last_date
        }

        # Логирование и возврат
        logger.debug(f"Final result: {json.dumps(result, indent=2)}")
        logger.debug("=" * 100)

        # Дополнительное логирование результата
        with open(os.path.join(LOG_DIR, "request_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"Result: {json.dumps(result, indent=2)}\n")

        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"CRITICAL ENDPOINT ERROR: {e}", exc_info=True)
        # Запись ошибки в отдельный лог
        with open(os.path.join(LOG_DIR, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n\n=== ERROR {datetime.datetime.now()} ===\n")
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": str(e)
            },
            media_type="application/json; charset=utf-8"
        )

@app.get("/health")
async def health_check():
    """Эндпоинт проверки работоспособности"""
    try:
        # Добавляем более подробную проверку статуса
        status = "healthy" if model is not None and tokenizer is not None else "unhealthy"

        # Логирование статуса
        logger.info(f"Health check - Status: {status}")

        return JSONResponse(
            content={
                "status": status,
                "model": MODEL_NAME,
                "message": "Review analysis service is operational",
                "details": {
                    "model_loaded": model is not None,
                    "tokenizer_loaded": tokenizer is not None
                }
            },
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500,
            media_type="application/json; charset=utf-8"
        )

# Для отладки - просто возвращает полученные данные
@app.post("/echo")
async def echo(data: Dict[str, Any] = Body(...)):
    return JSONResponse(
        content={
            "received": data,
            "message": "This is an echo endpoint for debugging"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)