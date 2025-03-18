from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import logging
import logging.handlers
import sys
import os
import json
import traceback
import torch
import datetime
from typing import Dict, Any
from fastapi import Body
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import gc  # Для очистки памяти

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

    logging.getLogger('transformers').setLevel(logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)

setup_logger()
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Определение Pydantic моделей
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

# Инициализация FastAPI
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

# Конфигурация модели
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Глобальные переменные
model = None
tokenizer = None

def initialize_model():
    global model, tokenizer
    try:
        logger.info(f"🚀 Initializing model: {MODEL_NAME}")

        if HF_TOKEN:
            logger.info("🔑 Logging in to Hugging Face")
            login(token=HF_TOKEN)
        else:
            logger.warning("⚠️ No Hugging Face token provided")

        # Очистка памяти перед загрузкой модели
        logger.info("🧹 Clearing unused memory...")
        gc.collect()
        torch.cuda.empty_cache()  # Если используешь GPU

        logger.info("📖 Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # ✅ Фиксируем проблему с отсутствием `pad_token`
        if tokenizer.pad_token is None:
            logger.warning("⚠️ Tokenizer has no `pad_token`, setting `eos_token` as padding.")
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("⚙️ Configuring quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        logger.info("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )

        logger.info("✅ Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model initialization error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return False

initialize_model()

# Функция для генерации саммари
def generate_mistral_summary(reviews: list, business_name: str) -> str:
    """Генерация саммари отзывов"""
    try:
        if not model or not tokenizer:
            logger.error("🚨 Model is not initialized")
            return "Model initialization failed"

        if not reviews:
            logger.warning("⚠️ No reviews provided")
            return "No reviews available for analysis"

        reviews = [review[:500] for review in reviews]
        prompt = f"<s>[INST] Summarize customer reviews about {business_name}:\n"
        for review in reviews[:50]:
            prompt += f"- {review}\n"
        prompt += "[/INST]"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=400,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.split("[/INST]")[-1].strip() if "[/INST]" in summary else summary.strip()

        return summary or "Unable to generate summary."

    except Exception as e:
        logger.error(f"🛑 Error generating summary: {e}")
        return "Unable to process reviews and generate summary at this time."

# Обработчик запроса
@app.post("/analyze-business-reviews-json")
async def analyze_business_reviews_json(payload: Dict[str, Any] = Body(...)):
    try:
        with open(os.path.join(LOG_DIR, "request_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n=== NEW REQUEST {datetime.datetime.now()} ===\n")
            f.write(f"Payload keys: {list(payload.keys())}\n")

        logger.debug("📥 Received payload")

        business_name = payload.get('Business Name', payload.get('business_name', 'Unknown Place'))
        reviews = payload.get('Extracted Reviews', [])

        review_texts = [r["text"] for r in reviews if "text" in r]

        summary = generate_mistral_summary(review_texts, business_name)

        result = {
            "nameOfPlace": business_name,
            "summary": summary,
            "averageScore": 4.0,  # Заглушка, рассчитай при необходимости
            "keyThemes": ["Service Quality", "Customer Experience"]
        }

        logger.debug(f"✅ Analysis result: {json.dumps(result, indent=2)}")

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"⚠️ Critical error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Проверка работоспособности
@app.get("/health")
async def health_check():
    status = "healthy" if model else "unhealthy"
    return JSONResponse({"status": status})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
