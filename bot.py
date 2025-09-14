import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from dotenv import load_dotenv
from telegram import (
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from textwrap import dedent

# Для ИИ
try:
    from openai import OpenAI  # type: ignore
except Exception:  # библиотека может быть не установлена локально
    OpenAI = None  # будет проверка в рантайме

# =====================
# Настройка логирования
# =====================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env, если есть
load_dotenv()

# Часовой пояс Ташкента (UTC+5)
TASHKENT_TZ = timezone(timedelta(hours=5))

# =====================
# Данные расписания
# =====================
# Нормализованные ключи: понедельник, вторник, среда, четверг, пятница, суббота
SCHEDULE: Dict[str, List[str]] = {
    "понедельник": [
        "1 Час будущего",
        "2 Английский язык",
        "3 Алгебра",
        "4 Узбекский язык",
        "5 Физкультура",
        "6 Воспитание",
    ],
    "вторник": [
        "Основы предпринимательства",
        "Алгебра",
        "Химия",
        "История Узбекистана",
    ],
    "среда": [
        "1 Химия",
        "2 Биология",
        "3 Узбекский язык",
        "4 Астрономия",
        "5 Родной язык",
        "6 ОГП",
    ],
    "четверг": [
        "1 Алгебра",
        "2 НВП",
        "3 Геометрия",
        "4 Английский язык",
        "5 Физика",
    ],
    "пятница": [
        "1 Узбекский язык",
        "2 Геометрия",
        "3 НВП",
        "4 Всемирная история",
        "5 Информатика",
        "6 Литература",
    ],
    "суббота": [
        "1 ытзтеп",
        "2 Физкультура",
        "3 Информатика",
        "4 Биология",
        "5 Литература",
    ],
}

# Синонимы и возможные варианты написаний
DAY_ALIASES = {
    "пн": "понедельник",
    "вт": "вторник",
    "ср": "среда",
    "чт": "четверг",
    "пт": "пятница",
    "сб": "суббота",
    # Возможные опечатки/регистр
    "понедельник": "понедельник",
    "вторник": "вторник",
    "среда": "среда",
    "четверг": "четверг",
    "пятница": "пятница",
    "суббота": "суббота",
    "субота": "суббота",
}

RUS_WEEK_ORDER = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота"]


def normalize_day(text: str) -> str | None:
    if not text:
        return None
    key = text.strip().lower()
    return DAY_ALIASES.get(key)


def weekday_ru_from_date(dt: datetime) -> str:
    # datetime.weekday(): Monday=0 ... Sunday=6
    idx = dt.weekday()
    if idx >= 6:
        # воскресенье — нет занятий, но вернём пустой ключ
        return "воскресенье"
    return RUS_WEEK_ORDER[idx]


def format_schedule_for_day(day_key: str) -> str:
    lessons = SCHEDULE.get(day_key)
    day_title = day_key.capitalize()
    if not lessons:
        return f"Расписание на {day_title}:\nНет занятий или информация отсутствует."
    lines = [f"<b>Расписание на {day_title}</b>:"]
    for item in lessons:
        lines.append(f"• {item}")
    return "\n".join(lines)


def full_week_schedule() -> str:
    parts: List[str] = []
    for day in RUS_WEEK_ORDER:
        parts.append(format_schedule_for_day(day))
    return "\n\n".join(parts)


# =====================
# ИИ помощник (домашка)
# =====================
SUPPORTED_SUBJECTS = {
    "алгебра": "Алгебра",
    "геометрия": "Геометрия",
    "русский": "Русский язык",
    "литература": "Литература",
    "информатика": "Информатика",
    "физика": "Физика",
    "химия": "Химия",
    "биология": "Биология",
    "английский": "Английский язык",
    "узбекский": "Узбекский язык",
    "астрономия": "Астрономия",
    "история": "История",
}


def get_openai_client() -> "OpenAI":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY. Задайте ключ для использования ИИ.")
    if OpenAI is None:
        raise RuntimeError("Библиотека openai не установлена. Установите 'openai' в requirements.txt.")
    return OpenAI(api_key=api_key)


async def ai_generate(subject: str, task_text: str) -> str:
    """Запрос к ИИ с безопасным промптом и ограничением длины ответа."""
    client = get_openai_client()
    system = dedent(f"""
        Ты — дружелюбный репетитор. Предмет: {SUPPORTED_SUBJECTS.get(subject, subject)}.
        Объясняй пошагово, ясно и кратко. Если запрос — сочинение, делай оригинальный текст.
        Если задача математическая — показывай ход решения и проверку.
        Если данных недостаточно — попроси уточнения.
        Язык ответа: русский.
    """)
    user = task_text.strip()

    # Модель можно заменить на любую доступную
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        logger.exception("Ошибка вызова OpenAI")
        content = f"Ошибка ИИ: {e}"
    return content


def split_long(text: str, limit: int = 3500) -> List[str]:
    parts: List[str] = []
    buf = []
    length = 0
    for line in text.splitlines():
        if length + len(line) + 1 > limit and buf:
            parts.append("\n".join(buf))
            buf = []
            length = 0
        buf.append(line)
        length += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    if not parts:
        parts = [text]
    return parts


# =====================
# Handlers
# =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет! Я бот с расписанием уроков.\n\n"
        "Команды:\n"
        "/today — расписание на сегодня\n"
        "/tomorrow — расписание на завтра\n"
        "/week — расписание на всю неделю\n"
        "ИИ помощь: /solve — решить задачу, /essay — написать сочинение, /cancel — сброс предмета.\n"
    )
    if update.message:
        await update.message.reply_text(text)
    elif update.callback_query:
        await update.callback_query.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


# ===== ИИ команды =====
async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("ai_subject", None)
    await update.message.reply_text("Сбросил выбранный предмет. Можешь выбрать заново, написав, например: Алгебра")


async def solve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = context.user_data.get("ai_subject", "алгебра")
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text(
            "Пришлите задачу текстом после команды. Пример:\n/solve Найди корни уравнения x^2 - 5x + 6 = 0"
        )
        return
    await update.message.reply_text("Думаю над решением…")
    answer = await ai_generate(subject, f"Задача по предмету '{subject}': {query}")
    for part in split_long(answer):
        await update.message.reply_text(part, parse_mode=ParseMode.HTML)


async def essay_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = context.user_data.get("ai_subject", "литература")
    topic = " ".join(context.args) if context.args else ""
    if not topic:
        await update.message.reply_text(
            "Пришлите тему сочинения после команды. Пример:\n/essay Дружба и честь в романе"
        )
        return
    await update.message.reply_text("Пишу черновик сочинения…")
    prompt = dedent(f"""
        Напиши оригинальное сочинение по теме: {topic}
       Объём: 200–300 слов. Структура: вступление, 2–3 абзаца основная часть, заключение.
        Избегай плагиата, не используй клише. Язык: русский.
    """)
    answer = await ai_generate(subject, prompt)
    for part in split_long(answer):
        await update.message.reply_text(part, parse_mode=ParseMode.HTML)


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now(TASHKENT_TZ)
    day_key = weekday_ru_from_date(now)
    if day_key == "воскресенье":
        await update.message.reply_text(
            "Сегодня воскресенье. Уроков нет."
        )
        return
    await update.message.reply_text(
        format_schedule_for_day(day_key), parse_mode=ParseMode.HTML
    )


async def tomorrow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now(TASHKENT_TZ) + timedelta(days=1)
    day_key = weekday_ru_from_date(now)
    if day_key == "воскресенье":
        await update.message.reply_text(
            "Завтра воскресенье. Уроков нет."
        )
        return
    await update.message.reply_text(
        format_schedule_for_day(day_key), parse_mode=ParseMode.HTML
    )


async def week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        full_week_schedule(), parse_mode=ParseMode.HTML
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip().lower()

    # Разрешим обработку текстов и в группах (если у бота есть доступ по настройкам privacy)

    if text in {"сегодня"}:
        return await today(update, context)
    if text in {"завтра"}:
        return await tomorrow(update, context)
    if text in {"вся неделя", "неделя"}:
        return await week(update, context)

    # Выбор предмета для ИИ: если пользователь пишет, например, "алгебра"
    if text in SUPPORTED_SUBJECTS:
        context.user_data["ai_subject"] = text
        await update.message.reply_text(
            f"Выбран предмет: {SUPPORTED_SUBJECTS[text]}. Пришлите задачу (или используйте /solve)."
        )
        return

    # Короткие триггеры ИИ: "реши ...", "сочинение ..."
    if text.startswith("реши "):
        context.args = [text[5:]]  # передадим как аргументы в /solve
        return await solve_cmd(update, context)
    if text.startswith("сочинение "):
        context.args = [text[10:]]
        return await essay_cmd(update, context)

    day_key = normalize_day(text)
    if day_key in SCHEDULE:
        await update.message.reply_text(
            format_schedule_for_day(day_key), parse_mode=ParseMode.HTML
        )
        return

    await update.message.reply_text(
        "Не понял запрос. Используйте команды /today, /tomorrow, /week или напишите день недели (например: Понедельник).",
    )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error(
            "Не найден TELEGRAM_BOT_TOKEN. Установите переменную окружения или создайте файл .env"
        )
        print(
            "Пожалуйста, установите токен: в PowerShell -> $env:TELEGRAM_BOT_TOKEN=\"<ВАШ_ТОКЕН>\""
        )
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("tomorrow", tomorrow))
    app.add_handler(CommandHandler("week", week))
    app.add_handler(CommandHandler("solve", solve_cmd))
    app.add_handler(CommandHandler("essay", essay_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Определяем режим работы: webhook (Render) или polling (локально)
    webhook_url = os.getenv("WEBHOOK_URL")  # например: https://your-service.onrender.com/webhook
    webhook_secret = os.getenv("WEBHOOK_SECRET")  # опционально
    port = int(os.getenv("PORT", "10000"))  # Render задаёт PORT автоматически

    if webhook_url:
        logger.info("Запуск в режиме Webhook")
        logger.info(f"WEBHOOK_URL = {webhook_url}")
        try:
            # PTB поднимет встроенный aiohttp-сервер и установит webhook у Telegram
            app.run_webhook(
                listen="0.0.0.0",
                port=port,
                webhook_url=webhook_url,
                secret_token=webhook_secret,
                drop_pending_updates=True,
                close_loop=False,
            )
        except Exception:
            logger.exception("Ошибка запуска в режиме webhook")
            raise
    else:
        logger.info("WEBHOOK_URL не задан. Запуск в режиме Polling")
        app.run_polling(drop_pending_updates=True, close_loop=False)


if __name__ == "__main__":
    main()
