#!/usr/bin/env python3
"""
SuperAIGamingBot v6.2.9 - Compatible con plan gratuito de X API y flujo corregido para tweets bilingües
Autor: Jose Manuel (con asistencia de Grok 3)
Versión: 6.2.9
Fecha: Abril 2025
Python: 3.11.9 a 3.13.2
"""
import asyncio
import logging
import logging.handlers
import random
import time
import json
import os
import sys
from queue import PriorityQueue
import shutil
from colorlog import ColoredFormatter
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from requests.exceptions import RequestException
from cachetools import TTLCache
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import aiohttp
import aiosqlite
import sqlite3
import discord
from discord.ext import commands
import pytz
import feedparser
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification
import spacy
from deep_translator import GoogleTranslator
import langdetect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import trafilatura
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googleapiclient.discovery import build
import concurrent.futures
import urllib.parse
import pickle
import hashlib
import tweepy
from textblob import TextBlob
from dotenv import load_dotenv

# Configurar el bucle de eventos para Windows al inicio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Cargar variables de entorno desde datos.env
load_dotenv(dotenv_path="datos.env")

# Configuración Base
BASE = {
    "RSS_SOURCES": [
        "https://www.ign.com/rss",
        "https://www.gamespot.com/feeds/news/",
        "https://kotaku.com/rss",
        "https://www.polygon.com/rss/index.xml"
    ],
    "RSS_TWEET_SOURCES_EN": [
        "https://nitter.net/Xbox/rss",
        "https://nitter.net/RockstarGames/rss",
        "https://nitter.net/CallofDuty/rss",
        "https://nitter.net/HelloGames/rss",
        "https://nitter.net/PlayStation/rss",
        "https://nitter.net/Steam/rss",
        "https://nitter.net/EpicGames/rss",
        "https://nitter.net/NintendoAmerica/rss",
        "https://nitter.net/IGN/rss",
        "https://nitter.net/GameSpot/rss"
    ],
    "RSS_TWEET_SOURCES_ES": [
        "https://nitter.net/Xbox_Spain/rss",
        "https://nitter.net/PlayStationES/rss",
        "https://nitter.net/NintendoES/rss",
        "https://nitter.net/3DJuegos/rss",
        "https://nitter.net/VidaExtra/rss",
        "https://nitter.net/Xataka/rss",
        "https://nitter.net/Espinof/rss"
    ]
}
RSS_SOURCES = BASE["RSS_SOURCES"]
RSS_TWEET_SOURCES_EN = BASE["RSS_TWEET_SOURCES_EN"]
RSS_TWEET_SOURCES_ES = BASE["RSS_TWEET_SOURCES_ES"]

# Palabras clave relevantes
RELEVANT_KEYWORDS = {
    "game", "gaming", "videojuego", "console", "ps5", "xbox", "nintendo", "release", "update", "trailer", "sales",
    "tech", "technology", "hardware", "gpu", "rocket", "space", "nasa", "spacex", "cine", "película", "film",
    "patch", "hotfix", "playstation", "sony", "switch", "nintendo direct", "pc gaming", "steam deck", "valve", "geforce",
    "rtx", "amd", "gpu", "hardware", "frame rate", "fps", "benchmark", "gameplay", "review", "análisis", "pre-order",
    "demo", "beta", "alpha", "roadmap", "leak", "filtración", "rumor", "remake", "remaster", "reboot", "port", "crossplay",
    "indie", "triple A", "AAA", "metroidvania", "roguelike", "soulslike", "patch notes", "parche", "balance", "buffs",
    "nerfs", "matchmaking", "ranked", "ladder", "meta", "battle pass", "loot box", "skin", "cosmetic", "crossover",
    "collab", "mod", "custom", "speedrun", "glitch", "exploit", "e3", "gamescom", "tga", "summer game fest", "showcase",
    "fanfest", "convention", "torneo", "competition", "championship", "twitch", "youtube gaming", "streamer", "rpg",
    "mmo", "mmorpg", "battle royale", "vr", "ar", "ubisoft", "activision", "blizzard", "bethesda", "square enix", "capcom",
    "sega", "bandai namco", "cd projekt", "rockstar", "take-two", "epic games", "unity", "unreal engine", "game pass",
    "subscription", "live service", "lore", "historia", "narrativa", "open world", "sandbox", "co-op", "multiplayer",
    "singleplayer", "pvp", "pve", "raids", "dungeon", "bosses", "quest", "loot", "crafting", "economy", "ranking",
    "leaderboard"
}

# Silenciar advertencias de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configuración de Logging
logger = logging.getLogger("SuperAIGamingBot")
logger.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red"}
)

file_handler = logging.handlers.RotatingFileHandler(
    f'bot_log_{datetime.now().strftime("%Y%m%d")}.log', maxBytes=10**6, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

discord_logger = logging.getLogger('discord')
discord_logger.setLevel(logging.WARNING)

# Directorios y Archivos
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
DB_FILE = BASE_DIR / "tweets_queue.db"
HISTORY_FILE = BASE_DIR / "tweet_history.json"
LIMIT_FILE = BASE_DIR / "tweet_limit.json"
STATS_FILE = BASE_DIR / "bot_stats.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
TRAINING_DATA_FILE = BASE_DIR / "training_data.json"

# Semáforo para tareas múltiples
db_semaphore = asyncio.Semaphore(1)

# Caché en memoria
CACHE = TTLCache(maxsize=1000, ttl=24 * 3600)

# Descargar recursos de NLTK
nltk.download('vader_lexicon', quiet=True)

# Cargar modelos
nlp = spacy.load("es_core_news_md")
sia = SentimentIntensityAnalyzer()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Validación y Carga de Configuración
def validate_config(config: Dict[str, any]) -> bool:
    required_keys = ["plan_level", "x_api_en", "x_api_es", "youtube_api_key", "discord_token", "discord_channel_ids"]
    for key in required_keys:
        if key not in config or not config[key]:
            logger.error(f"❌ Configuración inválida: Falta o está vacía la clave '{key}'")
            return False
    if config["plan_level"] not in ["free", "pro"]:
        logger.error("❌ Configuración inválida: 'plan_level' debe ser 'free' o 'pro'")
        return False
    for api_key in ["x_api_en", "x_api_es"]:
        if not all(k in config[api_key] for k in ["consumer_key", "consumer_secret", "access_token", "access_token_secret"]):
            logger.error(f"❌ Configuración inválida: Faltan claves en '{api_key}'")
            return False
    if not config["discord_channel_ids"]:
        logger.error("❌ Configuración inválida: 'discord_channel_ids' está vacío")
        return False
    return True

def load_config() -> Dict[str, any]:
    config = {
        "plan_level": os.getenv("PLAN_LEVEL", "pro"),
        "x_api_en": {
            "consumer_key": os.getenv("TWITTER_API_KEY_EN", ""),
            "consumer_secret": os.getenv("TWITTER_API_SECRET_EN", ""),
            "access_token": os.getenv("TWITTER_ACCESS_TOKEN_EN", ""),
            "access_token_secret": os.getenv("TWITTER_ACCESS_SECRET_EN", "")
        },
        "x_api_es": {
            "consumer_key": os.getenv("TWITTER_API_KEY_ES", ""),
            "consumer_secret": os.getenv("TWITTER_API_SECRET_ES", ""),
            "access_token": os.getenv("TWITTER_ACCESS_TOKEN_ES", ""),
            "access_token_secret": os.getenv("TWITTER_ACCESS_SECRET_ES", "")
        },
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY", ""),
        "discord_token": os.getenv("DISCORD_TOKEN", ""),
        "discord_channel_ids": [
            int(os.getenv("DISCORD_CHANNEL_1", 0)),
            int(os.getenv("DISCORD_CHANNEL_2", 0))
        ],
        "daily_tweet_limit": int(os.getenv("DAILY_TWEET_LIMIT", 50)),
        "youtube_quota_limit": int(os.getenv("YOUTUBE_QUOTA_LIMIT", 10000)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 0.0003)),
        "relevance_threshold": float(os.getenv("RELEVANCE_THRESHOLD", 0.4)),
        "min_attempts_for_dl": int(os.getenv("MIN_ATTEMPTS_FOR_DL", 100)),
        "sleep_times": {
            "heartbeat": int(os.getenv("SLEEP_HEARTBEAT", 60)),
            "model_save_interval": int(os.getenv("SLEEP_MODEL_SAVE_INTERVAL", 3600))
        }
    }
    try:
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open('r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
            logger.info("✅ Configuración cargada desde config.json y combinada con datos.env")
        else:
            with CONFIG_FILE.open('w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logger.info("📝 Configuración inicial guardada en config.json desde datos.env")
        if not validate_config(config):
            raise ValueError("Configuración inválida, revisa los logs.")
    except Exception as e:
        logger.error(f"❌ Error cargando configuración: {e}", exc_info=True)
        raise
    return config

CONFIG = load_config()

# Constantes
YOUTUBE_CHANNELS = ["UCvC4D8onUfXzvjTOM-dBfEA", "UCdykEBeltanKGS7oXOB1QZQ"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
]
SPAM_FILTER = {"ad", "sponsor", "giveaway", "promo"}
UNWANTED_DOMAINS = {"login", "signup", "advertisement", "forum", "wiki"}
TRUSTED_GAMING_SITES = {
    "www.gamespot.com", "xbox.com", "direct.playstation.com", "ign.com", "kotaku.com", "eurogamer.es", "snail.com",
    "ign.com", "famitsu.com", "polygon.com", "rockpapershotgun.com", "vg247.com", "engadget.com", "theverge.com",
    "destructoid.com", "gematsu.com", "gameinformer.com", "escapistmagazine.com", "pcgamer.com", "mmorpg.com",
    "gamesindustry.biz", "gamasutra.com", "gameranx.com", "gamesradar.com", "rockstargames.com", "ea.com", "ubisoft.com",
    "blizzard.com", "nintendo.com", "playstation.com", "xbox.com", "steampowered.com", "epicgames.com", "gog.com",
    "humblebundle.com", "youtube.com"
}
FUENTES_CONFIABLES = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.gameinformer.com/rss.xml",
    "https://www.reutersagency.com/feed/?taxonomy=sections&term=technology",
    "https://www.gamespot.com/feeds/news/",
    "https://news.xbox.com/en-us/feed",
    "https://blog.playstation.com/feed/",
    "https://www.ign.com/rss/articles.xml",
    "https://kotaku.com/rss",
    "https://www.eurogamer.es/rss",
    "https://snail.com/news/rss",
    "https://es.ign.com/rss",
    "https://www.famitsu.com/rss/famitsu.rss",
    "https://www.polygon.com/rss/index.xml",
    "https://www.rockpapershotgun.com/feed",
    "https://www.vg247.com/feed",
    "https://www.engadget.com/rss.xml",
    "https://www.theverge.com/rss/gaming/index.xml",
    "https://www.destructoid.com/feed/",
    "https://www.gematsu.com/feed",
    "https://www.gameinformer.com/feeds/all.xml",
    "https://www.escapistmagazine.com/v2/feed/",
    "https://www.pcgamer.com/rss",
    "https://www.mmorpg.com/rss",
    "https://www.gamesindustry.biz/rss",
    "https://www.gamasutra.com/static2/gamasutra.rss",
    "https://www.gameranx.com/feed/",
    "https://www.gamesradar.com/rss/",
    "https://www.rockstargames.com/newswire/RSS",
    "https://www.ea.com/news/rss",
    "https://www.ubisoft.com/en-us/company/newsroom/rss",
    "https://www.blizzard.com/en-us/news/rss",
    "https://www.nintendo.com/news/rss",
    "https://www.playstation.com/en-us/news/rss",
    "https://www.xbox.com/en-us/news/rss",
    "https://www.steampowered.com/news/rss",
    "https://www.epicgames.com/store/en-US/news/rss",
    "https://www.gog.com/news/rss",
    "https://www.humblebundle.com/blog/rss"
]

# Funciones de Carga y Guardado
def load_history() -> set:
    try:
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open('r', encoding='utf-8') as f:
                return set(json.load(f))
        else:
            with HISTORY_FILE.open('w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("📜 Historial inicial creado como vacío")
        return set()
    except Exception as e:
        logger.error(f"❌ Error cargando historial: {e}", exc_info=True)
        return set()

def save_history(history: set) -> None:
    try:
        with HISTORY_FILE.open('w', encoding='utf-8') as f:
            json.dump(list(history), f)
        logger.debug("💾 Historial guardado")
    except Exception as e:
        logger.error(f"❌ Error guardando historial: {e}", exc_info=True)

def load_tweet_limit() -> Dict[str, Union[int, str]]:
    default = {"daily_en": 0, "daily_es": 0, "reset_time": datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()}
    try:
        if LIMIT_FILE.exists():
            with LIMIT_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return data if "daily_en" in data and "daily_es" in data and "reset_time" in data else default
        return default
    except Exception as e:
        logger.error(f"❌ Error cargando límite diario: {e}", exc_info=True)
        save_tweet_limit(default)
        return default

def save_tweet_limit(limit_data: Dict[str, Union[int, str]]) -> None:
    try:
        with LIMIT_FILE.open('w', encoding='utf-8') as f:
            json.dump(limit_data, f)
        logger.debug("💾 Límite diario guardado")
    except Exception as e:
        logger.error(f"❌ Error guardando límite diario: {e}", exc_info=True)

def load_stats() -> Dict[str, any]:
    default_stats = {
        "total_tweets": 0,
        "tweet_history": [],
        "performance_metrics": {"latency": 0.0, "success_rate": 0.0}
    }
    try:
        if STATS_FILE.exists():
            with STATS_FILE.open("r", encoding='utf-8') as f:
                loaded_stats = json.load(f)
                for key in default_stats:
                    if key not in loaded_stats:
                        loaded_stats[key] = default_stats[key]
                    elif isinstance(default_stats[key], dict):
                        for subkey in default_stats[key]:
                            if subkey not in loaded_stats[key]:
                                loaded_stats[key][subkey] = default_stats[key][subkey]
                return loaded_stats
        return default_stats
    except Exception as e:
        logger.error(f"❌ Error cargando stats: {e}", exc_info=True)
        save_stats(default_stats)
        return default_stats

def save_stats(stats: Dict[str, any]) -> None:
    try:
        with STATS_FILE.open("w", encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        logger.debug("💾 Estadísticas guardadas")
    except Exception as e:
        logger.error(f"❌ Error guardando estadísticas: {e}", exc_info=True)

TWEET_HISTORY = load_history()
TWEET_LIMIT = load_tweet_limit()
STATS = load_stats()

# Modelos de Deep Learning
class AdvancedGamingNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relevance_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.generator_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.text_generator = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.relevance_head = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(self.device)
        self.relevance_head.trained = False
        self.tweet_queue = PriorityQueue()
        self.state_encoder = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()).to(self.device)
        self.combined_head = nn.Sequential(nn.Linear(768 + 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()).to(self.device)
        self.policy_head = nn.Linear(128, action_size).to(self.device)
        self.value_head = nn.Linear(128, 1).to(self.device)
        self.scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("es_core_news_md")
        self.generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

    def forward(self, texts: List[str], states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            inputs = self.relevance_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            text_embeds = self.relevance_head.distilbert(**inputs).last_hidden_state.mean(dim=1)
            relevance = torch.sigmoid(self.relevance_head(**inputs).logits)
            state_embeds = self.state_encoder(states.to(self.device))
            combined = torch.cat([text_embeds, state_embeds], dim=-1)
            features = self.combined_head(combined)
            policy = self.policy_head(features)
            value = self.value_head(features)
            return policy, value, relevance

    def is_coherent(self, text: str) -> bool:
        if not text:
            return False
        if re.search(r"(\b\w+\b)(\s+\1){2,}", text) or re.search(r"([!?.])\1{3,}", text):
            logger.debug(f"⚠️ Texto incoherente detectado: '{text[:50]}...'")
            return False
        if len(text.split()) < 3:
            return False
        return True

    def is_spanish(self, text: str) -> bool:
        try:
            doc = self.nlp(text)
            spanish_tokens = sum(1 for token in doc if token.lang_ == "es" or token.is_punct or token.is_space)
            total_tokens = len(doc)
            return spanish_tokens / total_tokens > 0.8
        except Exception:
            logger.error(f"❌ Error detectando idioma en '{text}'")
            return False

    async def shorten_url_tinyurl(self, url: str) -> str:
        if not url or not url.startswith("http"):
            logger.debug(f"⚠️ URL inválida para acortar: {url}")
            return url
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(url)}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        short_url = await resp.text()
                        if not short_url.startswith("https://"):
                            short_url = "https://" + short_url
                        logger.debug(f"🔗 URL acortada: {url} -> {short_url}")
                        return short_url
                    logger.warning(f"⚠️ Error acortando URL {url}: {resp.status}")
        except Exception as e:
            logger.error(f"❌ Error acortando URL {url}: {e}")
        return url

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ").strip()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"([!?.])\1{2,}", r"\1\1", text)
        text = re.sub(r"[^\w\s.,!?¡¿áéíóúÁÉÍÓÚñÑ-]", "", text)
        text = " ".join(text.split()).strip()
        logger.debug(f"🧹 Texto limpio: '{text[:50]}...'")
        return text

    async def generate_tweet(self, text: str, category: str = "general", url: str = "", language: str = "en", max_length: int = 280) -> str:
        emoji_dict = {
            "noticias": ["📰", "✨", "🔥"], "video": ["🎥", "🎬", "🔊"],
            "ventas": ["📈", "💰", "🏆"], "general": ["🎮", "🕹️", "💥"]
        }
        hashtag_dict = {
            "noticias": ["#GamingNews", "#NoticiasGaming"], "video": ["#GamingVideos", "#Videojuegos"],
            "ventas": ["#GameSales", "#OfertasGaming"], "general": ["#Gaming", "#Videojuegos"]
        }
        
        clean_text = self.clean_text(text)
        if not clean_text:
            clean_text = "Latest gaming update!" if language == "en" else "¡Última actualización gaming!"
        
        is_gaming = any(kw in clean_text.lower() for kw in RELEVANT_KEYWORDS)
        category = category if category in emoji_dict else ("noticias" if is_gaming else "general")
        emojis = random.sample(emoji_dict.get(category, ["🎮"]), 2)
        trends = list(self.bot.trends) if hasattr(self.bot, 'trends') else ["Gaming"]
        trend_tag = next((f"#{trend.replace(' ', '')}" for trend in trends if trend.lower() in clean_text.lower()), 
                        "#Videojuegos" if language == "es" else "#Gaming")
        hashtags = f"{random.choice(hashtag_dict.get(category, ['#Gaming']))} {trend_tag}"
        
        prompt = f"Create an engaging gaming tweet based on: {clean_text}. Keep it concise, exciting, and informative, no emojis or hashtags."
        for attempt in range(3):
            try:
                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    generated = self.generator(prompt, max_new_tokens=80, temperature=0.85, top_p=0.9, 
                                            do_sample=True, num_return_sequences=1)[0]["generated_text"]
                generated_en = generated.replace(prompt, "").strip()
                
                if self.is_coherent(generated_en) and len(generated_en.split()) > 5:
                    final_text = generated_en
                    if language == "es":
                        final_text = await self.translate_to_spanish(generated_en)
                        if not self.is_coherent(final_text):
                            final_text = GoogleTranslator(source="en", target="es").translate(generated_en)
                    
                    short_url = await self.shorten_url_tinyurl(url) if url else ""
                    emoji_str = " ".join(emojis)
                    hashtag_str = hashtags
                    
                    base_components = f"{emoji_str} {final_text} {hashtag_str} {short_url}"
                    base_len = len(base_components)
                    target_min, target_max = 260, 280
                    
                    if base_len < target_min:
                        extra_context = {
                            "en": ["Check this out!", "Big news for gamers!", "Hot gaming update!"],
                            "es": ["¡Mira esto!", "¡Gran noticia para gamers!", "¡Actualización caliente!"]
                        }
                        addition = random.choice(extra_context[language])
                        final_text += f" {addition}"
                        base_components = f"{emoji_str} {final_text} {hashtag_str} {short_url}"
                        base_len = len(base_components)
                        
                        if base_len < target_min:
                            padding = " Don't miss it!" if language == "en" else " ¡No te lo pierdas!"
                            final_text += padding[:target_min - base_len]
                    
                    elif base_len > target_max:
                        words = final_text.split()
                        while len(f"{emoji_str} {' '.join(words)} {hashtag_str} {short_url}") > target_max:
                            words.pop()
                        final_text = " ".join(words) + "..."
                    
                    tweet = f"{emoji_str} {final_text} {hashtag_str} {short_url}".strip()
                    tweet_len = len(tweet)
                    
                    if target_min <= tweet_len <= target_max and self.is_coherent(tweet):
                        logger.debug(f"📝 Tweet generado: '{tweet}' ({tweet_len} chars)")
                        return tweet
                    
            except Exception as e:
                logger.error(f"❌ Error generando tweet (intento {attempt + 1}): {e}")
        
        fallback_text = clean_text[:80]
        if language == "es":
            fallback_text = await self.translate_to_spanish(fallback_text)
        fallback = f"{' '.join(emojis)} {fallback_text} Exciting news! {hashtags} {await self.shorten_url_tinyurl(url)}"
        if len(fallback) > target_max:
            fallback = fallback[:target_max-3] + "..."
        elif len(fallback) < target_min:
            padding = " Don't miss it!" if language == "en" else " ¡No te lo pierdas!"
            fallback += padding[:target_min - len(fallback)]
        
        final_len = len(fallback)
        if target_min <= final_len <= target_max:
            logger.warning(f"⚠️ Usando tweet fallback: '{fallback}' ({final_len} chars)")
            return fallback
        
        while len(fallback) > target_max:
            fallback = fallback[:target_max-3] + "..."
        while len(fallback) < target_min:
            fallback += "!"
        logger.warning(f"⚠️ Fallback ajustado: '{fallback}' ({len(fallback)} chars)")
        return fallback

    async def translate_to_spanish(self, text: str) -> str:
        clean_text = self.clean_text(text)
        if not clean_text:
            logger.debug("⚠️ Texto vacío tras limpieza, devolviendo original")
            return text
        try:
            result = GoogleTranslator(source="auto", target="es").translate(clean_text)
            logger.debug(f"📝 Traducción Google: '{clean_text}' -> '{result}'")
            return result
        except Exception as e:
            logger.error(f"❌ Error en GoogleTranslator: {e}")
            return clean_text

    def train_relevance_head(self, train_dataset: List[Dict[str, Union[str, float]]], epochs: int = 5, batch_size: int = 16) -> None:
        if not train_dataset:
            logger.warning("⚠️ Dataset vacío, generando datos sintéticos.")
            train_dataset = self._generate_synthetic_data()
        logger.info(f"📚 Entrenando relevance_head con {len(train_dataset)} ejemplos")
        self.relevance_head.train()
        optimizer = optim.AdamW(self.relevance_head.parameters(), lr=CONFIG["learning_rate"])
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            random.shuffle(train_dataset)
            total_loss = 0
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i + batch_size]
                texts = [item["text"] for item in batch]
                labels = torch.FloatTensor([item["label"] for item in batch]).to(self.device)
                inputs = self.relevance_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                outputs = self.relevance_head(**inputs)
                loss = loss_fn(outputs.logits.squeeze(), labels)
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_dataset) // batch_size)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        self.relevance_head.eval()
        self.relevance_head.trained = True
        torch.save(self.relevance_head.state_dict(), MODEL_DIR / "relevance_head.pth")
        logger.info("✅ Modelo de relevancia entrenado y guardado")

    def _generate_synthetic_data(self) -> List[Dict[str, Union[str, float]]]:
        return [
            {"text": "New PS5 game released today", "label": 1.0},
            {"text": "Xbox Series X update coming soon", "label": 1.0},
            {"text": "Win a free gift card", "label": 0.0},
            {"text": "Gaming news: Nintendo Switch 2 rumors", "label": 1.0},
            {"text": "Sign up for our newsletter", "label": 0.0},
        ]

class ResourceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x.to(self.device))
        return self.fc(hn[-1])

# Decorador de Reintentos
def retry_with_backoff(max_attempts: int = 3):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, RequestException)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )

# Clase Principal del Bot
class SuperAIGamingBot(commands.Cog):
    def __init__(self):
        self.bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
        self.bot.bot_instance = self
        self.is_paused = False
        self.tweet_count = 0
        self.tweet_cache = set()
        self.max_cache_size = 1000
        self.db_semaphore = asyncio.Semaphore(1)
        self.twitter_enabled = True
        self.youtube = build('youtube', 'v3', developerKey=CONFIG["youtube_api_key"])
        self.youtube_quota_used = 0
        self.last_quota_reset = datetime.now(pytz.UTC)
        self.model = AdvancedGamingNet(state_size=10, action_size=4)
        self.model.bot = self
        self.resource_predictor = ResourceLSTM()
        self.resource_optimizer = optim.Adam(self.resource_predictor.parameters(), lr=CONFIG["learning_rate"])
        self.actions = ["post_now", "enqueue", "skip", "adjust_relevance"]
        self.env = DummyVecEnv([self._make_env])
        self.ppo = PPO("MlpPolicy", self.env, learning_rate=CONFIG["learning_rate"], verbose=0)
        self.tweet_cache = load_history()
        self.stats = load_stats()
        self.tweet_ids = []
        self.error_count = 0
        self.last_action_time = time.time()
        self.url_cache = {}
        self.nlp = spacy.load("es_core_news_sm")
        logger.info("✅ Modelo NLP de spaCy cargado")
        self.image_cache = {}
        self.trends = {"gaming", "ps5", "xbox", "steam"}
        self.last_trend_update = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.priority_queue = PriorityQueue()
        self.budget = {"youtube": 0.3, "rss": 0.4, "vgchartz": 0.3}
        self.connection_pool = None
        self.running = True
        self.paused = False
        self.memory = []
        self.BOT_DOMAINS = {"patchbot.io", "otherbot.com"}
        self.setup_twitter_clients()
        self.translation_tokenizer = None
        self.translation_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_relevance_head()
        self.new_training_data = []
        self.reads_count = 0
        self.DB_FILE = DB_FILE
        self.reads_request_count = 0
        self.last_reads_reset = datetime.now(pytz.UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    def setup_twitter_clients(self):
        try:
            auth_en = tweepy.OAuthHandler(CONFIG["x_api_en"]["consumer_key"], CONFIG["x_api_en"]["consumer_secret"])
            auth_en.set_access_token(CONFIG["x_api_en"]["access_token"], CONFIG["x_api_en"]["access_token_secret"])
            self.twitter_api_en = tweepy.API(auth_en, wait_on_rate_limit=True)
            self.twitter_client_en = tweepy.Client(
                consumer_key=CONFIG["x_api_en"]["consumer_key"],
                consumer_secret=CONFIG["x_api_en"]["consumer_secret"],
                access_token=CONFIG["x_api_en"]["access_token"],
                access_token_secret=CONFIG["x_api_en"]["access_token_secret"]
            )
            auth_es = tweepy.OAuthHandler(CONFIG["x_api_es"]["consumer_key"], CONFIG["x_api_es"]["consumer_secret"])
            auth_es.set_access_token(CONFIG["x_api_es"]["access_token"], CONFIG["x_api_es"]["access_token_secret"])
            self.twitter_api_es = tweepy.API(auth_es, wait_on_rate_limit=True)
            self.twitter_client_es = tweepy.Client(
                consumer_key=CONFIG["x_api_es"]["consumer_key"],
                consumer_secret=CONFIG["x_api_es"]["consumer_secret"],
                access_token=CONFIG["x_api_es"]["access_token"],
                access_token_secret=CONFIG["x_api_es"]["access_token_secret"]
            )
            logger.info("✅ Clientes de Twitter inicializados (EN y ES)")
        except Exception as e:
            logger.error(f"❌ Error inicializando clientes de Twitter: {e}")
            self.twitter_enabled = False

    async def enqueue_tweet(self, content: str, language: str, media_url: Optional[str] = None, source_type: int = 0, priority: int = 0):
        if content in self.tweet_cache:
            logger.debug(f"⚠️ Tweet duplicado: '{content[:50]}...'")
            return False
        try:
            final_media_url = media_url if (media_url and isinstance(media_url, str) and media_url.startswith("http")) else None
            if media_url and final_media_url is None:
                logger.warning(f"⚠️ Media URL inválida descartada: {media_url}")
            url_match = re.search(r'https?://\S+', content)
            final_url = url_match.group(0) if url_match else None
            if final_url and (not final_media_url or not await self.validate_image(final_media_url)):
                final_media_url, final_url = await self.ensure_relevant_image(content, final_url, final_media_url)
            base_priority = priority
            _, _, relevance_score = self.model([content], torch.FloatTensor(self._get_state(await self.get_queue_size(), TWEET_LIMIT["daily_en"] + TWEET_LIMIT["daily_es"])).unsqueeze(0))
            is_recent_bonus = 5 if self.is_recent(datetime.now(pytz.UTC), threshold=timedelta(hours=6)) else 0
            final_priority = base_priority + int(10 * relevance_score.item()) + is_recent_bonus
            async with self.db_semaphore:
                async with aiosqlite.connect(self.DB_FILE, timeout=10) as db:
                    await db.execute(
                        "INSERT OR IGNORE INTO tweets_queue (content, language, media_url, source_type, priority) VALUES (?, ?, ?, ?, ?)",
                        (content, language, final_media_url, source_type, final_priority)
                    )
                    await db.commit()
            self.tweet_cache.add(content)
            logger.info(f"📥 Encolado: '{content[:50]}...' (Idioma: {language}, Prioridad: {final_priority})")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error encolando: '{content[:50]}...' - {e}")
            return False

    async def process_rss_tweets(self, language: str):
     rss_sources = RSS_TWEET_SOURCES_EN if language == "en" else RSS_TWEET_SOURCES_ES
     async with aiohttp.ClientSession() as session:
        for feed_url in rss_sources:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    summary = entry.get("summary", title)
                    tweet_match = re.search(r'https://(nitter\.net|twitter\.com|x\.com)/(\w+)/status/(\d+)', link)
                    if tweet_match:
                        tweet_id = tweet_match.group(2)  # Captura el ID
                        await self.repost_tweet(tweet_id, language)
                    elif await self.is_relevant(summary, link):
                        tweet = await self.model.generate_tweet(summary, category="general", url=link, language=language)
                        await self.enqueue_tweet(tweet, language, None, source_type=6, priority=15)
                    logger.debug(f"📡 Procesado RSS tweet: {title[:50]}...")
            except Exception as e:
                logger.error(f"❌ Error procesando feed de tweets {feed_url}: {e}")
                self.error_count += 1

    async def repost_tweet(self, tweet_id: str, language: str):
        try:
            twitter_client = self.twitter_client_en if language == "en" else self.twitter_client_es
            twitter_client.retweet(tweet_id)
            logger.info(f"📤 Tweet reposteado: ID {tweet_id} en {'inglés' if language == 'en' else 'español'}")
            self.stats["total_tweets"] += 1
            self.stats["tweet_history"].append({
                "content": f"Retweet ID: {tweet_id}",
                "language": language,
                "source": "rss_repost"
            })
            save_stats(self.stats)
        except tweepy.TweepyException as e:
            logger.error(f"❌ Error reposteando tweet {tweet_id}: {e}")
            self.error_count += 1

    async def repost_tweet_from_url(self, url: str, title: str):
        tweet_match = re.search(r'twitter\.com/\w+/status/(\d+)', url)
        if tweet_match:
            tweet_id = tweet_match.group(1)
            await self.repost_tweet(tweet_id, "en")
            await self.repost_tweet(tweet_id, "es")

    async def fetch_reddit_posts(self):
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.reddit.com/r/gaming/new.json?limit=10") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for post in data["data"]["children"]:
                        title = post["data"]["title"]
                        url = post["data"]["url"]
                        tweet_match = re.search(r'https://twitter\.com/\w+/status/(\d+)', url)
                        if tweet_match:
                            await self.repost_tweet_from_url(url, title)
                        elif await self.is_relevant(title, url):
                            tweet_en = await self.model.generate_tweet(title, category="general", url=url, language="en")
                            tweet_es = await self.model.generate_tweet(title, category="general", url=url, language="es")
                            await self.enqueue_tweet(tweet_en, "en", None, source_type=4, priority=1)
                            await self.enqueue_tweet(tweet_es, "es", None, source_type=4, priority=1)

    async def process_content_periodically(self):
        while self.running:
            logger.info("⏳ Procesando contenido periódicamente...")
            try:
                await self.check_tweet_limit()
                trends = list(self.trends)
                for feed_url in RSS_SOURCES:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:3]:
                        title = entry.get("title", "")
                        link = entry.get("link", "")
                        summary = entry.get("summary", title)
                        entry_dict = {
                            "title": title,
                            "link": link,
                            "summary": summary,
                            "source": feed_url,
                            "date": datetime.now(pytz.UTC)
                        }
                        if await self.is_relevant(title, link):
                            await self.process_single_news(entry_dict, trends)
                await self.process_rss_tweets("en")
                await self.process_rss_tweets("es")
                await self.fetch_reddit_posts()
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"❌ Error en process_content_periodically: {e}")
                await asyncio.sleep(300)

    async def score_news(self, entry: Dict[str, str], trends: List[str]) -> float:
        try:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            url = entry.get("link", "")
            source = entry.get("source", "unknown")
            news_date = entry.get("date", datetime.now(pytz.UTC))
            if await self.is_duplicate(title, summary):
                logger.debug(f"🗑️ Noticia duplicada: {title[:50]}")
                return 0
            content = (title + " " + summary).lower()
            all_keywords = RELEVANT_KEYWORDS
            if not any(kw in content for kw in all_keywords):
                return 0
            negative_keywords = {"appliance", "refrigerator", "oven", "microwave", "washer", "dryer", "kitchen"}
            if any(nkw in content for nkw in negative_keywords):
                return 0
            keyword_score = sum(1 for kw in all_keywords if kw in content) * 2.0
            sentiment = TextBlob(content).sentiment.polarity
            sentiment_bonus = max(0, sentiment) * 2
            freshness = self.calculate_freshness(news_date)
            trend_score = sum(2 for trend in trends if trend.lower() in content)
            diversity = self.calculate_diversity(source)
            async with aiosqlite.connect(DB_FILE) as db:
                async with db.execute("SELECT content FROM tweet_history WHERE posted_at > ?", 
                                    ((datetime.now(pytz.UTC) - timedelta(hours=48)).isoformat(),)) as cursor:
                    past_news = [row[0] async for row in cursor]
            if past_news:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([content] + past_news)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0].max()
                similarity_penalty = -10 if similarity > 0.8 else 0
            else:
                similarity_penalty = 0
            w_relevance, w_freshness, w_sentiment, w_trends, w_diversity, w_similarity = 0.5, 0.2, 0.15, 0.1, 0.05, 0.1
            total_score = (w_relevance * keyword_score + w_freshness * freshness + w_sentiment * sentiment_bonus +
                           w_trends * trend_score + w_diversity * diversity + w_similarity * similarity_penalty)
            return min(100, total_score * 10)
        except Exception as e:
            logger.error(f"❌ Error calculando puntuación: {e}")
            return 0

    async def is_duplicate(self, title: str, summary: str) -> bool:
        content = (title + " " + summary).lower()
        hash_key = hashlib.md5(content.encode()).hexdigest()
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT tweet_id FROM tweet_history WHERE tweet_id = ?", (hash_key,)) as cursor:
                if await cursor.fetchone():
                    return True
            async with db.execute("SELECT content FROM tweet_history WHERE posted_at > ?", 
                                ((datetime.now(pytz.UTC) - timedelta(hours=48)).isoformat(),)) as cursor:
                past_news = [row[0] async for row in cursor]
            if past_news:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([content] + past_news)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0].max()
                if similarity > 0.8:
                    return True
        return False

    def calculate_freshness(self, news_date: datetime) -> float:
        try:
            age_hours = (datetime.now(pytz.UTC) - news_date).total_seconds() / 3600
            return max(0, np.exp(-0.1 * age_hours))
        except Exception:
            return 0

    def calculate_diversity(self, source: str) -> float:
        source_count = sum(1 for item in self.stats["tweet_history"] if item.get("source") == source)
        return max(0.1, 1 - source_count * 0.05)

    async def process_single_news(self, entry: Dict[str, str], trends: List[str]):
        try:
            title = entry.get("title", "")
            url = entry.get("link", "")
            summary = entry.get("summary", "")
            source = entry.get("source", "unknown")
            news_hash = hashlib.sha256((title + url).encode()).hexdigest()
            score = await self.score_news(entry, trends)
            if score < 50:
                logger.info(f"🗑️ Noticia descartada (puntuación {score:.1f}): {title[:50]}")
                return
            image_url = await self.extract_image_from_page(url)
            media_url = image_url if image_url else None
            tweet_en = await self.model.generate_tweet(summary, category="general", url=url, language="en")
            tweet_es = await self.model.generate_tweet(summary, category="general", url=url, language="es")
            await self.enqueue_tweet(tweet_en, "en", media_url, source_type=1, priority=int(score))
            await self.enqueue_tweet(tweet_es, "es", media_url, source_type=1, priority=int(score))
            logger.info(f"✅ Noticia procesada: EN: '{tweet_en[:50]}...', ES: '{tweet_es[:50]}...'")
        except Exception as e:
            logger.error(f"❌ Error procesando noticia: {e}")
            self.error_count += 1

    async def post_tweet(self):
     await self.check_tweet_limit()
     async with self.db_semaphore:
        async with aiosqlite.connect(self.DB_FILE, timeout=10) as db:
            # Verificar si la columna 'language' existe y añadirla si no está
            async with db.execute("PRAGMA table_info(tweets_queue)") as cursor:
                columns = [row[1] async for row in cursor]
                if "language" not in columns:
                    await db.execute("ALTER TABLE tweets_queue ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
                    logger.info("✅ Columna 'language' añadida a tweets_queue con valor por defecto 'en'")
                    await db.commit()

            # Seleccionar el tweet con mayor prioridad
            async with db.execute("SELECT id, content, language, media_url FROM tweets_queue ORDER BY priority DESC, created_at ASC LIMIT 1") as cursor:
                tweet_data = await cursor.fetchone()
            if not tweet_data:
                logger.debug("⚠️ Cola de tweets vacía")
                return
            tweet_id, content, language, media_url = tweet_data
            await db.execute("DELETE FROM tweets_queue WHERE id = ?", (tweet_id,))
            await db.commit()

     twitter_client = self.twitter_client_en if language == "en" else self.twitter_client_es
     twitter_api = self.twitter_api_en if language == "en" else self.twitter_api_es
     limit_key = "daily_en" if language == "en" else "daily_es"
    
     if TWEET_LIMIT[limit_key] >= CONFIG["daily_tweet_limit"]:
        logger.warning(f"⚠️ Límite diario alcanzado para {language}: {TWEET_LIMIT[limit_key]}/{CONFIG['daily_tweet_limit']}")
        return

     try:
        if media_url and await self.validate_image(media_url):
            async with aiohttp.ClientSession() as session:
                async with session.get(media_url) as resp:
                    if resp.status == 200:
                        media_data = await resp.read()
                        media_id = twitter_api.media_upload(filename="image.jpg", file=media_data).media_id_string
                        twitter_client.create_tweet(text=content, media_ids=[media_id])
        else:
            twitter_client.create_tweet(text=content)
        
        TWEET_LIMIT[limit_key] += 1
        save_tweet_limit(TWEET_LIMIT)
        self.stats["total_tweets"] += 1
        self.stats["tweet_history"].append({
            "content": content,
            "language": language,
            "source": "queue",
            "posted_at": datetime.now(pytz.UTC).isoformat()
        })
        save_stats(self.stats)
        logger.info(f"📤 Tweet publicado: '{content[:50]}...' en {language}")
     except tweepy.TweepyException as e:
        logger.error(f"❌ Error publicando tweet: {e}")
        self.error_count += 1

    async def post_tweets_periodically(self):
        while self.running:
            try:
                await self.post_tweet()
                await asyncio.sleep(300)  # Publicar cada 5 minutos
            except Exception as e:
                logger.error(f"❌ Error en post_tweets_periodically: {e}")
                await asyncio.sleep(60)

    def _initialize_relevance_head(self) -> None:
        if TRAINING_DATA_FILE.exists():
            with TRAINING_DATA_FILE.open("r", encoding="utf-8") as f:
                train_data = json.load(f)
            if not train_data:
                logger.warning("⚠️ El archivo training_data.json está vacío. Generando datos sintéticos.")
                train_data = self.model._generate_synthetic_data()
                with TRAINING_DATA_FILE.open("w", encoding="utf-8") as f:
                    json.dump(train_data, f, indent=4)
            else:
                logger.info(f"📂 Encontrados {len(train_data)} ejemplos de entrenamiento")
        else:
            logger.warning("⚠️ No se encontró training_data.json, generando datos sintéticos")
            train_data = self.model._generate_synthetic_data()
            with TRAINING_DATA_FILE.open("w", encoding="utf-8") as f:
                json.dump(train_data, f, indent=4)
            logger.info("📝 Datos sintéticos generados y guardados")
        self.model.train_relevance_head(train_data)

    def _make_env(self) -> gym.Env:
        class BotEnv(gym.Env):
            def __init__(self, bot):
                super().__init__()
                self.bot = bot
                self.action_space = spaces.Discrete(len(self.bot.actions))
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
                self.state = None

            def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
                if seed is not None:
                    np.random.seed(seed)
                self.state = self.bot._get_state(self.bot._get_queue_size_sync(), TWEET_LIMIT["daily_en"] + TWEET_LIMIT["daily_es"])
                return self.state, {}

            def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
                queue_size = self.bot._get_queue_size_sync()
                self.state = self.bot._get_state(queue_size, TWEET_LIMIT["daily_en"] + TWEET_LIMIT["daily_es"])
                reward = self.bot._compute_reward(action)
                terminated = TWEET_LIMIT["daily_en"] >= CONFIG["daily_tweet_limit"] and TWEET_LIMIT["daily_es"] >= CONFIG["daily_tweet_limit"]
                truncated = False
                info = {}
                self.bot._apply_action(action)
                next_state = self.bot._get_state(queue_size, TWEET_LIMIT["daily_en"] + TWEET_LIMIT["daily_es"])
                return next_state, reward, terminated, truncated, info

        return BotEnv(self)

    async def load_caches(self) -> None:
        try:
            url_cache_path = BASE_DIR / "url_cache.pkl"
            if url_cache_path.exists():
                with url_cache_path.open("rb") as f:
                    self.url_cache = pickle.load(f)
                logger.debug("📂 Cache de URLs cargado")
            else:
                self.url_cache = {}
            image_cache_path = BASE_DIR / "image_cache.pkl"
            if image_cache_path.exists():
                with image_cache_path.open("rb") as f:
                    self.image_cache = pickle.load(f)
                logger.debug("📂 Cache de imágenes cargado")
            else:
                self.image_cache = {}
            trends_cache_path = BASE_DIR / "trends_cache.pkl"
            if trends_cache_path.exists():
                with trends_cache_path.open("rb") as f:
                    self.trends = pickle.load(f)
                logger.debug("📂 Cache de tendencias cargado")
            else:
                self.trends = {"gaming", "ps5", "xbox", "steam"}
            self._load_model(MODEL_DIR / "advanced_net.pth", self.model)
            self._load_model(MODEL_DIR / "resource_lstm.pth", self.resource_predictor)
            relevance_head_path = MODEL_DIR / "relevance_head.pth"
            if relevance_head_path.exists():
                self._load_model(relevance_head_path, self.model.relevance_head)
                self.model.relevance_head.trained = True
            logger.info("✅ Modelos y cachés cargados")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando cachés o modelos: {e}, inicializando nuevos", exc_info=True)

    def _load_model(self, model_path: Path, model_instance):
        if model_path.exists():
            try:
                model_instance.load_state_dict(torch.load(model_path))
                logger.info(f"✅ Modelo {model_path.name} cargado correctamente")
            except Exception as e:
                logger.error(f"❌ Error cargando {model_path.name}: {e}")
                torch.save(model_instance.state_dict(), model_path)

    async def save_url_cache(self) -> None:
        try:
            with (BASE_DIR / "url_cache.pkl").open("wb") as f:
                pickle.dump(self.url_cache, f)
            with (BASE_DIR / "image_cache.pkl").open("wb") as f:
                pickle.dump(self.image_cache, f)
            with (BASE_DIR / "trends_cache.pkl").open("wb") as f:
                pickle.dump(self.trends, f)
            logger.debug("💾 Cachés de URLs, imágenes y tendencias guardados")
            asyncio.create_task(self.schedule_url_cache_save())
        except Exception as e:
            logger.error(f"❌ Error guardando cachés: {e}", exc_info=True)

    async def schedule_url_cache_save(self) -> None:
        await asyncio.sleep(3600)
        if self.running:
            await self.save_url_cache()

    async def update_trends(self) -> None:
        try:
            trends = await self.scrape_trends()
            if not trends:
                trends = {"gaming", "ps5", "xbox", "steam"}
            self.trends = trends
            logger.info(f"✅ Tendencias actualizadas desde scraping: {self.trends}")
        except Exception as e:
            logger.error(f"❌ Error actualizando tendencias: {e}")
            self.trends = {"gaming", "ps5", "xbox", "steam"}
        await self.save_trends_cache()

    async def scrape_trends(self) -> set:
        trends = set()
        try:
            async with aiohttp.ClientSession() as session:
                for url, selector in [
                    ("https://www.playstation.com/es-es/ps5/games/", ".game-title"),
                    ("https://www.xbox.com/es-ES/games/all-games", ".game-title"),
                    ("https://store.steampowered.com/explore/new/", ".tab_item_name"),
                    ("https://store.epicgames.com/en-US/browse", ".css-1ony6zw")
                ]:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            soup = BeautifulSoup(await resp.text(), "html.parser")
                            games = soup.select(selector)[:10]
                            trends.update(game.text.strip().lower() for game in games)
        except Exception as e:
            logger.error(f"❌ Error scrapeando tendencias: {e}")
        return trends

    async def save_trends_cache(self) -> None:
        try:
            with (BASE_DIR / "trends_cache.pkl").open("wb") as f:
                pickle.dump(self.trends, f)
            logger.debug("💾 Cache de tendencias guardado")
        except Exception as e:
            logger.error(f"❌ Error guardando cache de tendencias: {e}", exc_info=True)

    async def schedule_trend_update(self) -> None:
        await asyncio.sleep(86400)
        if self.running:
            await self.update_trends()

    async def initialize(self) -> None:
        try:
            logger.info("🚀 Iniciando inicialización del bot")
            self.connection_pool = aiohttp.ClientSession()
            await self._init_db()
            await self.check_tweet_limit()
            await self.load_caches()
            await self.update_trends()
            await self.backup_state()
            logger.info("✅ Bot inicializado correctamente")
        except Exception as e:
            logger.critical(f"💥 Fallo crítico durante inicialización: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        self.running = False
        self.executor.shutdown(wait=True)
        if self.connection_pool and not self.connection_pool.closed:
            await self.connection_pool.close()
        await self.save_url_cache()
        await self.save_models()
        logger.info("🛑 Bot apagado correctamente")

    async def _init_db(self) -> None:
     try:
        async with aiosqlite.connect(DB_FILE) as db:
            # Crear tabla tweets_queue con la columna language desde el inicio
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tweets_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL DEFAULT 'en',
                    media_url TEXT,
                    source_type INTEGER,
                    priority INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Crear tabla tweet_history
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tweet_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tweet_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    posted_at TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en'
                )
            """)
            # Verificar y migrar tweets_queue si falta la columna language
            async with db.execute("PRAGMA table_info(tweets_queue)") as cursor:
                columns = [row[1] async for row in cursor]
                if "language" not in columns:
                    await db.execute("ALTER TABLE tweets_queue ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
                    logger.info("✅ Columna 'language' añadida a tweets_queue con valor por defecto 'en'")
            # Verificar y migrar tweet_history si falta la columna language
            async with db.execute("PRAGMA table_info(tweet_history)") as cursor:
                columns = [row[1] async for row in cursor]
                if "language" not in columns:
                    await db.execute("ALTER TABLE tweet_history ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
                    logger.info("✅ Columna 'language' añadida a tweet_history con valor por defecto 'en'")
            await db.commit()
        logger.info("✅ Base de datos inicializada y migrada si era necesario")
     except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {e}", exc_info=True)

    async def save_models(self) -> None:
        try:
            torch.save(self.model.state_dict(), MODEL_DIR / "advanced_net.pth")
            torch.save(self.resource_predictor.state_dict(), MODEL_DIR / "resource_lstm.pth")
            torch.save(self.model.relevance_head.state_dict(), MODEL_DIR / "relevance_head.pth")
            logger.info("✅ Modelos guardados")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}", exc_info=True)

    async def save_models_periodically(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(CONFIG["sleep_times"]["model_save_interval"])
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: self.ppo.learn(total_timesteps=10000))
                self._train_resource_predictor()
                await self.save_models()
                logger.info("✅ Modelos entrenados y guardados periódicamente")
            except Exception as e:
                logger.error(f"❌ Error en save_models_periodically: {e}", exc_info=True)

    async def heartbeat(self) -> None:
        while self.running:
            try:
                sleep_time = CONFIG["sleep_times"].get("heartbeat", 60)
                await asyncio.sleep(sleep_time)
                queue_size = await self.get_queue_size()
                success_rate = self.stats["performance_metrics"].get("success_rate", 0.0)
                logger.info(f"💓 Heartbeat: Bot activo - Tweets hoy EN: {TWEET_LIMIT['daily_en']}/{CONFIG['daily_tweet_limit']}, ES: {TWEET_LIMIT['daily_es']}/{CONFIG['daily_tweet_limit']}, Cola: {queue_size}, Tasa de éxito: {success_rate:.2%}")
            except Exception as e:
                logger.error(f"❌ Error en heartbeat: {e}", exc_info=True)

    async def backup_state(self) -> None:
        try:
            backup_dir = BASE_DIR / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if DB_FILE.exists():
                shutil.copy(DB_FILE, backup_dir / f"tweets_queue_{timestamp}.db")
            if HISTORY_FILE.exists():
                shutil.copy(HISTORY_FILE, backup_dir / f"tweet_history_{timestamp}.json")
            if STATS_FILE.exists():
                shutil.copy(STATS_FILE, backup_dir / f"bot_stats_{timestamp}.json")
            logger.info(f"💾 Backup creado en {timestamp}")
        except Exception as e:
            logger.error(f"❌ Error creando backup: {e}", exc_info=True)

    def _get_state(self, queue_size: int, tweets_today: int) -> np.ndarray:
        now = datetime.now(timezone.utc)
        hour = now.hour / 24.0
        success_rate = self.stats["performance_metrics"].get("success_rate", 0.0)
        error_rate = self.error_count / max(1, len(self.stats["tweet_history"]) + 1)
        latency = time.time() - self.last_action_time
        quota_usage = self.youtube_quota_used / CONFIG["youtube_quota_limit"]
        return np.array([queue_size / 100, tweets_today / CONFIG["daily_tweet_limit"], hour, success_rate, 0.5,
                         error_rate, latency / 10, quota_usage, len(self.trends) / 50, self.budget["rss"]], dtype=np.float32)

    def _compute_reward(self, action: int) -> float:
        if self.actions[action] == "post_now":
            return 10.0 if TWEET_LIMIT["daily_en"] < CONFIG["daily_tweet_limit"] or TWEET_LIMIT["daily_es"] < CONFIG["daily_tweet_limit"] else -5.0
        elif self.actions[action] == "enqueue":
            return 2.0
        elif self.actions[action] == "skip":
            return -1.0
        elif self.actions[action] == "adjust_relevance":
            return 5.0
        return 0.0

    def _apply_action(self, action: int) -> None:
        if self.actions[action] == "post_now" and (TWEET_LIMIT["daily_en"] < CONFIG["daily_tweet_limit"] or TWEET_LIMIT["daily_es"] < CONFIG["daily_tweet_limit"]):
            TWEET_LIMIT["daily_en"] += 1

    def _train_resource_predictor(self) -> None:
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)
        states, _, _, _, _ = zip(*batch)
        states = torch.FloatTensor(states).unsqueeze(1).to(self.model.device)
        targets = torch.FloatTensor([random.random() for _ in range(32)]).unsqueeze(-1).to(self.model.device)
        self.resource_optimizer.zero_grad()
        preds = self.resource_predictor(states[:, :, :3])
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.resource_optimizer.step()
        logger.debug(f"📉 Pérdida ResourceLSTM: {loss.item():.4f}")

    def _get_queue_size_sync(self) -> int:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM tweets_queue")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Error obteniendo tamaño de la cola (sync): {e}", exc_info=True)
            return 0

    async def get_queue_size(self) -> int:
        try:
            async with aiosqlite.connect(DB_FILE) as db:
                async with db.execute("SELECT COUNT(*) FROM tweets_queue") as cursor:
                    result = await cursor.fetchone()
                    return result[0]
        except Exception as e:
            logger.error(f"❌ Error obteniendo tamaño de la cola: {e}", exc_info=True)
            return 0

    async def check_tweet_limit(self) -> None:
        now = datetime.now(timezone.utc)
        reset_time = datetime.fromisoformat(TWEET_LIMIT["reset_time"]) if TWEET_LIMIT["reset_time"] else now
        if now >= reset_time:
            TWEET_LIMIT["daily_en"] = 0
            TWEET_LIMIT["daily_es"] = 0
            TWEET_LIMIT["reset_time"] = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
            save_tweet_limit(TWEET_LIMIT)
            logger.info("🔄 Límites diarios de tweets reiniciados (EN y ES)")

    async def is_relevant(self, text: str, url: Optional[str] = None) -> bool:
        clean_input = self.model.clean_text(text)
        _, _, relevance = self.model([clean_input], torch.FloatTensor(self._get_state(await self.get_queue_size(), TWEET_LIMIT["daily_en"] + TWEET_LIMIT["daily_es"])).unsqueeze(0))
        rule_based = any(kw in clean_input.lower() for kw in RELEVANT_KEYWORDS) and not any(spam in clean_input.lower() for spam in SPAM_FILTER)
        if len(self.stats["tweet_history"]) < CONFIG["min_attempts_for_dl"] or not self.model.relevance_head.trained:
            logger.debug(f"📏 Usando relevancia basada en reglas para '{clean_input}': {rule_based}")
            return rule_based
        is_valid = await self.validate_url_content(url, clean_input) if url else True
        final_relevance = relevance.item() >= CONFIG["relevance_threshold"] and is_valid
        logger.debug(f"🧠 Relevancia DL para '{clean_input}': {relevance.item():.2f}, URL válida: {is_valid}, Resultado: {final_relevance}")
        return final_relevance

    async def validate_image(self, image_url: str) -> bool:
        try:
            async with self.connection_pool.head(image_url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200 and "image" in resp.headers.get("Content-Type", ""):
                    content_length = int(resp.headers.get("Content-Length", 0))
                    if content_length > 100000:
                        return True
            return False
        except Exception:
            logger.error(f"❌ Error validando imagen {image_url}")
            return False

    async def extract_image_from_page(self, url: str) -> Optional[str]:
        if url in self.image_cache:
            return self.image_cache[url]
        try:
            async with self.connection_pool.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    img = soup.find("img", {"src": re.compile(r".*\.(jpg|png|jpeg)")})
                    if img and img.get("src"):
                        img_url = urllib.parse.urljoin(url, img["src"])
                        if await self.validate_image(img_url):
                            self.image_cache[url] = img_url
                            return img_url
        except Exception as e:
            logger.error(f"❌ Error extrayendo imagen de {url}: {e}")
        return None

    async def validate_url_content(self, url: str, text: str) -> bool:
        if not url:
            return True
        if any(domain in url for domain in UNWANTED_DOMAINS):
            return False
        if any(domain in url for domain in TRUSTED_GAMING_SITES):
            return True
        try:
            async with self.connection_pool.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    content_lower = content.lower()
                    return any(kw in content_lower for kw in RELEVANT_KEYWORDS) and not any(spam in content_lower for spam in SPAM_FILTER)
        except Exception:
            logger.error(f"❌ Error validando contenido de URL {url}")
            return False

    async def ensure_relevant_image(self, content: str, url: str, media_url: Optional[str]) -> Tuple[Optional[str], str]:
        if media_url and await self.validate_image(media_url):
            return media_url, url
        new_media_url = await self.extract_image_from_page(url)
        return new_media_url, url

    def is_recent(self, news_date: datetime, threshold: timedelta) -> bool:
        return (datetime.now(pytz.UTC) - news_date) <= threshold

async def main():
    bot = SuperAIGamingBot()
    try:
        await bot.initialize()
        tasks = [
            bot.heartbeat(),
            bot.save_models_periodically(),
            bot.process_content_periodically(),
            bot.post_tweets_periodically(),
            bot.schedule_trend_update(),
            bot.schedule_url_cache_save()
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.critical(f"💥 Error crítico en main: {e}", exc_info=True)
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())