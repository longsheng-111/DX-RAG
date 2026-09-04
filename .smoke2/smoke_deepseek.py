# -*- coding: utf-8 -*-
"""DeepSeek API 连通性冒烟测试（阶段 2 环境准备）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from openai import OpenAI
from app.core.config import settings

print("key configured:", bool(settings.deepseek_api_key))
print("base_url:", settings.deepseek_base_url)

client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
resp = client.chat.completions.create(
    model="deepseek-chat",  # 与 token_tracker.compress_messages_with_llm 一致
    messages=[{"role": "user", "content": "用一句话回答：1+1等于几？"}],
    max_tokens=5,
    timeout=60.0,
)
print("reply:", resp.choices[0].message.content)
print("usage:", resp.usage)
print("API_OK")
