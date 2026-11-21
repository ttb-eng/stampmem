# -*- coding: utf-8 -*-
"""
LLM 抽象层：让记忆库不绑死任何一家大模型。

设计思想（可插拔、能适配未来）：
  1. 定义抽象接口 LLMProvider（只声明 chat 方法）
  2. 各家模型实现这个接口（OpenAI 兼容的都能用）
  3. 核心逻辑（LLMJudge）只依赖接口，不依赖具体实现
  4. API Key 获取优先级：显式传参 > 环境变量 > .env 文件

这样未来出现新模型，只需写一个新的 Provider 类，核心代码无需改动。
"""

import os
import json
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ============================================================
# 1. 抽象接口
# ============================================================
class LLMProvider(ABC):
    """所有大模型 Provider 必须实现的接口"""

    @abstractmethod
    def chat(self, messages: list, temperature: float = 0.3) -> str:
        """输入 messages，返回模型的文本回复"""
        raise NotImplementedError


# ============================================================
# 2. OpenAI 兼容 Provider（适配 DeepSeek / OpenAI / 通义 / 智谱 ...）
# ============================================================
class OpenAICompatProvider(LLMProvider):
    """
    适配所有 OpenAI 兼容接口的通用 Provider。
    换模型只需改 base_url + model 即可。

    设计原则（库不碰 env）：
      - api_key 必须由【调用方】显式传入，库自己不去读环境变量/.env
      - 读 .env 是【应用层】的职责，不是库的职责
    """

    def __init__(self, api_key: str, base_url: str = None,
                 model: str = "deepseek-chat"):
        if OpenAI is None:
            raise ImportError("请先安装 openai：pip install openai")

        if not api_key:
            raise ValueError(
                "必须显式传入 api_key。库不会自动读环境变量/.env，"
                "请由应用层读取后传入。"
            )

        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com"
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list, temperature: float = 0.3) -> str:
        """发一次对话请求，返回模型回复的纯文本（已 strip）。

        messages 采用 OpenAI 格式：[{"role": "user", "content": "..."}]。
        网络异常与限流会原样抛出，由调用方决定重试策略 —— 库不替你做重试。
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


# ============================================================
# 3. 应用层辅助函数：从 .env / 环境变量读 key（给用户用，不是库内部用）
# ============================================================
def load_api_key(env_key_name: str = "DEEPSEEK_API_KEY", env_file: str = None) -> str:
    """
    供【应用层】使用的辅助函数：从环境变量或 .env 文件读 API Key。
    注意：这是给"调用库的人"用的，库内部不会自动调用它。
    """
    # 1. 环境变量
    key = os.environ.get(env_key_name)
    if key:
        return key

    # 2. 尝试读取 .env 文件
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)  # 加载 .env
        key = os.environ.get(env_key_name)
        if key:
            return key
    except ImportError:
        pass

    return ""


# ============================================================
# 4. 记忆状态判断（只依赖抽象接口，不绑死具体模型）
# ============================================================
class LLMJudge:
    """用 LLM 判断记忆状态，并生成自然的确认问句"""

    def __init__(self, provider: LLMProvider):
        # 必须显式传入一个已配置好的 Provider（库不自动创建、不读 env）
        self.provider = provider

    def judge(self, memory_text: str, created_at: str, current_text: str) -> dict:
        """
        输入：一条记忆 + 它的记录时间 + 用户当前说的话
        输出：dict { 'status': 'done'/'ongoing'/'uncertain',
                     'reply': '自然语言确认问句' }
        核心要求：回复里【不要】出现具体时间戳，用"之前/上次"等模糊词。
        """
        prompt = f"""你是 AI 的长期记忆管理助手。

背景信息：
- 用户曾说过/做过这样一件事（记忆）："{memory_text}"
- 这条记忆被记录的时间是：{created_at}

现在用户说："{current_text}"
请判断这条记忆现在的状态，并生成一句自然的回复：
1. 状态 status 只能是：done（已完成/已过时）、ongoing（仍在进行）、uncertain（不确定）
2. 回复 reply 要自然、口语化，像人一样，不要提具体时间戳、不要提"某年某月某日"
3. 只有当你认为这条记忆和用户当前说的话可能相关时，才在 reply 里向用户确认；否则 reply 可以为空

只输出 JSON，格式：{{"status":"...", "reply":"..."}}"""

        content = self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # 去掉可能的 ```json ... ``` 包裹
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"status": "uncertain", "reply": content}

    def detect_conflict(self, old_memory: str, new_memory: str) -> dict:
        """
        判断新旧两条记忆是否冲突，以及如何处理。

        输入：旧记忆、新记忆
        输出：dict {
            'is_conflict': true/false,     # 是否冲突
            'action': 'update'/'keep'/'ask', # update=新替旧, keep=保留旧, ask=需确认
            'reason': '一句话解释'
        }
        """
        prompt = f"""你是 AI 的长期记忆冲突检测器。

有两条关于同一用户的记忆：
- 旧记忆："{old_memory}"
- 新记忆："{new_memory}"

请判断这两条记忆是否矛盾/冲突，并决定如何处理：
1. is_conflict：true 表示矛盾，false 表示不矛盾
2. action 只能是：
   - update：新记忆是真实改变，应更新旧记忆
   - keep：新记忆只是一时情绪/气话，应保留旧记忆
   - ask：无法确定，需要向用户确认
3. reason：用一句话解释你的判断

只输出 JSON，格式：{{"is_conflict":true/false,"action":"...","reason":"..."}}"""

        content = self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"is_conflict": True, "action": "ask", "reason": content}
