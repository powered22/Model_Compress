import os
import re
import signal
from typing import Tuple, List
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ollama import AsyncClient as OllamaClient
import asyncio
from asyncio import Semaphore, TimeoutError
import openai


current_dir = os.path.dirname(__file__)

MAX_TOTAL_TOKENS = 4096


async def talk_to_openai(message: str, fewshot: str = None, model: str = "gpt-4-turbo") -> str:
    """
    Talk to OpenAI's GPT models
    :param message: Prompt
    :param fewshot: Initial prompt for few-shot learning (if not None)
    :param model: GPT model to use
    :return: LLM response
    """

    load_dotenv()
    print("load dotenv ", load_dotenv())
    openai_api_key = os.getenv('OPENAI_API_KEY')
    client = AsyncOpenAI(api_key=openai_api_key)
    # load_dotenv()
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # client = OpenAI(api_key=openai_api_key)

    if not fewshot:
        messages = [
            {
                'role': 'user',
                'content': message,
            },
        ]
    else:
        messages = [
            {
                'role': 'system',
                'content': fewshot,
            },
            {
                'role': 'user',
                'content': message,
            },
        ]

    resp = (await client.chat.completions.create(
        model=model,
        messages=messages , temperature=0.5,
    )).choices[0].message.content

    return resp


async def talk_to_llm(message: str, fewshot: str = None, model: str = 'llama3') -> str:

    is_mistral = "mistral" in model.lower()
    is_llama = "llama" in model.lower()
    is_qwen = "qwen" in model.lower()

    if 'gpt' in model:
        return await talk_to_openai(message, fewshot, model)

    elif is_mistral or is_llama:
        if fewshot:
            messages = [
                {
                    "role": "user",
                    "content": f"\n{fewshot.strip()}\n\n{message.strip()}\n"
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": f"\n{message.strip()}\n"
                }
            ]
    elif is_qwen:
        if fewshot:
            fewshot_final = fewshot.strip()
            if not fewshot_final.endswith((".", "!", ":")):
                fewshot_final += "\n\nNow answer the following task."
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant. Answer the user's request based on the provided context and question. Follow the requested format when one is given."},
                {'role': 'user', 'content': f"{fewshot_final}\n\n{message.strip()}"}
            ]
        else:
            messages = [{'role': 'user', 'content': message.strip()}]

    response = await OllamaClient().chat(model=model, messages=messages)

    return str(response['message']['content'])


class Prompting:
    def __init__(self, initial_prompt_path: str, prompting: str):
        self.initial_prompt_path = os.path.join(current_dir, f'../initial_prompts/{initial_prompt_path}')
        self.initial_prompt_file = initial_prompt_path
        self.prompting = prompting

    def get_initial_prompt(self):
        return open(self.initial_prompt_path).read()

    def get_prompt(self, datum: Tuple[str]):
        raise NotImplementedError

class ResponseProcessor:
    def __init__(self, remove_characters: str = r'[!@#$\'"+]'):
        self.remove_characters = remove_characters

    def process_response(self, task: str, response: str or int, ans: str or int) -> str or int:
        raw = str(response) if response is not None else ""
        cleaned = re.sub(self.remove_characters, '', raw).lower().replace("\n", "")

        if task == 'weather':
            return raw.strip()

        if task == 'weather-extreme':
            # Try to find yes/no
            has = None
            if re.search(r"\b(yes|true)\b", cleaned):
                has = True
            if re.search(r"\b(no|false)\b", cleaned):
                # if both appear, keep the last one
                has = False

            # Try to find event type label from allowed list
            allowed = [
                "hail", "thunderstorm wind", "flash flood",
                "tornado", "lightning", "flood", "funnel cloud"
            ]
            et = "NA"
            for t in allowed:
                if t in response:
                    et = t
                    break
            # normalize title case
            et = " ".join(w.capitalize() for w in et.split()) if et != "NA" else "NA"
            if has is False:
                et = "NA"

            # return as compact JSON string (easy to log and parse later)
            if has is None:
                has = False
            return json.dumps({"has_extreme_weather": bool(has), "event_type": et}, ensure_ascii=False)


        if task == "healthcare":
            # Try JSON first (recommended for future)
            try:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    return json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
            # fallback: keep raw
            return raw.strip()
        else:
            return raw.strip()


class WeatherPrompting(Prompting):
    def __init__(self, prompting: str, task: str, format_dbase: str):
        self.task = task
        self.format_dbase = format_dbase
        filename = f'{task}/{prompting}-{task}.txt'

        super().__init__(initial_prompt_path=filename, prompting=prompting)

    def get_prompt(self, datum) -> str:

        baseWeather = (
            "Given the weather observations below, generate a forecast describing the upcoming weather conditions "
            "based on the given question, and make sure you answer in the same format as in the question.\n"
            # "Forecast the requested weather conditions using the observations.\n"
        )

        baseWeatherExtreme = (
            # "You are given the weather observations from the previous hours.\n"
            # "Answer the question strictly as a binary classifier decision (yes or no), and also provide the event type.\n"
            # "Allowed event types: Hail, Thunderstorm Wind, Flash Flood, Tornado, Lightning, Flood, Funnel Cloud, NA.\n\n"
            # "Output format:\n"
            # "Line 1: Yes or No\n"
            # "Line 2: EventType: <one of the allowed types>\n"
            "Determine whether an extreme weather event will occur using the observations. "
            "Provide a direct and concise answer. "
            "Do not repeat the instructions or restate the output format. "
            "Do not list multiple extreme weather events. "
            "Follow the required output format exactly.\n"
        )

        if self.format_dbase == "jsonl" and self.task == "weather":
            obs_json = datum.get("obs_json")
            question = datum.get("question") or datum.get("obs_json", {}).get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Missing question in datum.")
            obs_str = json.dumps(obs_json, ensure_ascii=False, indent=2)
            return f"{baseWeather}\n\nObservations:\n{obs_str}\n\nQuestion: {question} Please report them in the exact same format as the previous weather observations, including all variables (east–west wind speed, north–south wind speed, dewpoint temperature, temperature, mean sea level pressure, surface pressure, and total precipitation). Make sure to provide the results hourly, in the same time-stamped format as the observations."

        if self.format_dbase == "sentence" and self.task == "weather":
            observation = datum.get("observation", "").strip()
            return baseWeather + "Observations: \n" + observation + f" Please report them in the exact same format as the previous weather observations, including all variables (east–west wind speed, north–south wind speed, dewpoint temperature, temperature, mean sea level pressure, surface pressure, and total precipitation). Make sure to provide the results hourly, in the same time-stamped format as the observations."


        if self.task == "weather-extreme":
            observation = datum.get("observation", "")
            question = datum.get("question", "")
            return f"{baseWeatherExtreme}\nObservations:\n{observation}\n\nQuestion: {question}\n"

        else:
            raise ValueError(f"Unknown task: {self.task}")



