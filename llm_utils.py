import json
import time
import logging
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running.")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _is_config_error(e):
    """Returns True for errors that won't be fixed by retrying (bad key, missing key, etc.)."""
    msg = str(e).lower()
    return "openai_api_key" in msg or "api key" in msg or "authentication" in msg or "401" in msg


def call_llm(prompt, system_prompt=None, temperature=None, max_tokens=None, expect_json=False):
    """
    Thin wrapper around the OpenAI chat completions endpoint.

    Set expect_json=True to get a parsed dict back instead of raw text - this
    also enables JSON mode at the API level, which is more reliable than just
    asking the model nicely in the prompt.

    Retries up to LLM_RETRY_ATTEMPTS times with exponential backoff before
    giving up and raising.
    """
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens = max_tokens or config.LLM_MAX_TOKENS

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(1, config.LLM_RETRY_ATTEMPTS + 1):
        try:
            client = _get_client()
            kwargs = {
                "model": config.OPENAI_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()

            if expect_json:
                return json.loads(content)
            return content

        except json.JSONDecodeError as e:
            logger.warning("Got invalid JSON from the model (attempt %d): %s", attempt, e)
            last_error = e
        except Exception as e:
            logger.warning("API call failed (attempt %d): %s", attempt, e)
            last_error = e
            if _is_config_error(e):
                break  # no point retrying a bad/missing key

        if attempt < config.LLM_RETRY_ATTEMPTS:
            delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)

    raise RuntimeError(f"Gave up after {config.LLM_RETRY_ATTEMPTS} attempts: {last_error}")


def call_llm_safe(prompt, system_prompt=None, temperature=None, max_tokens=None, expect_json=False, default=None):
    """
    Same as call_llm, but swallows exceptions and returns `default` instead.
    Useful in batch jobs where one bad row shouldn't kill the whole run.
    """
    try:
        return call_llm(prompt, system_prompt, temperature, max_tokens, expect_json)
    except Exception as e:
        logger.error("LLM call failed permanently, using default: %s", e)
        return default
