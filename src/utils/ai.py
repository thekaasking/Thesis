"""
All AI utils.

Used for NLP, LLMs, LMMs, etc.
"""

import base64
from copy import deepcopy
import re
import spacy
import logging.config
from openai import OpenAI
from typing import Literal
from openai.types.chat import ChatCompletion
from openai.types.images_response import ImagesResponse

from src.config import LOGGING_CONFIG, config

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# NOTE: if installing via requirements does not work, try running the following commands from your virtual environment:
# dutch and english models
# python -m spacy download nl_core_news_sm
# python -m spacy download en_core_web_sm


def get_pos_tags(level):
    tags = {
        0: [],
        1: ["PRON"],
        2: ["PRON", "DET"],
        3: ["PRON", "DET", "ADJ"],
        4: ["PRON", "DET", "ADJ", "ADV"],
        5: ["PRON", "DET", "ADJ", "ADV", "VERB"],
        6: ["PRON", "DET", "ADJ", "ADV", "NOUN"],
    }
    return tags.get(level, [])


def get_words_to_mask(
    text_corpus: str,
    wordtypes_to_mask: list[str],
    language: str = "nl",
) -> list[str]:
    """
    For masking, we use NLP models from spaCy to identify the words to mask.
    We use Part-of-Speech (POS) tagging to identify the words to mask.

    For full list of supported POS tags, see README.md.

    Args:
        text_corpus: the text corpus to mask
        language: the language of the text corpus
        wordtypes_to_mask: the word types to mask. Example: ["NOUN", "VERB"]

    Returns:
        words_to_mask: the words to mask in the text corpus

    Docs for POS tagging: https://spacy.io/usage/linguistic-features/
    """
    nlp = get_language(language)

    doc = nlp(text_corpus)
    words_to_mask = [token.text for token in doc if token.pos_ in wordtypes_to_mask]

    return words_to_mask


def mask_words(
    text_to_mask: str, words_to_mask: list[str], mask: str = config.DEFAULT_MASK
) -> str:
    """Replaces all occurrences of the words in the list with the mask string, ensuring only whole words are masked."""
    for word in words_to_mask:
        # Use a regular expression to define word boundaries around the word to ensure it's not a substring of another word
        word_regex = r"\b" + re.escape(word) + r"\b"
        text_to_mask = re.sub(word_regex, mask, text_to_mask)
    return text_to_mask


def load_languages(languages: list[str]) -> dict[str, spacy.language.Language]:
    """
    Load the languages in the languages list
    """
    loaded_languages = {}
    for language in languages:
        try:
            if language == "nl":
                nlp = spacy.load("nl_core_news_sm")
            elif language == "en":
                nlp = spacy.load("en_core_web_sm")
            else:
                logger.error(f"Language {language} is not supported.")
                raise ValueError(f"Language {language} is not supported.")
        except Exception:
            logger.error(
                f"Error loading language: {language}. It is likely not installed (correctly)."
            )
            raise ValueError(
                f"Error loading language: {language}. It is likely not installed (correctly)."
            )
        loaded_languages[language] = nlp

    return loaded_languages


languages: dict[str, spacy.language.Language] = load_languages(
    config.SUPPORTED_LANGUAGES
)


def get_language(language: str):
    return languages.get(language)


def create_client(api_key: str, organization_id: str = None, project_id: str = None):
    logger.debug(
        f"Creating OpenAI client with API key: {api_key[:4]}, {organization_id=}, {project_id=}"
    )
    return OpenAI(api_key=api_key, organization=organization_id, project=project_id)


def get_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[str],
    temperature: float = 0,
    max_tokens: int = 152,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    logprobs: bool = False,
    n: int = 1,
    seed: int = config.OPENAI_SEED,
    top_logprobs: int = None,
) -> ChatCompletion:
    # logger.debug(
    #     f"Getting chat completion with {model=}, {temperature=}, {max_tokens=}, {top_p=}, {logprobs=}, {n=}, {seed=}, {top_logprobs=}, {frequency_penalty=}, {presence_penalty=}"
    #     f" \nand messages: {messages}"
    # )
    response: ChatCompletion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        logprobs=logprobs,
        seed=seed,
        n=n,
        stream=False,
        top_logprobs=top_logprobs,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    logger.debug(f"Chat completion response: {response.to_dict()}")
    return response


def generate_image(
    client: OpenAI,
    prompt: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
    style: Literal["vivid", "natural"] = "vivid",
) -> ImagesResponse:
    logger.debug(
        f"Generating image with {model=}, {prompt=}, {size=}, {quality=}, {n=}, {style=}"
    )
    if model == "dall-e-3":
        n = 1
        # check characters are max 4k
        if len(prompt) > 4000:
            raise ValueError(
                "Prompt must be less than 4000 characters for DALL-E-3 model"
            )

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
        style=style,
        response_format="b64_json",
    )
    logger.debug(f"Image generation response: {response.to_dict()}")
    return response


def encode_image(image_path):
    """
    Source: https://platform.openai.com/docs/guides/vision
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def create_image_content(image: str, detail: Literal["high", "low", "auto"] = "auto"):
    """
    Source: https://community.openai.com/t/how-to-send-base64-images-to-assistant-api/752440/3
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}", "detail": detail},
    }


def add_sys_msg(system_message: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_message}]


def add_user_msg(
    prompt: list[dict[str, str]], user_message: str, image: str = None
) -> list[dict[str, str]]:
    if image:
        prompt.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    create_image_content(image),
                ],
            },
        )
    else:
        prompt.append({"role": "user", "content": user_message})
    return prompt


def add_assistant_msg(
    prompt: list[dict[str, str]], assistant_message: str
) -> list[dict[str, str]]:
    prompt.append({"role": "assistant", "content": assistant_message})
    return prompt


def string_prompt_without_image(prompt: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(prompt)

    for message in result:
        if message["role"] == "user":
            content = message["content"]
            if isinstance(content, list) and len(content) > 1:
                # image url location: content[1].get("image_url")
                # shorten the base64 image url by only keeping 30 characters
                image_url_dict = content[1].get("image_url")
                if image_url_dict and "url" in image_url_dict:
                    image_url_dict["url"] = image_url_dict["url"][:30]

    return result
