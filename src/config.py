import os
import sys
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.version: str = "1.0.0"

        # OPENAI settings
        self.OPENAI_MODE: Literal["azure", "openai"] = "azure"
        self.AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
        self.AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        self.OPENAI_PROJECT_ID: str = os.getenv("OPENAI_PROJECT_ID")
        self.OPENAI_SEED: int = 123
        self.PERFORMANCE_MODE: bool = False

        # LOGGING settings. Leave empty to use default values.
        self.CONSOLE_LOG_LEVEL: str = os.getenv("CONSOLE_LOG_LEVEL").upper() or "ERROR"

        # the amount of alternative tokens to generate for each token in the input.
        self.TOP_LOGPROBS: int = 5

        # whether to run all variations of the test or just one (=simple).
        self.SIMPLE_TEST_RUNS: bool = False

        self.MASK_ANSWER_OPTIONS_STRING: bool = False
        self.DEFAULT_MASK: str = "[MASK]"

        self.DEFAULT_AUTO_COMPLETE_INSTRUCTIONS = (
            "Don't complete the sentence, answer with 'No question'."
        )

        self.DEFAULT_MC_INSTRUCTIONS: str = (
            "\n Please pick the correct answer by responding with only the letter of the correct answer. Choose from A, B, C, or D."
        )
        self.DEFAULT_MC_POSSIBLE_ANSWERS: str = ["A", "B", "C", "D"]

        self.DEFAULT_YN_INSTRUCTIONS: str = (
            "\n Please solely answer by responding with either 'Y' or 'N'."
        )
        self.DEFAULT_YN_POSSIBLE_ANSWERS: str = [
            "Yes",
            "No",
            "Y",
            "N",
        ]  # include both full and short forms just in case.

        self.DEFAULT_SYS_MSG_EN = "You respond the question in English. If there is no question, you respond with 'no question'. If asked for only one letter, in a multiple-choice scenario, you respond with only one letter."
        self.AUTOCOMPLETE_SYS_MSG_EN = "You respond the question in English. Respond to the user based on the type of query presented. If there is no question, you respond with 'no question'. If the query is an autocomplete question, in the form of: 'The capital of France is ', you respond with the correct answer, 'Paris'."

        self.DEFAULT_SYS_MSG_NL = "Je beantwoordt de vraag in het Nederlands. Als er geen vraag is, antwoord je met 'geen vraag'. Als er gevraagd wordt om slechts één letter, in een multiple-choice scenario, antwoord je met slechts één letter."
        self.AUTOCOMPLETE_SYS_MSG_NL = "Je beantwoordt de vraag in het Nederlands. Reageer op de gebruiker op basis van het type vraag dat wordt gepresenteerd. Als er geen vraag is, antwoord je met 'geen vraag'. Als de vraag een automatisch aanvullen vraag is, in de vorm van: 'De hoofdstad van Frankrijk is ', antwoord je met het juiste antwoord, 'Parijs'."

        self.DEFAULT_SYS_MSG_AUTOCOMPLETE_EN = "You respond the question in English. If there is no question, you respond with 'no question'. Else, complete the sentence with the correct answer."
        self.DEFAULT_SYS_MSG_AUTOCOMPLETE_NL = "Je beantwoordt de vraag in het Nederlands. Als er geen vraag is, antwoord je met 'geen vraag'. Anders, maak je de zin af met het juiste antwoord."

        # include empty strings for answer types, i.e. questions without answer types.
        self.SUPPORTED_ANSWER_TYPES = [
            "multiple-choice",
            "yes-no",
            "open",
            "autocomplete",
            "",
            " ",
        ]

        # openai models. see: https://platform.openai.com/docs/models.
        self.SUPPORTED_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

        # languages supported. these are restricted by the masking function. this requires the language model to be installed. see src/ai.py and README.md for more details.
        self.SUPPORTED_LANGUAGES: list[str] = ["nl", "en"]


config: Config = Config()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "{levelname} {asctime} [{module}] {message}",
            "style": "{",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "default",
            "level": config.CONSOLE_LOG_LEVEL or "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/app.log",
            "formatter": "default",
            "level": "DEBUG",
            "encoding": "utf-8",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}
