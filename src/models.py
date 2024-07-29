from dataclasses import dataclass
import logging.config
from typing import Optional
import uuid
from src.config import LOGGING_CONFIG, config

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Question:
    def __init__(
        self,
        question: str,
        correct_answers: list[str],
        answer_options: list[str] = [],
        b64_image: str = None,
        question_type: str = "",
        answer_type: str = None,
        answer_instructions: str = None,
        possible_answers: list[str] = None,
    ):
        if possible_answers is None:
            if answer_type == "multiple-choice":
                possible_answers = config.DEFAULT_MC_POSSIBLE_ANSWERS
            elif answer_type == "yes-no":
                possible_answers = config.DEFAULT_YN_POSSIBLE_ANSWERS
            else:
                possible_answers = []

        if answer_instructions is None:
            if answer_type == "multiple-choice":
                answer_instructions = config.DEFAULT_MC_INSTRUCTIONS
            elif answer_type == "yes-no":
                answer_instructions = config.DEFAULT_YN_INSTRUCTIONS
            else:
                answer_instructions = ""

        self.question = question
        self.correct_answers = correct_answers
        self.b64_image = b64_image
        self.question_type = question_type
        self.answer_type = answer_type
        self.answer_instructions = answer_instructions
        self.answer_options = answer_options
        self.possible_answers = possible_answers

        self.question_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{question}{b64_image}")

    def __repr__(self):
        if self.b64_image is None:
            return f"Question: {self.question}\nAnswer: {self.correct_answers}\nQuestion Type: {self.question_type}\nAnswer Type: {self.answer_type}\n Answer Instructions: {self.answer_instructions}\n Answer Options: {self.answer_options}\n Possible Answers: {self.possible_answers}\n, Image: None"
        else:
            return f"Question: {self.question}\nAnswer: {self.correct_answers}\nQuestion Type: {self.question_type}\nAnswer Type: {self.answer_type}\n Answer Instructions: {self.answer_instructions}\n Answer Options: {self.answer_options}\n Possible Answers: {self.possible_answers}\n, Image: {self.b64_image[:10]}..."

    def json(self, with_image: bool = False):
        return {
            "question": self.question,
            "correct_answers": self.correct_answers,
            "b64_image": self.b64_image if with_image else "[Image placeholder]",
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "answer_instructions": self.answer_instructions,
            "answer_options": self.answer_options,
            "possible_answers": self.possible_answers,
            "question_id": str(self.question_id),
        }


@dataclass
class TestMetadata:
    test_id: str
    sys_msg: str
    questions: list[Question]
    model: str
    test_language: str
    test_name: Optional[str] = ""
    test_answer_type: Optional[str] = ""
    test_description: Optional[str] = ""


@dataclass
class LevelResults:
    results: dict


@dataclass
class VariationResults:
    with_image: dict
    without_image: dict

    def __init__(self):
        self.with_image = {}
        self.without_image = {}

    def add_results(self, level: int, image: bool, results: dict):
        if image:
            self.with_image[level] = results
        else:
            self.without_image[level] = results


class TestRunResults:
    def __init__(self, test_id: str, test_name: str = ""):
        logger.debug(f"Creating TestRunResults with {test_id=}")
        self.test_id = test_id
        self.test_name = test_name
        self.variation_results = VariationResults()
        self.output_filename = f"test_results_{test_name}.json"

    def update_results(
        self, masking_level: int, image: bool, question_uuid: str, response: dict
    ):
        level_results = (
            self.variation_results.with_image
            if image
            else self.variation_results.without_image
        )
        if masking_level not in level_results:
            level_results[masking_level] = {}
        level_results[masking_level][question_uuid] = response

    def get_results(self):
        return self.variation_results

    def __str__(self):
        return f"Test ID: {self.test_id}\nResults: {self.variation_results}\n"

    def __repr__(self):
        return f"Test ID: {self.test_id}\nResults: {self.variation_results}\n"


@dataclass
class TestCollectionRunResults:
    test_collection_id: str
    test_collection_file_name: str
    test_run_results: list[TestRunResults]


class Test:
    def __init__(
        self,
        sys_msg: str,
        questions: list[Question],
        model: str,
        language: str,
        test_name: str = "",
        test_description: str = "",
        test_answer_type: str = "",
    ):
        if language not in config.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language {language} is not supported. Supported languages are {config.SUPPORTED_LANGUAGES}"
            )
        if model not in config.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model} is not supported. Supported models are {config.SUPPORTED_MODELS}"
            )
        self.sys_msg = sys_msg
        self.questions = questions
        self.model = model
        self.language = language
        self.test_id = str(uuid.uuid4())
        self.test_name = test_name
        self.test_answer_type = test_answer_type
        self.test_description = test_description

        self.test_metadata = TestMetadata(
            test_id=self.test_id,
            sys_msg=sys_msg,
            questions=questions,
            model=model,
            test_name=test_name,
            test_answer_type=test_answer_type,
            test_language=language,
            test_description=test_description,
        )
        logger.debug(
            f"Creating test with {sys_msg=}, {questions=} {model=}, {self.test_id=}"
        )

    def __str__(self):
        return f"Description: {self.test_description}\nTest ID: {self.test_id}\nSystem Message: {self.sys_msg}\nQuestions: {self.questions}\nModel: {self.model}\n"


@dataclass
class TestCollection:
    tests: list[Test]
    test_collection_description: str
    test_collection_id: Optional[str] = None
    test_collection_origin_filename: Optional[str] = None
