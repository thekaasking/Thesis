import json
import logging.config
from copy import deepcopy
from functools import wraps
from typing import Literal, Union
import numpy as np
import warnings

from src.config import LOGGING_CONFIG, config

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def startup_message() -> None:
    print(
        f"\n \
  _      __  __ _______    __  __        ______          _  \n \
 | |    |  \/  |__   __|  |  \/  |      |  ____|        | | \n \
 | |    | \  / |  | | ___ | \  / |______| |____   ____ _| | \n \
 | |    | |\/| |  | |/ _ \| |\/| |______|  __\ \ / / _` | | \n \
 | |____| |  | |  | | (_) | |  | |      | |___\ V / (_| | | \n \
 |______|_|  |_|  |_|\___/|_|  |_|      |______\_/ \__,_|_| \n \
\n LMToMEval version {config.version} \n \
Written by: Razo van Berkel(2024) \n \
Leiden Institute of Advanced Computer Science (LIACS) \n \
\n Environment info: \n \
Supported languages: {config.SUPPORTED_LANGUAGES} \n \
OpenAI API key: {config.OPENAI_API_KEY[:4]}.... \n \
OpenAI project ID: {config.OPENAI_PROJECT_ID} \n \
OpenAI seed: {config.OPENAI_SEED} \n \
Supported OpenAI models: {config.SUPPORTED_MODELS} \n \
Simple test runs: {config.SIMPLE_TEST_RUNS} \n \
Performance mode: {config.PERFORMANCE_MODE} \n \
\nPress <Ctrl+C> to exit. \n"
    )


def dict_to_string(d: dict) -> str:
    return json.dumps(d, indent=4, sort_keys=True)


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with {args=} and {kwargs=}")
        return func(*args, **kwargs)

    return wrapper


def calculate_accuracy_score(predicted_answers: list[str], correct_answers: list[str]):
    """
    Method that calculates the accuracy score for multiple choice question answering tasks.

    Args:
    - predicted_answers: list of predicted answers. in form: [a, b, c, d, ...] or [y, n, ...]
    - correct_answers: list of correct answers. in form: [a, b, c, d, ...] or [y, n, ...]

    Returns:

    - accuracy_score: the accuracy score of the predicted answers.
    """
    print(predicted_answers, correct_answers)
    if len(predicted_answers) != len(correct_answers):
        raise ValueError(
            "The number of predicted answers must be equal to the number of correct answers."
        )
    if not predicted_answers or not correct_answers:
        return 0

    # all answers to uppercase
    predicted_answers = [answer.upper() for answer in predicted_answers]
    correct_answers = [answer.upper() for answer in correct_answers]

    correct = 0
    for i in range(len(predicted_answers)):
        if predicted_answers[i] == correct_answers[i]:
            correct += 1

    try:
        accuracy_score = correct / len(predicted_answers)
    except ZeroDivisionError:
        accuracy_score = 0

    return accuracy_score


def calculate_confidence_score(
    logprobs: list, approach: str = "base", percentage: bool = False
) -> float:
    """Calculates the confidence score of the predicted answer based on the logprobs of the alternatives.
    returns actual probability of the predicted answer, not the logprobs.

    Args:
        logprobs (list): list of logprobs of the predicted answer.
        approach (str): the approach to calculate the confidence score. Default is "base".
        percentage (bool): if True, return the confidence score as a percentage. Default is False.

    Returns:
        float: the confidence score of the predicted answer.
    """
    assert approach in ["base", "compare"], f"Invalid approach {approach=}"
    assert isinstance(logprobs, list), f"Invalid logprobs {logprobs=}"

    average_logprobs_requested = True if len(logprobs) > 1 else False
    average_logprobs: Union[float, list] = 0 if approach == "base" else []

    for logprob in logprobs:

        top_logprobs = logprob["top_logprobs"]

        # sort the top logprobs by logprob value
        sorted_top_logprobs = sorted(
            top_logprobs, key=lambda x: x.get("logprob"), reverse=True
        )

        try:
            correct_logprob = sorted_top_logprobs[0].get("logprob")

            next_best_logprob = sorted_top_logprobs[1].get("logprob")
        except IndexError:
            logger.error(
                "Could not calculate confidence score. Probably no filtered alternatives remaining."
            )
            next_best_logprob = 0

        # calculate the confidence score
        if average_logprobs_requested:
            if approach == "compare":
                if percentage:
                    confidence_score = np.round(
                        (np.exp(correct_logprob) * 100)
                        - (np.exp(next_best_logprob) * 100),
                        3,
                    )
                else:
                    confidence_score = np.round(
                        np.exp(correct_logprob) - np.exp(next_best_logprob), 3
                    )

                average_logprobs.append(confidence_score)
            elif approach == "base":
                average_logprobs += correct_logprob

        else:
            if approach == "compare":
                if percentage:
                    confidence_score = np.round(
                        (np.exp(correct_logprob) * 100)
                        - (np.exp(next_best_logprob) * 100),
                        3,
                    )
                else:
                    confidence_score = np.round(
                        np.exp(correct_logprob) - np.exp(next_best_logprob), 3
                    )
            elif approach == "base":
                if percentage:
                    confidence_score = np.round(np.exp(correct_logprob) * 100, 3)
                else:
                    confidence_score = np.round(np.exp(correct_logprob), 3)

        return confidence_score

    if approach == "compare":
        # multiply the probabilities
        average_logprobs = np.prod(average_logprobs)
    elif approach == "base":
        if percentage:
            average_logprobs = np.round(np.exp(average_logprobs) * 100, 3)
        else:
            average_logprobs = np.round(np.exp(average_logprobs), 3)

    return average_logprobs


def calculate_logprob_confidence(logprobs: dict) -> float:
    """Calculates the distance between the logprobs of the correct answer and the next best alternative.
    Assumes the logprobs are already filtered to only contain actual alternatives.
    # NOTE: old version, calculate confidence score based on the logprobs of the predicted answer.

    Args:
        logprobs (dict): dict containing the logprobs of the predicted answer.

    Returns:
        float: the confidence score of the predicted answer.
    """
    warnings.warn(
        "calculate_logprob_confidence is deprecated. Use calculate_confidence_score instead.",
        DeprecationWarning,
    )
    top_logprobs = logprobs.get("top_logprobs", [])
    if not top_logprobs:
        return 0

    # sort the top logprobs by logprob value
    sorted_top_logprobs = sorted(
        top_logprobs, key=lambda x: x.get("logprob"), reverse=True
    )

    try:
        correct_logprob = sorted_top_logprobs[0].get("logprob")

        next_best_logprob = sorted_top_logprobs[1].get("logprob")
    except IndexError:
        logger.error(
            "Could not calculate confidence score. Probably no filtered alternatives remaining."
        )
        next_best_logprob = 0

    # calculate the confidence score
    confidence_score = next_best_logprob - correct_logprob
    return confidence_score


def filter_logprob_alternatives(
    answer_type: str,
    allowed_answers: list[str],
    logprobs: dict,
):
    """
    Filters the alternatives from the logprobs dict based on the type of question.
    E.g. if type==multiple-choice, the allowed_alternatives are [A, B, C, D]
    if type==yes-no, the allowed_alternatives are [Yes, No, y, n]

    logprobs: pass the q_id-choices[0].get("logprobs").get("content)[]

    Args:
    - answer_type: the type of question. e.g. multiple-choice, yes-no, etc.
    - allowed_answers: the list of allowed answers for the question.
    - logprobs: the logprobs dict from the OpenAI API response.

    Returns:
    - filtered_logprobs: the filtered logprobs dict.
    """
    logger.debug("Starting logprob filtering...")
    if answer_type not in ["multiple-choice", "yes-no"]:
        raise ValueError(
            f"Answer type {answer_type} is not supported for filtering logprobs."
        )

    if not allowed_answers or len(allowed_answers) == 0:
        # use defaults
        if answer_type == "multiple-choice":
            allowed_answers = config.DEFAULT_MC_POSSIBLE_ANSWERS

        elif answer_type == "yes-no":
            allowed_answers = config.DEFAULT_YN_POSSIBLE_ANSWERS

    # Normalize allowed answers to be lowercase and stripped of whitespace
    normalized_allowed_answers = {answer.strip().lower() for answer in allowed_answers}

    # create a copy
    filtered_logprobs = deepcopy(logprobs)

    top_logprobs: list[dict] = filtered_logprobs.get("top_logprobs", [])
    index = 0
    while index < len(top_logprobs):
        token = top_logprobs[index].get("token", "").strip().lower()
        if token not in normalized_allowed_answers:
            top_logprobs.pop(index)
        else:
            index += 1

    logger.debug("Finished logprob filtering.")
    logger.debug(f"Filtered logprobs: {filtered_logprobs}")
    logger.debug(f"Original logprobs: {logprobs}")

    return filtered_logprobs


def clean_response(
    openai_response: dict,
    cleaning_approach: Literal["azure", "openai"] = config.OPENAI_MODE,
) -> dict:
    """Removes unnecessary keys from the OpenAI API response.
    Removes keys: "object", "bytes", "index" from the response.

    cleaning approach is different for Azure and OpenAI API responses.
    in azure, we remove also the prompt_filter_results key.

    Args:
        openai_response (dict): raw OpenAI API response.

    Returns:
        cleaned_response (dict): the cleaned OpenAI API response.
    """
    logger.debug(f"Cleaning OpenAI API response with {cleaning_approach=}...")
    cleaned_response = deepcopy(openai_response)
    # remove the object key
    cleaned_response.pop("object", None)
    cleaned_response.pop("created", None)

    if cleaning_approach == "azure":
        cleaned_response.pop("prompt_filter_results", None)

    def recursive_remove_bytes(item):
        if isinstance(item, dict):
            item.pop("bytes", None)  # Remove "bytes" key if present at this level
            for key, value in list(item.items()):
                item[key] = recursive_remove_bytes(value)  # Recurse into nested dict
        elif isinstance(item, list):
            return [
                recursive_remove_bytes(elem) for elem in item
            ]  # Recurse into list elements
        return item

    # remove the bytes key from choices[0].logprobs.content and all occurrences in choices[0].logprobs.top_logprobs
    byte_cleaned_response = recursive_remove_bytes(cleaned_response.copy())

    choices = byte_cleaned_response.get("choices", [])
    if choices:
        for choice in choices:
            choice.pop("index", None)

    logger.debug("OpenAI API response cleaned.")
    return byte_cleaned_response
