from copy import deepcopy
import os
import json
import logging.config
from openai import AzureOpenAI
import yaml
from time import sleep, time

import yaml.parser

from src.config import LOGGING_CONFIG, config
from tqdm import tqdm
from src.utils.ai import (
    create_client,
    get_chat_completion,
    OpenAI,
    ChatCompletion,
    add_sys_msg,
    add_assistant_msg,
    add_user_msg,
    string_prompt_without_image,
    get_words_to_mask,
    mask_words,
    get_pos_tags,
)
from src.utils.misc import clean_response, startup_message
from src.utils.files import load_images, detect_test_collections
from src.models import (
    TestCollection,
    TestCollectionRunResults,
    TestRunResults,
    Test,
    TestMetadata,
    Question,
)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.ERROR)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def dump_results(
    testrun_results: TestRunResults,
    test_metadata: TestMetadata,
    full_masked_prompts: dict[str],
):
    """
    Dump results to json with structure
    key: question_uuid
    value: raw openai response

    testrun_results: TestRunResults object
    testrun_results.results dictionary structure:
    full_masked_prompts (dict[dict[str]]): key: question_uuid, value: dict with key: masking_level, value: masked_question


    args:
        testrun_results: TestRunResults object
    """
    logger.debug(f"Dumping results to {testrun_results.output_filename}")

    # Convert the results to json, stringify UUID. Loop over self.results and convert the keys to strings.
    # Convert UUID keys to string and prepare for JSON serialization
    # Currently, we have the question UUID key and value is the raw openai response. We want to add the metadata.
    # we want format: {test_metadata: {test_id, sys_msg, questions, model}, results: {question_uuid: raw_openai_response}}
    # openai_responses = {
    #     str(key): value for key, value in testrun_results.results.items()
    # }

    # Serialize the results
    serialized_results = {
        "test_metadata": {
            "test_id": str(test_metadata.test_id),
            "test_name": test_metadata.test_name,
            "test_description": test_metadata.test_description,
            "test_answer_type": test_metadata.test_answer_type,
            "test_language": test_metadata.test_language,
            "sys_msg": test_metadata.sys_msg,
            "questions": [question.json() for question in test_metadata.questions],
            "model": test_metadata.model,
            "full_masked_prompts": full_masked_prompts,
        },
        "results": {
            "with_image": {
                str(level): {str(q_uuid): resp for q_uuid, resp in results.items()}
                for level, results in testrun_results.variation_results.with_image.items()
            },
            "without_image": {
                str(level): {str(q_uuid): resp for q_uuid, resp in results.items()}
                for level, results in testrun_results.variation_results.without_image.items()
            },
        },
    }

    # debug log all the json keys and their child keys
    logger.debug("[BEGIN] Stringified results keys and their child keys:")
    for key, value in serialized_results.items():
        logger.debug(f"{key=}")
        for k, v in value.items():
            logger.debug(f"{k=}")
    logger.debug("[END] Stringified results keys and their child keys.")

    # Convert the results to JSON
    json_results = json.dumps(serialized_results, indent=4)

    with open(
        os.path.join(os.getcwd(), "data", "output", testrun_results.output_filename),
        "w",
    ) as f:
        f.write(json_results)

    logger.debug(f"Results dumped to {testrun_results.output_filename}.")


def run_test(
    test: Test, client: OpenAI, simple_run: bool = False, add_log_probs: bool = True
) -> TestRunResults:
    """
    This function runs a single test. It iterates over the questions in the test and generates responses for each question.
    If simple is false, it runs each test with, and without images, and it runs with all 6 levels of masking.
    If simple is true, it runs just the plain test with images.

    Args:
        test (Test): the test object to run
        client (OpenAI): the OpenAI client
        simple_run (bool, optional): if True, only the bare test. if false, it runs all test variations. Defaults to False.
        add_log_probs (bool, optional): if True, adds logprobs to the response. Defaults to True.

    Returns:
        TestRunResults: the results of the test run
    """
    logger.debug(f"Starting testrun for test id={test.test_id=}")
    result = TestRunResults(test_id=test.test_id, test_name=test.test_name)
    # dict to store the masked questions.
    full_masked_prompts = (
        {}
    )  # structure: key: question_uuid, value: dict with key: masking_level, value: masked_question

    # define if we use the test variations
    if simple_run:
        images = 1
        masking_levels = 1
    else:
        images = 2
        masking_levels = 7

    # initialize the prompt with the system message, to be used in each test variation.
    base_prompt: list[dict[str, str]] = add_sys_msg(test.sys_msg)

    # RUN TEST WITH IMAGES ON AND OFF
    for image_level in range(images):
        for masking_level in range(masking_levels):
            # for each masking level, representing one test variation, reset the prompt and system message.
            prompt = deepcopy(base_prompt)
            logger.debug(
                f"Reset the prompt+system message. New Sys msg for testrun: {prompt=}"
            )

            for question in tqdm(
                test.questions,
                desc=f"Running test with {len(test.questions)} questions for {masking_level=}, {image_level=}",
            ):
                question_text = question.question
                answer_type = question.answer_type
                answer_instr = question.answer_instructions
                answer_options = question.answer_options
                question_id = question.question_id
                str_question_id = str(question_id)
                logger.debug(
                    f"Generating answer for {question_id=}: {question_text=}, using image level {image_level=} and {masking_level=}.\
                        {answer_type=}, {answer_options=}, {answer_instr=}."
                )

                if isinstance(answer_options, list):
                    answer_options = "\n".join(answer_options)

                # apply the masking levels
                if config.MASK_ANSWER_OPTIONS_STRING:
                    # append the answer options to the question text
                    question_without_masking = f"{question_text}\n{answer_options}"
                else:
                    question_without_masking: str = question_text

                # initialize the masked question dict
                if str_question_id not in full_masked_prompts:
                    full_masked_prompts[str_question_id] = {}

                # mask the words in the question text
                word_types_to_mask: list[str] = get_pos_tags(level=masking_level)

                logger.debug(
                    f"On language {test.language}, applying masking level {masking_level}. Applying {word_types_to_mask=}."
                )

                words_to_mask: list[str] = get_words_to_mask(
                    text_corpus=question_without_masking,
                    wordtypes_to_mask=word_types_to_mask,
                    language=test.language,
                )
                masked_question: str = mask_words(
                    text_to_mask=question_without_masking,
                    words_to_mask=words_to_mask,
                    mask=config.DEFAULT_MASK,
                )
                logger.debug(
                    f"Masked question: {masked_question}. Original question: {question_text}."
                )
                if (
                    config.MASK_ANSWER_OPTIONS_STRING
                ):  # answer options are already added to the question text.
                    full_question = f"{masked_question}\n{answer_instr}"
                else:
                    full_question = f"{masked_question}\n{answer_options}{answer_instr}"

                # add to the full_masked_prompts dict
                full_masked_prompts[str_question_id][masking_level] = full_question

                # Use the image if image_level is 0, if not, dont use the image.
                image = question.b64_image if image_level == 0 else None

                # Format the prompt, add the question and image, get the response and add it back to the prompt.
                prompt: list[dict[str, str]] = add_user_msg(
                    prompt=prompt, user_message=full_question, image=image
                )
                logger.debug(
                    f"Prompt after adding question: {string_prompt_without_image(prompt)}"
                )
                response: ChatCompletion = get_chat_completion(
                    client=client,
                    messages=prompt,
                    model=test.model,
                    logprobs=add_log_probs,
                    top_logprobs=config.TOP_LOGPROBS if add_log_probs else None,
                )
                cleaned_response = clean_response(response.to_dict())

                result.update_results(
                    question_uuid=question_id,
                    response=cleaned_response,
                    masking_level=masking_level,
                    image=True if image_level == 0 else False,
                )
                response_text = response.choices[0].message.content
                prompt = add_assistant_msg(
                    prompt=prompt, assistant_message=response_text
                )

                logger.debug(f"Response to question {question=} == {response_text}.")

                # sleep for a bit to avoid rate limiting after each question
                sleep(1)

            if not config.PERFORMANCE_MODE:
                # sleep for a bit to avoid rate limiting after each masking level
                sleep(2)

        if not config.PERFORMANCE_MODE:
            # sleep for a bit to avoid rate limiting after each image level
            sleep(5)

    dump_results(
        testrun_results=result,
        test_metadata=test.test_metadata,
        full_masked_prompts=full_masked_prompts,
    )

    return result


def run_tom_test_pipeline(
    client: OpenAI, test_collections_to_run: list[str]
) -> dict[str, TestCollectionRunResults]:
    """Runs the pipeline for the Tom tests. Uses the filepaths as provided in the tests_to_run list.
    Loads them from yaml files and executes the tests.

    Args:
        client (OpenAI): the OpenAI client
        test_collections_to_run (list[str]): list of absolute filepaths to the yaml files containing the tests.

    Returns:
        dict[str, TestCollectionRunResults]: results of the test runs
    """
    all_collection_results: dict[str, TestCollectionRunResults] = {}

    all_test_collections: dict[str, list[TestCollection]] = {}

    # load all the tests from the yaml files
    try:
        for test_collection_file in test_collections_to_run:
            all_test_collections[test_collection_file] = (
                load_test_collections_from_yaml(test_file_path=test_collection_file)
            )
    except AttributeError as e:
        logger.error(f"Syntax Error in yaml file. Error Details: {e}. Exiting...")
        return {}
    except yaml.parser.ParserError as e:
        logger.error(f"Syntax Error in yaml file. Error Details: {e}. Exiting...")
        return {}
    except KeyError as e:
        logger.error(f"Syntax Error in yaml file. Error Details: {e}. Exiting...")
        return {}
    except TypeError as e:
        logger.error(
            f"Error loading test collection. Likely SyntaxError in yaml file. Error details: {e}. Exiting..."
        )
        return {}
    except FileNotFoundError as e:
        logger.error(f"File is not found. Error Details: {e}. Exiting...")
        return {}

    # NOTE: Debug print statements
    # print(test_collections_to_run)
    # print(test_collections)
    # iterate over all the test collections (each containing multiple tests)
    # TQDM is just an iterator wrap for visualizing a loading bar.
    # this says: for test_collection in test_collections:
    for test_collections_id, test_collections in all_test_collections.items():

        for test_collection in tqdm(
            test_collections,
            desc=f"Running {len(test_collections)} test collection(s)...",
        ):
            t1_collection = time()
            # print(test_collection)
            # print(test_collections[test_collection])

            collection_results = {}

            # iterate over the tests in the test collection
            for test in tqdm(
                test_collection.tests,
                desc=f"Running tests in test_collection_id: {test_collection.test_collection_id}...",
            ):
                t1_test = time()
                # print(test)
                # run the test
                result = run_test(
                    test=test, client=client, simple_run=config.SIMPLE_TEST_RUNS
                )
                collection_results[test.test_id] = result
                t2_test = time()
                exp_time_test = t2_test - t1_test
                logger.debug(
                    f"Test {test.test_id} took {exp_time_test} seconds to run."
                )
                print(
                    f"Succesfully ran test {test.test_id}. Time taken: {exp_time_test} seconds."
                )

            # create a TestCollectionRunResults object and add it to the all_collection_results dictionary
            all_collection_results[test_collection.test_collection_id] = (
                TestCollectionRunResults(
                    test_collection_id=test_collection.test_collection_id,
                    test_collection_file_name=test_collection.test_collection_origin_filename,
                    test_run_results=collection_results,
                )
            )

            t2_collection = time()
            exp_time_collection = t2_collection - t1_collection
            logger.debug(
                f"Test collection {test_collection.test_collection_id} took {exp_time_collection} seconds to run."
            )
            print(
                f"Succesfully ran test collection {test_collection.test_collection_id}. Time taken: {exp_time_collection} seconds."
            )

    return all_collection_results


def load_test_collections_from_yaml(test_file_path: str) -> list[TestCollection]:
    """Loads all test_collections from a yaml file. A test collection is a list of tests.

    Args:
        test_file_path (str): absolute path to the yaml file

    Raises:
        FileNotFoundError: if the file does not exist

    Returns:
        list[TestCollection]: list of TestCollection objects
    """
    if not os.path.exists(test_file_path):
        logger.error(f"Test file {test_file_path} does not exist.")
        raise FileNotFoundError(f"Test file {test_file_path} does not exist.")

    with open(test_file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        test_collections: list[TestCollection] = []

        for test_coll_key, test_coll_value in data.items():
            collection_description = test_coll_value.get("description", "")
            tests = []

            for test_data in test_coll_value.get("tests", []):
                test_description = test_data.get("test_description", "")
                sys_msg = test_data["sys_msg"]
                test_answer_type = test_data.get("test_answer_type").lower()
                questions = []

                image_keys = [
                    q["image_name"] for q in test_data["questions"] if "image_name" in q
                ]

                images = load_images(
                    image_keys=image_keys,
                    test_folder=test_data["test_images_folder"],
                )

                for question in test_data["questions"]:
                    answer_type = question[
                        "answer_type"
                    ]  # answertype is validated, i.e. multiple-choice, yes-no, open, autocomplete. required.

                    if answer_type not in config.SUPPORTED_ANSWER_TYPES:
                        logger.error(
                            f"Unsupported answer type: {answer_type}. Exiting..."
                        )
                        raise ValueError(
                            f"Unsupported answer type: {answer_type}. Exiting..."
                        )

                    question_type = question.get(
                        "question_type", ""
                    )  # question type is not validated, i.e. question, statement, control etc. optional.

                    question_text = question["question"]

                    # if the question is a statement, we don't need to provide an answer.
                    if answer_type == "" or answer_type == " ":
                        correct_answers = []
                        possible_answers = []
                        answer_instructions = ""
                        answer_options = ""
                    else:
                        if answer_type == "multiple-choice":
                            answer_options = question[
                                "answer_options"
                            ]  # Required, only for multiple-choice
                            correct_answers = question["correct_answers"]
                        elif answer_type == "yes-no":
                            answer_options = question.get("answer_options", "")
                            correct_answers = question["correct_answers"]
                        else:
                            answer_options = question.get("answer_options", "")
                            correct_answers = question.get("correct_answers", "")

                        # the followjng three are optional. if None is provided, the default values are used. These are set in the Question class.
                        # the defaults are loaded from the config file.
                        answer_instructions = question.get(
                            "answer_instructions", None
                        )  # optional
                        possible_answers = question.get(
                            "possible_answers", None
                        )  # optional
                        answer_instructions = question.get(
                            "answer_instructions", None
                        )  # optional

                    if question is None or question == "":
                        logger.error(
                            f"Question in {test_file_path} is empty, but can't be empty. Exiting..."
                        )
                        raise ValueError("Question is empty. Exiting...")

                    if (
                        test_answer_type == "autocomplete"
                        and answer_instructions is None
                        and question_type == ""
                        and (answer_type == "" or answer_type == " ")
                    ):
                        answer_instructions = config.DEFAULT_AUTO_COMPLETE_INSTRUCTIONS

                    questions.append(
                        Question(
                            question=question_text,
                            correct_answers=correct_answers,
                            question_type=question_type,
                            possible_answers=possible_answers,
                            answer_instructions=answer_instructions,
                            answer_options=answer_options,
                            answer_type=answer_type,
                            b64_image=(images.get(question.get("image_name"))),
                        )
                    )

                if sys_msg == "DEFAULT_SYS_MSG_EN":
                    sys_msg = config.DEFAULT_SYS_MSG_EN
                elif sys_msg == "DEFAULT_SYS_MSG_NL":
                    sys_msg = config.DEFAULT_SYS_MSG_NL
                elif sys_msg == "AUTOCOMPLETE_SYS_MSG_EN":
                    sys_msg = config.AUTOCOMPLETE_SYS_MSG_EN
                elif sys_msg == "AUTOCOMPLETE_SYS_MSG_NL":
                    sys_msg = config.AUTOCOMPLETE_SYS_MSG_NL
                elif sys_msg == "DEFAULT_SYS_MSG_AUTOCOMPLETE_EN":
                    sys_msg = config.DEFAULT_SYS_MSG_AUTOCOMPLETE_EN
                elif sys_msg == "DEFAULT_SYS_MSG_AUTOCOMPLETE_NL":
                    sys_msg = config.DEFAULT_SYS_MSG_AUTOCOMPLETE_NL

                test = Test(
                    sys_msg=sys_msg,
                    questions=questions,
                    model=test_data["model"],
                    language=test_data["language"],
                    test_name=test_data["test_name"],
                    test_answer_type=test_answer_type,
                )
                tests.append(test)

            test_collections.append(
                TestCollection(
                    test_collection_id=test_coll_key,
                    test_collection_description=collection_description,
                    tests=tests,
                    test_collection_origin_filename=test_file_path,
                )
            )

        return test_collections


def run():
    startup_message()
    tests: list[str] = detect_test_collections(test_filename_prefix="")

    print("Detected test collections: ")
    for index, test_path in enumerate(tests):
        printed_test = test_path.split("/")[-1]
        if printed_test == test_path:
            printed_test = test_path.split("\\")[-1]
        print(f" - ({index}) {printed_test}")

    print(
        f"Please select a test collection to run (0-{len(tests)-1}), or select all by entering (A). Enter (Q) to quit."
    )
    input_options = [str(i) for i in range(len(tests))] + ["A", "ALL"] + ["Q", "QUIT"]
    selected_test = ""
    test_collections_to_run: list[str] = []

    while selected_test not in input_options:
        selected_test = input("\nEnter selection: ").upper()
        if selected_test == "Q" or selected_test == "QUIT":
            print("Exiting...")
            exit(0)
        if selected_test not in input_options:
            print("Invalid selection. Please select a valid option.")
        elif selected_test == "A" or selected_test == "ALL":
            test_collections_to_run = tests
            break
        else:
            test_collections_to_run.append(tests[int(selected_test)])
            break

    print(f"Selected test collection(s): {test_collections_to_run}")
    print("Running selected test collection(s)...")

    if config.OPENAI_MODE == "azure":
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
        )
    else:
        client: OpenAI = create_client(api_key=config.OPENAI_API_KEY)
    results: dict[str, TestCollectionRunResults] = run_tom_test_pipeline(
        client=client, test_collections_to_run=test_collections_to_run
    )

    print("Finished running tests. Results have been saved to the data/output folder.")
    display_results = input("Do you want to view the results? (Y/N)").upper()

    if display_results == "Y" or display_results == "YES":
        print("Results: ")
        for test_id, test_results in results.items():
            print(f"Test ID: {test_id}")
            print(f"Results: {test_results.test_run_results}\n\n")

    print("Exiting LMToM-EVAL..")
    return


if __name__ == "__main__":
    logger.error("Please run from run.py. Exiting...")
    exit(1)
