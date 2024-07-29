import os
import json
import numpy as np
import pandas as pd
import logging.config
from datetime import datetime
import matplotlib.pyplot as plt


from src.config import LOGGING_CONFIG, config
from src.result_filenames import (
    SA_open_files,
    SA_autocomplete_files,
    SS_open_files,
    SS_autocomplete_files,
    SA_mc_files,
    SA_yn_files,
    SS_mc_files,
    SS_yn_files,
    sally_anne_rated_open,
    sally_anne_rated_autocomplete,
    strange_stories_rated_open,
    strange_stories_rated_autocomplete,
    IM_complete_files,
    IM_all_files,
    IM_filtered_files,
    testrun_id_dict,
)
from src.utils.misc import (
    filter_logprob_alternatives,
    calculate_logprob_confidence,
    calculate_confidence_score,
)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def create_one_answer_logprobs_plot(filename: str, only_filtered: bool = True) -> None:
    """Log probs for mc and yn answer alternatives
    Plots only a single question, with one level and one image presence variation.

    Args:
        filename (str): filename of the test results
    """
    allowed_answer_types = ["multiple-choice", "yes-no"]

    results, test_metadata = load_results_and_questions(filename)
    questions = {q["question_id"]: q for q in test_metadata.get("questions", [])}

    for image_var in ["with_image", "without_image"]:
        for level, level_results in results[image_var].items():

            # Process each question in the test
            for question_id, response in level_results.items():
                logger.debug(f"Processing question {question_id=}")
                question = questions.get(question_id)

                if question is None:
                    continue

                question_answer_type = question.get("answer_type")
                if question_answer_type not in allowed_answer_types:
                    logger.debug(
                        f"Skipping question {question_id=}. Not a {allowed_answer_types=} question."
                    )
                    continue

                # given_answer = response.get("choices")[0].get("message").get("content")
                logprobs_content: list = (
                    response.get("choices")[0].get("logprobs").get("content")
                )
                tokens = []
                logprobs = []
                top_tokens = []
                top_logprobs = []
                confidence_scores = []

                # Define answer type and allowed answers based on the question context
                # TODO: Add support for other answer types, based on test metadata
                if question_answer_type == "multiple-choice":
                    allowed_answers = config.DEFAULT_MC_POSSIBLE_ANSWERS
                else:
                    allowed_answers = config.DEFAULT_YN_POSSIBLE_ANSWERS

                for content in logprobs_content:
                    context = content["top_logprobs"]
                    print(f"context: {context}")

                    filtered_logprobs: dict = filter_logprob_alternatives(
                        question_answer_type, allowed_answers, content
                    )
                    print(f"f_logprobs: {filtered_logprobs}")
                    # Calculate confidence score for each token
                    confidence_score = calculate_logprob_confidence(filtered_logprobs)
                    confidence_scores.append(confidence_score)
                    print("confidence score: ", confidence_score)
                    logger.debug(f"{confidence_score=} for {content['token']}")
                    logger.debug(f"For token {content['token']}: {filtered_logprobs=}")

                    # fill the lists for the df / plot
                    tokens.append(content["token"])
                    logprobs.append(content["logprob"])

                    if only_filtered:
                        logprobs_to_plot = filtered_logprobs["top_logprobs"]
                    else:
                        logprobs_to_plot = content["top_logprobs"]

                    for top in logprobs_to_plot:
                        print(f"top: {top}")
                        top_tokens.append(f"{content['token']} ({top['token']})")
                        top_logprobs.append(top["logprob"])

                    context = content["top_logprobs"]

                # Combine lists into a dictionary and create a DataFrame
                data_dict = {
                    "Token Pair": top_tokens,
                    "Top Log Probability": top_logprobs,
                }
                df_top = pd.DataFrame(data_dict)

                # create logprobs plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Top alternatives plot
                bars_top = ax.bar(
                    df_top["Token Pair"],
                    df_top["Top Log Probability"],
                    color="green",
                    label="Top Alternative Tokens",
                )
                ax.set_title(
                    f"Top Log Probabilities of possible answers for {question_id}, {image_var}, with masking {level=}",
                    pad=20,
                )
                ax.set_xlabel("Log Probability")
                ax.set_ylabel("Token Pair")
                ax.legend(
                    title="Top alternatives (lower is better). Format: Chosen token (Alternative)."
                )

                # Add text annotations for top tokens
                for bar in bars_top:
                    yval = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval,
                        f"{yval:.5f}",
                        va="bottom",
                        ha="center",
                    )

                plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        os.getcwd(), "data", "output", "plots", f"{question_id}.png"
                    )
                )
                plt.show()
                # stop after processing the first question
                break
            break
        break


def plot_accuracy_one_question(accuracy_data: dict, save_fig: bool = True):
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8
    level_padding = 0.1  # Additional space between different levels

    positions = []
    level_labels = []

    # Calculate the number of unique levels across all questions for spacing
    num_levels = len({lvl for data in accuracy_data.values() for lvl in data["1"]})

    for i, (question_id, levels_data) in enumerate(accuracy_data.items()):
        base_position = i * num_levels * (2 * bar_width + level_padding) * 2
        for j, (level, score) in enumerate(sorted(levels_data["1"].items())):
            pos = base_position + j * (2 * bar_width + level_padding)

            ax.bar(
                pos,
                score,
                bar_width,
                alpha=opacity,
                color="b",
                label="With Image" if i == 0 and j == 0 else "",
            )
            ax.bar(
                pos + bar_width,
                levels_data["0"][level],
                bar_width,
                alpha=opacity,
                color="r",
                label="Without Image" if i == 0 and j == 0 else "",
            )

            positions.append(pos + bar_width / 2)
            level_labels.append(level)

        # Add the question UUID below the first bar of each group
        ax.text(
            base_position,
            -0.5,
            question_id,
            ha="center",
            va="top",
            rotation=0,
            # add some negative padding to avoid overlapping with x-axis labels
            bbox=dict(facecolor="white", alpha=0.5, pad=5),
            fontsize=8,
        )

        # Set axes properties and layout
        ax.set_xlabel("Question ID and Level")
        ax.set_ylabel("Accuracy Score")
        ax.set_title("Accuracy Scores by Question ID, Level, and Image Presence")
        ax.set_xticks(positions)
        ax.set_xticklabels(level_labels, rotation=45)
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save_fig:
        fig_id: str = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(
            os.path.join(
                os.getcwd(), "data", "output", "plots", f"accuracy_scores_{fig_id}.png"
            )
        )
    plt.show()


def format_question_data(
    answer_type: str,
    data_type: str,
    files: list[str] = None,
    confidence_approach: str = None,
) -> dict:
    """Formats the accuracy scores of multiple-choice or yes-no questions by question ID, level, and image presence.

    Enforces to only process questions of the specified answer type.

    Args:
        answer_type (str): the type of answer to visualize. Can be 'multiple-choice' or 'yes-no'.
        data_type (str): the type of data to format. Can be 'accuracy' or 'confidence'.
        filename (str, optional): filename of the test results. Defaults to None.
        confidence_approach (str, optional): the approach to calculate the confidence score. Defaults to None.

    Returns:
        dict: the accuracy data in the format {UUID: {image_level: {level: accuracy_score}}}
    """
    assert data_type in ["accuracy", "confidence"], "Invalid data type."
    assert answer_type in [
        "multiple-choice",
        "yes-no",
        "autocomplete",
        "open",
    ], "Invalid answer type."
    assert (
        data_type == "confidence" if answer_type == "autocomplete" else True
    ), "Invalid combination of answer type and data type."
    assert isinstance(files, list), "Please provide a list of filenames to process."

    if data_type == "confidence" and confidence_approach is None:
        confidence_approach = "compare"

    logger.debug(
        f"Formatting {answer_type} {data_type} results for the following {len(files)} files: \n{files}"
    )

    formatted_data = {}
    for file in files:
        logger.debug(f"Starting to process {file=}...")
        results, test_metadata = load_results_and_questions(
            answer_type=answer_type, file=file
        )
        questions = {q["question_id"]: q for q in test_metadata.get("questions", [])}
        file_uuid = test_metadata.get("test_id")
        formatted_data[file_uuid] = {}

        # Process both image and no-image variations
        for image_var in ["with_image", "without_image"]:
            for level, level_results in results[image_var].items():
                question_counter = 0
                # Process each question in the test
                for question_id, response in level_results.items():
                    logger.debug(f"Processing question {question_id=}")
                    question_counter += 1
                    question = questions.get(question_id)

                    if question is None:
                        continue

                    question_answer_type = question.get("answer_type")
                    if answer_type != question_answer_type:
                        logger.debug(
                            f"Skipping question {question_id=}. Not a {answer_type=} question."
                        )
                        continue

                    ## CONFIDENCE CALCULATION
                    if data_type == "confidence":
                        # get all logprob alternatives
                        logprobs = (
                            response.get("choices")[0].get("logprobs").get("content")
                        )
                        # Define answer type and allowed answers based on the question context
                        # TODO: Add support for other answer types, based on test metadata
                        if question_answer_type == "multiple-choice":
                            allowed_answers = config.DEFAULT_MC_POSSIBLE_ANSWERS
                        elif question_answer_type == "yes-no":
                            allowed_answers = config.DEFAULT_YN_POSSIBLE_ANSWERS
                        else:
                            allowed_answers = []

                        if (
                            question_answer_type == "multiple-choice"
                            or question_answer_type == "yes-no"
                        ):
                            filtered_logprobs: dict = filter_logprob_alternatives(
                                answer_type=answer_type,
                                allowed_answers=allowed_answers,
                                logprobs=logprobs[0],
                            )
                            calculated_score: float = calculate_confidence_score(
                                [filtered_logprobs], approach=confidence_approach
                            )
                        elif (
                            question_answer_type == "autocomplete"
                            or question_answer_type == "open"
                        ):
                            calculated_score: float = calculate_confidence_score(
                                logprobs=logprobs, approach=confidence_approach
                            )

                    ## ACCURACY CALCULATION
                    elif data_type == "accuracy":
                        correct_answer = question.get("correct_answers")
                        if (
                            not correct_answer
                            or not isinstance(correct_answer, list)
                            or not correct_answer
                        ):
                            logger.debug(
                                f"No correct answer found for {question_id}. Skipping..."
                            )
                            continue

                        given_answer = (
                            response.get("choices")[0].get("message").get("content")
                        )
                        logger.debug(
                            f"For {question_id=}: {given_answer=} {correct_answer=}"
                        )

                        if given_answer in correct_answer:
                            calculated_score = 1
                        else:
                            calculated_score = 0
                    else:
                        raise ValueError(
                            f"Invalid data type '{data_type}'. Please provide a valid data type."
                        )

                    question_id = question.get("question_id")
                    image_level = "1" if image_var == "with_image" else "0"
                    level_key = f"{level}"

                    # Store data in a temporary structure with UUID keys
                    if question_id not in formatted_data[file_uuid]:
                        formatted_data[file_uuid][question_id] = {
                            "1": {},
                            "0": {},
                        }
                    # Initialize nested dictionary if level key not present
                    if (
                        level_key
                        not in formatted_data[file_uuid][question_id][image_level]
                    ):
                        formatted_data[file_uuid][question_id][image_level][
                            level_key
                        ] = calculated_score
                    else:
                        # Average or replace existing score, based on your use case
                        formatted_data[file_uuid][question_id][image_level][
                            level_key
                        ] = calculated_score

    logger.debug(f"Formatted {data_type} data: {formatted_data}")
    return formatted_data


def validate_answertype_filenames(answer_type: str, filename: str = None) -> list:
    """Validate the answer type and filename for the test results.

    Args:
        answer_type (str): the type of answer to visualize.
        filename (str): filename of the test results. Defaults to None.

    Returns:
        list: the list of filenames to process
    """
    assert answer_type in [
        "open",
        "autocomplete",
        "multiple-choice",
        "yes-no",
    ], "Invalid answer type."

    endings = {
        "multiple-choice": "_mc",
        "yes-no": "_yn",
        "open": "_open",
        "autocomplete": "_autocomplete",
    }
    ending = endings[answer_type]

    # Check for input files
    if filename is None:
        files = [
            file
            for file in os.listdir(os.path.join(os.getcwd(), "data", "output"))
            if file.endswith(f"{ending}.json") and file.startswith("test_results_")
        ]
    else:
        files = [filename]
    return files


def plot_average_data_levels(
    multiple_test_data: dict,
    data_type: str,
    answer_type: str = "",
    save_fig: bool = True,
    show_labels: bool = True,
    show_fig: bool = True,
) -> None:
    """Visualizes average the accuracy or confidence scores of multiple-choice or yes-no questions by level and image presence.
    Input is a dictionary containing multiple tests, each with multiple questions of the same answer type.

    Args:
        multiple_test_data (dict): the accuracy or confidence data in the format {Test_UUID: {Q_UUID: {image_level: {level: accuracy_score}}}}
        data_type (str): the type of data to visualize. Can be 'accuracy' or 'confidence'.
        save_fig (bool): whether to save the figure to disk or not.
        show_labels (bool): whether to show the average scores on the plot or not.
        show_fig (bool): whether to show the plot or not.

    """
    assert data_type in ["accuracy", "confidence"], "Invalid data type."

    logger.debug(f"Visualizing {len(multiple_test_data)} tests for {answer_type=}")
    print(f"Visualizing {len(multiple_test_data)} tests for {answer_type=}")

    for test_uuid, data in multiple_test_data.items():

        levels = range(7)  # From Lvl 0 to Lvl 6
        avg_scores_img = [0] * 7  # To store average scores for image level 0
        avg_scores_no_img = [0] * 7  # To store average scores for image level 1

        # Calculating averages for each level
        for uuid in data:
            for lvl in levels:
                lvl_key = f"{lvl}"
                avg_scores_img[lvl] += data[uuid]["0"][lvl_key]
                avg_scores_no_img[lvl] += data[uuid]["1"][lvl_key]

        num_questions = len(data)
        avg_scores_img = [score / num_questions for score in avg_scores_img]
        avg_scores_no_img = [score / num_questions for score in avg_scores_no_img]

        plt.figure(figsize=(10, 5))
        test_name = testrun_id_dict.get(test_uuid, test_uuid)
        plt.title(f"{data_type} by Level for {answer_type} test {test_name}")
        plt.plot(
            levels,
            avg_scores_img,
            label="With Image",
            marker="o",
            color="r",
            linestyle="-",
        )
        plt.plot(
            levels,
            avg_scores_no_img,
            label="Without Image",
            marker="o",
            color="b",
            linestyle="-",
        )
        plt.ylim(0, 1.1)
        if show_labels:
            offset = 0
            for i in range(len(avg_scores_img)):
                plt.text(
                    levels[i],
                    avg_scores_img[i] - offset,
                    f"{avg_scores_img[i]:.2f}",
                    ha="center",
                    va="bottom",
                )
                plt.text(
                    levels[i],
                    avg_scores_no_img[i] - offset,
                    f"{avg_scores_no_img[i]:.2f}",
                    ha="center",
                    va="bottom",
                )
        plt.xlabel("Level")
        plt.ylabel(f"Average {data_type} Score")
        plt.xticks(levels, [f"{i}" for i in levels])
        plt.legend()

        plt.grid(True)
        if save_fig:

            plt.savefig(
                os.path.join(
                    os.getcwd(),
                    "data",
                    "output",
                    "plots",
                    f"{data_type}_by_level_test_{test_name}.png",
                )
            )
        if show_fig:
            plt.show()
        plt.close()


def plot_combined_accuracy_confidence_data_1(
    accuracy_data: dict,
    confidence_data: dict,
    answer_type: str = "",
    save_fig: bool = True,
    accuracy_threshold: int = 0.5,
) -> None:
    """Visualizes average the accuracy or confidence scores of multiple-choice or yes-no questions by level and image presence.
    Input is a dictionary containing multiple tests, each with multiple questions of the same answer type.

    Args:
        accuracy_data (dict): the accuracy data in the format {Test_UUID: {Q_UUID: {image_level: {level: accuracy_score}}}}
        confidence_data (dict): the confidence data in the format {Test_UUID: {Q_UUID: {image_level: {level: confidence_score}}}}
        answer_type (str): the type of answer to visualize. Can be 'multiple-choice' or 'yes-no', 'autocomplete', 'open'.
        save_fig (bool): whether to save the figure to disk or not.
        accuracy_threshold (int): the threshold for the accuracy score to color the confidence points.
    """
    assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
    assert accuracy_data.keys() == confidence_data.keys(), "Data keys do not match."
    assert answer_type in [
        "multiple-choice",
        "yes-no",
        "autocomplete",
        "open",
    ], "Invalid answer type."

    logger.debug(f"Visualizing {len(accuracy_data)} tests for {answer_type=}")
    print(f"Visualizing {len(accuracy_data)} tests for {answer_type=}")

    levels = range(7)  # From Lvl 0 to Lvl 6
    for test_uuid, data in accuracy_data.items():

        avg_confidence_img = [0] * 7  # To store average scores for image level 0
        avg_confidence_no_img = [0] * 7  # To store average scores for image level 1

        avg_accuracy_img = [0] * 7  # To store average scores for image level 0
        avg_accuracy_no_img = [0] * 7  # To store average scores for image level 1

        # Calculating averages for each level
        for uuid in data:
            for lvl in levels:
                # print(f"{data[uuid]=}")
                lvl_key = f"{lvl}"
                avg_accuracy_img[lvl] += data[uuid]["0"][lvl_key]
                avg_accuracy_no_img[lvl] += data[uuid]["1"][lvl_key]

                # fetch confidence data
                avg_confidence_img[lvl] += confidence_data[test_uuid][uuid]["0"][
                    lvl_key
                ]
                avg_confidence_no_img[lvl] += confidence_data[test_uuid][uuid]["1"][
                    lvl_key
                ]

        num_questions = len(data)
        print(f"NUM QUESTIONS: {num_questions} in {test_uuid=}")

        avg_confidence_img = [score / num_questions for score in avg_confidence_img]
        avg_confidence_no_img = [
            score / num_questions for score in avg_confidence_no_img
        ]

        avg_accuracy_img = [score / num_questions for score in avg_accuracy_img]
        avg_accuracy_no_img = [score / num_questions for score in avg_accuracy_no_img]

        # check if there is still a plot open
        if plt.fignum_exists(1):
            plt.close()

        plt.figure(figsize=(10, 5))
        test_name = testrun_id_dict.get(test_uuid, test_uuid)
        plt.title(f"Confidence by Level for {answer_type} test: {test_name}")
        plt.ylim(0, 110)

        # plot confidence
        plt.plot(
            levels,
            avg_confidence_img,
            label="With Image",
            marker=",",
            color="purple",
            linestyle="-",
        )
        plt.plot(
            levels,
            avg_confidence_no_img,
            label="Without Image",
            marker=",",
            color="blue",
            linestyle="-",
        )
        # plot the accuracy
        offset = 0
        for i in range(len(avg_accuracy_img)):
            plt.text(
                levels[i],
                avg_confidence_img[i] - offset,
                f"{avg_accuracy_img[i]:.2f}",
                ha="center",
                va="bottom",
            )
            plt.text(
                levels[i],
                avg_confidence_no_img[i] - offset,
                f"{avg_accuracy_no_img[i]:.2f}",
                ha="center",
                va="bottom",
            )

        # plot scatter points for accuracy
        accuracy_range = range(len(avg_accuracy_img))

        for i in accuracy_range:
            plt.scatter(
                levels[i],
                avg_confidence_img[i],
                color="g" if avg_accuracy_img[i] > accuracy_threshold else "r",
                s=100,
                marker="o",
            )
            plt.scatter(
                levels[i],
                avg_confidence_no_img[i],
                color="g" if avg_accuracy_no_img[i] > accuracy_threshold else "r",
                s=100,
                marker="o",
            )

        plt.xlabel("Level")
        plt.ylabel("Average Confidence Score (%)")
        plt.xticks(levels, [f"{i}" for i in levels])
        plt.legend()

        plt.grid(True)
        if save_fig:
            plt.savefig(
                os.path.join(
                    os.getcwd(),
                    "data",
                    "output",
                    "plots",
                    f"confidence_accuracy_by_level_test_{test_name}.png",
                )
            )
        plt.show()
        plt.close()


def plot_combined_accuracy_confidence_data_2(
    accuracy_data: dict,
    confidence_data: dict,
    save_fig: bool = True,
    use_bars: bool = False,
    show_fig: bool = True,
):
    assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
    assert accuracy_data.keys() == confidence_data.keys(), "Data keys do not match."

    logger.debug(f"Visualizing {len(accuracy_data)} ")
    print(f"Visualizing {len(accuracy_data)} ")

    levels = range(7)  # From Lvl 0 to Lvl 6

    for test_uuid, data in accuracy_data.items():

        avg_confidence_img = [0] * 7  # To store average scores for image level 0
        avg_confidence_no_img = [0] * 7  # To store average scores for image level 1

        avg_accuracy_img = [0] * 7  # To store average scores for image level 0
        avg_accuracy_no_img = [0] * 7  # To store average scores for image level 1

        # Calculating averages for each level
        for uuid in data:
            for lvl in levels:
                # print(f"{data[uuid]=}")
                lvl_key = f"{lvl}"
                avg_accuracy_img[lvl] += data[uuid]["0"][lvl_key]
                avg_accuracy_no_img[lvl] += data[uuid]["1"][lvl_key]

                # fetch confidence data
                avg_confidence_img[lvl] += confidence_data[test_uuid][uuid]["0"][
                    lvl_key
                ]
                avg_confidence_no_img[lvl] += confidence_data[test_uuid][uuid]["1"][
                    lvl_key
                ]

        num_questions = len(data)
        print(f"NUM QUESTIONS: {num_questions} in {test_uuid=}")

        avg_confidence_img = [score / num_questions for score in avg_confidence_img]
        avg_confidence_no_img = [
            score / num_questions for score in avg_confidence_no_img
        ]

        avg_accuracy_img = [score / num_questions for score in avg_accuracy_img]
        avg_accuracy_no_img = [score / num_questions for score in avg_accuracy_no_img]

        # Main plot: Grouped Bar Chart
        if plt.fignum_exists(1):
            plt.close()
        fig, main_ax = plt.subplots(figsize=(14, 8))

        index = np.arange(7)

        main_ax.set_ylim(0, 1.1)

        if use_bars:
            # accuracy bars
            bar_width = 0.2
            # Plotting bars for accuracy and confidence side by side
            main_ax.bar(
                index,
                avg_accuracy_img,
                bar_width,
                label="Accuracy with Image",
                color="blue",
            )
            main_ax.bar(
                index + bar_width,
                avg_accuracy_no_img,
                bar_width,
                label="Accuracy without Image",
                color="lightblue",
            )
            main_ax.bar(
                index + 2 * bar_width,
                avg_confidence_img,
                bar_width,
                label="Confidence with Image",
                color="green",
            )
            main_ax.bar(
                index + 3 * bar_width,
                avg_confidence_no_img,
                bar_width,
                label="Confidence without Image",
                color="lightgreen",
            )

            main_ax.set_xticks(index + 1.5 * bar_width)
            main_ax.set_xticklabels(np.arange(0, 7))

        else:
            main_ax.plot(
                np.arange(0, 7),
                avg_accuracy_img,
                label="Accuracy with Image",
                color="blue",
                marker="o",
            )
            main_ax.plot(
                np.arange(0, 7),
                avg_accuracy_no_img,
                label="Accuracy without Image",
                color="lightblue",
                marker="o",
            )
            main_ax.plot(
                np.arange(0, 7),
                avg_confidence_img,
                label="Confidence with Image",
                color="green",
                marker="o",
            )
            main_ax.plot(
                np.arange(0, 7),
                avg_confidence_no_img,
                label="Confidence without Image",
                color="lightgreen",
                marker="o",
            )

            main_ax.set_xticks(np.arange(0, 7))

        test_name = testrun_id_dict.get(test_uuid, test_uuid)
        main_ax.grid(True)
        main_ax.set_xlabel("Masking Levels")
        main_ax.set_ylabel("Scores")
        main_ax.set_title(f"Comparison of Accuracy and Confidence for test {test_name}")
        main_ax.legend()

        if save_fig:

            plt.savefig(
                os.path.join(
                    os.getcwd(),
                    "data",
                    "output",
                    "plots",
                    f"combined_accuracy_confidence_{test_name}.png",
                )
            )

        if show_fig:
            plt.show()
        plt.close()


def plot_average_combined_accuracy_confidence_multiple_tests(
    accuracy_data: dict,
    confidence_data: dict,
    save_fig: bool = True,
    test_collection: str = "unknown",
    show_fig: bool = True,
):
    """Same as plot_combined_accuracy_confidence_data_2, but for multiple tests at once.

    Args:
        accuracy_data (dict): accuracy data in the format {Test_UUID: {Q_UUID: {image_level: {level: accuracy_score}}}
        confidence_data (dict): confidence data in the format {Test_UUID: {Q_UUID: {image_level: {level: confidence_score}}}
        save_fig (bool, optional): Defaults to True.
    """
    assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
    assert accuracy_data.keys() == confidence_data.keys(), "Data keys do not match."

    logger.debug(f"Visualizing {len(accuracy_data)} tests for {test_collection=}")
    print(f"Visualizing {len(accuracy_data)} tests for {test_collection=}")

    levels = range(7)  # From Lvl 0 to Lvl 6

    total_avg_confidence_img = np.zeros(
        7
    )  # To store cumulative scores for image level 0
    total_avg_confidence_no_img = np.zeros(
        7
    )  # To store cumulative scores for image level 1

    total_avg_accuracy_img = np.zeros(7)  # To store cumulative scores for image level 0
    total_avg_accuracy_no_img = np.zeros(
        7
    )  # To store cumulative scores for image level 1

    total_num_questions = 0

    for test_uuid, data in accuracy_data.items():

        avg_confidence_img = np.zeros(7)  # To store average scores for image level 0
        avg_confidence_no_img = np.zeros(7)  # To store average scores for image level 1

        avg_accuracy_img = np.zeros(7)  # To store average scores for image level 0
        avg_accuracy_no_img = np.zeros(7)  # To store average scores for image level 1

        # Calculating averages for each level within a test
        for uuid in data:
            for lvl in levels:
                lvl_key = f"{lvl}"
                avg_accuracy_img[lvl] += data[uuid]["0"][lvl_key]
                avg_accuracy_no_img[lvl] += data[uuid]["1"][lvl_key]

                avg_confidence_img[lvl] += confidence_data[test_uuid][uuid]["0"][
                    lvl_key
                ]
                avg_confidence_no_img[lvl] += confidence_data[test_uuid][uuid]["1"][
                    lvl_key
                ]

        num_questions = len(data)
        total_num_questions += num_questions

        avg_confidence_img = [score / num_questions for score in avg_confidence_img]
        avg_confidence_no_img = [
            score / num_questions for score in avg_confidence_no_img
        ]

        avg_accuracy_img = [score / num_questions for score in avg_accuracy_img]
        avg_accuracy_no_img = [score / num_questions for score in avg_accuracy_no_img]

        total_avg_confidence_img += avg_confidence_img
        total_avg_confidence_no_img += avg_confidence_no_img

        total_avg_accuracy_img += avg_accuracy_img
        total_avg_accuracy_no_img += avg_accuracy_no_img

    # Calculating overall averages across all tests
    overall_avg_confidence_img = total_avg_confidence_img / len(accuracy_data)
    overall_avg_confidence_no_img = total_avg_confidence_no_img / len(accuracy_data)

    overall_avg_accuracy_img = total_avg_accuracy_img / len(accuracy_data)
    overall_avg_accuracy_no_img = total_avg_accuracy_no_img / len(accuracy_data)

    # Main plot: Line Chart
    if plt.fignum_exists(1):
        plt.close()
    fig, main_ax = plt.subplots(figsize=(14, 8))

    index = np.arange(7)

    main_ax.set_ylim(0, 1.1)

    main_ax.plot(
        np.arange(0, 7),
        overall_avg_accuracy_img,
        label="Accuracy with Image",
        color="blue",
        marker="o",
    )
    main_ax.plot(
        np.arange(0, 7),
        overall_avg_accuracy_no_img,
        label="Accuracy without Image",
        color="lightblue",
        marker="o",
    )
    main_ax.plot(
        np.arange(0, 7),
        overall_avg_confidence_img,
        label="Confidence with Image",
        color="green",
        marker="o",
    )
    main_ax.plot(
        np.arange(0, 7),
        overall_avg_confidence_no_img,
        label="Confidence without Image",
        color="lightgreen",
        marker="o",
    )

    main_ax.set_xticks(np.arange(0, 7))
    main_ax.grid(True)
    main_ax.set_xlabel("Masking Levels")
    main_ax.set_ylabel("Scores: Confidence and Accuracy")
    main_ax.set_title(
        f"Comparison of Accuracy and Confidence across all Tests in {test_collection=}"
    )
    main_ax.legend()

    if save_fig:
        plt.savefig(
            os.path.join(
                os.getcwd(),
                "data",
                "output",
                "plots",
                f"combined_accuracy_confidence_all_tests_{test_collection}.png",
            )
        )
    if show_fig:
        plt.show()
    plt.close()


def plot_average_combined_accuracy_confidence_per_test(
    accuracy_data: dict,
    confidence_data: dict,
    save_fig: bool = True,
    test_collection: str = "unknown",
    show_fig: bool = True,
):
    assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
    assert accuracy_data.keys() == confidence_data.keys(), "Data keys do not match."

    logger.debug(f"Visualizing {len(accuracy_data)} tests for {test_collection=}")
    print(f"Visualizing {len(accuracy_data)} tests for {test_collection=}")

    levels = range(7)  # From Lvl 0 to Lvl 6
    test_names = []

    avg_confidence_img = []
    avg_confidence_no_img = []
    avg_accuracy_img = []
    avg_accuracy_no_img = []

    for test_uuid, data in accuracy_data.items():

        sum_confidence_img = 0  # To store cumulative scores for image level 0
        sum_confidence_no_img = 0  # To store cumulative scores for image level 1

        sum_accuracy_img = 0  # To store cumulative scores for image level 0
        sum_accuracy_no_img = 0  # To store cumulative scores for image level 1

        num_questions = 0

        # Calculating sums for each level within a test
        for uuid in data:
            for lvl in levels:
                lvl_key = f"{lvl}"
                sum_accuracy_img += data[uuid]["0"][lvl_key]
                sum_accuracy_no_img += data[uuid]["1"][lvl_key]

                sum_confidence_img += confidence_data[test_uuid][uuid]["0"][lvl_key]
                sum_confidence_no_img += confidence_data[test_uuid][uuid]["1"][lvl_key]

            num_questions += 1

        avg_confidence_img.append(sum_confidence_img / (num_questions * 7))
        avg_confidence_no_img.append(sum_confidence_no_img / (num_questions * 7))

        avg_accuracy_img.append(sum_accuracy_img / (num_questions * 7))
        avg_accuracy_no_img.append(sum_accuracy_no_img / (num_questions * 7))

        test_names.append(testrun_id_dict.get(test_uuid, test_uuid))

    # Main plot: Line Chart
    if plt.fignum_exists(1):
        plt.close()
    fig, main_ax = plt.subplots(figsize=(14, 8))

    index = np.arange(len(test_names))

    main_ax.set_ylim(0, 1.1)

    main_ax.plot(
        index, avg_accuracy_img, label="Accuracy with Image", color="blue", marker="o"
    )
    main_ax.plot(
        index,
        avg_accuracy_no_img,
        label="Accuracy without Image",
        color="lightblue",
        marker="o",
    )
    main_ax.plot(
        index,
        avg_confidence_img,
        label="Confidence with Image",
        color="green",
        marker="o",
    )
    main_ax.plot(
        index,
        avg_confidence_no_img,
        label="Confidence without Image",
        color="lightgreen",
        marker="o",
    )

    main_ax.set_xticks(index)
    main_ax.set_xticklabels(test_names, rotation=45, ha="right")
    main_ax.grid(True)
    main_ax.set_xlabel("Test Names")
    main_ax.set_ylabel("Scores")
    main_ax.set_title(
        f"Comparison of Accuracy and Confidence per test in {test_collection=}"
    )
    main_ax.legend()

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(
                os.getcwd(),
                "data",
                "output",
                "plots",
                f"average_combined_accuracy_confidence_per_test_{test_collection}.png",
            )
        )

    if show_fig:
        plt.show()
    plt.close()


def load_results_and_questions(file: str, answer_type: str = None) -> tuple:
    """Load test results and questions from a JSON file.

    Args:
        file (str): filename of the JSON file containing test results
        answer_type (str, optional): just description. Defaults to None.

    Returns:
        tuple: results and test_metadata
    """
    logger.debug(f"Loading data for {answer_type=} for {file=}...")
    with open(os.path.join(os.getcwd(), "data", "output", file), "r") as f:
        data = json.load(f)

    test_metadata = data.get("test_metadata")
    results = data.get("results")

    return results, test_metadata


def calculate_confidence_difference(confidence_data: dict) -> dict:
    # Calculation 1: Confidence difference per masking level for each question
    confidence_differences = {}
    global_differences = {f"{i}": [] for i in range(7)}  # initialize global differences

    for test_uuid, question_dicts in confidence_data.items():
        for question_uuid, image_level_dicts in question_dicts.items():
            lvl0 = image_level_dicts["0"]  # with image
            lvl1 = image_level_dicts["1"]  # without image
            confidence_differences[question_uuid] = {}
            for mask_level in lvl0.keys():
                diff = lvl0[mask_level] - lvl1[mask_level]
                confidence_differences[question_uuid][mask_level] = diff
                global_differences[mask_level].append(
                    diff
                )  # append difference to global list

    # Calculation 2: Average confidence difference per image level
    average_confidence_differences = {}
    for question_uuid, differences in confidence_differences.items():
        average_confidence_differences[question_uuid] = sum(differences.values()) / len(
            differences
        )

    # Calculation 3: Global average difference per masking level
    global_average_differences = {
        k: sum(v) / len(v) for k, v in global_differences.items()
    }

    print("Confidence Differences Per Masking Level:")
    for question_uuid, differences in confidence_differences.items():
        print(f"Question {question_uuid}:")
        for level, diff in differences.items():
            if diff > 0:
                print(f"  {level}: +{diff:.4f} (higher confidence with image)")
            else:
                print(f"  {level}: {diff:.4f} (higher confidence without image)")

    print("\nAverage Confidence Differences Per Image Level:")
    for question_uuid, avg_diff in average_confidence_differences.items():
        if avg_diff > 0:
            print(
                f"Question {question_uuid}: +{avg_diff:.4f} (on average, higher confidence with image)"
            )
        else:
            print(
                f"Question {question_uuid}: {avg_diff:.4f} (on average, higher confidence without image)"
            )

    print("\nGlobal Average Confidence Differences Per Masking Level:")
    for level, avg_diff in global_average_differences.items():
        if avg_diff > 0:
            print(
                f"{level}: +{avg_diff:.4f} (on average, higher confidence with image)"
            )
        else:
            print(
                f"{level}: {avg_diff:.4f} (on average, higher confidence without image)"
            )

    return (
        confidence_differences,
        average_confidence_differences,
        global_average_differences,
    )


def visualize_global_average_differences(
    global_average_differences: dict,
    test_collection: str,
    tests: str | list = None,
    show: bool = False,
) -> None:
    levels = list(global_average_differences.keys())
    values = list(global_average_differences.values())

    logger.debug(f"Visualizing global average confidence differences for {tests=}")
    print(f"Visualizing global average confidence differences for {tests=}")

    plot_title = f"Global Average Confidence Differences Per Masking Level for {len(tests)} Tests"
    tests_text = f"Test collection: {test_collection}"

    plt.figure(figsize=(10, 16))
    plt.plot(levels, values, marker="o", linestyle="-", color="b")
    plt.title(plot_title)
    plt.xlabel("Masking Level")
    plt.ylabel("Average Confidence Difference")
    plt.grid(True)

    for i in range(len(values)):
        plt.text(levels[i], values[i], f"{values[i]:.2f}", ha="center", va="bottom")

    summary_text = "Positive: Higher confidence with image\nNegative: Higher confidence without image"
    plt.figtext(
        0.5,
        0.06,
        summary_text,
        fontsize=12,
        va="top",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="wheat"),
    )

    # Add tests text under the plot
    plt.figtext(
        0.5,
        0.035,
        tests_text,
        fontsize=10,
        va="top",
        ha="center",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"),
    )

    plt.savefig(
        os.path.join(
            os.getcwd(),
            "data",
            "output",
            "plots",
            f"global_average_confidence_differences_{test_collection}.png",
        )
    )
    if show:
        plt.show()


def format_rated_question_data(
    answer_type: str, files: list[str] = None, max_rating: int = 2
) -> dict:
    """Formats the accuracy scores of autocomplete and open questions by question ID, level, and image presence.
    Takes the average of the ratings of the raters for each question.

    Args:
        answer_type (str): the type of answer to visualize. Can be 'autocomplete' or 'open'.
        files (list[str], optional): List of files. Defaults to None.
        max_rating (int, optional): Maximum rating value. Defaults to 2.

    Returns:
        dict: the accuracy data in the format {UUID: {image_level: {level: accuracy_score}}}
    """
    assert answer_type in ["autocomplete", "open"], "Invalid answer type."
    assert isinstance(files, list), "Please provide a list of filenames to process."

    logger.debug(
        f"Formatting {answer_type} ratings for the following {len(files)} files: \n{files}"
    )

    formatted_data = {}

    for file in files:
        with open(os.path.join(os.getcwd(), "data", "output", file), "r") as f:
            data = json.load(f)

        for testname, questions in data.items():
            formatted_data.setdefault(testname, {})

            for question_uuid, image_levels in questions.items():
                question_data = {"0": {}, "1": {}}

                for image_level, levels in image_levels.items():
                    for level, ratings in levels.items():
                        total_scores = sum(int(ratings[rater]) for rater in ratings)
                        average_score = (total_scores / len(ratings)) / max_rating

                        if "with_image" in image_level:
                            question_data["0"][level] = average_score
                        else:
                            question_data["1"][level] = average_score

                formatted_data[testname][question_uuid] = question_data

    logger.debug(f"Formatted {answer_type} data: {formatted_data}")
    return formatted_data  # [testname]


def visualize_rated_question_data() -> None:
    """Important: distinguish between accuracy and confidence for rated questions.
    The rated questions are rated manually by human raters, so the accuracy is the average rating of the raters.
    Therefore, the confidence is fetched from the original test results, and the accuracy from a separate file.
    """
    logger.debug("Visualizing rated question data...")
    print("Visualizing rated question data...")
    accuracy_files = strange_stories_rated_autocomplete
    # accuracy_files = [accuracy_files[0]]

    confidence_files = SS_autocomplete_files
    # confidence_files = [confidence_files[0]]

    answer_type = "autocomplete"

    confidence_data = format_question_data(
        answer_type=answer_type,
        data_type="confidence",
        files=confidence_files,
    )
    accuracy_data = format_rated_question_data(
        answer_type=answer_type, files=accuracy_files
    )

    assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
    # replace accuracy data main keys by the main keys of the confidence data for comparison
    accuracy_data = {
        k: v for k, v in zip(confidence_data.keys(), accuracy_data.values())
    }
    print("Confidence Data: \n", confidence_data)
    print("Accuracy Data: \n", accuracy_data)

    # plot_average_data_levels(accuracy_data, data_type="accuracy", answer_type="open")
    plot_combined_accuracy_confidence_data_2(
        accuracy_data=accuracy_data, confidence_data=confidence_data
    )


def create_strange_stories_plots_1(show=False) -> None:
    """Creates the four plots of the SS test collection,
    where the accuracy and confidence scores are visualized for each test combined, as averages.
    Creates four plots, one for each answer type: autocomplete, yes-no, multiple-choice, and open.
    """
    logger.debug("Creating Strange Stories plots...")
    print("Creating Strange Stories plots...")

    open_answer_types = ["autocomplete", "open"]

    for answer_type in open_answer_types:

        accuracy_files = (
            strange_stories_rated_open
            if answer_type == "open"
            else strange_stories_rated_autocomplete
        )
        confidence_files = (
            SS_open_files if answer_type == "open" else SS_autocomplete_files
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=confidence_files,
        )
        accuracy_data = format_rated_question_data(
            answer_type=answer_type, files=accuracy_files
        )

        assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
        # replace accuracy data main keys by the main keys of the confidence data for comparison
        accuracy_data = {
            k: v for k, v in zip(confidence_data.keys(), accuracy_data.values())
        }
        plot_average_data_levels(
            multiple_test_data=confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_combined_accuracy_confidence_multiple_tests(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            test_collection=f"SS_{answer_type}",
            show_fig=show,
        )

    closed_answer_types = ["yes-no", "multiple-choice"]
    for answer_type in closed_answer_types:
        accuracy_files = (
            SS_mc_files if answer_type == "multiple-choice" else SS_yn_files
        )
        confidence_files = (
            SS_mc_files if answer_type == "multiple-choice" else SS_yn_files
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=confidence_files,
        )
        accuracy_data = format_question_data(
            answer_type=answer_type,
            data_type="accuracy",
            files=accuracy_files,
        )
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )

        assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."

        plot_average_combined_accuracy_confidence_multiple_tests(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            test_collection=f"SS_{answer_type}",
            show_fig=show,
        )


def create_strange_stories_plots_2(show=False) -> None:
    """Creates the four plots of the SS test collection,
    in contrast to create_strange_stories_plots_1, this function creates the plots with the test names as x-axis labels.
    Creates four plots, one for each answer type: autocomplete, yes-no, multiple-choice, and open.
    """
    logger.debug("Creating Strange Stories plots...")
    print("Creating Strange Stories plots...")

    open_answer_types = ["open", "autocomplete"]

    for answer_type in open_answer_types:
        accuracy_files = (
            strange_stories_rated_open
            if answer_type == "open"
            else strange_stories_rated_autocomplete
        )
        confidence_files = (
            SS_open_files if answer_type == "open" else SS_autocomplete_files
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=confidence_files,
        )
        accuracy_data = format_rated_question_data(
            answer_type=answer_type, files=accuracy_files
        )

        assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
        # replace accuracy data main keys by the main keys of the confidence data for comparison
        accuracy_data = {
            k: v for k, v in zip(confidence_data.keys(), accuracy_data.values())
        }
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_combined_accuracy_confidence_per_test(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            test_collection=f"SS_{answer_type}",
            show_fig=show,
        )

    closed_answer_types = ["yes-no", "multiple-choice"]

    for answer_type in closed_answer_types:
        accuracy_files = (
            SS_mc_files if answer_type == "multiple-choice" else SS_yn_files
        )
        confidence_files = (
            SS_mc_files if answer_type == "multiple-choice" else SS_yn_files
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=confidence_files,
        )
        accuracy_data = format_question_data(
            answer_type=answer_type,
            data_type="accuracy",
            files=accuracy_files,
        )
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )

        assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."
        # replace accuracy data main keys by the main keys of the confidence data for comparison

        plot_average_combined_accuracy_confidence_per_test(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            test_collection=f"SS_{answer_type}",
            show_fig=show,
        )


def create_sally_anne_plots(show=False) -> None:
    logger.debug("Creating Sally Anne plots...")
    print("Creating Sally Anne plots...")

    closed_answer_types = ["yes-no", "multiple-choice"]
    for answer_type in closed_answer_types:
        files = SA_mc_files if answer_type == "multiple-choice" else SA_yn_files

        accuracy_data = format_question_data(
            answer_type=answer_type,
            data_type="accuracy",
            files=files,
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=files,
        )
        plot_combined_accuracy_confidence_data_2(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            use_bars=False,
            show_fig=show,
        )
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )

    open_answer_types = ["open", "autocomplete"]
    for answer_type in open_answer_types:
        confidence_files = (
            SA_open_files if answer_type == "open" else SA_autocomplete_files
        )
        accuracy_files = (
            sally_anne_rated_open
            if answer_type == "open"
            else sally_anne_rated_autocomplete
        )

        accuracy_data = format_rated_question_data(
            answer_type=answer_type, files=accuracy_files
        )

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=confidence_files,
        )
        assert len(accuracy_data) == len(confidence_data), "Data lengths do not match."

        # replace accuracy data main keys by the main keys of the confidence data for comparison
        accuracy_data = {
            k: v for k, v in zip(confidence_data.keys(), accuracy_data.values())
        }
        plot_combined_accuracy_confidence_data_2(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            use_bars=False,
            show_fig=show,
        )
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )


def create_imposing_memory_plots(show=False) -> None:
    """Creates all indiviual plots and one combined plot for all four tests, i.e. filtered and non-filtered tests."""
    logger.debug("Creating Imposing Memory plots...")
    print("Creating Imposing Memory plots...")

    answer_type = "yes-no"
    IM_source_files = [IM_complete_files, IM_filtered_files]

    for IM_files in IM_source_files:
        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=IM_files,
        )
        accuracy_data = format_question_data(
            answer_type=answer_type,
            data_type="accuracy",
            files=IM_files,
        )
        plot_combined_accuracy_confidence_data_2(
            accuracy_data=accuracy_data,
            confidence_data=confidence_data,
            use_bars=False,
            show_fig=show,
        )
        plot_average_data_levels(
            confidence_data,
            data_type="confidence",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )
        plot_average_data_levels(
            multiple_test_data=accuracy_data,
            data_type="accuracy",
            answer_type=answer_type,
            show_labels=False,
            show_fig=show,
        )


def print_avg_accuracy(files: list[str], answer_type: str) -> None:

    if answer_type in ["autocomplete", "open"]:
        accuracy_data = format_rated_question_data(answer_type=answer_type, files=files)
    else:
        accuracy_data = format_question_data(
            answer_type=answer_type,
            files=files,
            data_type="accuracy",
        )

    for test_uuid, data in accuracy_data.items():

        levels = range(7)  # From Lvl 0 to Lvl 6
        avg_scores_img = [0] * 7  # To store average scores for image level 0
        avg_scores_no_img = [0] * 7  # To store average scores for image level 1

        # Calculating averages for each level
        for uuid in data:
            for lvl in levels:
                lvl_key = f"{lvl}"
                avg_scores_img[lvl] += data[uuid]["0"][lvl_key]
                avg_scores_no_img[lvl] += data[uuid]["1"][lvl_key]

        num_questions = len(data)
        avg_scores_img = [score / num_questions for score in avg_scores_img]
        avg_scores_no_img = [score / num_questions for score in avg_scores_no_img]

        # average scores for all levels
        avg_scores_img = sum(avg_scores_img) / len(avg_scores_img)
        avg_scores_no_img = sum(avg_scores_no_img) / len(avg_scores_no_img)

        #  round to 3 decimal places
        avg_scores_img = round(avg_scores_img, 3)
        avg_scores_no_img = round(avg_scores_no_img, 3)

        # print(
        #     f"Average Scores for test {testrun_id_dict.get(test_uuid, test_uuid)} and {answer_type=}:"
        # )
        # print(f"  With Image: {avg_scores_img}")
        # print(f"  No Image: {avg_scores_no_img}\n\n")

        print(
            f"{testrun_id_dict.get(test_uuid, test_uuid)} & {answer_type} & {avg_scores_img=} & {avg_scores_no_img=}"
        )


def print_all_accuracy() -> None:
    answer_type = "yes-no"
    IM_source_files = [IM_complete_files, IM_filtered_files]

    # IM
    for IM_files in IM_source_files:
        print_avg_accuracy(IM_files, answer_type)

    open_answer_types = ["open", "autocomplete"]
    closed_answer_types = ["yes-no", "multiple-choice"]

    # OPEN TYPES
    for answer_type in open_answer_types:
        accuracy_files = (
            sally_anne_rated_open
            if answer_type == "open"
            else sally_anne_rated_autocomplete
        )
        print_avg_accuracy(accuracy_files, answer_type)

        accuracy_files = (
            strange_stories_rated_open
            if answer_type == "open"
            else strange_stories_rated_autocomplete
        )
        print_avg_accuracy(accuracy_files, answer_type)

    # CLOSED TYPES
    for answer_type in closed_answer_types:
        files = SA_mc_files if answer_type == "multiple-choice" else SA_yn_files

        print_avg_accuracy(files, answer_type)

        files = SS_mc_files if answer_type == "multiple-choice" else SS_yn_files
        print_avg_accuracy(files, answer_type)


def visualize_global_confidence_differences(show=False) -> None:
    answer_types = ["autocomplete", "open", "yes-no", "multiple-choice"]

    for answer_type in answer_types:
        files = validate_answertype_filenames(answer_type=answer_type, filename=None)

        confidence_data = format_question_data(
            answer_type=answer_type,
            data_type="confidence",
            files=files,
        )
        print("Confidence Data: \n", confidence_data)
        print("Evaluated Files: \n", files)
        print("Answer Type: \n", answer_type)
        (
            confidence_differences,
            average_confidence_differences,
            global_average_differences,
        ) = calculate_confidence_difference(confidence_data)

        visualize_global_average_differences(
            global_average_differences=global_average_differences,
            test_collection=answer_type,
            tests=files,
            show=show,
        )


def run():

    print_all_accuracy()

    return
    show_plots = False
    create_imposing_memory_plots(show=show_plots)  # final plots for IM
    create_sally_anne_plots(show=show_plots)  # final plots for SA
    create_strange_stories_plots_2(show=show_plots)  # final plots for SS - per test
    create_strange_stories_plots_1(
        show=show_plots
    )  # final plots for SS - combined tests
    return

    # visualize_global_confidence_differences()  # visualize global differences
    # visualize_rated_question_data() # visualize rated questions


if __name__ == "__main__":
    print(
        "Please run from the visualize.py file in the project root directory.\nExiting..."
    )
    import sys

    sys.exit(1)
