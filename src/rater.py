import os
import json
import glob
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_results(results, output_file):
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)


def rate_tests(directory):
    """Open test results files in a directory and rate the questions in each test.
    Prompts the user to rate each question in each test, and saves the results to a new file.

    Args:
        directory (str or Path): Directory containing test results files to rate.
    """
    print(f"Rating all tests in directory: {directory}")
    files = [
        f
        for f in os.listdir(directory)
        if (
            f.startswith("test_results_")
            and f.endswith(("_autocomplete.json", "_open.json"))
        )
    ]

    print("Files to rate: ")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")

    # check which file to rate, or all
    file_number = input("Enter file number to rate (or enter 'all'): ")
    if file_number.lower() == "all":
        files_to_rate = files
    else:
        file_number = int(file_number) - 1
        while file_number not in range(len(files)):
            print("Invalid file number. Please enter a valid file number.")
            file_number = int(input("Enter file number to rate: "))
        files_to_rate = [files[file_number]]

    rater_number = int(input("Enter rater number (1 or 2): "))
    while rater_number not in [1, 2]:
        print("Invalid rater number. Please enter 1 or 2.")
        rater_number = int(input("Enter rater number (1 or 2): "))
    print(f"Rater {rater_number} is rating the tests.\n\n")

    for file in files_to_rate:
        file_path = os.path.join(directory, file)

        test_data = load_json_file(file_path)
        blind = input("Blind rating? (y/n): ").upper() == "Y"
        _ = rate_questions(test_data, directory, rater_number, blind=blind)


def rate_questions(
    data: dict,
    directory,
    rater_number: int,
    blind: bool = False,
    overwrite: bool = True,
):
    """Rate the questions in a test and save the results to a new file.

    Args:
        data (dict): loaded json data from test results file
        directory (str or Path): path to directory to save results file
        rater_number (int): rater number, 1 or 2
        blind (bool, optional): whether the rating occurs blind. Defaults to False.
        overwrite (bool, optional): whether to override pre existing rater files. Defaults to True.

    Returns:
        results (dict):
    """
    test_metadata = data.get("test_metadata")
    result_data = data.get("results")

    if test_metadata:
        test_name = test_metadata.get("test_name")
        test_id = test_metadata.get("test_id")
        print(f"Test Name: {test_name} - Test ID: {test_id}\n")

        results_file_name = os.path.join(directory, f"output_rated_{test_name}.json")

        if overwrite:
            results = {}

        # check if results file already exists
        else:
            print(
                f"Results file already exists for test: {test_name}. Loading existing results."
            )
            results = load_json_file(results_file_name)
            # check if the rater number already rated the test
            if test_name in results:
                if (
                    f"score_rater_{rater_number}"
                    in results[test_name]["with_image"]["0"]
                    and f"score_rater_{rater_number}"
                    in results[test_name]["without_image"]["0"]
                ):
                    print(
                        f"Rater {rater_number} has already rated this test. Skipping."
                    )
                    return results

        if test_name not in results:
            results[test_name] = {}

        for question in test_metadata.get("questions", []):
            question_id = question.get("question_id")
            question_answer_type = question.get("answer_type")
            if question_answer_type in ["autocomplete", "open"]:
                correct_answers = question.get("correct_answers")
                prompts = test_metadata["full_masked_prompts"].get(question_id, {})

                if question_id not in results[test_name]:
                    results[test_name][question_id] = {
                        "with_image": {},
                        "without_image": {},
                    }

                for image_level in ["with_image", "without_image"]:
                    for lvl in range(7):
                        # print(f"debug {result_data[image_level]}")
                        model_answer_text = result_data[image_level][f"{lvl}"][
                            question_id
                        ]["choices"][0]["message"]["content"]
                        print("_" * 80, "\n")
                        if blind:
                            print(
                                f"Rate the following question: \n"
                                f"Prompt: [{prompts[str(lvl)]}] \n"
                                f"Model Answer: [{model_answer_text}]"
                            )
                        else:
                            print(
                                f"Rate the following question: \n"
                                f"Question ID: {question_id} \n"
                                f"Image Level: {image_level} - Masking Level: {lvl} \n"
                                f"Prompt: [{prompts[str(lvl)]}] \n"
                                f"Correct Answer: {correct_answers} \n"
                                f"Model Answer: [{model_answer_text}]"
                            )

                        for rater in [1, 2]:
                            score = input(f"\nRater {rater}\nEnter score (0-2): ")
                            while score not in ["0", "1", "2"]:
                                print(
                                    "Invalid score. Please enter a score between 0 and 2."
                                )
                                score = input("Enter score (0-2): ")

                            if rater == 1:
                                score_rater_1 = score
                            else:
                                score_rater_2 = score

                        results[test_name][question_id][image_level][f"{lvl}"] = {
                            "score_rater_1": score_rater_1,
                            "score_rater_2": score_rater_2,
                        }

                        # results[test_name][question_id][image_level][f"{lvl}"] = {
                        #     f"score_rater_{rater_number}": score,
                        # }
            else:
                print(
                    f"Invalid answer type {question_answer_type} for question: {question_id}. Skipping."
                )
                continue

        print(f"Finished rating test: {test_name}\n")
    else:
        print(f"Invalid file format for test in {directory}")

    save_results(results, output_file=results_file_name)

    return results


def calculate_kappa(scores_rater_1, scores_rater_2):
    return cohen_kappa_score(scores_rater_1, scores_rater_2)


def calc_cohens_kappa(directory):
    general_scores_rater_1 = []
    general_scores_rater_2 = []
    difficulty_scores = {}
    image_scores = {
        "with_image": {"rater_1": [], "rater_2": []},
        "without_image": {"rater_1": [], "rater_2": []},
    }

    # Read all JSON files
    files = glob.glob(os.path.join(directory, "output_rated_*.json"))
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            for test_name, questions in data.items():
                for question_uuid, conditions in questions.items():
                    for condition, difficulties in conditions.items():
                        for difficulty, scores in difficulties.items():
                            score_rater_1 = int(scores["score_rater_1"])
                            score_rater_2 = int(scores["score_rater_2"])

                            # Aggregate general scores
                            general_scores_rater_1.append(score_rater_1)
                            general_scores_rater_2.append(score_rater_2)

                            # Aggregate scores per difficulty level
                            if difficulty not in difficulty_scores:
                                difficulty_scores[difficulty] = {
                                    "rater_1": [],
                                    "rater_2": [],
                                }
                            difficulty_scores[difficulty]["rater_1"].append(
                                score_rater_1
                            )
                            difficulty_scores[difficulty]["rater_2"].append(
                                score_rater_2
                            )

                            # Aggregate scores per image level
                            image_scores[condition]["rater_1"].append(score_rater_1)
                            image_scores[condition]["rater_2"].append(score_rater_2)

    kappa_general = calculate_kappa(general_scores_rater_1, general_scores_rater_2)

    kappas_per_difficulty = {}
    for difficulty, scores in difficulty_scores.items():
        kappas_per_difficulty[difficulty] = calculate_kappa(
            scores["rater_1"], scores["rater_2"]
        )

    kappas_per_image = {}
    for condition, scores in image_scores.items():
        kappas_per_image[condition] = calculate_kappa(
            scores["rater_1"], scores["rater_2"]
        )

    # Print the results
    print(f"Cohen's kappa (general): {kappa_general}")
    for difficulty, kappa in kappas_per_difficulty.items():
        print(f"Cohen's kappa (difficulty level {difficulty}): {kappa}")
    for condition, kappa in kappas_per_image.items():
        print(f"Cohen's kappa ({condition}): {kappa}")

    # Visualize the results
    visualize_kappas(kappa_general, kappas_per_difficulty, kappas_per_image)


def visualize_kappas(kappa_general, kappas_per_difficulty, kappas_per_image):
    categories = (
        ["General"]
        + [f"Difficulty {d}" for d in sorted(kappas_per_difficulty.keys())]
        + ["With Image", "Without Image"]
    )
    kappa_values = (
        [kappa_general]
        + [kappas_per_difficulty[d] for d in sorted(kappas_per_difficulty.keys())]
        + [kappas_per_image["with_image"], kappas_per_image["without_image"]]
    )

    plt.figure(figsize=(10, 6))
    plt.bar(categories, kappa_values, color="blue")
    plt.xlabel("Categories")
    plt.ylabel("Cohen's Kappa")
    plt.title("Cohen's Kappa Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(kappa_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "output", "kappa_scores.png"))
    plt.show()


def run():
    output_filepath = os.path.join(os.getcwd(), "data", "output")

    print("RATER PROGRAM.\n")
    mode = input("Do you want to rate tests or calculate Cohen's kappa? (rate/kappa): ")
    while mode not in ["rate", "kappa"]:
        print("Invalid mode. Please enter 'rate' or 'kappa'.")
        mode = input(
            "Do you want to rate tests or calculate Cohen's kappa? (rate/kappa): "
        )

    if mode == "rate":
        rate_tests(directory=output_filepath)

    elif mode == "kappa":
        print("Calculating Cohen's kappa.")
        calc_cohens_kappa(directory=output_filepath)
