import os
import json
import pandas as pd
import logging.config
import matplotlib.pyplot as plt


from src.config import LOGGING_CONFIG
from src.utils.ai import (
    OpenAI,
    get_chat_completion,
    ChatCompletion,
)
from src.utils.misc import (
    dict_to_string,
)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def log_probs_testrun(client: OpenAI):

    messages = [
        {
            "role": "system",
            "content": "Je antwoordt op de vraag in het Nederlands. Als er geen vraag is, antwoord je met 'geen vraag' Als er een incomplete zin wordt gegeven, maak je de zin af.",
        },
        {"role": "user", "content": "De hoofdstad van Nederland is "},
    ]
    model = "gpt-4-turbo"

    response: ChatCompletion = get_chat_completion(
        client=client, messages=messages, model=model, logprobs=True, top_logprobs=5
    )
    response_string = dict_to_string(response.to_dict())
    # dump the response string to a file
    filepath = os.path.join(
        os.getcwd(), "data", "output", "openai_logprobs_testrun.json"
    )
    with open(filepath, "w") as f:
        f.write(response_string)

    logger.debug(response_string)
    # print(response.to_dict())


def create_logprobs_plot(data: dict, output_file: str):
    """
    Create a logprobsp lot from raw openai response data.

    Args:
        data (dict): raw openai response data
        output_file (str): filename for the output plot
    """

    tokens = []
    logprobs = []
    top_tokens = []
    top_logprobs = []

    for content in data["choices"][0]["logprobs"]["content"]:
        tokens.append(content["token"])
        logprobs.append(content["logprob"])
        for top in content["top_logprobs"]:
            top_tokens.append(f"{content['token']} ({top['token']})")
            top_logprobs.append(top["logprob"])

    df = pd.DataFrame({"Token": tokens, "Log Probability": logprobs})
    df_top = pd.DataFrame(
        {"Token Pair": top_tokens, "Top Log Probability": top_logprobs}
    )

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Main tokens plot
    bars = axs[0].bar(
        df["Token"], df["Log Probability"], color="blue", label="Main Tokens"
    )
    axs[0].set_title("Log Probabilities of Tokens")
    # add padding below the title
    axs[0].title.set_y(1.2)
    axs[0].set_xlabel("Token")
    axs[0].set_ylabel("Log Probability")
    axs[0].legend(title="Lower log probability indicates higher model confidence.")

    # Add text annotations for main tokens
    for bar in bars:
        yval = bar.get_height()
        axs[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.5f}",
            va="bottom",
            ha="center",
        )  # Adjust alignment and formatting

    # Top alternatives plot
    bars_top = axs[1].bar(
        df_top["Token Pair"],
        df_top["Top Log Probability"],
        color="green",
        label="Top Alternative Tokens",
    )
    axs[1].set_title("Top Log Probabilities of Token Pairs")
    axs[1].set_xlabel("Log Probability")
    axs[1].set_ylabel("Token Pair")
    axs[1].legend(
        title="Top alternatives (lower is better). Token (Alternative) explains the comparison basis."
    )

    # Add text annotations for top tokens
    for bar in bars_top:
        yval = bar.get_height()
        axs[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.5f}",
            va="bottom",
            ha="center",
        )  # Adjust alignment and formatting

    plt.setp(axs[1].get_xticklabels(), rotation=45, horizontalalignment="right")

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "data", "output", f"{output_file}.png"))
    plt.show()


def run():
    client = OpenAI()

    log_probs_testrun(client=client)

    filepath = os.path.join(
        os.getcwd(), "data", "output", "openai_logprobs_testrun.json"
    )
    with open(filepath, "r") as f:
        data = json.load(f)

    create_logprobs_plot(data=data, output_file="logprobs_plot")
