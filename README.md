# Multimodal AI vs Theory of mind

Razo van Berkel

## Project structure

.
├── LICENSE \
├── README.md \
├── data \
│   ├── input \
│   │   ├── images \
│   │   │   ├── {test image folders} \
│   │   │ \
│   │   ├── {sample test collection}.yaml \
│   │   ├── source \
│   │   │   ├── {source test files}.yaml \
│   └── output \
│       └── {sample test results}.json \
├── logs \          
│   └── app.log         # program logs file\
├── requirements.txt    # python dependencies \
├── run.py              # main entry point for the LMToM-Eval program \
├── scripts \
│   └── explore_data.ipynb \
├── src \
│   ├── config.py       # containing all pipeline configuration and defaults \
│   ├── main.py         # main LMToM-EVAL program code \
│   ├── models.py       # data models and classes as used in the script \
│   ├── test.py         # scripts used in testing and development \   
│   ├── utils \
│   │   ├── ai.py       # contains all AI-helper utils; e.g. for NLP and OpenAI\
│   │   ├── files.py    # contains all file helper utils\
│   │   └── misc.py     # contains miscellaneous helper functions\
│   ├── rater.py         # scripts and program for manual grading/rating of results/questions \   
│   └── visualize.py    # Contains all scripts for analyzing and visualising data \
├── test.py             # entry point for test script\
├── rate.py             # entry point for rating script\
└── visualize.py        # entry point for visualize script \

## Unpacking Tests and Results.

To protect against webscrapers scraping the contents of the tests, they are provided in a zipped manner and password-protected. The password is `mediatech`. Note: because the dump of the test results contain the test-metadata, which includes the questions, these were also zipped, using the same password.


## Running the Tests

DISCLAIMER: This project has been developped on `Python 3.11.8`, on Windows 11 Pro. No other Python versions are tested, and therefore there is no guarantee for correct program execution.

Ensure Python is installed. Check Python version by running in a command shell: \

    py --version

or \

    python --version

or on a Linux-based system: \

    python3 --version

If the proper version is installed, create and activate a Python Virtual Environment. \
For POSIX (Windows) systems: \

    py -m venv .venv
    .venv\Scripts\activate

For UNIX (MacOS or Linux-based) \

    python3 -m venv .venv
    source .venv/scripts/activate

When activated, install the Python project dependencies from requirements.txt file. Run: \ 

    pip install -r requirements.txt

If no errors, you can run the program. From the root (Thesis) dir, run \

    py run.py

This starts the program.



## Notes

Turned off in libs/openai/_utils/_logs.py: \
def setup_logging() -> None:
    env = "info"

### Json path finder

Used https://jsonpathfinder.com/ to find the json 

## Defining tests


### Programatically for the Pipeline

Defining tests in the Pipeline, programmatically, is possible. This is done by fillign the Test object. See the model definition below.


### Defining Tests in YAML

The preferred way to define tests is using the built-in test loader. It loads tests from YAML format. 

The YAML structure is defined as follows.

- test_collection: List of tests.
    - sys_msg: System message for the test.
    - model: Model name (e.g., "gpt-4o").
    - test_name: Name of the test.
    - test_answer_type: Type of answers expected (e.g., "open", "multiple-choice").
    - test_images_folder: Folder where images are stored.
    - questions: List of questions with properties:
        - question: The text of the question.
        - correct_answers: List of correct answers.
        - image_name: Name of the image file (if any).
        - question_type: Type of question (e.g., "control", "justification").
        - answer_type: Type of answer (e.g., "open", "multiple-choice").


### Model Definitions

    class Question:
        def __init__(
            self,
            question: str,
            correct_answers: list[str],
            b64_image: str = None,
            question_type: str = "",
            answer_type: str = "",
        ):
            self.question = question
            self.correct_answers = correct_answers
            self.b64_image = b64_image
            self.question_type = question_type
            self.answer_type = answer_type
            self.question_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{question}{b64_image}")

    @dataclass
    class TestMetadata:
        test_id: str
        sys_msg: str
        questions: list[Question]
        model: str
        test_name: str = ""
        test_answer_type: str = ""

    class Test:
        def __init__(
            self,
            sys_msg: str,
            questions: list[Question],
            model: str,
            test_name: str = "",
            test_answer_type: str = "",
        ):
            self.sys_msg = sys_msg
            self.questions = questions
            self.model = model
            self.test_id = str(uuid.uuid4())
            self.test_name = test_name
            self.test_answer_type = test_answer_type

            self.test_metadata = TestMetadata(
                test_id=self.test_id,
                sys_msg=sys_msg,
                questions=questions,
                model=model,
                test_name=test_name,
                test_answer_type=test_answer_type,
            )

### Defining tests

Note: add answer_instructions to MC quesiton if deviating from standard four options.

### On Masking words

Words can be masked to evaluate model performance with incomplete information.
We mask words by replacing the word to mask with a mask, "[MASK]".

#### Using NLP

Using NLP, we parse sentences and get the Parts-of-Speech (POS). This is useful for masking certain POS types programmatically. We define four levels of masking words.

- Level 0: (no mask)
- Level 1: PRON
- Level 2: PRON, DET
- Level 3: PRON, DET, ADJ
- Level 4: PRON, DET, ADJ, ADV
- Level 5: PRON, DET, ADJ, ADV, VERB
- Level 6: PRON, DET, ADJ, ADV, VERB, NOUN, PROPN


Supported POS tags to mask:
- ADJ: adjective
- ADP: adposition
- ADV: adverb
- AUX: auxiliary
- CCONJ: coordinating conjunction
- DET: determiner
- INTJ: interjection
- NOUN: noun
- NUM: numeral
- PART: particle
- PRON: pronoun
- PROPN: proper noun
- PUNCT: punctuation
- SCONJ: subordinating conjunction
- SYM: symbol
- VERB: verb
- X: other

Source of Universal POS Tags: https://universaldependencies.org/u/pos/