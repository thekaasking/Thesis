"""Util for handling files and file operations"""

import os
import logging.config

from src.config import LOGGING_CONFIG
from src.utils.ai import encode_image

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def load_images(
    image_keys: list[str], test_folder: str, ext: str = "png"
) -> tuple[dict, dict]:
    """load images from the test folder provided

    Args:
        image_keys (list[str]): list of image filenames to load
        test_folder (str): folder name where the images are stored
        ext (str, optional): image file extension. Defaults to "png".

    Returns:
        tuple[dict, dict]: dictionary of images and blurred images with the keys
        provided in image_keys and the values as the base64 encoded images.
    """

    try:
        image_paths = {
            key: f"a.{ext}" if key == "a" else f"{key}.{ext}" for key in image_keys
        }
        logger.debug(f"Reading images from {image_paths=}")

        images = {}
        for key, value in image_paths.items():
            try:
                if value is None or value == f"None.{ext}":
                    logger.debug(f"Skipping image {value} as it is None.")
                    continue
                b64_image = encode_image(
                    image_path=os.path.join(
                        os.getcwd(), "data", "input", "images", test_folder, value
                    )
                )
                # strip the key to remove the .{ext} extension
                stripped_key = key.split(".")[0]
                images[stripped_key] = b64_image
            except Exception as e:
                logger.error(f"Error loading single image {value}: {e}. Skipping.")
                continue

        logger.debug(f"Successfully loaded {len(images)} images.")
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        images = None
        # logger.debug(f"{images=}")

    return images


def detect_test_collections(
    input_dir: str = os.path.join(os.getcwd(), "data", "input"),
    test_filename_prefix: str = "test_collection_",
) -> list[str]:
    """
    Detect test collections in the input directory. Adds the absolute path to the list.

    Args:
        input_dir (str, optional):  Defaults to os.path.join("data", "input").
        test_filename_prefix (str, optional): Defaults to "test_collection_".

    Returns:
        list[str]: list of all paths of the test collections.
    """
    test_collection_files = []
    for f in os.listdir(input_dir):
        if (
            os.path.isfile(os.path.join(input_dir, f))
            and f.startswith(test_filename_prefix)
            and f.endswith(".yaml")
        ):
            # add abs path to the list
            test_collection_files.append(os.path.join(input_dir, f))

    return test_collection_files
