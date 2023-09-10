import os

from bardapi import Bard
import uuid
import re
from tqdm import tqdm
import argparse


def get_fairytale(bard_obj):
    response = bard_obj.get_answer("Napisz bajkę dla dzieci")["content"]

    return response


def save_fairytale(data, path):
    name = str(uuid.uuid4())

    with open(path + name + ".txt", "w") as f:
        f.write(data)


def clean_string(data):
    pattern = r"[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\.,!?;:\"\' ]"

    # Use re.sub to replace all non-matching characters with an empty string
    result_string = re.sub(pattern, "", " ".join(data.split()))
    return result_string


def gather_fairy_tales(bard_obj, n_iter, path):
    for i in tqdm(range(n_iter)):
        fairy_tale = get_fairytale(bard_obj)
        cleaned = clean_string(fairy_tale)
        save_fairytale(cleaned, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n-iter", type=int, default=1, help="Number of iterations of gathering"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        required=True,
        help="Path to directory where fairy tales will be saved",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (If not provided defaults to BARD_API_KEY environment variable)",
    )
    args = parser.parse_args()

    if args.api_key is None:
        apikey = os.environ.get("BARD_API_KEY")
        if apikey is None:
            raise ValueError(
                "Bard API environment variable not set. Please set BARD_API_KEY environment variable or provide "
                "--api-key argument"
            )
        bard = Bard(apikey)
    else:
        bard = Bard(token=args.api_key)

    gather_fairy_tales(
        bard_obj=bard,
        n_iter=args.n_iter,
        path=args.path,
    )
