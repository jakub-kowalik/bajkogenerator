import argparse
import os
import openai
import uuid
from tqdm import tqdm


def get_fairytale(n_tales=1, maximum_tokens=3700):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś generatorem bajek dla dzieci"},
            {"role": "user", "content": "Napisz bajkę dla dzieci"},
        ],
        temperature=1,
        max_tokens=maximum_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n_tales,
    )

    return response


def prepare_fairytale(response):
    tales = []
    for i in range(len(response["choices"])):
        tales.append(" ".join(response["choices"][i]["message"]["content"].split()))

    return tales


def save_fairytale(fairy_tales, path):
    for ft in fairy_tales:
        name = str(uuid.uuid4())

        with open(path + name + ".txt", "w") as f:
            f.write(ft.encode().decode())


def gather_fairy_tales(n_iter, n_tales, path, maximum_tokens=3700):
    for i in tqdm(range(n_iter)):
        fairy_tales = prepare_fairytale(get_fairytale(n_tales=n_tales), maximum_tokens)
        save_fairytale(fairy_tales, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n-iter", type=int, default=1, help="Number of iterations of gathering"
    )
    parser.add_argument(
        "--n-tales", type=int, default=1, help="Number of tales in one iteration"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        required=True,
        help="Path to directory where fairy tales will be saved",
    )
    parser.add_argument(
        "--maximum-tokens",
        type=int,
        default=3700,
        help="Maximum number of tokens in one fairy tale (Needs to be scaled with n_tales to meet OpenAI API limits)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (If not provided defaults to OPENAI_API_KEY environment variable)",
    )
    args = parser.parse_args()

    if args.api_key is None:
        apikey = os.environ.get("OPENAI_API_KEY")
        if apikey is None:
            raise ValueError(
                "OpenAI API environment variable not set. Please set OPENAI_API_KEY environment variable or provide "
                "--api-key argument"
            )
        openai.api_key = apikey
    else:
        openai.api_key = args.api_key

    gather_fairy_tales(
        n_iter=args.n_iter,
        n_tales=args.n_tales,
        path=args.path,
        maximum_tokens=args.maximum_tokens,
    )
