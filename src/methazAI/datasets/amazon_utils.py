import gzip
import pickle
import pandas as pd
import numpy as np
import csv
import requests
import zlib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm
import os
import re
import nltk
import string
from os import listdir
from os.path import isfile, join
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("popular")


def download_dataset(dataset_name, wd, chunk_size=8192):
    endpoint = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_"
    )
    endpoint += dataset_name + "_5.json.gz"

    print("Downloading dataset " + dataset_name + "...")
    r = requests.get(endpoint, allow_redirects=True, stream=True)
    progr_bar = tqdm(
        total=int(r.headers.get("content-length", 0)), unit="iB", unit_scale=True
    )
    if r.status_code == 200:
        with open(
            wd + "/data/SA_amazon_data/external/" + dataset_name + ".bin", "wb"
        ) as extfile:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progr_bar.update(len(chunk))
                extfile.write(chunk)
    elif r.status_code == 404:
        raise ValueError("Requested dataset does not exists on server.")


def fetch_raw_dataset(dataset_name, wd):
    stop_words = set(stopwords.words("english"))
    try:
        with open(
            wd + "/data/SA_amazon_data/external/" + dataset_name + ".bin", "rb"
        ) as extfile:
            data = zlib.decompress(extfile.read(), zlib.MAX_WBITS | 32).decode("utf-8")
            data = data.split("\n")

            with open(
                wd + "/data/SA_amazon_data/interim/" + dataset_name + ".csv", "w"
            ) as outfile:
                for review in data:
                    try:
                        obj = json.loads(review)
                        try:
                            sentence = f'{obj["textReview"]}'
                            sentence = handle_string(sentence, stop_words)
                            line = f"{sentence};{obj['overall']};{dataset_name}\n"
                        except KeyError:
                            sentence = f'{obj["reviewText"]}'
                            sentence = handle_string(sentence, stop_words)
                            line = f"{sentence};{obj['overall']};{dataset_name}\n"

                        outfile.write(line)
                    except:
                        pass  # warnings.warn("A record in dataset "+dataset_name+" has been skipped as it was corrupted.")
    except FileNotFoundError:
        download_dataset(dataset_name, wd)
        fetch_raw_dataset(dataset_name, wd)


def handle_string(text_input, stop_words):
    text_input = remove_URL(text_input)
    text_input = remove_html(text_input)
    text_input = remove_emojis(text_input)
    text_input = remove_punct(text_input)
    text_input = re.sub(" +", " ", text_input)

    filtered_sentence = []
    tokenized_sentence = word_tokenize(text_input)
    for w in tokenized_sentence:
        if w not in stop_words:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)


def download_if_not_existing(datasets, wd=""):

    print(listdir(wd + "/data/SA_amazon_data/external/"))
    try:
        available_datasets = [
            f[:-4]
            for f in listdir(wd + "/data/SA_amazon_data/external/")
            if isfile(join(wd + "/data/SA_amazon_data/external", f))
            and f[:-4] in datasets
        ]
        to_download = [item for item in datasets if item not in available_datasets]

        for dataset in to_download:
            fetch_raw_dataset(dataset, wd)
    except Exception as ex:
        if type(ex) == FileNotFoundError:
            raise FileNotFoundError(
                f"The {wd}/data/SA_amazon_data/ directory does not exist. Create it before moving on."
            )


def check_and_create_data_subfolders(
    root="/data/SA_amazon_data/", subfolders=["raw", "interim", "processed", "external"]
):
    for folder in subfolders:
        if not os.path.exists(root + folder):
            os.makedirs(root + folder)


def ensemble(config, wd):
    check_and_create_data_subfolders(root=wd + "/data/SA_amazon_data/")
    datasets = parse_datasets(config)

    name = config["name"]

    download_if_not_existing(datasets, wd)
    check_and_create_data_subfolders(
        wd + "/data/SA_amazon_data/raw/", subfolders=[str(name)]
    )

    f = open(
        wd + "/data/SA_amazon_data/raw/" + str(name) + "/AmazonProductReviews.csv", "w"
    )
    for filename in datasets:
        fetch_raw_dataset(filename, wd)
        with open(wd + "/data/SA_amazon_data/interim/" + filename + ".csv") as subfile:
            f.write(subfile.read())

        os.remove(wd + "/data/SA_amazon_data/interim/" + filename + ".csv")

    with open(wd + "/data/SA_amazon_data/raw/" + str(name) + "/datasets.txt", "w") as f:
        f.write(f"Used datasets: {datasets}")


def parse_datasets(config):

    flags = config["used_datasets"]
    try:
        datasets = [k for (k, v) in flags.items() if int(v) == 1]
    except ValueError:
        raise ValueError("Insert only 0 (not wanted) or 1 (wanted) in the config file")

    return datasets


def clean_data(config, wd):
    # Getting the rest of configs
    dataset_name = config["name"]
    seed = config["seed"]
    splits = config["train_val_test_splits"]
    max_length = config["max_seq_length"]
    datasets = parse_datasets(config)
    print("Using following datasets: {}".format(datasets))

    # load raw csv file for given reviews at supplied path
    df = check_and_load_raw(
        wd
        + "/data/SA_amazon_data/raw/"
        + str(dataset_name)
        + "/AmazonProductReviews.csv"
    )

    try:
        f = gzip.open(
            wd
            + "/data/SA_amazon_data/processed/"
            + str(dataset_name)
            + "/used_datasets.pklz",
            "rb",
        )
        existing_datasets = pickle.load(f, encoding="bytes")
        if existing_datasets == datasets:
            print("Datasets are allready prepared!:)")
            return
    except Exception as ex:
        print("Generating new datasets...")
        pass

    # drop any rows which have missing reviews, class or a class which is not in our class dict

    nrows = df.shape[0]
    df["review"].replace("", np.nan, inplace=True)
    df.dropna(subset=["review"], inplace=True)
    df["score"].replace("", np.nan, inplace=True)
    df.dropna(subset=["score"], inplace=True)
    print("Nr. rows dropped because containing NaN:", nrows - df.shape[0])

    # nrows = df.shape[0]
    # df = df[df['score'].isin()]

    # print('Nr. rows dropped because score label was incorrect:',
    #      nrows - df.shape[0])

    # One hot encode score labels
    labelencoder = LabelEncoder()
    df["score"] = labelencoder.fit_transform(df["score"])

    # Run this if we want to see some info on string lengths
    # check_string_lengths(df)
    split1, split2 = check_splits(splits)

    # split train dataset into train, validation and test sets
    train_text, test_text, train_labels, test_labels = train_test_split(
        df["review"],
        df["score"],
        random_state=seed,
        test_size=split1,
        stratify=df["score"],
    )

    train_text, val_text, train_labels, val_labels = train_test_split(
        train_text,
        train_labels,
        random_state=seed,
        test_size=split2,
        stratify=train_labels,
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    tokens_train = tokenizer(
        train_text.tolist(), max_length=max_length, padding=True, truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer(
        val_text.tolist(), max_length=max_length, padding=True, truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer(
        test_text.tolist(), max_length=max_length, padding=True, truncation=True
    )

    train_data = TensorDataset(
        torch.tensor(tokens_train["input_ids"]),
        torch.tensor(tokens_train["attention_mask"]),
        torch.tensor(train_labels.tolist()),
    )
    val_data = TensorDataset(
        torch.tensor(tokens_val["input_ids"]),
        torch.tensor(tokens_val["attention_mask"]),
        torch.tensor(val_labels.tolist()),
    )
    test_data = TensorDataset(
        torch.tensor(tokens_test["input_ids"]),
        torch.tensor(tokens_test["attention_mask"]),
        torch.tensor(test_labels.tolist()),
    )

    pickle_TensorDataset(train_data, dataset_name, "train", wd)
    pickle_TensorDataset(val_data, dataset_name, "validate", wd)
    pickle_TensorDataset(test_data, dataset_name, "test", wd)

    f = gzip.open(
        wd
        + "/data/SA_amazon_data/processed/"
        + str(dataset_name)
        + "/used_datasets.pklz",
        "wb",
    )
    pickle.dump(datasets, f)
    f.close()


def check_and_load_raw(file):

    try:
        df = pd.read_csv(
            file,
            error_bad_lines=False,
            delimiter=";",
            quoting=csv.QUOTE_NONE,
            names=["review", "score", "category"],
        )
        return df
    except Exception as ex:
        if type(ex) == FileNotFoundError:
            raise FileNotFoundError(
                "The /data/SA_amazon_data/raw/"
                + str(file)
                + "file does not exist. Fetch the dataset before contiunuing"
            )


def check_string_lengths(df):
    # get length of all the messages in the train set
    seq_len = [len(i.split()) for i in df["review"]]

    plot = pd.Series(seq_len).hist(bins=30)
    plot.figure.savefig("./reports/figures/hist_of_string_lengths.pdf")
    print("Mean seq-len:", np.mean(seq_len))
    print("Median seq-len:", np.median(seq_len))


def check_splits(splits):
    assert int(np.sum(splits)) == 1, "Splits must sum to one"
    first = splits[2]
    second = splits[1] / (1 - splits[2])
    return first, second


def pickle_TensorDataset(dataset, experiment_name, dataset_name, wd):
    check_and_create_data_subfolders(
        wd + "/data/SA_amazon_data/processed/", subfolders=[str(experiment_name)]
    )
    f = gzip.open(
        f"{wd}/data/SA_amazon_data/processed/{experiment_name}/{dataset_name}.pklz",
        "wb",
    )
    pickle.dump(dataset, f)
    f.close()


def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


### CODE STOLEN FROM https://medium.com/analytics-vidhya/data-cleaning-in-natural-language-processing-1f77ec1f6406 ####
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emojis(text):
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", text)


def remove_punct(text):
    translation_dict = dict(zip(string.punctuation, [" "] * len(string.punctuation)))
    table = str.maketrans(translation_dict)
    return text.translate(table)
