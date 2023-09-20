import requests
import os
import tarfile
import pickle
import pandas as pd
from typing import Dict


def extract_cifar_data(url: str, filename: str = "cifar.tar.gz") -> requests.Response:
    """
    Downloads CIFAR data from the specified URL and saves it to a file.

    Args:
        url (str): The URL to download the CIFAR data from.
        filename (str, optional): The name of the file to save the downloaded data.
            Defaults to "cifar.tar.gz".
    """

    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return response


def construct_dataframe(dataset: Dict[bytes, any]) -> pd.DataFrame:
    # Construct dataframe
    df = pd.DataFrame({
        "filenames": dataset[b'filenames'],
        "labels": dataset[b'fine_labels'],
        "row": range(len(dataset[b'filenames']))
    })

    # Drop rows where label is not 8 (bicycle) or 48 (motorbike)
    df = df[df['labels'].isin([8, 48])].copy()
    # Decode filenames so that they are regular strings
    df['filenames'] = df['filenames'].str.decode("utf-8")

    return df


def main():
    # Checks if file exists in local directory and if not downloads it
    filename = "cifar.tar.gz"
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    if os.path.exists(filename):
        print(f"CIFAR data already downloaded to {filename}")
    else:
        print("Downloading CIFAR data...")
        extract_cifar_data(cifar_url)
        if os.path.exists(filename):
            print(f"CIFAR data successfully downloaded to {filename}")
        # Extract all files to local directories
        print("Extracting data...")
        with tarfile.open("cifar.tar.gz", "r:gz") as tar:
            tar.extractall()
            print("Data extracted successfully")

    with open("./cifar-100-python/meta", "rb") as f:
        dataset_meta = pickle.load(f, encoding='bytes')

    with open("./cifar-100-python/test", "rb") as f:
        dataset_test = pickle.load(f, encoding='bytes')

    with open("./cifar-100-python/train", "rb") as f:
        dataset_train = pickle.load(f, encoding='bytes')

    df_train = construct_dataframe(dataset_train)

    print(df_train.head())



if __name__ == "__main__":
    main()
