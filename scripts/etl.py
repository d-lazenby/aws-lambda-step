import requests
import os
import tarfile
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Union, List, Dict


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


def save_images(dataset: Dict[str, Any], dataframe: pd.DataFrame, path: str) -> None:
    # Grabs image data
    for row in dataframe['row'].values:
        img = dataset[b'data'][row]

        # Stacks the data into suitable format
        target = np.dstack((
            img[0:1024].reshape(32, 32),
            img[1024:2048].reshape(32, 32),
            img[2048:].reshape(32, 32)
        ))

        # Saves the image
        plt.imsave(path + "/" + dataframe['filenames'][row], target)


def to_metadata_file(dataframe: pd.DataFrame, prefix: str) -> None:
    dataframe["s3_path"] = dataframe["filenames"]
    # Relabels bicycle as 0 and motorbike as 1 for MXNet
    dataframe["labels"] = np.where(dataframe["labels"] == 8, 0, 1)
    # Output result to CSV
    dataframe[["row", "labels", "s3_path"]].to_csv(
        f"./metadata/{prefix}.lst", sep="\t", index=False, header=False
    )


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

    os.makedirs('./metadata')
    ml_split = ["train", "test"]
    for split in ml_split:
        with open(f"./cifar-100-python/{split}", "rb") as f:
            dataset = pickle.load(f, encoding='bytes')

        # Make training directory for images
        os.makedirs(f"./{split}")
        # Makes dataframe and saves images
        df = construct_dataframe(dataset)
        save_images(dataset, df, f"./{split}")
        # Makes metadata file for training with MXNet
        to_metadata_file(df, split)


if __name__ == "__main__":
    main()
