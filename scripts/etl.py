import requests
import os
import tarfile
import pickle


def extract_cifar_data(url: str, filename: str = "cifar.tar.gz") -> requests.Response:
    """
    Downloads CIFAR data from the specified URL and saves it to a file.

    Args:
        url: The URL to download the CIFAR data from.
        filename: The name of the file to save the downloaded data.
            Defaults to "cifar.tar.gz".
    
    Returns:
        The HTTP response object containing information about the download.
    """

    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return response


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

    print(type(dataset_meta), type(dataset_test), type(dataset_train))


if __name__ == "__main__":
    main()
