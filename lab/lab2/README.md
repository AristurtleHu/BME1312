# BME 1312 Lab2

## Program requirements
If you are familiar with jupyter notebook and conda. You may configure the environment by yourself.

Recommended python version: `3.8`

Necessary packages:

- jupyter
- scikit-image
- scikit-learn
- numPy
- pytorch
- torchvision
- matplotlib
- tqdm

## Guidence of environment setup.

If you have created the environment successfully in Lab1, you only need to install the scikit-learn package, using the command "conda install scikit-learn".

Otherwise, you need follow the steps below:
### Step-1 `conda` installation
You may install `conda` on your computer as python packages manager. Either [Anaconda]( https://www.anaconda.com/ ) or [Miniconda]( https://docs.conda.io/en/latest/miniconda.html ) is okay.

### Step-2 packages installtion
Create the lab environment
```plaintext
conda create -n bme1312lab2 python=3.8
conda activate bme1312lab2
```

Install packages except PyTorch
```plaintext
conda install jupyter scikit-image scikit-learn numpy matplotlib tqdm
```

Install PyTorch: we recommend to install PyTorch under the guidence of [PyTorch GET STARTED]( https://pytorch.org/get-started/locally/) to adapt your platform.

### Step-3 setup jupyter and do experiment
Run the following command and go for the site `http://localhost:8888` in your browser.

```plaintext
cd /path/to/lab/directory
jupyter notebook
```

`lab2.ipynb` is the content of this lab.