# BME 1312 Lab0

## Program requirements
If you are familiar with jupyter notebook and conda. You may configure the environment by yourself.

Necessary packages:

- jupyter
- scikit-image
- numPy
- pytorch
- torchvision
- matplotlib
- tqdm

## Guidence of environment setup.
### Step-1 `conda` installation
You may install `conda` on your computer as python packages manager. Either [Anaconda]( https://www.anaconda.com/ ) or [Miniconda]( https://docs.conda.io/en/latest/miniconda.html ) is okay.

### Step-2 packages installtion
Create the lab environment
```plaintext
conda create -n bme1312lab0 python=3.9
conda activate bme1312lab0
```

Install packages except PyTorch
```plaintext
conda install jupyter numpy matplotlib tqdm
```

Install PyTorch: we recommend to install PyTorch under the guidence of [PyTorch GET STARTED]( https://pytorch.org/get-started/locally/) to adapt your platform.

### Step-3 setup jupyter and do experiment
Run the following command and go for the site `http://localhost:8888` in your browser.

```plaintext
cd /path/to/lab/directory
jupyter notebook
```

`lab0.ipynb` is the content of this lab.

