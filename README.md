# **Deep scSTAR documentation**

Next, we attempted to effectively integrate the two samples based on existing annotations, designing a semi-supervised sample integration algorithm based on deep learning—**BIPACT** (Batch Integration via Precision Annotation and Consistency Testing). This scheme, based on existing cell type annotation results, through DsAno and MetaNeighbor consistency verification, allowed us to accurately identify reliable cell type annotations. Based on this, we identified highly credible shared cell types between batches.
![Image text](https://github.com/Hao-Zou-lab/BIPACT/blob/main/BIPACT.png)

## **Dependency**

```shell
    python >= 3.10.0
    pytorch >= 2.2.1
    scikit-learn >= 1.2.2
    numpy >= 1.25.2
    pandas >= 1.5.3
    seaborn >- 0.13.1
    matplotlib >= 3.5.2
    imbalanced-learn >= 0.10.1
    umap-learn >= 0.5.5
```

We recommend three primary platforms for installing and running BIPACT:

1. **Google Colab**: Given that BIPACT was developed on Colab, running it on this platform allows you to bypass the environment setup process entirely. This method is highly recommended for users looking for quick deployment and execution without the need to manage dependencies and environments locally.
2. **Linux**: For users preferring local installation on Linux, it's advised to install dependencies in a sub-environment using miniconda3. 
3. **Windows via Anaconda Prompt**: For Windows users, utilizing Anaconda Prompt offers a straightforward method to run BIPACT. After setting up the conda environment with all necessary dependencies, you can execute the BIPACT scripts directly in the prompt.



## **Running Deep scSTAR**

```shell
!python BIPACT_main.py
```




## **Adjustable Model Parameters**

The following parameters can be adjusted within `BIPACT_main.py`:

`batch_size` - The number of samples per mini-batch. Select a number that is approximately 5% of the total number of samples.

`num_epochs` - The number of iterations for training the model. You can choose between 200-400. A higher number may lead to overfitting of the model.

`gamma` - If the results appear distorted, you can appropriately increase the gamma value. A higher gamma value can constrain the model to retain more of the original information.

