### reranking

#### Setup

To setup the environment, you can install all the required dependencies listed in the `requirements.txt` file by running the following command:

```
pip install -r requirements.txt
```

Specifically, we have tested the project with `torch` version 2.3, if you encounter any compilation issues while using your own environment, please strictly follow our dependencies.

#### Dataset

Extracted image features (https://drive.google.com/drive/folders/1u3ZN1ItC__IqIk-_hn8apkXCn7MnpoYs?usp=drive_link), put in `revisitop/dataset/roxford5k/features`.

#### CUDA code compilation

Navigate to the folder `reranker/matrix_utils` and `reranker/sparse_divergence`, run the following command in the terminal:

```
python setup.py build_ext --inplace
```

If the compilation completes without errors, you will see the complied files like `sparse_divergence.cpython-312-x86_64-linux-gnu.so` in the corresponding folders. These files can be imported as libraries in the project.
