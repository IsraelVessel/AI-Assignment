# Colab instructions — Run MNIST with GPU

This file contains a small snippet you can paste into a Colab notebook to mount Google Drive, install needed packages, and run the MNIST training cell with GPU.

1) Open https://colab.research.google.com and create a new notebook.
2) In the first cell, paste the following (it installs TensorFlow and mounts your Drive if you want to save models):

```python
# Install a compatible TF version in Colab (Colab already includes TF; these are optional)
!pip install -q tensorflow

# Mount Google Drive to save model/artifacts (optional)
from google.colab import drive
drive.mount('/content/drive')

# Example: save model to /content/drive/MyDrive/mnist_model
``` 

3) Copy the MNIST CNN training cell from `AIToolsAssignment.ipynb` into a code cell and run. Use Runtime -> Change runtime type -> GPU for faster training.

4) After training, save the model to Drive:

```python
model.save('/content/drive/MyDrive/mnist_cnn.h5')
```

Notes:
- Colab is recommended if your local machine doesn't have a compatible TensorFlow setup.
- If you want, I can produce a direct Colab link (notebook upload) you can click to open the prepared notebook there.