# AI Tools Assignment — Mastering the AI Toolkit

Contents:
- `AIToolsAssignment.ipynb` — Notebook with theory, code cells for Iris, MNIST, spaCy, ethics, and debugging notes.
- `iris_classification.py` — Standalone script performing Iris Decision Tree training/evaluation.
- `mnist_cnn.py` — Standalone TensorFlow Keras CNN for MNIST (saves model).
- `spacy_ner_sentiment.py` — spaCy demonstration of NER and simple rule-based sentiment.
- `streamlit_app.py` — Small Streamlit app skeleton to load `mnist_cnn.h5` and predict images (bonus).
- `buggy_tensorflow_fixed.py` — Fixed TensorFlow script addressing common errors.
- `requirements.txt` — Python dependencies.

How to run (recommended: Google Colab for MNIST training):

1. Create environment locally (optional):
```
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
2. Open `AIToolsAssignment.ipynb` in Jupyter or upload to Colab. In Colab, enable Runtime -> Change runtime type -> GPU, then run the MNIST cell for faster training.

3. Run scripts locally (lower compute):
```
python iris_classification.py
python spacy_ner_sentiment.py
```

4. To train MNIST locally:
```
python mnist_cnn.py
```

5. To try the Streamlit app (after training and saving `mnist_cnn.h5`):
```
streamlit run streamlit_app.py
```

Notes:
- For spaCy, install the English model if needed:
```
python -m spacy download en_core_web_sm
```
- The notebook and scripts are commented to satisfy the assignment's submission requirements. Capture screenshots of outputs and include them in your PDF report.

Local environment note (important):

- Check your Python version before installing heavy packages like TensorFlow:
```
python --version
```
- TensorFlow often has OS / Python-version constraints. If you see errors such as "ModuleNotFoundError: No module named 'tensorflow'" or pip install failures on Windows, two recommended options are:
	1. Use Google Colab (recommended for MNIST training): upload `AIToolsAssignment.ipynb` to Colab and enable a GPU runtime (Runtime -> Change runtime type -> GPU). This avoids local installation issues.
	2. Create a dedicated virtual environment with a supported Python version (commonly 3.8–3.11) and install dependencies.

PowerShell commands to create a venv and install requirements (example):
```
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install --upgrade pip; pip install -r requirements.txt
# If spaCy model is still needed:
python -m spacy download en_core_web_sm
```

If you'd like, I can try installing the dependencies and re-run the smoke tests here, or prepare a Colab-ready notebook export. Tell me which you'd prefer.

PowerShell note: If you see an error about running scripts being disabled (ExecutionPolicy), run PowerShell as Administrator and enable scripts temporarily with:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
This sets the policy for the current PowerShell session only. Alternatively, bypass activation by calling the venv python directly, e.g.:
```
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Conda alternative (recommended for ease with TensorFlow on Windows):
```
conda env create -f environment.yml
conda activate aitools
python -m pip install -r requirements.txt
```
