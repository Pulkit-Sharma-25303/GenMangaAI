# GenManga AI: Manga Generator

GenManga AI is an intuitive web app that lets anyone generate their own manga panels using the power of artificial intelligence! Born from a love of manga and a fascination with AI, this project helps you bring original stories and characters to life—no drawing skills required.

## 🚀 Features

- **Text-to-Manga Panel Generation**: Describe a scene, and GenManga AI draws it for you.
- **Character Consistency**: Keep your main character’s look stable across panels.
- **Speech Bubbles**: Add dialogue wherever you want.
- **Manga Page Collage**: Arrange panels into full manga pages, export and share.
- **Easy-to-Use Interface**: Powered by Streamlit, ready to use in your browser.


## 🖼️ Demo

> _![WhatsApp Image 2025-07-18 at 13 16 37_beb87428](https://github.com/user-attachments/assets/f98cdddc-20c1-42ef-a459-857e82eb7ff4)
_

## 🛠️ Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/genmanga-ai.git
cd genmanga-ai
```


### 2. Set up and activate a virtual environment _(recommended)_

```bash
python -m venv genmanga_env
source genmanga_env/bin/activate       # On Windows: .\genmanga_env\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Add or Download Your Model

- Download or fine-tune your Stable Diffusion model (see instructions below).
- Place the resulting `genmanga_model` folder in your project directory.


### 5. Run GenManga AI!

```bash
streamlit run manga_generator_app.py
```


## ⚡ Making Model Loading Fast (with Google Colab)

1. **Run this in Google Colab to prepare your model:**

```python
!pip install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionPipeline
import torch
from google.colab import drive
drive.mount('/content/drive')
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.save_pretrained("/content/drive/MyDrive/genmanga_model")
```

2. **Download the `genmanga_model` folder from your Google Drive.**
3. **Copy it into your project directory (should look like below):**

```
genmanga-ai/
├── genmanga_model/
│   ├── model_index.json
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── manga_generator_app.py
├── requirements.txt
└── README.md
```


## 🧩 Dependencies

Main requirements (see `requirements.txt`):

- Python 3.8+
- streamlit
- diffusers
- torch
- transformers
- pillow
- safetensors
- accelerate


## ✨ Usage Overview

1. Describe your character, select style \& expression, and enter scene prompts.
2. Click **Generate Panel** to create manga art.
3. Add speech bubbles for dialogue.
4. Arrange panels with the collage builder.
5. Download and share your manga!

## 🐛 Troubleshooting

- **Slow model loading?**
Use the Google Colab steps above and store models on a fast SSD.
- **Not enough RAM?**
Lower the resolution or number of inference steps. For best results, use 16GB+ RAM (32GB for large models).
- **File not found/model errors?**
Ensure the `genmanga_model` folder is present and contains all the required files.


## 👤 Author

Created by Pulkit Sharma – manga lover, AI enthusiast, and builder at heart.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Stable Diffusion (CompVis)](https://huggingface.co/CompVis/stable-diffusion)
- [Diffusers (Hugging Face)](https://github.com/huggingface/diffusers)
- [Streamlit](https://streamlit.io/)
- Manga \& Open Source AI communities worldwide!

\#Manga \#AI \#StableDiffusion \#Streamlit \#Python \#GenerativeArt \#OpenSource


