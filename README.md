# CS553 GAN (CIFAR-10)

GAN implementation with TensorFlow: generator and discriminator on CIFAR-10, image saving every 10 epochs.

## Run with TensorFlow

```powershell
cd C:\Users\Keira\Documents\genAI\CS553-GAN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python gan.py
```

Generated images are saved to `generated_images/` every 10 epochs.

## Requirements

- Python 3.9–3.11
- TensorFlow 2.15.x (see `requirements.txt`)
- matplotlib
