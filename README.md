# Proyecto MLOps con PyTorch

## Estructura
- `model/train.py`: Entrena un modelo simple en MNIST (5 epochs).
- `validate/validate.py`: Evalúa precisión en test set.
- `run_every_10h.sh`: Ejecuta validación local cada 10 h.
- `.github/workflows/ci.yml`: CI/CD en GitHub Actions (cada push y cada 10 h).

## Uso local

```bash
git clone <TU-REPO-URL>
cd mlops-pytorch-setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python model/train.py
python validate/validate.py
./run_every_10h.sh
# trigger
