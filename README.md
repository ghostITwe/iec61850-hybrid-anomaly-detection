# iec61850-anomaly-detection

Гибридное обнаружение киберфизических аномалий в цифровых подстанциях методами обучения без учителя:
Isolation Forest, Autoencoder и объединение их оценок на датасете TU Delft CPPS (v2, 2026-10-19).
Протокол: `docs/experiment_protocol.md`. Сырые данные в Git не хранятся.

## Требования
Python ≥ 3.10 (проверено на 3.12), Wireshark/**tshark ≥ 4.x** в PATH, PyTorch (с CUDA) — по команде с pytorch.org.

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # команду сверить на pytorch.org
pip install -e .
$env:DATASET_ROOT = "D:\data\2026_10_19_CPPS_Dataset_CyberSec_DigitalSubstation_TUD"
```
Если tshark не в PATH — укажите `extract.tshark` в `config/experiment_v1.yaml`.

## Порядок запуска
```powershell
python scripts/00_inventory.py --sha256   # reports/inventory.csv, reports/catalog.csv  <- ПРОВЕРИТЬ catalog.csv глазами
python scripts/01_extract.py              # pcapng -> data/interim/frames/*.parquet (долго, параллельно)
python scripts/01b_eda.py                 # reports/eda_network.csv, eda_physical.csv, figures/signals_*.png
python scripts/02_prepare.py              # окна + метки + split по прогонам
python scripts/03_train_if.py
python scripts/04_train_ae.py
python scripts/05_hybrid.py               # reports/scores_all_methods.csv
python scripts/06_evaluate.py             # metrics.csv, metrics_by_scenario.csv, bootstrap_ap.csv, pr_curves.png
python -m pytest -q tests                 # тесты на утечки
```
Быстрая проверка конвейера без реальных данных: `python tests/make_synthetic.py <папка>` и тот же порядок команд.
