# Indian_Food_classification

IndianFoodAI is a YOLO-based computer vision project for detecting Indian food items from images and estimating meal calories in a Streamlit web interface.

## Highlights

- End-to-end pipeline: dataset -> training -> inference UI
- Streamlit app with confidence and IOU controls
- YOLO training script with GPU auto-detection
- Portable setup (relative dataset paths, no machine-specific assumptions)
- Ready structure for adding more food classes later

## Demo Screenshots

Store screenshots in `assets/screenshots/` and embed them in this README:

```markdown
![Detection Result 1](assets/screenshots/result-1.png)
![Detection Result 2](assets/screenshots/result-2.png)
```

## Tech Stack

- Python
- Ultralytics YOLO
- Streamlit
- OpenCV
- Pillow

## Repository Structure

- `app.py` - Streamlit inference and calorie summary dashboard
- `train.py` - training entry point with configurable CLI options
- `dataset/` - YOLO dataset files and labels
- `dataset/data.yaml` - dataset config used by training
- `assets/screenshots/` - README images/results
- `requirements.txt` - runtime dependencies

## Quick Start

### 1) Create virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3) Start the app

```bash
streamlit run app.py
```

By default, the app loads weights from `best.pt`.  
You can override this with `NUTRISCAN_MODEL`.

Linux/macOS:

```bash
export NUTRISCAN_MODEL="/absolute/path/to/best.pt"
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:NUTRISCAN_MODEL="C:\path\to\best.pt"
streamlit run app.py
```

## Training

Default:

```bash
python train.py
```

Custom example:

```bash
python train.py --epochs 60 --batch 8 --imgsz 640 --name exp_food_v2
```

The script uses GPU automatically when available; otherwise it falls back to CPU.

## Current Classes

`Bhatura`, `BhindiMasala`, `Biryani`, `Chole`, `ShahiPaneer`, `chicken`, `dal`, `dhokla`, `gulab_jamun`, `idli`, `jalebi`, `modak`, `palak_paneer`, `poha`, `rice`, `roti`, `samosa`

## Adding More Classes (Future Expansion)

To scale this project with new food categories:

1. Collect and annotate images for the new classes in YOLO format.
2. Add class names to `dataset/data.yaml` under `names`.
3. Update `nc` in `dataset/data.yaml` to match the total class count.
4. Retrain with `python train.py` and export the best weights.
5. Update calorie mapping in `app.py` (`CAL_VALS`) for the new labels.
6. Replace/point `best.pt` to the new trained model.

Recommended practices:

- Keep class names consistent across dataset labels and `CAL_VALS`.
- Add at least a few hundred samples per new class for stable performance.
- Track experiments using new run names (for example `--name exp_v3_more_classes`).

## Dataset

- Dataset format: YOLO
- Config file: `dataset/data.yaml`
- Paths are relative for portability
- Source metadata is documented in `dataset/README.dataset.txt` and `dataset/README.roboflow.txt`

## Portability Notes

- The repo avoids committing local virtual environments and training outputs.
- Use `.venv` on every machine instead of sharing environment folders.
- Keep model weights (`.pt`) outside git or use release assets if needed.

## GitHub Publishing

If your local commit is ready, run:

```bash
git remote add origin https://github.com/itsAtharvv/Indian_Food_classification.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/itsAtharvv/Indian_Food_classification.git
git push -u origin main
```

## License

Dataset references indicate CC BY 4.0 for the exported dataset metadata.  
Add a dedicated `LICENSE` file for repository code if you plan public reuse.
<<<<<<< HEAD
=======

>>>>>>> c57ab01 (Polish README with full project documentation.)
