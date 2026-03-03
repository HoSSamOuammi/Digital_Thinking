# Interactive Generative Studio

Interactive Generative Studio is a Flask-based creative platform built to match the `Term Project.pdf` brief for Digital Creativity using Python.

It combines:

- Three distinct generative art systems
- Data-driven artistic visualization
- Image manipulation tools
- Optional audio manipulation tools
- Gallery export and download workflow
- A K-means color extraction bonus module

## What Was Improved

Compared to the earlier version, this iteration adds:

- A stronger editorial interface and clearer site structure
- Three clearly differentiated generative art families:
  - `constellation`
  - `mosaic`
  - `kinetic`
- Live preview and click-or-drag overlay drawing in the generative module
- Seed-based generation for repeatable outputs
- More image effects:
  - neon
  - glitch
  - watercolor
  - contour
- More data-art controls:
  - focus column
  - colormap
  - smoothing window
  - radial bloom style
- More audio options:
  - pitch shifting
  - fade in/out
- Smoke tests for the main routes and workflows

## Project Structure

```text
interactive-generative-studio/
|-- app.py
|-- README.md
|-- REPORT.md
|-- requirements.txt
|-- modules/
|   |-- __init__.py
|   |-- generative_art.py
|   |-- data_visualization.py
|   |-- image_processing.py
|   `-- audio_processing.py
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- gallery.html
|   |-- generative.html
|   |-- data_art.html
|   `-- media_tools.html
|-- static/
|   |-- css/
|   |   `-- style.css
|   |-- generated/
|   `-- uploads/
`-- tests/
    `-- test_app.py
```

## Features

### 1. Generative Art Collection

- Loop- and randomness-based composition
- OOP shape system with `Shape`, `Circle`, `Square`, and `Triangle`
- Three visual series with different logic and appearance
- Interactivity through:
  - palette changes
  - shape count
  - density
  - size variation
  - background mode
  - seeded randomness
  - animation trails
  - live drawing overlay by click/drag

### 2. Data-Driven Creative Visualization

- CSV upload or synthetic dataset fallback
- Pandas preprocessing and cleanup
- Artistic render modes:
  - landscape
  - heatmap
  - gradient
  - radial
  - composite poster (`all`)
- Adjustable focus column, colormap, and smoothing

### 3. Image and Audio Manipulation

Image tools:

- grayscale
- sepia
- invert
- blur
- edge detection
- pixelation
- mirror
- rotation
- neon
- glitch
- watercolor
- contour
- K-means palette extraction

Audio tools:

- reverse
- speed change
- echo
- merge
- pitch shift
- fade in/out

## Installation

### 1. Create and activate a virtual environment

PowerShell:

```powershell
cd interactive-generative-studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Optional audio support

`pydub` requires `ffmpeg` for most real-world audio formats.

- Install `ffmpeg`
- Make sure it is available in your system `PATH`

## Run the App

```powershell
python app.py
```

Open:

- `http://127.0.0.1:5000/`

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Technical Notes

- The app is fully modularized by domain
- Generated media is written to `static/generated`
- Uploaded files are written to `static/uploads`
- The home, gallery, generative, data-art, and media-tools routes are all server-rendered with Jinja2
- The generative page uses lightweight browser-side preview logic for faster feedback before server export
- The optional machine-learning bonus is implemented with K-means color extraction

## Deliverables Alignment

This project now covers the main PDF requirements:

- Flask app with home, gallery, module pages, and file-upload workflows
- Generative art using loops, conditionals, randomness, and OOP
- Data processing with Pandas and artistic visualization with Matplotlib
- Image or audio manipulation tools
- Interactivity through control panels, gallery browsing, live preview, and drawing overlay
- Downloadable visual and audio outputs

## Report

See `REPORT.md` for a concise report covering:

- artistic direction
- implemented modules
- tools used
- pipeline
- challenges and solutions
