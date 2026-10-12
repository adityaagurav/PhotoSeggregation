# AI-Based Photo Segregation Using Merge Sort

An AI-powered web application that automatically organizes event photographs
person-wise.

## Overview

A user uploads a collection of event photographs containing multiple people.
The system:

1. Detects faces in every uploaded photograph.
2. Generates facial embeddings using a pretrained face model.
3. Compares embeddings and clusters similar faces together.
4. Associates matching photographs with the same temporary **Person ID**
   (the application initially knows no names).
5. Displays representative faces for each discovered person and asks the user
   to enter their names.
6. Creates person-wise photo collections dynamically — no names or people are
   ever hardcoded.

## DAA component

After AI processing, the system produces person-photo records such as
`(Person_ID, Photo_ID)`. These records are sorted by **Person ID** using our
own **Merge Sort** implementation (`app/algorithms/merge_sort.py`) before they
are grouped into person-wise collections. Python's built-in sorting is not used
to replace this implementation for that step.

## Planned technology

- **Backend / AI:** Python
- **Web interface:** Streamlit (planned)
- **Face model:** a pretrained face detection/embedding model, to be selected
  after experimentation in `ml/`

## Project layout

- `app/` — production application code
  - `ai/` — face detection, embedding and clustering integration
  - `algorithms/` — our DAA components, including custom Merge Sort
  - `processing/` — pipeline, person-photo records, person management
  - `web/` — web interface
  - `utils/` — shared helpers
- `ml/` — model research: datasets, models, training, inference, experiments,
  embeddings (kept separate from the application)
- `data/` — runtime application data (uploads, face crops, outputs, temp)
- `tests/` — tests for components and the full pipeline
- `docs/` — project documentation, diagrams, screenshots and results

## Status

Initial project scaffolding only. Face recognition, clustering, Merge Sort and
the web interface are not implemented yet.
