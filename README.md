# Multi X-Y DAE Dashboard

A full-stack dashboard for **Multi-Input, Multi-Output modeling** using a **Denoising AutoEncoder (DAE)**.

---

## What does this project do?

You upload a dataset (CSV/Excel), pick which columns are inputs (**X**) and which are outputs (**Y**), and train a neural network that learns to predict Y from X — even when the input data is noisy or imperfect.

---

## Folder Structure

```
Multi X-Y/
│
├── backend/                    ← Python FastAPI server
│   ├── app/
│   │   ├── main.py             ← App entry point, registers all routes
│   │   ├── database.py         ← SQLite database connection
│   │   ├── models/
│   │   │   └── db_models.py    ← Database table definitions
│   │   ├── routers/
│   │   │   ├── data.py         ← API: upload & manage datasets
│   │   │   └── model.py        ← API: train models & predict
│   │   ├── schemas/
│   │   │   └── schemas.py      ← Data shapes for API requests/responses
│   │   └── services/
│   │       └── autoencoder.py  ← The actual DAE neural network logic
│   ├── uploads/                ← Uploaded files + saved model files
│   ├── requirements.txt        ← Python package list
│   └── .env                    ← Config (database path, upload dir)
│
├── frontend/                   ← React JS web app
│   ├── src/
│   │   ├── App.jsx             ← Main app, sets up page routing
│   │   ├── main.jsx            ← React entry point
│   │   ├── index.css           ← Global styles (Tailwind)
│   │   ├── components/
│   │   │   └── Navbar.jsx      ← Top navigation bar
│   │   ├── pages/
│   │   │   ├── Home.jsx        ← Dashboard home page
│   │   │   ├── Upload.jsx      ← Upload CSV/Excel files
│   │   │   ├── Train.jsx       ← Configure and train the DAE model
│   │   │   ├── Predict.jsx     ← Run predictions with trained model
│   │   │   └── History.jsx     ← View all past training runs
│   │   └── services/
│   │       └── api.js          ← All HTTP calls to the backend
│   ├── package.json            ← Node.js package list
│   ├── vite.config.js          ← Vite dev server config
│   └── index.html              ← HTML entry point
│
├── start_backend.bat           ← Double-click to start backend
├── start_frontend.bat          ← Double-click to start frontend
└── start_all.bat               ← Double-click to start BOTH
```

---

## Setup Instructions (Step by Step)

### Prerequisites
- Python 3.10 or newer → https://www.python.org/downloads/
- Node.js 18 or newer  → https://nodejs.org/

### First-time setup

**Option A — Easiest: just double-click `start_all.bat`**
It installs everything and starts both servers automatically.

**Option B — Manual:**

1. Open a terminal in the `backend` folder:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

2. Open another terminal in the `frontend` folder:
```bash
cd frontend
npm install
npm run dev
```

### Open the dashboard
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs

---

## How to use

1. **Upload Data** — Go to "Upload Data", upload a CSV file with numeric columns
2. **Train Model** — Go to "Train Model", pick X and Y columns, set parameters, click Train
3. **Predict** — Go to "Predict", select your trained model, enter X values, get Y predictions
4. **History** — View all past runs and their accuracy (R² score, loss)

---

## Understanding the Model Parameters

| Parameter | What it means |
|-----------|--------------|
| **Noise Factor** | How much random noise is added during training (0.0–1.0). Higher = more robust but harder to train. Start with 0.1. |
| **Epochs** | How many times the model trains over the full dataset. More = better accuracy but takes longer. Start with 100. |
| **Hidden Dim** | Size of the neural network's hidden layer. Larger = more powerful. Start with 64. |

---

## Understanding R² Score

- **R² = 1.0** → Perfect predictions
- **R² > 0.9**  → Excellent
- **R² > 0.7**  → Good
- **R² < 0.5**  → Model needs more data or tuning
