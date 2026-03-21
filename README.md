# 📄 Contract Analysis System

A lightweight, privacy-first contract analysis tool using **spaCy NER** and **Streamlit**.  
No internet required. No transformer models. Runs fast on modest hardware.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **NLP Engine** | spaCy `en_core_web_sm` — small, fast, offline |
| **Entities** | Person, Organization, Date, Money, Location |
| **Rule-Based** | Effective Date, Contract Value, Contract Term (regex) |
| **Highlights** | displaCy in-text entity visualization |
| **Download** | Extracted entities as JSON or plain TXT |
| **File Input** | `.txt` and `.pdf` upload support |

---

## 🔧 Setup

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit spacy pymupdf
```

### 2 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3 — Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## 🗂️ Project Structure

```
contract_analysis/
├── app.py           # Main Streamlit application (single file)
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## 🧠 How It Works

```
User Input (text / file)
        │
        ▼
  spaCy en_core_web_sm
  (NER pipeline)
        │
        ▼
  Entity Grouping     +   Regex Rule Engine
  PERSON / ORG            Effective Date
  DATE / TIME             Contract Value
  MONEY                   Contract Term
  GPE / LOC / FAC
        │
        ▼
  Streamlit UI
  (Tabs · Highlights · Download)
```

### Entity label mapping

| spaCy Label | Display Category |
|-------------|-----------------|
| `PERSON`    | Names |
| `ORG`       | Names |
| `DATE`      | Dates |
| `TIME`      | Dates |
| `MONEY`     | Money |
| `GPE`       | Locations |
| `LOC`       | Locations |
| `FAC`       | Locations |

---

## 💡 Sample Contract Snippet (for testing)

```
This Service Agreement ("Agreement") is entered into as of January 15, 2024,
between Acme Corporation, a Delaware corporation ("Company"), headquartered
in New York, and Jane Doe ("Contractor"), residing at 456 Oak Avenue, Boston, MA.

The total contract value is $75,000 per annum. The initial term of this
agreement is 2 years, commencing on the Effective Date.

Payment shall be made by the Company to the Contractor on a monthly basis
in equal installments of $6,250.
```

Paste this into the app to see entity extraction in action.

---

## ⚙️ System Requirements

- Python **3.9+**
- RAM: **2 GB minimum** (comfortably runs on 8 GB)
- CPU: Any modern i5 or equivalent
- No GPU required
- No API keys or internet connection needed

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `spacy` | NLP / NER engine |
| `en_core_web_sm` | Pre-trained English language model |
| `pymupdf` | PDF text extraction (optional) |
