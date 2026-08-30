# Voice AI Audio Attribute Service

## 1. Requirements

* Python 3.11
* FastAPI
* Uvicorn
* NumPy
* SoundFile
* PyTorch
* Transformers
* Requests
* WebSockets
* FFmpeg
* Docker Desktop
* WSL2 on Windows

---

## 2. Project Structure

```text
voice_AI_agents/
│
│
└── Voice_app/
    ├── README.md
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py
    │
    ├── Voice_api/
    │   ├── Voice_audio.py
    │   ├── text_to_audio.py
    │
    │
    └── Voice_services/
        ├── audio_processor.py
        └── attribute_model.py
        ├── audio_session.py
        └── audio_decoder.py
```

---

# 3. Local Setup

### Create virtual environment

```cmd
cd /d D:\Python\voice_AI_agents

python -m venv venv
```

### Activate

```cmd
venv\Scripts\activate
```

### Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Install dependencies

```cmd
pip install fastapi
pip install "uvicorn[standard]"
pip install python-multipart
pip install numpy
pip install soundfile
pip install pydantic-settings
pip install torch
pip install transformers
pip install requests
pip install websockets
```
# 4. FFmpeg Setup

### Windows

```cmd
winget install Gyan.FFmpeg
```

Verify:

```cmd
ffmpeg -version
```

### Ubuntu/Linux

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

---

# 5. Run Application

From project root:

```cmd
cd /d D:\Python\voice_AI_agents
venv\Scripts\activate
```

Start:

```cmd
uvicorn Voice_app.main:Voice_app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 6. API Endpoints

```text
GET  /health
GET  /ready

POST /audio/session
POST /audio/predict
OPTIONS /text-to-audio

