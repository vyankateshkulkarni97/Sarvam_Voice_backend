# Sarvam Voice
<img width="1908" height="891" alt="image" src="https://github.com/user-attachments/assets/cb737678-c4b9-41d7-9db1-3cd353eb77b8" />

<img width="1897" height="890" alt="image" src="https://github.com/user-attachments/assets/33ee5691-e8c5-47da-bfe8-bbfe9492e40d" />



# Audio Attribute Inference
<img width="1918" height="896" alt="image" src="https://github.com/user-attachments/assets/c4e1ffdb-bdde-4c77-83fe-379dff2ddb17" />



Text to Audio 
<img width="1916" height="892" alt="image" src="https://github.com/user-attachments/assets/2bdef34f-53bb-46d1-bd5b-19f3b836a950" />


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

