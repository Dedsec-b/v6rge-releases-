# V6rge

<p align="center">
  <img src="app/icon.png" alt="V6rge Logo" width="128">
</p>

<h3 align="center">The Local-First AI Suite</h3>

<p align="center">
  <b>Run AI models 100% on your GPU. No cloud. No API keys. No limits.</b>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#development">Development</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Chat** | Qwen 2.5 (3B to 72B) with vision support |
| 🖼️ **Image Generation** | FLUX.1 (Schnell & Dev) + Qwen-Image |
| 🎵 **Music Generation** | MusicGen (Medium & Large) |
| 🗣️ **Text-to-Speech** | Chatterbox Turbo TTS with emotion tags |
| 🎬 **Video Generation** | HunyuanVideo |
| 📦 **3D Model Generation** | Hunyuan3D 2.5 |
| ✂️ **Background Removal** | U2Net |
| 📈 **Image Upscaling** | RealESRGAN 4x |
| 🎤 **Vocal Separation** | RoFormer |
| ⚡ **God Mode (Agent)** | AI controls your computer via terminal |
| 🌍 **Hotspot Mode** | Share your AI with other devices (Cloudflare Tunnel) |

## 🚀 Installation

### For Users (Pre-built)
1. Download the latest installer from [Releases](https://github.com/YourUsername/V6rge/releases).
2. Run the installer. The backend will extract on first launch.
3. Download models from the Dashboard tab.

### For Developers
See [Development](#development) below.

## 🛠️ Development

### Prerequisites
- **Node.js** 18+
- **Python** 3.10+
- **NVIDIA GPU** (Recommended, 8GB+ VRAM)
- **CUDA Toolkit** 12.1+ (for GPU acceleration)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/V6rge.git
cd V6rge/desktop-app

# 2. Install frontend dependencies
npm install

# 3. Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# 4. Create your .env file
cp .env.example .env
# Edit .env and add your Hugging Face token

# 5. Run in development mode
npm start
```

### Project Structure

```
desktop-app/
├── app/                  # Frontend (HTML/CSS/JS)
│   ├── js/modules/       # Core JS modules (chat, api, tools)
│   ├── index.html        # Main UI
│   └── styles.css        # Styling
├── backend/              # Python Flask server
│   ├── services/         # AI service modules
│   ├── config.py         # Configuration (reads from .env)
│   └── server_modular.py # Main API server
├── main.js               # Electron main process
├── preload.js            # Electron preload script
└── package.json
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Fork** this repository.
2. Create a **feature branch**: `git checkout -b feat/my-feature`
3. **Commit** your changes: `git commit -m 'Add some feature'`
4. **Push** to your branch: `git push origin feat/my-feature`
5. Open a **Pull Request**.

### Good First Issues
- Look for issues tagged `help wanted` or `good first issue`.
- We especially need help with:
  - AMD GPU support (ROCm)
  - Linux packaging
  - Performance optimizations

### Code Style
- **Python**: Follow PEP8. Use `black` for formatting.
- **JavaScript**: Use ES6+ syntax. No semicolons.

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by the V6rge community.
</p>
