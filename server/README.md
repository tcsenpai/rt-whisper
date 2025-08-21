# Whisper Streaming Server

This is the **server component** that runs on the machine with the RTX 4070 Ti Super GPU.

## 📍 Where to Run This

**Run this on**: The machine with NVIDIA RTX 4070 Ti Super (or other CUDA-capable GPU)

## 🚀 Quick Start

### Method 1: Direct Python

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python3 streaming_server.py
```

3. The server will start on port 8765 and listen on all network interfaces.

### Method 2: Docker (Optional)

If you prefer containerized deployment:

```bash
# Build
docker-compose build

# Run
docker-compose up
```

## 🔧 Configuration

Edit `streaming_server.py` to change:

```python
TranscriptionServer(
    model_size="large-v3",        # Whisper model (large-v3 recommended for 4070 Ti)
    device="cuda",                 # Use "cuda" for GPU, "cpu" for CPU
    compute_type="float16",        # float16 for faster GPU inference
    vad_aggressiveness=2,          # Voice Activity Detection (0-3, higher = more aggressive)
    sample_rate=16000,             # Audio sample rate
    buffer_duration=0.5,           # Audio buffer size in seconds
    silence_duration=0.5           # Silence duration before transcription triggers
)
```

## 📊 Performance Notes

With RTX 4070 Ti Super (16GB VRAM):
- Model loading: ~5-10 seconds
- Real-time factor: >5x (processes 5 seconds of audio in <1 second)
- Recommended model: `large-v3` for best accuracy
- Can handle multiple concurrent connections

## 🔒 Network Security

1. **Firewall**: Open port 8765 for incoming connections:
```bash
# Ubuntu/Debian
sudo ufw allow 8765

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT
```

2. **Find your server IP**:
```bash
# Linux
hostname -I

# Or
ip addr show

# Windows
ipconfig
```

3. **Test locally first**:
```bash
# On the same machine
python3 ../client/client_example.py --server ws://localhost:8765
```

## 📈 Monitoring

The server logs will show:
- Client connections/disconnections
- Transcriptions being processed
- Language detection results
- Any errors or warnings

## 🐛 Troubleshooting

### CUDA Not Available
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi
```

### Port Already in Use
```bash
# Check what's using port 8765
lsof -i :8765

# Or
netstat -tulpn | grep 8765
```

### Model Download
The first run will download the model (~1.5GB for large-v3). Models are cached in `./models/`

## 📝 API Endpoints

The server accepts WebSocket connections on port 8765.

### Incoming Messages

**Audio Data** (binary):
- Format: 16-bit PCM
- Sample Rate: 16000 Hz
- Channels: Mono

**Control Commands** (JSON):
```json
{
  "command": "set_language",
  "language": "en"
}
```

### Outgoing Messages

**Transcription** (JSON):
```json
{
  "type": "partial",
  "text": "Hello world",
  "language": "en",
  "timestamp": 1234567890.123,
  "duration": 2.5
}
```

## 🔄 Updates

To update the Whisper model:
```bash
pip install --upgrade faster-whisper
```

## 📊 Resource Usage

- **RAM**: ~4-6GB for large-v3 model
- **VRAM**: ~5GB for large-v3 with float16
- **CPU**: Minimal when using GPU
- **Network**: Low bandwidth (~2KB/s per client)