# AI Waifu# AI Waifu# AI Waifu



Simple AI chatbot with automatic GPU detection and smart installation.



## Quick StartSimple AI chatbot with automatic GPU detection and platform-aware installation.Cross-platform AI chatbot with automatic GPU detection.



### Step 1: Automatic Installation

```bash

python install.py## Quick Start## Quick Start (Lần đầu chạy)

```



**What it does automatically:**

- Detects your OS (Windows/macOS/Linux)### Step 1: Install (First time)### Bước 1: Cài đặt tự động

- Detects GPU (NVIDIA CUDA/Apple Metal/AMD ROCm/CPU-only)

- Installs PyTorch with correct GPU support```bash```bash

- Installs all required AI libraries

- No user interaction needed - fully automatic!python install.py# Chạy installer - tự động kiểm tra hệ thống và cài đặt thư viện



### Step 2: Start Chat```python install.py

```bash

python main.py```

```

**Installer Options:**

That's it! The installer runs automatically without asking questions.

1. **Full Setup** (Recommended) - Platform detection + Smart installInstaller sẽ:

## Manual Installation Modes

2. **Smart Install** - Auto-detect GPU and install optimal packages- Tự động phát hiện hệ điều hành (Windows/macOS/Linux)

If you need manual control, use command-line arguments:

3. **Quick Install** - Basic packages only- Tự động phát hiện GPU (NVIDIA/Apple/AMD)

```bash

python install.py full      # Full automatic setup (default)4. **Platform Detection Only** - Check system without installing- Tự động tải và cài đặt tất cả thư viện cần thiết

python install.py smart     # Smart install without platform detection

python install.py quick     # Basic packages only- Tự động cài đặt PyTorch phù hợp với GPU của bạn

python install.py detect    # Platform detection only (no install)

```**What installer does:**



## Project Structure- Detects OS (Windows/macOS/Linux)Chọn mode:



```- Detects GPU (NVIDIA CUDA/Apple Metal/AMD ROCm)- **Option 1** (Smart Install): Tự động phát hiện GPU và cài đặt thư viện tối ưu (Khuyến nghị)

AI_Waifu/

├── src/- Installs PyTorch with correct GPU support- **Option 2** (Quick Install): Cài đặt nhanh các thư viện cơ bản

│   ├── ai_waifu_core.py          # AI chat engine

│   ├── system_detector.py         # Platform & GPU detection- Installs all required AI libraries

│   ├── unified_setup_manager.py   # Smart package installer

│   └── utils.py                   # Utilities- Saves installation report### Bước 2: Chạy chương trình

├── data/                          # Data storage

├── models/                        # Model cache (auto-downloaded)```bash

├── logs/                          # Chat history

├── install.py                     # Automatic installer### Step 2: Run Chat# Sau khi cài đặt xong, chạy chương trình

├── main.py                        # Chat application

├── test.py                        # Test suite```bashpython main.py

├── config.yaml                    # Configuration

└── requirements.txt               # Dependenciespython main.py```

```

```

## Features

Hoặc chạy trực tiếp:

- **Fully Automatic**: No questions, no choices - just install and run

- **Cross-platform**: Windows, macOS, LinuxThat's it! Start chatting with your AI.```bash

- **Smart GPU Detection**: NVIDIA CUDA, Apple Metal, AMD ROCm

- **Platform-Aware**: Installs optimal packages for your systempython main.py --chat       # Chat CLI

- **CLI Chat**: Simple terminal chat interface

- **Chat Logging**: Automatic conversation history## Project Structurepython main.py --web        # Web interface (http://localhost:7860)

- **Lightweight**: No web dependencies

python main.py --detect     # Xem thông tin hệ thống

## Testing

``````

```bash

# Test all componentsAI_Waifu/

python test.py --all

├── src/                           # Source code## Các lệnh hữu ích

# Test system detection only

python test.py --system --detailed│   ├── ai_waifu_core.py          # AI core functionality



# Test chat only│   ├── system_detector.py         # Platform & GPU detection### Kiểm tra hệ thống

python test.py --chat

```│   ├── unified_setup_manager.py   # Installation manager```bash



## Configuration│   └── utils.py                   # Utilities# Kiểm tra GPU và hệ thống



Edit `config.yaml`:├── data/                          # Data storagepython main.py --detect



```yaml├── models/                        # Model cache (auto-downloaded)

model:

  name: "microsoft/DialoGPT-medium"├── logs/                          # Chat history# Test tất cả chức năng

  temperature: 0.7        # Response creativity (0-1)

  max_length: 1000       # Max response length├── install.py                     # Installer (run first)python test.py --all

  device: "auto"         # auto/cuda/mps/cpu

├── main.py                        # Chat application

logging:

  enabled: true├── test.py                        # Test suite# Test riêng từng phần

  save_path: "logs/"

```├── config.yaml                    # Configurationpython test.py --system     # Test phát hiện hệ thống



## System Requirements└── requirements.txt               # Dependenciespython test.py --chat       # Test chat



### Minimum```python test.py --web        # Test web interface

- **Python**: 3.8+

- **RAM**: 8GB```

- **Storage**: 10GB free

- **Internet**: Required for first-time setup## Features



### Recommended### Cài đặt lại

- **Python**: 3.10+

- **RAM**: 16GB- **Cross-platform**: Windows, macOS, Linux```bash

- **GPU**: NVIDIA GPU with CUDA 11.8+ or Apple Silicon M1+

- **Storage**: 15GB free- **GPU Auto-detection**: NVIDIA CUDA, Apple Metal, AMD ROCm# Nếu cài đặt lỗi, chạy lại



### Supported Platforms- **Smart Installation**: Platform-aware package installationpython install.py

- Windows 10/11 (x64)

- macOS (Intel/Apple Silicon)- **CLI Chat**: Simple terminal chat interface

- Linux (x64/ARM64)

- **Chat Logging**: Automatic conversation history# Hoặc cài thủ công

## Troubleshooting

- **Lightweight**: No web dependencies, pure CLIpip install -r requirements.txt

### Installation Failed

```bash```

# Check what went wrong:

python install.py detect## Installation Modes



# Try quick install:## Cấu trúc thư mục

python install.py quick

### Mode 1: Full Setup (Recommended)

# Manual install:

pip install -r requirements.txt```bash```

```

python install.pyAI_Waifu/

### GPU Not Detected

```bash# Choose: 1├── src/                    # Source code

# Check system info:

python test.py --system --detailed│   ├── ai_waifu_core.py           # AI core



# For NVIDIA:# This will:│   ├── system_detector.py          # Phát hiện hệ thống & GPU

# 1. Run: nvidia-smi

# 2. Reinstall PyTorch:# 1. Detect your platform and GPU│   ├── unified_setup_manager.py    # Quản lý cài đặt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```# 2. Show detailed system information│   ├── utils.py                    # Tiện ích



### Chat Not Working# 3. Ask for confirmation│   └── web_interface.py            # Giao diện web

```bash

# Test dependencies:# 4. Install optimal packages for your system├── data/                   # Dữ liệu

python main.py

# 5. Save installation report├── models/                 # Model cache

# If model download fails, check internet connection

# Model is ~500MB, downloads on first run```├── logs/                   # Chat logs

```

├── main.py                 # Entry point chính

## Usage Examples

### Mode 2: Smart Install├── install.py              # Installer (chạy đầu tiên)

### Basic Chat

```bash```bash├── test.py                 # Test suite

python main.py

```python install.py├── config.yaml             # Cấu hình



```# Choose: 2└── requirements.txt        # Danh sách thư viện

You: Hello!

AI: Hello! How can I help you today?```



You: Tell me a joke# Auto-detects GPU and installs optimal PyTorch version

AI: Why did the programmer quit? He didn't get arrays!

```## Tính năng

You: quit

[*] Goodbye!

```

### Mode 3: Quick Install- **Cross-platform**: Windows, macOS, Linux

### Installation Workflow

```bash- **Auto GPU detection**: Tự động phát hiện NVIDIA CUDA, Apple Metal, AMD ROCm

```bash

# 1. Clone/Download projectpython install.py- **Smart Installation**: Tự động cài đặt thư viện phù hợp

cd AI_Waifu

# Choose: 3- **Web interface**: Giao diện web với Gradio

# 2. Run automatic installer

python install.py- **CLI chat**: Chat qua terminal

# (Wait 5-10 minutes - fully automatic)

# Installs basic packages without GPU detection- **Chat logging**: Lưu lịch sử chat

# 3. Test everything

python test.py --all```



# 4. Start chatting## Yêu cầu hệ thống

python main.py

```### Mode 4: Platform Detection Only



## FAQ```bash- **Python**: 3.8+ (Python 3.10 khuyến nghị)



**Q: Do I need to answer any questions during installation?**  python install.py- **RAM**: 8GB tối thiểu (16GB khuyến nghị)

A: No! The installer is fully automatic. Just run `python install.py` and wait.

# Choose: 4- **GPU**: Không bắt buộc (tự động dùng CPU nếu không có GPU)

**Q: Do I need a GPU?**  

A: No, but it's faster. Auto-detects and uses CPU if no GPU available.  - NVIDIA: CUDA 11.8+ hoặc 12.1+



**Q: How long does installation take?**  # Shows system info without installing anything  - Apple: M1/M2/M3/M4 với Metal

A: 5-10 minutes depending on internet speed.

# Useful for checking GPU compatibility  - AMD: ROCm (Linux)

**Q: Do I need CUDA installed?**  

A: No, PyTorch with CUDA is installed automatically if you have NVIDIA GPU.```- **Storage**: 10GB trống



**Q: How big is the AI model?**  

A: ~500MB, downloaded automatically on first chat.

## Usage## Cài đặt chi tiết

**Q: Does it work offline?**  

A: Yes, after initial setup and model download.



**Q: Can I use a different AI model?**  ### Start Chat### Mode 1: Smart Install (Khuyến nghị)

A: Yes, edit `model.name` in `config.yaml`.

```bash```bash

**Q: Where are chat logs?**  

A: In `logs/` directory as JSON files.python main.pypython install.py



## Advanced Usage```# Chọn: 1



### Command-Line Options



```bash**Chat Commands:**# Installer sẽ:

# Installer modes

python install.py              # Automatic full setup (default)- Type your message and press Enter# 1. Phát hiện hệ thống của bạn

python install.py detect       # Show platform info only

python install.py quick        # Fast basic install- Type `quit`, `exit`, or `bye` to exit# 2. Phát hiện GPU (nếu có)

python install.py smart        # Smart install

- Press `Ctrl+C` to force quit# 3. Hỏi xác nhận

# Chat

python main.py                 # Start chat# 4. Tự động cài đặt PyTorch + CUDA (nếu có NVIDIA GPU)



# Testing**Example:**# 5. Cài đặt tất cả thư viện cần thiết

python test.py --all           # All tests

python test.py --system        # System detection```# 6. Lưu báo cáo cài đặt

python test.py --chat          # Chat functionality

```You: Hello!```



## LicenseAI: Hello! How can I help you today?



MIT License### Mode 2: Quick Install



---You: Tell me a joke```bash



**Simple. Automatic. Smart.**AI: Why did the programmer quit his job? Because he didn't get arrays!python install.py


# Chọn: 2

You: quit

[*] Goodbye!# Cài đặt nhanh các thư viện cơ bản

```# Không tự động phát hiện GPU

```

## Testing

### Cài đặt thủ công

### Test All Components```bash

```bash# Nếu installer không hoạt động

python test.py --allpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```pip install transformers gradio datasets accelerate

pip install -r requirements.txt

### Test System Detection```

```bash

# Basic system info## Sử dụng

python test.py --system

### 1. Menu tương tác (Interactive)

# Detailed system info```bash

python test.py --system --detailedpython main.py

```

# Save system info to fileHiển thị menu với các tùy chọn:

python test.py --system --save1. Run Setup (cài đặt/kiểm tra)

2. Start Chat (chat CLI)

# Show optimization recommendations3. Start Web Interface (mở web)

python test.py --system --recommendations4. Platform Detection (xem thông tin hệ thống)

```5. Exit



### Test Chat### 2. Chat CLI

```bash```bash

# Interactive chat testpython main.py --chat

python test.py --chat```

- Chat trực tiếp qua terminal

# Skip interactive mode- Gõ 'quit' hoặc 'exit' để thoát

python test.py --chat --skip-interactive

```### 3. Web Interface

```bash

## Configurationpython main.py --web

```

Edit `config.yaml`:- Mở trình duyệt: http://localhost:7860

- Giao diện đồ họa đẹp hơn

```yaml- Hỗ trợ nhiều người dùng

model:

  name: "microsoft/DialoGPT-medium"### 4. Kiểm tra hệ thống

  temperature: 0.7        # Response creativity (0-1)```bash

  max_length: 1000       # Max response lengthpython main.py --detect

  device: "auto"         # auto/cuda/mps/cpu```

Hiển thị:

logging:- Thông tin OS

  enabled: true- Thông tin CPU

  save_path: "logs/"- Thông tin GPU (nếu có)

```- Thông tin RAM

- Python version

## System Requirements- PyTorch version

- CUDA version (nếu có)

### Minimum

- **Python**: 3.8+## Cấu hình

- **RAM**: 8GB

- **Storage**: 10GB free spaceChỉnh sửa `config.yaml`:

- **Internet**: Required for first-time model download

```yaml

### Recommendedmodel:

- **Python**: 3.10+  name: "microsoft/DialoGPT-medium"

- **RAM**: 16GB  temperature: 0.7        # Độ sáng tạo (0-1)

- **GPU**: NVIDIA GPU with CUDA 11.8+ or Apple Silicon M1+  max_length: 1000       # Độ dài tối đa

- **Storage**: 15GB free space  device: "auto"         # auto/cuda/mps/cpu



### Supported Platformsweb:

- **Windows 10/11** (x64) - NVIDIA/AMD/Intel GPU or CPU  host: "0.0.0.0"

- **macOS** (Intel/Apple Silicon) - Metal/CPU  port: 7860

- **Linux** (x64/ARM64) - NVIDIA/AMD GPU or CPU  share: false           # true để tạo public link



## Troubleshootinglogging:

  enabled: true

### Dependencies Not Installed  save_path: "logs/"

```bash```

# If main.py says dependencies missing:

python install.py## Testing



# Or install manually:```bash

pip install -r requirements.txt# Test tất cả

```python test.py --all



### GPU Not Detected# Test từng phần

```bashpython test.py --system --detailed    # Xem chi tiết hệ thống

# Check detailed system info:python test.py --chat                 # Test chat

python test.py --system --detailedpython test.py --web --no-launch      # Test web (không mở trình duyệt)



# For NVIDIA GPU:# Lưu kết quả

# 1. Verify driver: nvidia-smipython test.py --system --save        # Lưu thông tin hệ thống

# 2. Reinstall PyTorch with CUDA:```

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```## Troubleshooting



### Chat Not Working### GPU không được phát hiện?

```bash```bash

# Test chat functionality:# Kiểm tra chi tiết

python test.py --chatpython test.py --system --detailed



# Check if model downloaded:# Nếu có NVIDIA GPU nhưng không nhận:

# First run needs internet to download ~500MB model# 1. Kiểm tra CUDA toolkit đã cài chưa

```# 2. Chạy: nvidia-smi

# 3. Cài lại PyTorch với CUDA:

### Installation Failedpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```bash```

# Try quick install:

python install.py### Lỗi khi cài đặt?

# Choose: 3```bash

# Cài lại từng package

# Or manual install:pip install --upgrade pip

pip install --upgrade pippip install torch torchvision torchaudio

pip install torch transformerspip install transformers gradio

pip install -r requirements.txtpip install -r requirements.txt

```

# Hoặc dùng conda

## Command Referenceconda create -n ai_waifu python=3.10

conda activate ai_waifu

### install.pypython install.py

```bash```

python install.py           # Interactive menu

python install.py full      # Full setup### Chat không hoạt động?

python install.py smart     # Smart install```bash

python install.py quick     # Quick install# Kiểm tra dependencies

python install.py detect    # Platform detection onlypython test.py --chat

```

# Kiểm tra model đã tải chưa

### main.py# Model sẽ tự động tải lần đầu chạy

```bash# Cần kết nối internet

python main.py              # Start chat```

```

### Web interface không mở?

### test.py```bash

```bash# Kiểm tra port

python test.py --all                    # Run all testspython main.py --web --port 8080

python test.py --system                 # Test system detection

python test.py --system --detailed      # Detailed system info# Kiểm tra gradio

python test.py --system --save          # Save to JSONpip install --upgrade gradio

python test.py --chat                   # Test chat

python test.py --chat --skip-interactive # No interactive chat# Test không mở trình duyệt

```python test.py --web --no-launch

```

## First-Time Setup Guide

## Quy trình chạy lần đầu (Step-by-step)

```bash

# 1. Clone/Download project```bash

cd AI_Waifu# Bước 1: Clone hoặc download project

cd AI_Waifu

# 2. Run installer (choose option 1)

python install.py# Bước 2: Cài đặt (tự động phát hiện và cài thư viện)

python install.py

# 3. Wait for installation (5-10 minutes)# Chọn: 1 (Smart Install)

# Installer will download:# Đợi 5-10 phút để tải thư viện

# - PyTorch (~2GB with CUDA)

# - Transformers & other libraries (~500MB)# Bước 3: Kiểm tra cài đặt

# - AI model (~500MB, downloaded on first chat)python main.py --detect

# Xem thông tin GPU, Python, PyTorch

# 4. Test installation

python test.py --all# Bước 4: Test

python test.py --all

# 5. Start chatting!# Kiểm tra tất cả chức năng

python main.py

```# Bước 5: Chạy

python main.py --chat

## FAQ# Hoặc

python main.py --web

**Q: Do I need a GPU?**  ```

A: No, but it's faster. The system auto-detects and uses CPU if no GPU is available.

## FAQ

**Q: How long does installation take?**  

A: 5-10 minutes depending on internet speed and system.**Q: Lần đầu chạy có cần internet không?**  

A: Có. Cần internet để tải thư viện và AI model (khoảng 500MB-1GB).

**Q: Do I need CUDA installed?**  

A: No, PyTorch with CUDA will be installed automatically.**Q: Chạy trên máy không có GPU được không?**  

A: Được. Tự động dùng CPU (chậm hơn nhưng vẫn hoạt động).

**Q: How big is the AI model?**  

A: ~500MB, automatically downloaded on first chat.**Q: Mất bao lâu để cài đặt?**  

A: 5-10 phút (tùy tốc độ internet và cấu hình máy).

**Q: Does it work offline?**  

A: After first-time setup and model download, yes.**Q: Cần cài CUDA không?**  

A: Không bắt buộc. Nếu có NVIDIA GPU, installer sẽ tự cài PyTorch với CUDA.

**Q: Can I change the AI model?**  

A: Yes, edit `model.name` in `config.yaml`.**Q: Model AI nặng bao nhiêu?**  

A: Khoảng 500MB (tải tự động lần đầu).

**Q: Where are chat logs saved?**  

A: In `logs/` directory as JSON files.## License



## LicenseMIT License



MIT License---



---**Dễ dàng. Nhanh chóng. Thông minh.**


**Simple. Fast. Smart.**
