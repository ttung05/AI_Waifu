#!/usr/bin/env python3
"""
AI Waifu Installer - Smart Installation with System Detection
Auto-detect hardware, test system, install compatible PyTorch + CUDA
"""

import sys
import subprocess
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def show_platform_detection():
    """Show detailed platform and GPU detection"""
    print("\n" + "="*60)
    print("PLATFORM DETECTION")
    print("="*60)
    
    try:
        from system_detector import SystemDetector
        detector = SystemDetector()
        detector.print_summary(detailed=True)
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        for i, rec in enumerate(detector.get_optimization_recommendations(), 1):
            print(f"  {i}. {rec}")
        
        return detector
    except ImportError as e:
        print(f"[-] Cannot import system_detector: {e}")
        print("[*] Basic platform info:")
        print(f"    OS: {platform.system()} {platform.release()}")
        print(f"    Python: {sys.version}")
        return None
    except Exception as e:
        print(f"[-] Error detecting platform: {e}")
        return None


def auto_install_pytorch(detector):
    """Auto-detect GPU and install compatible PyTorch version"""
    print("\n" + "="*60)
    print("PYTORCH AUTO-INSTALLER")
    print("="*60)
    
    gpu_info = detector.gpu_info
    
    # Determine PyTorch install command
    if gpu_info['has_gpu'] and gpu_info['gpu_type'] == 'nvidia':
        cuda_version = gpu_info.get('cuda_version', '12.9')
        
        # Map CUDA version to PyTorch index
        if cuda_version.startswith('12'):
            torch_index = 'cu121'
            print(f"[*] NVIDIA GPU detected: {gpu_info['gpu_names'][0]}")
            print(f"[*] CUDA Version: {cuda_version}")
            print(f"[*] Installing PyTorch with CUDA 12.1 support...")
        elif cuda_version.startswith('11'):
            torch_index = 'cu118'
            print(f"[*] NVIDIA GPU detected: {gpu_info['gpu_names'][0]}")
            print(f"[*] CUDA Version: {cuda_version}")
            print(f"[*] Installing PyTorch with CUDA 11.8 support...")
        else:
            torch_index = 'cu118'
            print(f"[*] NVIDIA GPU detected but CUDA version unclear")
            print(f"[*] Installing PyTorch with CUDA 11.8 (most compatible)...")
        
        # Try CUDA install
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', f'https://download.pytorch.org/whl/{torch_index}'
        ]
        
        try:
            print(f"[*] Running: pip install torch with {torch_index}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("[+] PyTorch with CUDA installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("[-] CUDA install failed, trying CPU version...")
            return install_pytorch_cpu()
            
    elif gpu_info['has_gpu'] and gpu_info['gpu_type'] == 'amd':
        print(f"[*] AMD GPU detected: {gpu_info['gpu_names'][0]}")
        print(f"[*] Installing PyTorch with ROCm support...")
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/rocm5.7'
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("[+] PyTorch with ROCm installed!")
            return True
        except:
            print("[-] ROCm install failed, using CPU version...")
            return install_pytorch_cpu()
            
    elif gpu_info['has_gpu'] and gpu_info['gpu_type'] == 'apple':
        print(f"[*] Apple Silicon detected: {gpu_info['gpu_names'][0]}")
        print(f"[*] Installing PyTorch with Metal support...")
        return install_pytorch_cpu()  # Apple uses same package with Metal support
        
    else:
        print("[*] No GPU detected or unsupported GPU")
        print("[*] Installing CPU-only PyTorch...")
        return install_pytorch_cpu()


def install_pytorch_cpu():
    """Install CPU-only PyTorch"""
    cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("[+] PyTorch (CPU) installed successfully!")
        return True
    except:
        print("[-] Failed to install PyTorch")
        return False


def quick_install():
    """Quick installation - basic packages only"""
    print("\n" + "="*60)
    print("AI WAIFU QUICK INSTALLER")
    print("="*60)
    
    packages = [
        ("numpy scipy pandas matplotlib pillow requests tqdm", "Core Libraries"),
        ("transformers datasets tokenizers accelerate scikit-learn huggingface-hub", "AI/ML"),
        ("psutil GPUtil pynvml", "Monitoring"),
        ("torch torchvision torchaudio", "PyTorch"),
    ]
    
    success = 0
    for pkg, desc in packages:
        print(f"\n[*] Installing {desc}...")
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + pkg.split()
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[+] {desc} - Success")
            success += 1
        except:
            print(f"[-] {desc} - Failed")
    
    print(f"\n{'='*60}")
    if success == len(packages):
        print("[+] INSTALLATION COMPLETE!")
    else:
        print(f"[!] WARNING: {success}/{len(packages)} packages installed")
    
    return success == len(packages)


def smart_install():
    """Smart installation - auto-detect system and GPU"""
    print("\n" + "="*60)
    print("AI WAIFU SMART INSTALLER")
    print("="*60)
    
    try:
        from system_detector import SystemDetector
        
        print("\n[*] Detecting system...")
        detector = SystemDetector()
        detector.print_summary(detailed=False)
        
        # Install core packages first
        print("\n[*] Installing core packages...")
        core_packages = [
            "numpy", "scipy", "pandas", "matplotlib", "pillow", 
            "requests", "tqdm", "psutil", "GPUtil"
        ]
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + core_packages
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("[+] Core packages installed")
        except:
            print("[-] Some core packages failed")
        
        # Install AI/ML packages
        print("\n[*] Installing AI/ML packages...")
        ml_packages = ["transformers", "datasets", "tokenizers", "accelerate", "scikit-learn"]
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + ml_packages
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("[+] AI/ML packages installed")
        except:
            print("[-] Some AI/ML packages failed")
        
        # Auto-install PyTorch
        print("\n[*] Auto-installing PyTorch...")
        pytorch_success = auto_install_pytorch(detector)
        
        if pytorch_success:
            print("\n[+] SMART INSTALLATION COMPLETE!")
            return True
        else:
            print("\n[!] Installation completed with warnings")
            return False
            
    except ImportError as e:
        print(f"[-] Smart installer requires system_detector module: {e}")
        print("[*] Falling back to quick install...")
        return quick_install()


def run_full_setup():
    """Run complete setup with platform detection"""
    print("\n" + "="*60)
    print("AI WAIFU FULL SETUP")
    print("="*60)
    
    # Step 1: Platform Detection
    print("\n[STEP 1/2] Platform Detection")
    detector = show_platform_detection()
    
    if not detector:
        print("\n[!] Could not detect platform, using basic install")
        return quick_install()
    
    # Step 2: Smart Installation
    print("\n[STEP 2/2] Smart Installation")
    print("[*] Starting automatic installation...")
    
    try:
        # Install core packages
        print("\n[*] Installing core packages...")
        core_packages = [
            "numpy", "scipy", "pandas", "matplotlib", "pillow",
            "requests", "tqdm", "psutil", "GPUtil"
        ]
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + core_packages
        subprocess.run(cmd, check=True, capture_output=True)
        print("[+] Core packages installed")
        
        # Install AI/ML packages
        print("\n[*] Installing AI/ML packages...")
        ml_packages = ["transformers", "datasets", "tokenizers", "accelerate", "scikit-learn"]
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + ml_packages
        subprocess.run(cmd, check=True, capture_output=True)
        print("[+] AI/ML packages installed")
        
        # Auto-install PyTorch
        print("\n[*] Auto-installing PyTorch based on GPU detection...")
        pytorch_success = auto_install_pytorch(detector)
        
        if pytorch_success:
            print("\n" + "="*60)
            print("[+] SETUP COMPLETE!")
            print("="*60)
            return True
        else:
            print("\n[!] Setup completed with warnings")
            return False
            
    except Exception as e:
        print(f"[-] Setup error: {e}")
        print("[*] Falling back to quick install...")
        return quick_install()


def main():
    """Main installer - automatically detects and installs"""
    print("\n" + "="*60)
    print("AI WAIFU AUTOMATIC INSTALLER")
    print("="*60)
    print("[*] Automatic platform detection and smart installation")
    print("="*60)
    
    # Check if manual mode specified via command line
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        # Default: Full automatic setup
        mode = "full"
    
    # Run selected mode
    if mode in ['full', '1', '-f', '--full']:
        success = run_full_setup()
    elif mode in ['smart', '2', '-s', '--smart']:
        success = smart_install()
    elif mode in ['detect', '4', '-d', '--detect']:
        show_platform_detection()
        success = True
    else:
        success = quick_install()
    
    if success:
        print("\n" + "="*60)
        print("[*] Next steps:")
        print("   python main.py              # Start chat")
        print("   python test.py --all        # Run tests")
        print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())