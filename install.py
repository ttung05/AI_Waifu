#!/usr/bin/env python3
"""AI Waifu Installer - Everything in one file"""
import sys, subprocess, platform, os

def check_gpu():
    """Check NVIDIA GPU"""
    print("\n[*] Checking GPU...")
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.split('\n'):
                if 'Quadro' in line or 'GeForce' in line or 'Tesla' in line or 'RTX' in line:
                    if '|' in line:
                        gpu = line.split('|')[1].strip().split()[0:3]
                        print(f"[+] GPU: {' '.join(gpu)}")
                        break
            for line in r.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda = line.split('CUDA Version:')[1].split()[0]
                    print(f"[+] CUDA: {cuda}")
                    return True, cuda
            return True, "12.0"
    except: pass
    print("[-] No GPU")
    return False, None

def check_pytorch():
    """Check PyTorch type"""
    print("\n[*] Checking PyTorch...")
    try:
        import torch
        ver = torch.__version__
        print(f"[+] PyTorch: {ver}")
        if '+cpu' in ver:
            print("[!] CPU version")
            return 'cpu'
        elif torch.cuda.is_available():
            print("[+] GPU version")
            return 'gpu'
        else:
            print("[!] No CUDA")
            return 'no_cuda'
    except:
        print("[-] Not installed")
        return 'not_installed'

def uninstall_pytorch():
    """Uninstall PyTorch"""
    print("\n[*] Uninstalling PyTorch...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'], check=True, capture_output=True)
        print("[+] Uninstalled")
        return True
    except:
        print("[-] Failed")
        return False

def install_pytorch_gpu(cuda):
    """Install PyTorch GPU"""
    print("\n[*] Installing PyTorch GPU...")
    idx = 'cu118' if cuda and cuda.startswith('11') else 'cu121'
    print(f"[*] Using {idx}")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', f'https://download.pytorch.org/whl/{idx}'], check=True)
        print("[+] Installed!")
        return True
    except:
        print("[-] Failed")
        return False

def install_pytorch_cpu():
    """Install PyTorch CPU"""
    print("\n[*] Installing PyTorch CPU...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'], check=True)
        print("[+] Installed!")
        return True
    except:
        print("[-] Failed")
        return False

def install_packages():
    """Install other packages"""
    print("\n[*] Installing packages...")
    pkgs = ['numpy', 'scipy', 'pandas', 'matplotlib', 'pillow', 'requests', 'tqdm', 'psutil', 'GPUtil', 'transformers', 'datasets', 'tokenizers', 'accelerate', 'scikit-learn']
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade'] + pkgs, check=True)
        print("[+] Packages installed!")
        return True
    except:
        print("[-] Some failed")
        return False

def verify():
    """Verify installation"""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    try:
        import torch
        print(f"[+] PyTorch: {torch.__version__}")
        print(f"[+] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
            x = torch.randn(100,100).cuda()
            y = x @ x
            print("[+] GPU test: PASSED")
        return True
    except Exception as e:
        print(f"[-] Failed: {e}")
        return False

def main():
    """Main installer"""
    print("\n" + "="*60)
    print("AI WAIFU INSTALLER")
    print("="*60)
    
    # Check GPU
    has_gpu, cuda = check_gpu()
    
    # Check PyTorch
    pt_status = check_pytorch()
    
    # Install PyTorch
    print("\n" + "="*60)
    print("PYTORCH INSTALLATION")
    print("="*60)
    
    if has_gpu:
        if pt_status == 'cpu':
            print("[*] GPU detected, CPU PyTorch installed")
            print("[*] Switching to GPU version...")
            uninstall_pytorch()
            install_pytorch_gpu(cuda)
        elif pt_status == 'not_installed':
            print("[*] Installing GPU version...")
            install_pytorch_gpu(cuda)
        elif pt_status == 'gpu':
            print("[+] GPU version already installed")
        else:
            print("[*] Reinstalling GPU version...")
            uninstall_pytorch()
            install_pytorch_gpu(cuda)
    else:
        if pt_status == 'not_installed':
            print("[*] Installing CPU version...")
            install_pytorch_cpu()
        else:
            print("[+] Already installed")
    
    # Install packages
    print("\n" + "="*60)
    print("OTHER PACKAGES")
    print("="*60)
    install_packages()
    
    # Verify
    if verify():
        print("\n" + "="*60)
        print("[+] INSTALLATION COMPLETE!")
        print("="*60)
        print("\nRun: python main.py")
        print("="*60)
        return 0
    else:
        print("\n[-] Verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
