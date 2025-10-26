#!/usr/bin/env python3
"""
AI Waifu Installer - Smart Installation with System Detection
Unified installer with platform detection and setup management
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
        from unified_setup_manager import UnifiedSetupManager
        
        print("\n[*] Detecting system...")
        detector = SystemDetector()
        detector.print_summary(detailed=False)
        
        print("\n[*] Starting smart installation...")
        setup_manager = UnifiedSetupManager(detector)
        results = setup_manager.full_setup(include_dev=False, include_optional=True)
        
        if results['success']:
            print("\n[+] SMART INSTALLATION COMPLETE!")
            setup_manager.save_setup_report(results, "installation_report.json")
            return True
        else:
            print("\n[!] Installation completed with warnings")
            return False
            
    except ImportError:
        print("[-] Smart installer requires system_detector module")
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
        from unified_setup_manager import UnifiedSetupManager
        
        setup_manager = UnifiedSetupManager(detector)
        results = setup_manager.full_setup(include_dev=False, include_optional=True)
        
        if results['success']:
            print("\n" + "="*60)
            print("[+] SETUP COMPLETE!")
            print("="*60)
            setup_manager.save_setup_report(results, "installation_report.json")
            print("[*] Report saved to: installation_report.json")
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