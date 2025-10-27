#!/usr/bin/env python3
"""
AI Waifu - Main Entry Point
Run system tests and verify GPU compatibility
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point - Test system and verify setup"""
    print("\n" + "="*60)
    print("AI WAIFU - SYSTEM VERIFICATION")
    print("="*60)
    
    # Check if system test module exists
    try:
        from system_test import SystemTest
        
        print("[*] Running comprehensive system tests...")
        print("="*60)
        
        tester = SystemTest()
        results = tester.run_all_tests()
        
        # Get recommendations
        recommendations = tester.get_recommendations()
        
        if recommendations:
            print("\n" + "="*60)
            print("RECOMMENDATIONS")
            print("="*60)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
            print("="*60)
            
            # Suggest running installer
            if not results['pytorch']['passed']:
                print("\n[!] PyTorch not properly configured")
                print("[*] Run: python install.py")
                print("    This will auto-detect your GPU and install compatible PyTorch")
        else:
            print("\n[+] SYSTEM FULLY CONFIGURED!")
            print("[*] All tests passed - ready for AI/ML workloads")
        
        return 0 if results['pytorch']['passed'] else 1
        
    except ImportError as e:
        print(f"[-] Error: Cannot import system_test module: {e}")
        print("[*] Running basic GPU test instead...")
        return basic_gpu_test()
    except Exception as e:
        print(f"[-] Error running tests: {e}")
        return 1


def basic_gpu_test():
    """Fallback: Basic GPU test if system_test unavailable"""
    print("\n" + "="*60)
    print("BASIC GPU TEST")
    print("="*60 + "\n")
    
    try:
        import torch
        print(f"[+] PyTorch version: {torch.__version__}")
        print(f"[+] CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"[+] CUDA version: {torch.version.cuda}")
            print(f"[+] GPU count: {torch.cuda.device_count()}")
            print(f"[+] Current GPU: {torch.cuda.current_device()}")
            print(f"[+] GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test GPU computation
            print("\n[*] Testing GPU computation...")
            device = torch.device("cuda")
            x = torch.rand(1000, 1000).to(device)
            y = torch.rand(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print(f"[+] Matrix multiplication on GPU: SUCCESS")
            print(f"[+] Result shape: {z.shape}")
            print(f"[+] Device: {z.device}")
            
            print("\n[+] GPU TEST PASSED!")
            return 0
        else:
            print("[-] CUDA not available - running on CPU")
            print("\n[!] To enable GPU:")
            print("    1. Make sure you have an NVIDIA GPU")
            print("    2. Install CUDA drivers")
            print("    3. Run: python install.py")
            return 1
            
    except ImportError as e:
        print(f"[-] Error: {e}")
        print("\n[!] PyTorch not installed")
        print("[*] Run: python install.py")
        return 1
    except Exception as e:
        print(f"[-] Error: {e}")
        return 1
    finally:
        print("\n" + "="*60)


if __name__ == "__main__":
    sys.exit(main())
