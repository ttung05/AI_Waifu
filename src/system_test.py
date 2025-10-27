#!/usr/bin/env python3
"""
System Test Module - Comprehensive Hardware Testing
Test CPU, RAM, GPU, CUDA compatibility before installation
"""

import sys
import platform
import subprocess
from pathlib import Path


class SystemTest:
    """Comprehensive system testing"""
    
    def __init__(self):
        self.results = {
            'cpu': {'passed': False, 'info': {}},
            'memory': {'passed': False, 'info': {}},
            'gpu': {'passed': False, 'info': {}},
            'cuda': {'passed': False, 'info': {}},
            'pytorch': {'passed': False, 'info': {}}
        }
    
    def test_cpu(self):
        """Test CPU capabilities"""
        print("\n[TEST] CPU Check")
        print("-" * 60)
        
        try:
            import os
            cpu_count = os.cpu_count()
            processor = platform.processor()
            machine = platform.machine()
            
            self.results['cpu']['info'] = {
                'count': cpu_count,
                'processor': processor,
                'architecture': machine
            }
            
            print(f"[+] CPU Count: {cpu_count} cores")
            print(f"[+] Processor: {processor}")
            print(f"[+] Architecture: {machine}")
            
            if cpu_count >= 2:
                self.results['cpu']['passed'] = True
                print("[+] CPU: PASSED (2+ cores)")
            else:
                print("[-] CPU: WARNING (only 1 core)")
                
        except Exception as e:
            print(f"[-] CPU Test Error: {e}")
        
        return self.results['cpu']['passed']
    
    def test_memory(self):
        """Test RAM availability"""
        print("\n[TEST] Memory (RAM) Check")
        print("-" * 60)
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = round(mem.total / (1024**3), 2)
            available_gb = round(mem.available / (1024**3), 2)
            
            self.results['memory']['info'] = {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'percent_used': mem.percent
            }
            
            print(f"[+] Total RAM: {total_gb} GB")
            print(f"[+] Available RAM: {available_gb} GB")
            print(f"[+] Usage: {mem.percent}%")
            
            if total_gb >= 8:
                self.results['memory']['passed'] = True
                print("[+] Memory: PASSED (8+ GB recommended)")
            elif total_gb >= 4:
                print("[!] Memory: WARNING (4-8 GB, may be slow)")
                self.results['memory']['passed'] = True
            else:
                print("[-] Memory: INSUFFICIENT (< 4 GB)")
                
        except ImportError:
            print("[-] psutil not installed, cannot check memory")
        except Exception as e:
            print(f"[-] Memory Test Error: {e}")
        
        return self.results['memory']['passed']
    
    def test_gpu_nvidia(self):
        """Test NVIDIA GPU"""
        print("\n[TEST] NVIDIA GPU Check")
        print("-" * 60)
        
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 3:
                    gpu_name = gpu_info[0]
                    driver = gpu_info[1]
                    memory = gpu_info[2]
                    
                    self.results['gpu']['info'] = {
                        'type': 'nvidia',
                        'name': gpu_name,
                        'driver': driver,
                        'memory': memory
                    }
                    
                    print(f"[+] GPU Found: {gpu_name}")
                    print(f"[+] Driver: {driver}")
                    print(f"[+] Memory: {memory}")
                    
                    self.results['gpu']['passed'] = True
                    print("[+] GPU: NVIDIA DETECTED")
                    return True
                    
        except FileNotFoundError:
            print("[-] nvidia-smi not found")
        except Exception as e:
            print(f"[-] GPU Test Error: {e}")
        
        print("[-] No NVIDIA GPU detected")
        return False
    
    def test_cuda(self):
        """Test CUDA availability"""
        print("\n[TEST] CUDA Check")
        print("-" * 60)
        
        try:
            # Try nvidia-smi for CUDA version
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                output = result.stdout
                # Parse CUDA version from output
                for line in output.split('\n'):
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        
                        self.results['cuda']['info'] = {
                            'version': cuda_version,
                            'available': True
                        }
                        
                        print(f"[+] CUDA Version: {cuda_version}")
                        
                        # Check if version is compatible
                        major_version = int(cuda_version.split('.')[0])
                        if major_version >= 11:
                            self.results['cuda']['passed'] = True
                            print(f"[+] CUDA: COMPATIBLE (v{cuda_version})")
                            return True
                        else:
                            print(f"[!] CUDA: OLD VERSION (v{cuda_version}, need 11+)")
                            return False
                            
        except Exception as e:
            print(f"[-] CUDA Test Error: {e}")
        
        print("[-] CUDA not available")
        return False
    
    def test_pytorch(self):
        """Test PyTorch installation and GPU support"""
        print("\n[TEST] PyTorch Check")
        print("-" * 60)
        
        try:
            import torch
            
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            self.results['pytorch']['info'] = {
                'version': version,
                'cuda_available': cuda_available
            }
            
            print(f"[+] PyTorch Version: {version}")
            print(f"[+] CUDA Available: {cuda_available}")
            
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                
                self.results['pytorch']['info'].update({
                    'cuda_version': cuda_version,
                    'gpu_count': gpu_count,
                    'gpu_name': gpu_name
                })
                
                print(f"[+] PyTorch CUDA Version: {cuda_version}")
                print(f"[+] GPU Count: {gpu_count}")
                print(f"[+] GPU Name: {gpu_name}")
                
                # Test GPU computation
                try:
                    device = torch.device("cuda")
                    x = torch.rand(100, 100).to(device)
                    y = torch.rand(100, 100).to(device)
                    z = torch.matmul(x, y)
                    
                    print(f"[+] GPU Computation: SUCCESS")
                    print(f"[+] Test tensor device: {z.device}")
                    
                    self.results['pytorch']['passed'] = True
                    print("[+] PyTorch: FULLY FUNCTIONAL ON GPU")
                    return True
                    
                except Exception as e:
                    print(f"[-] GPU Computation Failed: {e}")
                    print("[!] PyTorch installed but GPU not working")
                    return False
            else:
                print("[!] PyTorch installed but CUDA not available")
                if 'cpu' in version:
                    print("[!] CPU-only version detected")
                return False
                
        except ImportError:
            print("[-] PyTorch not installed")
            self.results['pytorch']['info'] = {'installed': False}
            return False
        except Exception as e:
            print(f"[-] PyTorch Test Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all system tests"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SYSTEM TEST")
        print("="*60)
        
        # Run tests in order
        self.test_cpu()
        self.test_memory()
        gpu_detected = self.test_gpu_nvidia()
        
        if gpu_detected:
            self.test_cuda()
        
        self.test_pytorch()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        tests = [
            ('CPU', self.results['cpu']['passed']),
            ('Memory', self.results['memory']['passed']),
            ('GPU', self.results['gpu']['passed']),
            ('CUDA', self.results['cuda']['passed']),
            ('PyTorch', self.results['pytorch']['passed'])
        ]
        
        passed = sum(1 for _, p in tests if p)
        total = len(tests)
        
        for name, status in tests:
            status_str = "[+] PASSED" if status else "[-] FAILED"
            print(f"{name:15} {status_str}")
        
        print("-" * 60)
        print(f"TOTAL: {passed}/{total} tests passed")
        
        if self.results['pytorch']['passed']:
            print("\n[+] SYSTEM READY - PyTorch running on GPU!")
        elif self.results['gpu']['passed']:
            print("\n[!] GPU detected but PyTorch not configured")
            print("[*] Recommendation: Run 'python install.py' to install PyTorch with GPU support")
        else:
            print("\n[!] No GPU detected - will use CPU")
            print("[*] Recommendation: Run 'python install.py' to install PyTorch (CPU)")
        
        print("="*60)
        
        return self.results
    
    def get_recommendations(self):
        """Get installation recommendations based on test results"""
        recommendations = []
        
        if not self.results['pytorch']['passed']:
            if self.results['cuda']['passed']:
                cuda_ver = self.results['cuda']['info'].get('version', '12.9')
                if cuda_ver.startswith('12'):
                    recommendations.append("Install PyTorch with CUDA 12.1 support")
                elif cuda_ver.startswith('11'):
                    recommendations.append("Install PyTorch with CUDA 11.8 support")
            else:
                recommendations.append("Install PyTorch (CPU-only)")
        
        if not self.results['memory']['passed']:
            recommendations.append("WARNING: Low RAM - consider upgrading to 8+ GB")
        
        return recommendations


def main():
    """Main test entry point"""
    tester = SystemTest()
    results = tester.run_all_tests()
    
    recommendations = tester.get_recommendations()
    if recommendations:
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        print("="*60)


if __name__ == "__main__":
    main()
