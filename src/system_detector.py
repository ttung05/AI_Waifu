#!/usr/bin/env python3
"""
System Detector - Unified Platform and GPU Detection
Hệ thống phát hiện nền tảng và GPU thống nhất
"""

import os
import sys
import platform
import subprocess
import json
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path


class SystemDetector:
    """Unified system, platform, and GPU detection"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.gpu_info = self._detect_gpu()
        self.python_info = self._detect_python()
        self.performance_info = self._detect_performance()
        
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform information"""
        return {
            'system': platform.system(),  # Windows, Darwin, Linux
            'machine': platform.machine(),  # AMD64, arm64, x86_64
            'architecture': platform.architecture()[0],  # 64bit, 32bit
            'processor': platform.processor(),
            'release': platform.release(),
            'version': platform.version(),
            'node': platform.node()
        }
    
    def _detect_python(self) -> Dict[str, Any]:
        """Detect Python information"""
        return {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable': sys.executable,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro
            }
        }
    
    def _detect_performance(self) -> Dict[str, Any]:
        """Detect system performance capabilities"""
        try:
            import psutil
        except ImportError:
            psutil = None
        
        info = {
            'cpu_count': os.cpu_count(),
            'memory_available': False,
            'total_memory_gb': 0
        }
        
        try:
            if psutil:
                memory = psutil.virtual_memory()
                info.update({
                    'memory_available': True,
                    'total_memory_gb': round(memory.total / (1024**3), 2),
                    'available_memory_gb': round(memory.available / (1024**3), 2)
                })
        except:
            pass
            
        return info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Comprehensive GPU detection"""
        gpu_info = {
            'has_gpu': False,
            'gpu_type': 'none',
            'gpu_count': 0,
            'gpu_names': [],
            'cuda_version': None,
            'rocm_version': None,
            'metal_support': False,
            'compute_capability': None,
            'memory_info': [],
            'details': []
        }
        
        # Try NVIDIA first (most common)
        nvidia_info = self._detect_nvidia_gpu()
        if nvidia_info['detected']:
            gpu_info.update({
                'has_gpu': True,
                'gpu_type': 'nvidia',
                'gpu_count': nvidia_info['count'],
                'gpu_names': nvidia_info['names'],
                'cuda_version': nvidia_info['cuda_version'],
                'compute_capability': nvidia_info['compute_capability'],
                'memory_info': nvidia_info['memory_info'],
                'details': nvidia_info['details']
            })
            return gpu_info
        
        # Try Apple Silicon
        if self.platform_info['system'] == 'Darwin':
            apple_info = self._detect_apple_gpu()
            if apple_info['detected']:
                gpu_info.update({
                    'has_gpu': True,
                    'gpu_type': 'apple',
                    'gpu_count': 1,
                    'gpu_names': [apple_info['chip']],
                    'metal_support': apple_info['metal_support'],
                    'details': apple_info['details']
                })
                return gpu_info
        
        # Try AMD GPU
        amd_info = self._detect_amd_gpu()
        if amd_info['detected']:
            gpu_info.update({
                'has_gpu': True,
                'gpu_type': 'amd',
                'gpu_count': amd_info['count'],
                'gpu_names': amd_info['names'],
                'rocm_version': amd_info['rocm_version'],
                'details': amd_info['details']
            })
            return gpu_info
        
        # Try Intel GPU (basic detection)
        intel_info = self._detect_intel_gpu()
        if intel_info['detected']:
            gpu_info.update({
                'has_gpu': True,
                'gpu_type': 'intel',
                'gpu_count': intel_info['count'],
                'gpu_names': intel_info['names'],
                'details': intel_info['details']
            })
        
        return gpu_info
    
    def _detect_nvidia_gpu(self) -> Dict[str, Any]:
        """Detect NVIDIA GPU with detailed info"""
        result = {
            'detected': False,
            'count': 0,
            'names': [],
            'cuda_version': None,
            'compute_capability': [],
            'memory_info': [],
            'details': []
        }
        
        try:
            # Use nvidia-ml-py if available, otherwise nvidia-smi
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                result['detected'] = True
                result['count'] = device_count
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    result['names'].append(name)
                    result['memory_info'].append({
                        'total_mb': memory_info.total // (1024**2),
                        'free_mb': memory_info.free // (1024**2),
                        'used_mb': memory_info.used // (1024**2)
                    })
                    result['details'].append({
                        'index': i,
                        'name': name,
                        'memory_total_gb': round(memory_info.total / (1024**3), 1)
                    })
                
                # Get CUDA version
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    result['cuda_version'] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    pass
                    
            except ImportError:
                # Fallback to nvidia-smi
                cmd_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'], 
                    capture_output=True, text=True, timeout=10
                )
                
                if cmd_result.returncode == 0:
                    lines = [line.strip() for line in cmd_result.stdout.strip().split('\n') if line.strip()]
                    result['detected'] = True
                    result['count'] = len(lines)
                    
                    for i, line in enumerate(lines):
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 2:
                            name = parts[0]
                            memory = parts[1]
                            compute_cap = parts[2] if len(parts) > 2 else 'Unknown'
                            
                            result['names'].append(name)
                            result['compute_capability'].append(compute_cap)
                            result['details'].append({
                                'index': i,
                                'name': name,
                                'memory_mb': memory,
                                'compute_capability': compute_cap
                            })
                    
                    # Get CUDA version
                    try:
                        cuda_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                        if cuda_result.returncode == 0:
                            for line in cuda_result.stdout.split('\n'):
                                if 'CUDA Version:' in line:
                                    result['cuda_version'] = line.split('CUDA Version:')[1].strip().split()[0]
                                    break
                    except:
                        pass
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            result['details'].append({'error': str(e)})
        
        return result
    
    def _detect_apple_gpu(self) -> Dict[str, Any]:
        """Detect Apple Silicon GPU"""
        result = {
            'detected': False,
            'chip': 'Unknown',
            'metal_support': False,
            'details': []
        }
        
        if self.platform_info['system'] == 'Darwin':
            try:
                # Check for Apple Silicon
                if self.platform_info['machine'] == 'arm64':
                    result['detected'] = True
                    result['metal_support'] = True
                    
                    # Get chip info
                    try:
                        cmd_result = subprocess.run(
                            ['system_profiler', 'SPHardwareDataType'], 
                            capture_output=True, text=True, timeout=10
                        )
                        if cmd_result.returncode == 0:
                            for line in cmd_result.stdout.split('\n'):
                                if 'Chip:' in line:
                                    result['chip'] = line.split('Chip:')[1].strip()
                                    break
                                elif 'Processor Name:' in line and any(m in line for m in ['M1', 'M2', 'M3', 'M4']):
                                    result['chip'] = line.split('Processor Name:')[1].strip()
                                    break
                    except:
                        result['chip'] = 'Apple Silicon'
                    
                    result['details'].append({
                        'chip': result['chip'],
                        'architecture': 'Apple Silicon ARM64',
                        'metal_support': True,
                        'unified_memory': True
                    })
                    
            except Exception as e:
                result['details'].append({'error': str(e)})
        
        return result
    
    def _detect_amd_gpu(self) -> Dict[str, Any]:
        """Detect AMD GPU"""
        result = {
            'detected': False,
            'count': 0,
            'names': [],
            'rocm_version': None,
            'details': []
        }
        
        try:
            system = self.platform_info['system']
            
            if system == 'Linux':
                # Try rocm-smi
                try:
                    cmd_result = subprocess.run(['rocm-smi', '--showproductname'], 
                                              capture_output=True, text=True, timeout=10)
                    if cmd_result.returncode == 0:
                        result['detected'] = True
                        # Parse ROCm output
                        for line in cmd_result.stdout.split('\n'):
                            if 'GPU' in line and ':' in line:
                                result['count'] += 1
                                gpu_name = line.split(':')[1].strip()
                                result['names'].append(gpu_name)
                except:
                    pass
                
                # Fallback to lspci
                if not result['detected']:
                    try:
                        cmd_result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                        if cmd_result.returncode == 0:
                            for line in cmd_result.stdout.split('\n'):
                                if any(keyword in line.upper() for keyword in ['AMD', 'RADEON', 'RX ']):
                                    if 'VGA' in line.upper() or 'DISPLAY' in line.upper():
                                        result['detected'] = True
                                        result['count'] += 1
                                        result['names'].append(line.strip())
                    except:
                        pass
                        
            elif system == 'Windows':
                # Use wmic for AMD detection
                try:
                    cmd_result = subprocess.run(
                        ['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                        capture_output=True, text=True, timeout=10
                    )
                    if cmd_result.returncode == 0:
                        for line in cmd_result.stdout.split('\n'):
                            if any(keyword in line.upper() for keyword in ['AMD', 'RADEON']):
                                result['detected'] = True
                                result['count'] += 1
                                result['names'].append(line.strip())
                except:
                    pass
                    
        except Exception as e:
            result['details'].append({'error': str(e)})
        
        return result
    
    def _detect_intel_gpu(self) -> Dict[str, Any]:
        """Detect Intel GPU (basic)"""
        result = {
            'detected': False,
            'count': 0,
            'names': [],
            'details': []
        }
        
        try:
            system = self.platform_info['system']
            
            if system == 'Windows':
                cmd_result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                    capture_output=True, text=True, timeout=10
                )
                if cmd_result.returncode == 0:
                    for line in cmd_result.stdout.split('\n'):
                        if 'Intel' in line and any(keyword in line.upper() for keyword in ['GRAPHICS', 'UHD', 'IRIS']):
                            result['detected'] = True
                            result['count'] += 1
                            result['names'].append(line.strip())
                            
            elif system == 'Linux':
                cmd_result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                if cmd_result.returncode == 0:
                    for line in cmd_result.stdout.split('\n'):
                        if 'Intel' in line and any(keyword in line.upper() for keyword in ['VGA', 'GRAPHICS']):
                            result['detected'] = True
                            result['count'] += 1
                            result['names'].append(line.strip())
                            
        except Exception as e:
            result['details'].append({'error': str(e)})
        
        return result
    
    def get_best_compute_device(self) -> str:
        """Get the best available compute device"""
        system = self.platform_info['system']
        
        # NVIDIA CUDA (highest priority)
        if self.gpu_info['gpu_type'] == 'nvidia':
            return 'cuda'
        
        # Apple Metal (macOS only)
        if system == 'Darwin' and self.gpu_info['gpu_type'] == 'apple':
            return 'mps'
        
        # AMD ROCm (Linux only)
        if system == 'Linux' and self.gpu_info['gpu_type'] == 'amd':
            return 'cuda'  # ROCm uses CUDA API
        
        # CPU fallback
        return 'cpu'
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        if not self.gpu_info['has_gpu']:
            recommendations.append("Consider getting a GPU for 5-20x AI performance boost")
            recommendations.append("Use CPU-optimized libraries (Intel MKL, OpenMP)")
        
        if self.gpu_info['gpu_type'] == 'nvidia':
            recommendations.append("Use CUDA-optimized libraries (CuPy, CuDNN)")
            recommendations.append("Enable mixed precision training for 2x speedup")
        
        if self.gpu_info['gpu_type'] == 'apple':
            recommendations.append("Use Metal Performance Shaders for optimization")
            recommendations.append("Leverage unified memory architecture")
        
        if self.performance_info['cpu_count'] > 4:
            recommendations.append("Enable multi-threading and parallel processing")
        
        return recommendations
    
    def print_summary(self, detailed: bool = True):
        """Print system summary"""
        print("[SYS]  SYSTEM INFORMATION")
        print("=" * 50)
        
        # Platform
        print(f"OS: {self.platform_info['system']} {self.platform_info['release']}")
        print(f"Architecture: {self.platform_info['architecture']} ({self.platform_info['machine']})")
        print(f"Processor: {self.platform_info['processor']}")
        
        # Python
        print(f"Python: {self.python_info['version']} ({self.python_info['implementation']})")
        
        # Performance
        print(f"CPU Cores: {self.performance_info['cpu_count']}")
        if self.performance_info['memory_available']:
            print(f"Total Memory: {self.performance_info['total_memory_gb']} GB")
        
        # GPU
        print(f"\n[GPU] GPU INFORMATION")
        print("=" * 30)
        
        if self.gpu_info['has_gpu']:
            print(f"[+] GPU Type: {self.gpu_info['gpu_type'].upper()}")
            print(f"GPU Count: {self.gpu_info['gpu_count']}")
            
            for i, name in enumerate(self.gpu_info['gpu_names']):
                print(f"GPU {i}: {name}")
                
                if detailed and self.gpu_info['memory_info']:
                    if i < len(self.gpu_info['memory_info']):
                        mem_info = self.gpu_info['memory_info'][i]
                        print(f"  Memory: {mem_info.get('total_mb', 'Unknown')} MB")
            
            if self.gpu_info['cuda_version']:
                print(f"CUDA Version: {self.gpu_info['cuda_version']}")
                
            if self.gpu_info['metal_support']:
                print("Metal Support: [+]")
                
        else:
            print("[-] No GPU detected - CPU only")
        
        # Compute device
        best_device = self.get_best_compute_device()
        print(f"\n[*] RECOMMENDED COMPUTE DEVICE: {best_device.upper()}")
        
        # Recommendations
        if detailed:
            recommendations = self.get_optimization_recommendations()
            if recommendations:
                print(f"\n[*] OPTIMIZATION RECOMMENDATIONS:")
                for rec in recommendations:
                    print(f"• {rec}")
    
    def save_to_file(self, filename: str = "system_info.json"):
        """Save system info to JSON file"""
        data = {
            'platform': self.platform_info,
            'python': self.python_info,
            'performance': self.performance_info,
            'gpu': self.gpu_info,
            'best_device': self.get_best_compute_device(),
            'recommendations': self.get_optimization_recommendations()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return filename


def main():
    """Test the system detector"""
    detector = SystemDetector()
    detector.print_summary(detailed=True)
    
    # Save results
    filename = detector.save_to_file()
    print(f"\n[*] System info saved to: {filename}")
    
    return detector


if __name__ == "__main__":
    main()
