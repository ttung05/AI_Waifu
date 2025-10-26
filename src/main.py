#!/usr/bin/env python3
"""
AI Waifu Main Application - Optimized Version
Auto-checks dependencies, GPU support, and system performance
"""

import os
import sys
import subprocess
import importlib
import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class GPUType(Enum):
    """GPU Types enumeration"""
    NVIDIA_CUDA = "nvidia_cuda"
    APPLE_METAL = "apple_metal"
    CPU_ONLY = "cpu_only"

class PerformanceLevel(Enum):
    """Performance level enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class GPUInfo:
    """GPU information structure"""
    available: bool
    gpu_type: GPUType
    name: str
    memory_total: float = 0.0
    memory_free: float = 0.0
    cuda_version: Optional[str] = None
    device_count: int = 0

@dataclass
class LibraryStatus:
    """Library GPU support status"""
    name: str
    gpu_ready: bool
    version: Optional[str] = None
    error: Optional[str] = None

@dataclass
class PerformanceResult:
    """Performance test result"""
    device: str
    basic_ops_time: float
    ai_ops_time: float
    memory_used: float = 0.0
    performance_level: PerformanceLevel = PerformanceLevel.FAIR

class SystemAnalyzer:
    """Optimized system analysis with caching"""
    
    def __init__(self):
        self._gpu_info = None
        self._libraries_status = None
        self._system_info = None
    
    @property
    def system_info(self) -> Dict:
        """Cached system information"""
        if self._system_info is None:
            self._system_info = {
                'os': platform.system(),
                'version': platform.release(),
                'architecture': platform.machine(),
                'python': sys.version.split()[0],
                'cpu_cores': os.cpu_count()
            }
        return self._system_info
    
    def display_system_info(self):
        """Display system information efficiently"""
        info = self.system_info
        print("=== System Information ===")
        print(f"OS: {info['os']} {info['version']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Python: {info['python']}")
        print(f"CPU Cores: {info['cpu_cores']}")

class DependencyManager:
    """Optimized dependency management"""
    
    REQUIRED_PACKAGES = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("cv2", "opencv-python"),
        ("gradio", "gradio"),
        ("requests", "requests"),
        ("tqdm", "tqdm")
    ]
    
    def __init__(self):
        self._installation_cache = {}
    
    def check_and_install_package(self, package_name: str, pip_name: str = None) -> bool:
        """Check package with caching"""
        if package_name in self._installation_cache:
            return self._installation_cache[package_name]
        
        pip_name = pip_name or package_name
        
        try:
            importlib.import_module(package_name)
            print(f"✓ {package_name} - OK")
            self._installation_cache[package_name] = True
            return True
        except ImportError:
            print(f"✗ {package_name} - Missing, installing...")
            return self._install_package(package_name, pip_name)
    
    def _install_package(self, package_name: str, pip_name: str) -> bool:
        """Install package with error handling"""
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✓ {package_name} - Installed successfully")
            self._installation_cache[package_name] = True
            return True
        except subprocess.CalledProcessError:
            print(f"✗ {package_name} - Installation failed")
            self._installation_cache[package_name] = False
            return False
    
    def check_all_dependencies(self) -> bool:
        """Check all required dependencies"""
        print("\n=== Dependency Check ===")
        
        results = [
            self.check_and_install_package(pkg, pip_name)
            for pkg, pip_name in self.REQUIRED_PACKAGES
        ]
        
        success_count = sum(results)
        total_count = len(results)
        
        if success_count == total_count:
            print(f"\n✓ All dependencies satisfied! ({success_count}/{total_count})")
            return True
        else:
            print(f"\n⚠ Dependencies: {success_count}/{total_count} available")
            return False

class GPUAnalyzer:
    """Optimized GPU analysis and testing"""
    
    def __init__(self):
        self._gpu_info = None
        self._libraries_status = None
    
    def detect_gpu_hardware(self) -> GPUInfo:
        """Detect GPU hardware with caching"""
        if self._gpu_info is not None:
            return self._gpu_info
        
        try:
            import torch
            
            if torch.cuda.is_available():
                self._gpu_info = GPUInfo(
                    available=True,
                    gpu_type=GPUType.NVIDIA_CUDA,
                    name=torch.cuda.get_device_name(0),
                    memory_total=torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    memory_free=(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
                    cuda_version=torch.version.cuda,
                    device_count=torch.cuda.device_count()
                )
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._gpu_info = GPUInfo(
                    available=True,
                    gpu_type=GPUType.APPLE_METAL,
                    name="Apple Silicon GPU"
                )
            else:
                self._gpu_info = GPUInfo(
                    available=False,
                    gpu_type=GPUType.CPU_ONLY,
                    name="CPU Only"
                )
        except ImportError:
            self._gpu_info = GPUInfo(
                available=False,
                gpu_type=GPUType.CPU_ONLY,
                name="PyTorch not available"
            )
        
        return self._gpu_info
    
    def analyze_library_gpu_support(self) -> List[LibraryStatus]:
        """Analyze GPU support for each library"""
        if self._libraries_status is not None:
            return self._libraries_status
        
        gpu_info = self.detect_gpu_hardware()
        self._libraries_status = []
        
        # PyTorch
        self._libraries_status.append(self._check_pytorch_gpu(gpu_info))
        
        # Transformers
        self._libraries_status.append(self._check_transformers_gpu(gpu_info))
        
        # OpenCV
        self._libraries_status.append(self._check_opencv_gpu())
        
        return self._libraries_status
    
    def _check_pytorch_gpu(self, gpu_info: GPUInfo) -> LibraryStatus:
        """Check PyTorch GPU support"""
        try:
            import torch
            gpu_ready = gpu_info.available and (torch.cuda.is_available() or 
                                               (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
            return LibraryStatus("PyTorch", gpu_ready, torch.__version__)
        except Exception as e:
            return LibraryStatus("PyTorch", False, error=str(e))
    
    def _check_transformers_gpu(self, gpu_info: GPUInfo) -> LibraryStatus:
        """Check Transformers GPU support"""
        try:
            import transformers
            # Transformers can use GPU if PyTorch can
            gpu_ready = gpu_info.available
            return LibraryStatus("Transformers", gpu_ready, transformers.__version__)
        except Exception as e:
            return LibraryStatus("Transformers", False, error=str(e))
    
    def _check_opencv_gpu(self) -> LibraryStatus:
        """Check OpenCV GPU support"""
        try:
            import cv2
            gpu_ready = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
            return LibraryStatus("OpenCV", gpu_ready, cv2.__version__)
        except Exception as e:
            return LibraryStatus("OpenCV", False, error=str(e))
    
    def display_gpu_analysis(self):
        """Display comprehensive GPU analysis"""
        print("\n=== COMPREHENSIVE GPU ANALYSIS ===")
        
        # Hardware detection
        gpu_info = self.detect_gpu_hardware()
        print("1. Hardware Detection:")
        
        if gpu_info.available:
            print(f"   ✓ GPU: {gpu_info.name}")
            if gpu_info.memory_total > 0:
                print(f"   ✓ Memory: {gpu_info.memory_total:.1f} GB total")
                print(f"   ✓ Memory Free: {gpu_info.memory_free:.1f} GB")
            if gpu_info.cuda_version:
                print(f"   ✓ CUDA Version: {gpu_info.cuda_version}")
            if gpu_info.device_count > 0:
                print(f"   ✓ GPU Count: {gpu_info.device_count}")
        else:
            print("   ⚠ Hardware GPU: Not detected")
        
        # Library analysis
        libraries = self.analyze_library_gpu_support()
        print("\n2. Library GPU Support:")
        
        gpu_ready_count = 0
        for lib in libraries:
            if lib.gpu_ready:
                print(f"   ✓ {lib.name}: GPU Ready")
                if lib.version:
                    print(f"     Version: {lib.version}")
                gpu_ready_count += 1
            else:
                print(f"   ⚠ {lib.name}: CPU Only")
                if lib.error:
                    print(f"     Error: {lib.error}")
        
        # Summary
        print(f"\n=== GPU SUMMARY ===")
        print(f"Hardware: {gpu_info.name} ({gpu_info.gpu_type.value.replace('_', ' ').title()})")
        print(f"GPU-Ready Libraries: {gpu_ready_count}/{len(libraries)}")
        
        for lib in libraries:
            status = "GPU" if lib.gpu_ready else "CPU"
            icon = "✓" if lib.gpu_ready else "⚠"
            print(f"  {icon} {lib.name}: {status}")
        
        # Final status
        if gpu_ready_count == len(libraries) and gpu_info.available:
            print("🚀 Status: FULL GPU ACCELERATION")
        elif gpu_ready_count > 0:
            print("⚡ Status: PARTIAL GPU ACCELERATION")
        else:
            print("💻 Status: CPU ONLY")

class PerformanceTester:
    """Optimized performance testing"""
    
    def __init__(self, gpu_analyzer: GPUAnalyzer):
        self.gpu_analyzer = gpu_analyzer
    
    def run_comprehensive_test(self) -> List[PerformanceResult]:
        """Run comprehensive performance test"""
        print("\n=== COMPREHENSIVE PERFORMANCE TEST ===")
        
        gpu_info = self.gpu_analyzer.detect_gpu_hardware()
        results = []
        
        # Test available devices
        devices_to_test = []
        
        if gpu_info.gpu_type == GPUType.NVIDIA_CUDA:
            devices_to_test.append(("cuda", "NVIDIA GPU"))
        elif gpu_info.gpu_type == GPUType.APPLE_METAL:
            devices_to_test.append(("mps", "Apple Metal"))
        
        devices_to_test.append(("cpu", "CPU"))
        
        for device_name, device_desc in devices_to_test:
            print(f"\n--- Testing {device_desc} ---")
            result = self._test_device_performance(device_name, device_desc)
            if result:
                results.append(result)
        
        # Performance comparison
        self._display_performance_comparison(results)
        
        # Final verdict
        self._display_final_verdict(gpu_info, results)
        
        return results
    
    def _test_device_performance(self, device_name: str, device_desc: str) -> Optional[PerformanceResult]:
        """Test performance on specific device"""
        try:
            import torch
            device = torch.device(device_name)
            
            # Basic operations test
            basic_time = self._test_basic_operations(device)
            print(f"   Matrix Operations: {basic_time:.4f}s")
            
            # AI workload test
            ai_time = self._test_ai_workload(device)
            print(f"   AI Workload: {ai_time:.4f}s")
            
            # Memory usage
            memory_used = 0.0
            if device.type == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                print(f"   GPU Memory Used: {memory_used:.1f} MB")
            
            # Determine performance level
            perf_level = self._determine_performance_level(ai_time)
            print(f"   ✓ {device_desc}: {perf_level.value.title()}")
            
            return PerformanceResult(
                device=device_name,
                basic_ops_time=basic_time,
                ai_ops_time=ai_time,
                memory_used=memory_used,
                performance_level=perf_level
            )
            
        except Exception as e:
            print(f"   ✗ {device_desc}: Failed - {e}")
            return None
    
    def _test_basic_operations(self, device) -> float:
        """Test basic tensor operations"""
        import torch
        
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        start_time = time.time()
        torch.mm(x, y)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        return time.time() - start_time
    
    def _test_ai_workload(self, device) -> float:
        """Test AI workload simulation"""
        import torch
        
        # Neural network simulation
        batch_size, features = 32, 512
        w1 = torch.randn(features, features, device=device)
        w2 = torch.randn(features, features, device=device)
        x = torch.randn(batch_size, features, device=device)
        
        start_time = time.time()
        
        # Forward pass simulation
        for _ in range(10):
            x = torch.relu(torch.mm(x, w1))
            x = torch.relu(torch.mm(x, w2))
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        return time.time() - start_time
    
    def _determine_performance_level(self, ai_time: float) -> PerformanceLevel:
        """Determine performance level based on AI workload time"""
        if ai_time < 0.005:
            return PerformanceLevel.EXCELLENT
        elif ai_time < 0.015:
            return PerformanceLevel.GOOD
        elif ai_time < 0.050:
            return PerformanceLevel.FAIR
        else:
            return PerformanceLevel.POOR
    
    def _display_performance_comparison(self, results: List[PerformanceResult]):
        """Display performance comparison"""
        if len(results) <= 1:
            return
        
        print(f"\n--- PERFORMANCE COMPARISON ---")
        
        # Find CPU baseline
        cpu_result = next((r for r in results if r.device == "cpu"), None)
        if not cpu_result:
            return
        
        print(f"CPU Baseline: {cpu_result.ai_ops_time:.4f}s")
        
        for result in results:
            if result.device != "cpu":
                # Avoid division by zero
                if result.ai_ops_time > 0:
                    speedup = cpu_result.ai_ops_time / result.ai_ops_time
                    device_name = "NVIDIA GPU" if result.device == "cuda" else "Apple Metal"
                    print(f"{device_name}: {result.ai_ops_time:.4f}s (🚀 {speedup:.1f}x {'faster' if speedup > 1 else 'slower'})")
                else:
                    device_name = "NVIDIA GPU" if result.device == "cuda" else "Apple Metal"
                    print(f"{device_name}: {result.ai_ops_time:.4f}s (🚀 Extremely fast)")
    
    def _display_final_verdict(self, gpu_info: GPUInfo, results: List[PerformanceResult]):
        """Display final performance verdict"""
        print(f"\n=== FINAL ANALYSIS ===")
        
        if gpu_info.gpu_type == GPUType.NVIDIA_CUDA:
            print("🚀 NVIDIA GPU: FULLY OPERATIONAL")
            print("   ✓ Your project WILL run on GPU")
            print("   ✓ Expect significant speedup for AI workloads")
        elif gpu_info.gpu_type == GPUType.APPLE_METAL:
            print("⚡ Apple Metal: OPERATIONAL")
            print("   ✓ Your project will use Metal acceleration")
            print("   ✓ Good performance for Apple Silicon")
        else:
            print("💻 CPU Only: No GPU acceleration")
            print("   ⚠ Project will run on CPU")
            print("   ⚠ Consider GPU setup for better performance")

class AIWaifuApp:
    """Main optimized application class"""
    
    def __init__(self):
        self.system_analyzer = SystemAnalyzer()
        self.dependency_manager = DependencyManager()
        self.gpu_analyzer = GPUAnalyzer()
        self.performance_tester = PerformanceTester(self.gpu_analyzer)
    
    def run(self):
        """Main application entry point"""
        print("🚀 AI WAIFU PROJECT - OPTIMIZED")
        print("=" * 40)
        
        # System analysis
        self.system_analyzer.display_system_info()
        
        # Dependency check
        deps_ok = self.dependency_manager.check_all_dependencies()
        
        # GPU analysis
        self.gpu_analyzer.display_gpu_analysis()
        
        # Performance testing
        if deps_ok:
            self.performance_tester.run_comprehensive_test()
        
        # Final summary
        self._display_final_summary(deps_ok)
    
    def _display_final_summary(self, deps_ok: bool):
        """Display final application summary"""
        gpu_info = self.gpu_analyzer.detect_gpu_hardware()
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Dependencies: {'✓ Ready' if deps_ok else '⚠ Issues detected'}")
        print(f"GPU Support: {'✓ ' + gpu_info.gpu_type.value.replace('_', ' ').title() if gpu_info.available else '⚠ CPU only'}")
        print(f"Status: {'🎉 Ready to use!' if deps_ok else '🔧 Setup needed'}")
        
        if deps_ok:
            print(f"\n=== NEXT STEPS ===")
            print("1. Your environment is ready!")
            print("2. You can now run AI models and training")
            print("3. Check DOCUMENTATION.md for detailed usage")
        else:
            print(f"\n=== TROUBLESHOOTING ===")
            print("1. Try: pip install -r requirements.txt")
            print("2. Or run: python install.py")
            print("3. Check conda environment is activated")

def main():
    """Optimized main function"""
    try:
        app = AIWaifuApp()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠ Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()