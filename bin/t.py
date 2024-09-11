import subprocess
import sys
import shlex
import platform
import multiprocessing
import os
import requests
from tqdm import tqdm
import tarfile
import shutil

class BayeSEDInterface:
    def __init__(self, version, executable_type='mn_1', openmpi_mirror=None):
        self.version = version
        self.executable_type = executable_type
        self.openmpi_mirror = openmpi_mirror
        self._get_system_info()
        self.mpi_cmd = self._setup_openmpi()
        self.num_processes = self._get_max_threads()
        self._setup_environment()

    def _get_system_info(self):
        self.os = platform.system().lower()
        self.arch = platform.machine().lower()

    def _get_max_threads(self):
        return multiprocessing.cpu_count()

    def _setup_openmpi(self):
        openmpi_version = "4.1.6"
        openmpi_url = self.openmpi_mirror or f"https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{openmpi_version}.tar.gz"
        openmpi_dir = f"openmpi-{openmpi_version}"
        openmpi_file = f"{openmpi_dir}.tar.gz"
        install_dir = os.path.abspath("openmpi")
        
        if not os.path.exists(install_dir):
            print(f"Downloading OpenMPI {openmpi_version}...")
            response = requests.get(openmpi_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(openmpi_file, 'wb') as file, tqdm(
                desc=openmpi_file,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            
            print("Extracting OpenMPI...")
            with tarfile.open(openmpi_file, 'r:gz') as tar:
                tar.extractall()
            
            print("Compiling and installing OpenMPI...")
            os.chdir(openmpi_dir)
            subprocess.run(["./configure", f"--prefix={install_dir}"], check=True)
            subprocess.run(["make", "-j", str(self._get_max_threads())], check=True)
            subprocess.run(["make", "install"], check=True)
            os.chdir("..")
            
            print("Cleaning up temporary files...")
            os.remove(openmpi_file)
            shutil.rmtree(openmpi_dir)
        
        mpirun_path = os.path.join(install_dir, "bin", "mpirun")
        if not os.path.exists(mpirun_path):
            raise FileNotFoundError(f"mpirun not found at {mpirun_path}. OpenMPI installation may have failed.")
        
        return mpirun_path

    def _setup_environment(self):
        openmpi_lib_path = os.path.join(os.path.dirname(self.mpi_cmd), "..", "lib")
        os.environ["LD_LIBRARY_PATH"] = f"{openmpi_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ["DYLD_LIBRARY_PATH"] = f"{openmpi_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

    def _get_executable(self):
        base_path = f"./BayeSED{self.version}/bin"
        executable = f"bayesed_{self.executable_type}"
        if self.os == "linux" or (self.os == "windows" and "microsoft" in platform.uname().release.lower()):
            platform_dir = "linux"
        elif self.os == "darwin":
            platform_dir = "mac"
        else:
            raise ValueError(f"Unsupported operating system: {self.os}")
        
        executable_path = os.path.join(base_path, platform_dir, executable)
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Executable not found: {executable_path}")
        
        return executable_path

    def run(self, args):
        executable = self._get_executable()
        if self.os == "darwin" and "arm" in self.arch:
            cmd = ["arch", "-x86_64", self.mpi_cmd, "--use-hwthread-cpus", "-np", str(self.num_processes), executable] + args
        else:
            cmd = [self.mpi_cmd, "--use-hwthread-cpus", "-np", str(self.num_processes), executable] + args
        
        try:
            print(f"Executing command: {shlex.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"BayeSED v{self.version} executed successfully")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Execution error: {e}")
            print(f"Error output: {e.stderr}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check if OpenMPI is correctly installed and mpirun is in the PATH.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bayesed.py [--np <num_processes>] [--exe <executable_type>] [BayeSED arguments...]")
        sys.exit(1)

    executable_type = 'mn_1'  # Default to bayesed_mn_1
    bayesed_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--np":
            num_processes = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--exe":
            executable_type = sys.argv[i+1]
            i += 2
        else:
            bayesed_args.append(sys.argv[i])
            i += 1

    bayesed = BayeSEDInterface(version="3-beta", executable_type=executable_type)
    if 'num_processes' in locals():
        bayesed.num_processes = num_processes

    bayesed.run(bayesed_args)
