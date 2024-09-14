import subprocess
import sys
import platform
import multiprocessing
import os
import requests
from tqdm import tqdm
import tarfile
import shutil

class BayeSEDInterface:
    def __init__(self, executable_type='mn_1', openmpi_mirror=None):
        self.executable_type = executable_type
        self.openmpi_mirror = openmpi_mirror
        self._get_system_info()
        self.mpi_cmd = self._setup_openmpi()
        self.num_processes = self._get_max_threads()
        self.executable_path = self._get_executable()  # 添加这一行

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
        
        # Set up environment variables for OpenMPI
        os.environ["PATH"] = f"{os.path.dirname(mpirun_path)}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(install_dir, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        return mpirun_path

    def _get_executable(self):
        base_path = "./bin"
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
        if isinstance(self.mpi_cmd, str):
            self.mpi_cmd = self.mpi_cmd.split()

        # Set TMPDIR environment variable
        os.environ['TMPDIR'] = '/tmp'

        cmd = self.mpi_cmd + ['--use-hwthread-cpus', '-np', str(self.num_processes), self.executable_path] + args
        print(f"Executing command: {' '.join(cmd)}")
        
        try:
            # Check if -h is in args
            if '-h' in args or '--help' in args:
                # For help command, don't use mpirun
                process = subprocess.Popen([self.executable_path] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            else:
                # Use mpirun for normal execution
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            # Read and print output in real-time
            for line in process.stdout:
                print(line, end='')  # Print each line of output directly
            
            # Wait for the process to finish
            process.wait()
            
            # Check the return code
            if process.returncode != 0:
                print(f"BayeSED execution failed, return code: {process.returncode}")
                return False
            else:
                print("BayeSED execution completed successfully")
                return True
        except Exception as e:
            print(f"Error occurred while executing BayeSED: {str(e)}")
            return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bayesed.py [--np <num_processes>] [--exe <executable_type>] [BayeSED arguments...]")
        sys.exit(1)

    executable_type = 'mn_1'  # Default to bayesed_mn_1
    bayesed_args = []
    i = 1
    num_processes = 1  # 默认设置为1
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

    bayesed = BayeSEDInterface(executable_type=executable_type)
    
    # 检查是否存在 -h 参数
    if '-h' in bayesed_args:
        num_processes = 1
    
    bayesed.num_processes = num_processes

    bayesed.run(bayesed_args)
