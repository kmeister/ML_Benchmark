from multiprocessing import Pool
from multiprocessing import Manager
import subprocess

class AsyncTask:

    def __init__(self, name, command, queue):
        self.command = command
        self.queue = queue
        self.name = name

    def execute(self):
        print(f"Starting Task: {self.name}")

        try:
            result = subprocess.check_output(self.command.split(), stderr=subprocess.STDOUT)


            self.queue.put(result.decode())

        except Exception as e:
            self.queue.put(str(e))
            pass

        print(f"Finished Task: {self.name}")



class BenchmarkCommandBuilder:

    def __init__(self):
        self._stats_filename = "RISCV.txt"
        self._config_filename= "CONFIG.ini"
        self._l1i_size = 32
        self._l1i_assoc = 4
        self._l1d_size = 32
        self._l1d_assoc = 4
        self._cacheline_size = 64
        self._l2cache = False
        self._l2_size = 1024
        self._l2_assoc = 8
        self._cpu_type = "DerivO3CPU"
        self._maxinsts=100000000
        self._benchmark_path= "../ML_Benchmark/Benchmarks/mlbench"

    def set_l2cache(self, value: bool):
        self._l2cache = value
        return self

    def set_stats_filename(self, value):
        self._stats_filename = value
        return self

    def set_config_filename(self, value):
        self._config_filename = value
        return self

    def set_l1i_size(self, value):
        self._l1i_size = value
        return self

    def set_l1i_assoc(self, value):
        self._l1i_assoc = value
        return self

    def set_l1d_size(self, value):
        self._l1d_size = value
        return self

    def set_l1d_assoc(self, value):
        self._l1d_assoc = value
        return self

    def set_cacheline_size(self, value):
        self._cacheline_size = value
        return self

    def set_l2_size(self, value):
        self._l2_size = value
        return self

    def set_l2_assoc(self, value):
        self._l2_assoc = value
        return self

    def set_cpu_type(self, value):
        self._cpu_type = value
        return self

    def set_maxinsts(self, value):
        self._maxinsts = value
        return self

    def set_benchmark_path(self, value):
        self._benchmark_path = value
        return self

    def build(self):
        str =   "build/RISCV/gem5.opt "
        str += f"--stats-file={self._stats_filename:s} "
        str += f"--dump-config={self._config_filename:s} "
        str += f"configs/example/se.py "
        str += f"-c {self._benchmark_path} "
        str += f"--caches "
        str += f"--l1i_size={self._l1i_size}kB "
        str += f"--l1i_assoc={self._l1d_assoc} "
        str += f"--l1d_size={self._l1d_size}kB "
        str += f"--l1d_assoc={self._l1d_assoc} "
        str += f"--cacheline_size={self._cacheline_size} "
        if self._l2cache:
            str += f"--l2cache "
            str += f"--l2_size={self._l2_size}kB "
            str += f"--l2_assoc={self._l2_assoc} "
        str += f"--cpu-clock=1.6GHz "
        str += f"--cpu-type={self._cpu_type} "
        str += f" -n 1"
        str += f" --maxinsts={self._maxinsts} "

        return str


def run_commands_async(commands, max_processes = 2):
    pool = Pool(max_processes)
    manager = Manager()
    queue = manager.Queue()

    tasks = []
    for command in commands:
        tasks.append(AsyncTask(command, command, queue))

    pool.map(AsyncTask.execute, tasks)

    while not queue.empty():
        print(queue.get())


