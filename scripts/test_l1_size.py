#!/usr/bin/env python3
from benchmark_utils import BenchmarkCommandBuilder
from benchmark_utils import run_commands_async
from benchmark_utils import run_commands_async

commands = []
cpus = ['MinorCPU', 'DerivO3CPU']
l1_size = [16, 32, 64, 128]
l1_assoc = [2, 4, 8, 16, 32]

print("generating commands")

for size in l1_size:
    for cpu in cpus:
        bench = BenchmarkCommandBuilder()\
            .set_cpu_type(cpu)\
            .set_stats_filename(f"l1_{size}kb_{cpu}.txt")\
            .set_config_filename(f"l1_{size}kb_{cpu}.ini")\
            .set_l1i_size(size).set_l1d_size(size).build()
        print(bench + "\n\n")
        commands.append(bench)

for assoc in l1_assoc:
    for cpu in cpus:
        bench = BenchmarkCommandBuilder()\
            .set_cpu_type(cpu)\
            .set_stats_filename(f"l1_assoc_{assoc}_{cpu}.txt")\
            .set_config_filename(f"l1_assoc_{assoc}_{cpu}.ini")\
            .set_l1i_assoc(assoc).set_l1d_assoc(assoc).build()
        print(bench + "\n\n")
        commands.append(bench)

run_commands_async(commands, max_processes=3)


