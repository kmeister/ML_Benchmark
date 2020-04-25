#!/usr/bin/env python3
from benchmark_utils import BenchmarkCommandBuilder
from benchmark_utils import run_commands_async
from benchmark_utils import run_commands_async

commands = []
l2_assoc = [1, 2, 4, 8, 16, 32, 64]

print("generating commands")

for assoc in l2_assoc:
    bench = BenchmarkCommandBuilder()\
        .set_stats_filename(f"assoc_{assoc}.txt")\
        .set_config_filename(f"assoc_{assoc}.ini")\
        .set_l2_assoc(assoc).build()
    print(bench + "\n\n")
    commands.append(bench)

run_commands_async(commands, max_processes=3)


