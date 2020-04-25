#!/usr/bin/env python3
from benchmark_utils import BenchmarkCommandBuilder
from benchmark_utils import run_commands_async
from benchmark_utils import run_commands_async

commands = []
l2_size = [512, 1024, 2048, 2048*2, 2048*4]

print("generating commands")

for size in l2_size:
    bench = BenchmarkCommandBuilder()\
        .set_stats_filename(f"l2size_{size}.txt")\
        .set_config_filename(f"l2size_{l2_size}.ini")\
        .set_l2_size(size).build()
    print(bench + "\n\n")
    commands.append(bench)

run_commands_async(commands, max_processes=3)


