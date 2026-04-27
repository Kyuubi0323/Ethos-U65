# Copyright 2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys
import re
import subprocess
import argparse
import platform
from ethosu.vela import architecture_features
from ethosu.vela import compiler_driver
from ethosu.vela import model_reader
from ethosu.vela import rawdata_writer
from ethosu.vela import scheduler
from ethosu.vela import tflite_writer
from ethosu.vela import errors
from ethosu.vela import nn_graph
from ethosu.vela.operation import Op
from ethosu.vela.operation import CustomType

cwd = os.path.dirname(os.path.abspath(__file__))

# Function to execute shell command and return output (while printing output in console)
def system_cmd(cmd, cwd='./', shell=True, output=False, exceptionOnError=False, show_cmd=False):
    process = subprocess.Popen(cmd, cwd=cwd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = process.communicate()
    if (output):
        print(':%s$ %s' % (cwd, cmd))
        print(cmd_out.strip().decode('utf-8'))
    else:
        if show_cmd:
            cmd_out = f"# {cmd}\n".encode('utf-8') + cmd_out
    if process.returncode != 0:
        print(cmd_err.strip().decode('utf-8'))
        if exceptionOnError:
            raise Exception('Error running command: "%s"' % cmd)

    try:
        return cmd_out.strip().decode('utf-8'), cmd_err.strip().decode('utf-8'), process.returncode
    except:
        return str(cmd_out.strip()), str(cmd_err.strip()), process.returncode

def vela_convert(input_model, converted_model):
    sys.setrecursionlimit(4000)

    output_dir = os.path.dirname(converted_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    arch = architecture_features.ArchitectureFeatures(
        vela_config_files=None,
        system_config=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        memory_mode=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        accelerator_config='ethos-u65-256',
        max_blockdep=architecture_features.ArchitectureFeatures.MAX_BLOCKDEP,
        verbose_config=False,
        arena_cache_size=384 * 1024,
    )

    compiler_options = compiler_driver.CompilerOptions(
        tensor_allocator = nn_graph.TensorAllocator.HillClimb,
        output_dir=output_dir,
    )

    scheduler_options = scheduler.SchedulerOptions(
        optimization_strategy=scheduler.OptimizationStrategy.Performance,
        sram_target=arch.arena_cache_size,
        verbose_schedule=False,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    sys.stdout = open(os.devnull, 'w')
    nng, network_type = model_reader.read_model(input_model, model_reader_options)
    sys.stdout = sys.__stdout__
    if not nng:
        raise InputFileError(input_model, "Input file could not be read")

    # Skip if model is converted model.
    for sg in nng.subgraphs:
        for op in sg.get_all_ops():
            if op.type == Op.Custom and op.attrs.get("custom_type") == CustomType.ExistingNpuOp:
                return None, input_model

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options, network_type, "results_vela")

    tflite_writer.write_tflite(nng, converted_model)

    nng.arch = arch
    return nng, converted_model


def get_counter(converted_model, delegate_path, pmu_events):
    pmus=[0, 0, 0, 0]
    cycle = 1

    delegate_option = "enable_cycle_counter:true;"
    for idx, pmu in enumerate(pmu_events):
        delegate_option += f"pmu_event{idx}:{pmu};"

    out, err, ret = system_cmd(f"/usr/bin/tensorflow-lite*/examples/benchmark_model --graph={converted_model} --external_delegate_path={delegate_path} --external_delegate_options=\"{delegate_option}\" --max_secs=0.001")

    #Ethos_u PMUs : [ 338698 62783 0 0 ]
    #Ethos-u cycle counter: 3607420
    pattern = r".*Ethos_u\s*PMUs\s*:\s*\[ ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) \].*"
    m = re.search(r"%s" % pattern, out)
    if m:
        pmus = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
    else:
        print("WARN: pmu count is not captured")

    pattern = r".*Ethos-u\s*cycle\s*counter:\s*([0-9]+).*"
    m = re.search(r"%s" % pattern, out)
    if m:
        cycle = int(m.group(1))
    else:
        print("WARN: cycle count is not captured")

    return cycle, pmus

#Ethos-U65 High-End: SRAM (16 GB/s) and DRAM (3.75 GB/s)
SRAM_BPC = 16.0
DRAM_BPC = 3.75

def main():
    machine = platform.machine().lower()
    if not 'aarch64' in machine:
        print("warn: running on imx93 board, please.")
        sys.exit(1)

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="vela-prof", description="Runtime model profiling tool for vela model on the i.MX93 platform.")
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Input tflite model')
    parser.add_argument(
        '-e',
        '--delegate',
        required=False,
        default='libethosu_delegate.so',
        help='External delegate library (default: %(default)s)')

    args = parser.parse_args(argv)
    nng = None

    if not os.path.exists(args.input):
        raise InputFileError(input_model_name, "No such file")

    dir_name, file_name = os.path.split(args.input)
    basename = os.path.splitext(file_name)[0]
    converted_model = os.path.join(dir_name, basename + "_vela.tflite")

    # convert model
    nng, converted_model = vela_convert(args.input, converted_model)

    print(f"Info: Profiling model {converted_model} with PMU:")
    print(f"================================================")

    cycle, pmus = get_counter(converted_model, args.delegate, [62, 67, 40, 45])

    if nng and nng.macs:
        print(f"NPU Utilization: {nng.macs*100/256/cycle:.1f}% = {nng.macs} / 256 / {cycle} ")
        print("")

    rd, wr  = pmus[0], pmus[1]
    print(f"DRAM Utilization: {(rd + wr) * 1600/cycle/DRAM_BPC:.1f}% = {(rd + wr)} * 16 / {cycle} / {DRAM_BPC}")
    print(f"   read : {rd * 1600/cycle/DRAM_BPC:.1f}% = {rd} * 16 / {cycle} / {DRAM_BPC}")
    print(f"   write: {wr * 1600/cycle/DRAM_BPC:.1f}% = {wr} * 16 / {cycle} / {DRAM_BPC}")
    print("")

    rd, wr  = pmus[2], pmus[3]
    print(f"SRAM Utilization: {(rd + wr )* 1600/cycle/SRAM_BPC:.1f}% = {(rd + wr)} * 16 / {cycle} / {SRAM_BPC}")
    print(f"   read : {rd * 1600/cycle/SRAM_BPC:.1f}% = {rd} * 16 / {cycle} / {SRAM_BPC}")
    print(f"   write: {wr * 1600/cycle/SRAM_BPC:.1f}% = {wr} * 16 / {cycle} / {SRAM_BPC}")

if __name__ == '__main__':
    main()
