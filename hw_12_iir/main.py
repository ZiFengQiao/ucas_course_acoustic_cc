#!/usr/bin/env python3
"""
定点 IIR 滤波器实验 — 跨平台构建脚本
替换 Makefile，保证在 Windows / Linux 上均可运行。

用法：
    python main.py           — 运行全流程（设计→编译→滤波→分析）
    python main.py design    — 仅 Python 滤波器设计 + 量化
    python main.py build     — 仅编译 C 程序
    python main.py filter    — 仅 C 定点滤波
    python main.py analyze   — 仅 Python 误差分析 + 绘图
    python main.py clean     — 清理编译产物 (obj/, bin/)
    python main.py distclean — 清理所有生成文件
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ── 项目路径配置 ────────────────────────────────────────────────────────

PROJ_DIR = Path(__file__).resolve().parent
SRCDIR   = PROJ_DIR / "src"
OBJDIR   = PROJ_DIR / "obj"
BINDIR   = PROJ_DIR / "bin"
PYDIR    = PROJ_DIR / "python"
DATADIR  = PROJ_DIR / "data"
IMGDIR   = PROJ_DIR / "img"

SRCS    = [SRCDIR / "main.c", SRCDIR / "iir_filter.c"]
OBJS    = [OBJDIR / s.with_suffix(".o").name for s in SRCS]

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
TARGET     = BINDIR / f"iir_filter{EXE_SUFFIX}"

CONDA_ENV  = "gpu_course"


# ── 工具函数 ────────────────────────────────────────────────────────────

def run(cmd, *, cwd=None, env=None):
    """运行命令，失败时退出。"""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  \033[2m{cmd_str}\033[0m")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"\033[31mError: 命令失败 (exit {result.returncode})\033[0m")
        sys.exit(result.returncode)


def run_python(script_rel):
    """在 conda 环境中运行 Python 脚本，conda 不可用时回退到当前 Python。"""
    script = (PYDIR / script_rel).resolve()
    if not script.is_file():
        print(f"\033[31mError: 脚本不存在: {script}\033[0m")
        sys.exit(1)

    # 优先用 conda run（跨平台，免去 shell 激活差异）
    if shutil.which("conda"):
        try:
            result = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", CONDA_ENV,
                 "python", str(script)],
                cwd=PROJ_DIR,
            )
            if result.returncode == 0:
                return
            print(f"\033[33mconda run 失败，回退到系统 Python...\033[0m")
        except FileNotFoundError:
            pass

    # 回退：直接用当前 Python 解释器
    run([sys.executable, str(script)], cwd=PROJ_DIR)


def find_cc():
    """查找可用的 C 编译器，优先 gcc。"""
    # Linux 优先 gcc
    if not IS_WINDOWS:
        for cc in ["gcc", "cc", "clang"]:
            if shutil.which(cc):
                return cc
        print("\033[31mError: 未找到 C 编译器 (gcc/cc/clang)\033[0m")
        sys.exit(1)

    # Windows：优先 MinGW gcc，其次 clang，最后 MSVC cl
    for cc in ["gcc", "clang", "cl"]:
        if shutil.which(cc):
            return cc
    print("\033[31mError: 未找到 C 编译器，请安装 MinGW-w64 或 Visual Studio\033[0m")
    sys.exit(1)


# ── 步骤实现 ────────────────────────────────────────────────────────────

def cmd_design():
    """[1/4] Python 滤波器设计 + 数据量化"""
    print("\033[1m[1/4] Python: 滤波器设计 + 数据量化...\033[0m")
    DATADIR.mkdir(parents=True, exist_ok=True)
    IMGDIR.mkdir(parents=True, exist_ok=True)
    run_python("design_filter.py")


def cmd_build():
    """[2/4] C 编译"""
    print("\033[1m[2/4] C: 编译...\033[0m")
    OBJDIR.mkdir(parents=True, exist_ok=True)
    BINDIR.mkdir(parents=True, exist_ok=True)

    cc = find_cc()
    cflags = ["-Wall", "-Wextra", "-O2", "-std=c99"]
    ldflags = ["-lm"]

    # 编译 .c -> .o
    for src, obj in zip(SRCS, OBJS):
        run([cc] + cflags + ["-c", "-o", str(obj), str(src)])

    # 链接
    print("\033[1m[2/4] C: 链接...\033[0m")
    obj_paths = [str(o) for o in OBJS]
    run([cc] + cflags + ["-o", str(TARGET)] + obj_paths + ldflags)


def cmd_filter():
    """[3/4] C 定点滤波"""
    print("\033[1m[3/4] C: 定点 IIR 滤波 (Q14 + Q30)...\033[0m")
    if not TARGET.is_file():
        print("\033[33m二进制文件不存在，先执行 build...\033[0m")
        cmd_build()

    run([
        str(TARGET),
        str(DATADIR / "input_16bit.raw"),
        str(DATADIR / "input_32bit.raw"),
        str(DATADIR / "coeffs.txt"),
    ])


def cmd_analyze():
    """[4/4] Python 误差分析 + 绘图"""
    print("\033[1m[4/4] Python: 误差分析 + 绘图...\033[0m")
    run_python("analyze_results.py")


def cmd_run():
    """运行全流程"""
    cmd_design()
    cmd_build()
    cmd_filter()
    cmd_analyze()
    print()
    print("=" * 48)
    print("  全流程完成：设计 → 编译 → 滤波 → 分析")
    print("=" * 48)


def cmd_clean():
    """清理编译产物"""
    for d in [OBJDIR, BINDIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"  已删除: {d}")


def cmd_distclean():
    """清理所有生成文件"""
    cmd_clean()
    for p in IMGDIR.glob("*.png"):
        p.unlink()
        print(f"  已删除: {p}")
    # 清理 data 目录中由脚本生成的文件
    generated_data = [
        "coeffs.txt", "input_16bit.raw", "input_32bit.raw",
        "output_python.raw", "output_16bit.raw", "output_32bit.raw",
    ]
    for name in generated_data:
        f = DATADIR / name
        if f.exists():
            f.unlink()
            print(f"  已删除: {f}")
    print("已清理所有生成文件。")


# ── 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="定点 IIR 滤波器实验 — 跨平台构建脚本"
    )
    sub = parser.add_subparsers(dest="command", help="可用命令")

    sub.add_parser("run",      help="运行全流程（默认）")
    sub.add_parser("all",      help="同 run")
    sub.add_parser("design",   help="仅 Python 滤波器设计 + 量化")
    sub.add_parser("build",    help="仅编译 C 程序")
    sub.add_parser("filter",   help="仅 C 定点滤波")
    sub.add_parser("analyze",  help="仅 Python 误差分析 + 绘图")
    sub.add_parser("clean",    help="清理编译产物 (obj/, bin/)")
    sub.add_parser("distclean", help="清理所有生成文件")
    sub.add_parser("help",     help="显示帮助")

    args = parser.parse_args()

    commands = {
        "run": cmd_run, "all": cmd_run,
        "design": cmd_design, "build": cmd_build,
        "filter": cmd_filter, "analyze": cmd_analyze,
        "clean": cmd_clean, "distclean": cmd_distclean,
        "help": lambda: parser.print_help(),
    }

    cmd = args.command or "run"
    if cmd not in commands:
        parser.print_help()
        sys.exit(1)
    commands[cmd]()


if __name__ == "__main__":
    main()
