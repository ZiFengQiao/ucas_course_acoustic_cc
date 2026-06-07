# ============================================================================
#  定点 IIR 滤波器实验 — 全流程 Makefile
# ============================================================================

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

SRCDIR  = src
OBJDIR  = obj
BINDIR  = bin
PYDIR   = python
DATADIR = data
IMGDIR  = img

SRCS    = $(SRCDIR)/main.c $(SRCDIR)/iir_filter.c
OBJS    = $(addprefix $(OBJDIR)/, $(notdir $(SRCS:.c=.o)))
TARGET  = $(BINDIR)/iir_filter

# Conda 环境
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)" && conda activate gpu_course &&

# 中间产物
DESIGN_OUT  = $(DATADIR)/coeffs.txt \
              $(DATADIR)/input_16bit.raw \
              $(DATADIR)/input_32bit.raw \
              $(DATADIR)/output_python.raw \
              $(IMGDIR)/filter_response.png

FILTER_OUT  = $(DATADIR)/output_16bit.raw \
              $(DATADIR)/output_32bit.raw

ANALYZE_OUT = $(IMGDIR)/time_domain_comparison.png \
              $(IMGDIR)/error_full.png \
              $(IMGDIR)/error_histogram.png \
              $(IMGDIR)/error_spectrum.png \
              $(IMGDIR)/output_spectrum.png \
              $(IMGDIR)/mse_comparison.png

# ── 顶层目标 ──────────────────────────────────────────────────────────

.PHONY: all run clean distclean
.PHONY: build design filter analyze

all: run

run: analyze
	@echo "============================================"
	@echo "  全流程完成：设计 → 编译 → 滤波 → 分析"
	@echo "============================================"

# ── 步骤 1：Python 滤波器设计 + 量化 ──────────────────────────────────

design: $(DESIGN_OUT)

$(DESIGN_OUT): $(PYDIR)/design_filter.py
	@echo "[1/4] Python: 滤波器设计 + 数据量化..."
	@mkdir -p $(DATADIR) $(IMGDIR)
	$(CONDA_ACTIVATE) python3 $(PYDIR)/design_filter.py

# ── 步骤 2：C 编译 ───────────────────────────────────────────────────

build: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	@echo "[2/4] C: 链接..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR) $(BINDIR):
	mkdir -p $@

# ── 步骤 3：C 定点滤波 ───────────────────────────────────────────────

filter: build design $(FILTER_OUT)

$(FILTER_OUT): $(TARGET) $(DESIGN_OUT)
	@echo "[3/4] C: 定点 IIR 滤波 (Q14 + Q30)..."
	./$(TARGET) $(DATADIR)/input_16bit.raw $(DATADIR)/input_32bit.raw $(DATADIR)/coeffs.txt

# ── 步骤 4：Python 误差分析 + 绘图 ───────────────────────────────────

analyze: filter $(ANALYZE_OUT)

$(ANALYZE_OUT): $(PYDIR)/analyze_results.py $(FILTER_OUT) $(DATADIR)/output_python.raw
	@echo "[4/4] Python: 误差分析 + 绘图..."
	$(CONDA_ACTIVATE) python3 $(PYDIR)/analyze_results.py

# ── 清理 ─────────────────────────────────────────────────────────────

clean:
	rm -rf $(OBJDIR) $(BINDIR)

distclean: clean
	rm -rf $(DATADIR) $(IMGDIR)/*.png
	@echo "Cleaned all generated files."

# ── 帮助 ─────────────────────────────────────────────────────────────

help:
	@echo "Usage:"
	@echo "  make            — 运行全流程"
	@echo "  make build      — 仅编译 C 程序"
	@echo "  make design     — 仅 Python 滤波器设计"
	@echo "  make filter     — 仅 C 定点滤波"
	@echo "  make analyze    — 仅 Python 结果分析"
	@echo "  make clean      — 清理 C 编译产物"
	@echo "  make distclean  — 清理所有生成文件"
