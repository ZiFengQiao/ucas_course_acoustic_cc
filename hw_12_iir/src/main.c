/*
 * @Author: Wang, Qiaoyu
 * @Date: 2026-06-02 20:42:44
 * @LastEditors: Wang, Qiaoyu
 * @LastEditTime: 2026-06-02 22:01:51
 * @Description: 定点 IIR 滤波器主程序，命令行入口
 */

#include "iir_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 512

/** @brief 读取二进制文件到数组 */
static void *read_raw(const char *path, size_t elem_size, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    *out_n = (int)((size_t)bytes / elem_size);
    void *buf = malloc((size_t)bytes);
    if (!buf || fread(buf, 1, (size_t)bytes, f) != (size_t)bytes) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/**
 * @brief 定点 IIR 滤波器命令行入口
 *
 * 用法：./iir_filter <input_16bit.raw> <input_32bit.raw> <coeffs.txt>
 *
 * 流程：
 *   1. 读取 Python 预量化的输入文件（int16_t / int32_t）
 *   2. 解析系数文件，获取 Q 格式和系数
 *   3. 初始化滤波器描述符 IIRFilter16 / IIRFilter32
 *   4. 滤波并输出
 */
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_16bit.raw> <input_32bit.raw> <coeffs.txt>\n", argv[0]);
        return 1;
    }

    /* 1. 读取预量化输入 */
    int N16 = 0, N32 = 0;
    int16_t *x16 = (int16_t *)read_raw(argv[1], sizeof(int16_t), &N16);
    int32_t *x32 = (int32_t *)read_raw(argv[2], sizeof(int32_t), &N32);
    if (!x16 || !x32) { free(x16); free(x32); return 1; }
    printf("Loaded: %d samples (int16_t), %d samples (int32_t)\n", N16, N32);

    /* 2. 解析系数文件 */
    FILE *fc = fopen(argv[3], "r");
    if (!fc) { perror(argv[3]); free(x16); free(x32); return 1; }

    int16_t coeffs_flat16[IIR16_MAX_SECTIONS * 5];
    int32_t coeffs_flat32[IIR32_MAX_SECTIONS * 5];
    int num_sections = 0, coeff_q16 = 0, coeff_q32 = 0;
    char line[MAX_LINE];
    int current_section = -1;

    while (fgets(line, sizeof(line), fc)) {
        if (line[0] == '#') {
            if (strstr(line, "coeff_q_bits_16:"))
                coeff_q16 = atoi(strchr(line, ':') + 1);
            else if (strstr(line, "coeff_q_bits_32:"))
                coeff_q32 = atoi(strchr(line, ':') + 1);
            continue;
        }
        if (line[0] == '\n') continue;

        if (strncmp(line, "[Section", 8) == 0) {
            current_section = atoi(line + 9);
            if (current_section >= IIR16_MAX_SECTIONS) goto fail;
            if (current_section + 1 > num_sections)
                num_sections = current_section + 1;
        } else if (current_section >= 0 && strncmp(line, "q16:", 4) == 0) {
            int v[5];
            if (sscanf(line + 4, "%d %d %d %d %d", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5) {
                int base = current_section * 5;
                for (int k = 0; k < 5; k++)
                    coeffs_flat16[base + k] = (int16_t)v[k];
            }
        } else if (current_section >= 0 && strncmp(line, "q32:", 4) == 0) {
            int v[5];
            if (sscanf(line + 4, "%d %d %d %d %d", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5) {
                int base = current_section * 5;
                for (int k = 0; k < 5; k++)
                    coeffs_flat32[base + k] = (int32_t)v[k];
            }
        }
    }
    fclose(fc);

    if (num_sections == 0) { fprintf(stderr, "No sections\n"); goto fail; }
    printf("Parsed %d sections, coeff Q16=%d, Q32=%d\n",
           num_sections, coeff_q16, coeff_q32);

    /* 3. 初始化滤波器描述符 */
    IIRFilter16 f16;
    iir_filter_16_init(&f16, coeffs_flat16, num_sections, coeff_q16);

    IIRFilter32 f32;
    iir_filter_32_init(&f32, coeffs_flat32, num_sections, coeff_q32);

    /* 4. 16-bit 滤波 */
    printf("Running 16-bit filter (Q%d)…\n", coeff_q16);
    iir_filter_16_process_signal(&f16, x16, N16);
    {
        FILE *f = fopen("data/output_16bit.raw", "wb");
        if (f) { fwrite(x16, sizeof(int16_t), (size_t)N16, f); fclose(f); }
        printf("  -> data/output_16bit.raw (%d samples, int16_t Q%d)\n", N16, coeff_q16);
    }

    /* 5. 32-bit 滤波 */
    printf("Running 32-bit filter (Q%d)…\n", coeff_q32);
    iir_filter_32_process_signal(&f32, x32, N32);
    {
        FILE *f = fopen("data/output_32bit.raw", "wb");
        if (f) { fwrite(x32, sizeof(int32_t), (size_t)N32, f); fclose(f); }
        printf("  -> data/output_32bit.raw (%d samples, int32_t Q%d)\n", N32, coeff_q32);
    }

    free(x16);
    free(x32);
    printf("Done.\n");
    return 0;

fail:
    fclose(fc);
    free(x16);
    free(x32);
    return 1;
}
