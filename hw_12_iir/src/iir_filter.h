/*
 * @Author: Wang, Qiaoyu
 * @Date: 2026-06-02 20:42:11
 * @LastEditors: Wang, Qiaoyu
 * @LastEditTime: 2026-06-03 21:56:30
 * @Description: 定点 IIR 滤波器头文件
 */

#ifndef IIR_FILTER_H
#define IIR_FILTER_H

#include <stdint.h>

// 4阶IIR，两个二阶节
#define IIR16_MAX_SECTIONS 2   /**< 16-bit 滤波器最大二阶节数 */
#define IIR32_MAX_SECTIONS 2   /**< 32-bit 滤波器最大二阶节数 */

/* ── 16-bit 类型 ───────────────────────────────────────────────────── */

/** @brief 单个双二阶节系数（Q 格式，a0=1 隐含） */
typedef struct {
    int16_t b0, b1, b2;     /**< 前馈系数（分子） */
    int16_t a1, a2;         /**< 反馈系数（分母） */
} BiquadCoeffs16;

/** @brief 单个双二阶节状态变量（直接II型转置） */
typedef struct {
    int16_t s1;             /**< 第一个延迟单元 */
    int16_t s2;             /**< 第二个延迟单元 */
} BiquadState16;

/** @brief 16-bit 定点 IIR 滤波器描述符 */
typedef struct {
    BiquadCoeffs16 sections[IIR16_MAX_SECTIONS];  /**< 各二阶节系数 */
    BiquadState16  states[IIR16_MAX_SECTIONS];    /**< 各二阶节状态 */
    int16_t num_sections;   /**< 实际二阶节数量 */
    int16_t Q;              /**< 系数量化小数位数 */
    int64_t clamp_min;      /**< 饱和下限 */
    int64_t clamp_max;      /**< 饱和上限 */
} IIRFilter16;

/* ── 32-bit 类型 ───────────────────────────────────────────────────── */

/** @brief 单个双二阶节系数（Q 格式，a0=1 隐含） */
typedef struct {
    int32_t b0, b1, b2;     /**< 前馈系数（分子） */
    int32_t a1, a2;         /**< 反馈系数（分母） */
} BiquadCoeffs32;

/** @brief 单个双二阶节状态变量（直接II型转置） */
typedef struct {
    int32_t s1;             /**< 第一个延迟单元 */
    int32_t s2;             /**< 第二个延迟单元 */
} BiquadState32;

/** @brief 32-bit 定点 IIR 滤波器描述符 */
typedef struct {
    BiquadCoeffs32 sections[IIR32_MAX_SECTIONS];  /**< 各二阶节系数 */
    BiquadState32  states[IIR32_MAX_SECTIONS];    /**< 各二阶节状态 */
    int32_t num_sections;   /**< 实际二阶节数量 */
    int32_t Q;              /**< 系数量化小数位数 */
    int64_t clamp_min;      /**< 饱和下限 */
    int64_t clamp_max;      /**< 饱和上限 */
} IIRFilter32;

/* ── 初始化接口 ────────────────────────────────────────────────────── */

/**
 * @brief 初始化 16-bit IIR 滤波器
 *
 * @param f            滤波器描述符指针
 * @param coeffs_flat  扁平系数数组 [b0,b1,b2,a1,a2, b0,b1,b2,a1,a2, ...]
 * @param num_sections 二阶节数量
 * @param Q            系数量化小数位数
 */
void iir_filter_16_init(IIRFilter16 *f, const int16_t *coeffs_flat,
                        int num_sections, int Q);

/**
 * @brief 初始化 32-bit IIR 滤波器
 *
 * @param f            滤波器描述符指针
 * @param coeffs_flat  扁平系数数组 [b0,b1,b2,a1,a2, ...]
 * @param num_sections 二阶节数量
 * @param Q            系数量化小数位数
 */
void iir_filter_32_init(IIRFilter32 *f, const int32_t *coeffs_flat,
                        int num_sections, int Q);

/* ── 逐采样处理 ────────────────────────────────────────────────────── */

/**
 * @brief 16-bit 滤波器处理单个采样
 *
 * @param f 滤波器描述符（状态就地更新）
 * @param x 输入采样（int16_t，Q 格式数据）
 * @return 滤波后输出采样
 */
int16_t iir_filter_16_process(IIRFilter16 *f, int16_t x);

/**
 * @brief 32-bit 滤波器处理单个采样
 *
 * @param f 滤波器描述符（状态就地更新）
 * @param x 输入采样（int32_t，Q 格式数据）
 * @return 滤波后输出采样
 */
int32_t iir_filter_32_process(IIRFilter32 *f, int32_t x);

/* ── 批量处理 ──────────────────────────────────────────────────────── */

/**
 * @brief 16-bit 滤波器处理整段信号（就地）
 *
 * @param f      滤波器描述符
 * @param signal 信号数组（int16_t），滤波结果写回
 * @param n      采样点数
 */
void iir_filter_16_process_signal(IIRFilter16 *f, int16_t *signal, int n);

/**
 * @brief 32-bit 滤波器处理整段信号（就地）
 *
 * @param f      滤波器描述符
 * @param signal 信号数组（int32_t），滤波结果写回
 * @param n      采样点数
 */
void iir_filter_32_process_signal(IIRFilter32 *f, int32_t *signal, int n);

#endif /* IIR_FILTER_H */
