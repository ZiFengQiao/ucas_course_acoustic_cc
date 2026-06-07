/*
 * @Author: Wang, Qiaoyu
 * @Date: 2026-06-02 20:42:44
 * @LastEditors: Wang, Qiaoyu
 * @LastEditTime: 2026-06-03 21:56:22
 * @Description: 定点 IIR 滤波器实现
 */

#include "iir_filter.h"
#include <string.h>

/* ── 内部工具函数 ──────────────────────────────────────────────────── */

/** @brief 带四舍五入的右移 */
static int64_t rshift_round(int64_t val, int Q) {
    return (val + ((int64_t)1 << (Q - 1))) >> Q;
}

/** @brief 饱和限幅 */
static int64_t sat(int64_t val, int64_t lo, int64_t hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

/* ── 16-bit 实现 ───────────────────────────────────────────────────── */

void iir_filter_16_init(IIRFilter16 *f, const int16_t *coeffs_flat,
                        int num_sections, int Q) {
    memset(f, 0, sizeof(*f));
    f->num_sections = (int16_t)num_sections;
    f->Q            = (int16_t)Q;
    f->clamp_min    = INT16_MIN;
    f->clamp_max    = INT16_MAX;

    for (int s = 0; s < num_sections; s++) {
        const int16_t *c = coeffs_flat + s * 5;
        f->sections[s].b0 = c[0];
        f->sections[s].b1 = c[1];
        f->sections[s].b2 = c[2];
        f->sections[s].a1 = c[3];
        f->sections[s].a2 = c[4];
    }
}

// 输出值与输入的Q格式相同
int16_t iir_filter_16_process(IIRFilter16 *f, int16_t x) {
    int64_t xn = (int64_t)x;
    int Q = f->Q;
    int64_t lo = f->clamp_min, hi = f->clamp_max;

    for (int s = 0; s < f->num_sections; s++) {
        BiquadCoeffs16 *c = &f->sections[s];
        BiquadState16  *st = &f->states[s];

        /* y[n] = b0*x[n] + s1 */
        int64_t acc = (int64_t)c->b0 * xn + ((int64_t)st->s1 << Q);
        int64_t yn = sat(rshift_round(acc, Q), lo, hi); 
        // Q14乘完之后，是Qin + 14，右移丢弃14个小数，并限制饱和值，结果是Qin

        /* s1_next = b1*x[n] - a1*y[n] + s2 */
        acc = (int64_t)c->b1 * xn - (int64_t)c->a1 * yn + ((int64_t)st->s2 << Q);
        st->s1 = (int16_t)sat(rshift_round(acc, Q), lo, hi);

        /* s2_next = b2*x[n] - a2*y[n] */
        acc = (int64_t)c->b2 * xn - (int64_t)c->a2 * yn;
        st->s2 = (int16_t)sat(rshift_round(acc, Q), lo, hi);

        xn = yn;
    }
    return (int16_t)xn;
}

void iir_filter_16_process_signal(IIRFilter16 *f, int16_t *signal, int n) {
    for (int i = 0; i < n; i++)
        signal[i] = iir_filter_16_process(f, signal[i]);
}

/* ── 32-bit 实现 ───────────────────────────────────────────────────── */

void iir_filter_32_init(IIRFilter32 *f, const int32_t *coeffs_flat,
                        int num_sections, int Q) {
    memset(f, 0, sizeof(*f));
    f->num_sections = num_sections;
    f->Q            = Q;
    f->clamp_min    = INT32_MIN;
    f->clamp_max    = INT32_MAX;

    for (int s = 0; s < num_sections; s++) {
        const int32_t *c = coeffs_flat + s * 5;
        f->sections[s].b0 = c[0];
        f->sections[s].b1 = c[1];
        f->sections[s].b2 = c[2];
        f->sections[s].a1 = c[3];
        f->sections[s].a2 = c[4];
    }
}

int32_t iir_filter_32_process(IIRFilter32 *f, int32_t x) {
    int64_t xn = (int64_t)x;
    int Q = f->Q;
    int64_t lo = f->clamp_min, hi = f->clamp_max;

    for (int s = 0; s < f->num_sections; s++) {
        BiquadCoeffs32 *c = &f->sections[s];
        BiquadState32  *st = &f->states[s];

        /* y[n] = b0*x[n] + s1 */
        int64_t acc = (int64_t)c->b0 * xn + ((int64_t)st->s1 << Q);
        int64_t yn = sat(rshift_round(acc, Q), lo, hi);

        /* s1_next = b1*x[n] - a1*y[n] + s2 */
        acc = (int64_t)c->b1 * xn - (int64_t)c->a1 * yn + ((int64_t)st->s2 << Q);
        st->s1 = (int32_t)sat(rshift_round(acc, Q), lo, hi);

        /* s2_next = b2*x[n] - a2*y[n] */
        acc = (int64_t)c->b2 * xn - (int64_t)c->a2 * yn;
        st->s2 = (int32_t)sat(rshift_round(acc, Q), lo, hi);

        xn = yn;
    }
    return (int32_t)xn;
}

void iir_filter_32_process_signal(IIRFilter32 *f, int32_t *signal, int n) {
    for (int i = 0; i < n; i++)
        signal[i] = iir_filter_32_process(f, signal[i]);
}
