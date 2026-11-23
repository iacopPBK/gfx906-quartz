// Image resampling shared by the encoders.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qz_img {

// Pillow's separable bicubic resampling in 22-bit fixed point, the weights of
// img_tool::resize_pillow in mtmd-image.cpp without its other filters.
inline void resize_bicubic(const unsigned char * src, int sw, int sh, std::vector<unsigned char> & dst, int dw, int dh) {
    const int PB = 22;
    auto filt = [](double x) {
        constexpr double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    auto clip8 = [](int v) -> unsigned char { return (unsigned char) (v < 0 ? 0 : v > 255 ? 255 : v); };
    auto weights = [&](int in, int out, std::vector<int> & bounds, std::vector<int32_t> & w) {
        double scale = (double) in / out, fs = scale < 1.0 ? 1.0 : scale;
        const double support = 2.0 * fs;
        const int ks = (int) std::ceil(support) * 2 + 1;
        std::vector<double> pw((size_t) out * ks);
        bounds.resize((size_t) out * 2);
        for (int xx = 0; xx < out; xx++) {
            const double center = (xx + 0.5) * scale, ss = 1.0 / fs;
            double ww = 0.0;
            int xmin = (int) (center - support + 0.5); if (xmin < 0) xmin = 0;
            int xmax = (int) (center + support + 0.5); if (xmax > in) xmax = in;
            xmax -= xmin;
            int x = 0;
            for (; x < xmax; x++) { const double v = filt((x + xmin - center + 0.5) * ss); pw[(size_t) xx * ks + x] = v; ww += v; }
            for (x = 0; x < xmax; x++) if (ww != 0.0) pw[(size_t) xx * ks + x] /= ww;
            for (; x < ks; x++) pw[(size_t) xx * ks + x] = 0;
            bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
        }
        w.resize((size_t) out * ks);
        const double fx = std::ldexp(1.0, PB);
        for (size_t i = 0; i < w.size(); i++) w[i] = (int32_t) (pw[i] * fx + (pw[i] < 0 ? -0.5 : 0.5));
        return ks;
    };
    std::vector<unsigned char> cur(src, src + (size_t) sw * sh * 3);
    int cw = sw;
    if (dw != sw) {
        std::vector<int> bd; std::vector<int32_t> k;
        const int ks = weights(sw, dw, bd, k);
        std::vector<unsigned char> o((size_t) dw * sh * 3);
        for (int yy = 0; yy < sh; yy++)
            for (int xx = 0; xx < dw; xx++) {
                const int xmin = bd[xx * 2], xc = bd[xx * 2 + 1];
                const int32_t * kk = &k[(size_t) xx * ks];
                const unsigned char * p = &cur[((size_t) yy * cw + xmin) * 3];
                int32_t s0 = 1 << (PB - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xc; x++, p += 3) { s0 += p[0] * kk[x]; s1 += p[1] * kk[x]; s2 += p[2] * kk[x]; }
                unsigned char * q = &o[((size_t) yy * dw + xx) * 3];
                q[0] = clip8(s0 >> PB); q[1] = clip8(s1 >> PB); q[2] = clip8(s2 >> PB);
            }
        cur.swap(o); cw = dw;
    }
    if (dh != sh) {
        std::vector<int> bd; std::vector<int32_t> k;
        const int ks = weights(sh, dh, bd, k);
        const size_t re = (size_t) cw * 3;
        std::vector<unsigned char> o(re * dh);
        std::vector<int32_t> acc(re);
        for (int yy = 0; yy < dh; yy++) {
            const int ymin = bd[yy * 2], yc = bd[yy * 2 + 1];
            const int32_t * kk = &k[(size_t) yy * ks];
            std::fill(acc.begin(), acc.end(), 1 << (PB - 1));
            for (int y = 0; y < yc; y++) {
                const unsigned char * r = &cur[(size_t) (ymin + y) * re];
                for (size_t i = 0; i < re; i++) acc[i] += r[i] * kk[y];
            }
            for (size_t i = 0; i < re; i++) o[(size_t) yy * re + i] = clip8(acc[i] >> PB);
        }
        cur.swap(o);
    }
    dst.swap(cur);
}


}   // namespace qz_img
