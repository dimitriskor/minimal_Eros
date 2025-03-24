#pragma once

#include "surface.h"

namespace ev {

class EROS : public surface {
public:
    inline void update(int x, int y, double t = 0, int p = 0) override {
        static double odecay = pow(parameter, 1.0 / kernel_size);
        surf(cv::Rect(x, y, kernel_size, kernel_size)) *= odecay;
        surf.at<double>(y + half_kernel, x + half_kernel) = 255.0;
    }
};

}  // namespace ev
