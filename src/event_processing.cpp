#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "surface.h"

namespace py = pybind11;
using namespace ev;

class PyEROS : public EROS {
public:
    using EROS::EROS;

    // Process multiple events at once
    void update_batch(py::array_t<double> events) {
        auto buf = events.request();
        double* ptr = static_cast<double*>(buf.ptr);
        size_t num_events = buf.shape[0];

        for (size_t i = 0; i < num_events; ++i) {
            int x = static_cast<int>(ptr[i * 4]);
            int y = static_cast<int>(ptr[i * 4 + 1]);
            double ts = ptr[i * 4 + 2];
            int p = static_cast<int>(ptr[i * 4 + 3]);
            update(x, y, ts, p);
        }
    }

    py::array_t<uint8_t> getSurfacePy() {
        cv::Mat mat = getSurface();
        return py::array_t<uint8_t>({mat.rows, mat.cols}, mat.data);
    }
};

PYBIND11_MODULE(event_processing, m) {
    py::class_<PyEROS>(m, "EROS")
        .def(py::init<>())
        .def("init", &PyEROS::init)
        .def("update_batch", &PyEROS::update_batch, "Batch update from event list")
        .def("get_surface", &PyEROS::getSurfacePy)
        .def("temporal_decay", &PyEROS::temporalDecay)
        .def("spatial_decay", &PyEROS::spatialDecay);
}