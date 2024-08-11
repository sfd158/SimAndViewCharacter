#pragma once
#include <Eigen/Eigen>

class AxisAngleToEuler
{
    public:
        std::vector<std::string> euler_order;
    public:
        AxisAngleToEuler(std::vector<std::string> & euler_order_);
        void convert_single(const double * action, double * euler);
};