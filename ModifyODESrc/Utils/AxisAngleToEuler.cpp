#include "AxisAngleToEuler.h"

AxisAngleToEuler::AxisAngleToEuler(std::vector<std::string> & euler_order_): euler_order(euler_order_) {}

void AxisAngleToEuler::convert_single(const double * action, double * euler)
{
    /* int tot_size = euler_order.size();
    int offset = 0;
    for(int i = 0; i < tot_size; i++)
    {
        std::string & s = euler_order[i];
        int curr_len = s.size();
        if (curr_len == 1)
        {
            euler[offset] = action[offset];
        }
        else if (curr_len == 3)
        {
            Eigen::Vector3d v(action[offset], action[offset+1], action[offset+2]);
            Eigen::AngleAxisd rot(v);
            Eigen::Matrix3d mat = rot.toRotationMatrix();
            Eigen::Vector3d e = mat.eulerAngles(s[2] - 'x', s[1] - 'x', s[0] - 'x');
            euler[offset] = e[0];
            euler[offset + 1] = e[1];
            euler[offset + 2] = e[2];
        }
        offset += curr_len;
    } */
}