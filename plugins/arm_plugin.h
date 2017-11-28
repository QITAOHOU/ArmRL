#pragma once

#include <vector>
#include <armadillo>

extern "C" {

  void arm_plugin_init(void);
  void arm_plugin_destroy(void);
  void arm_plugin_setPositions(
      double x1, double y1, double z1,
      double x2, double y2, double z2,
      double x3, double y3, double z3,
      double x4, double y4, double z4,
      double x5, double y5, double z5,
      double x6, double y6, double z6,
      double x7, double y7, double z7
    );

}
