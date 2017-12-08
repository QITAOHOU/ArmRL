#pragma once

#include <vector>

extern "C" {

  void ball_plugin_init(void);
  void ball_plugin_destroy(void);
  void ball_plugin_setPositions(double x, double y, double z);

}
