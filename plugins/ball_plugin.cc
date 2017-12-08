#include "ball_plugin.h"
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <sdf/sdf.hh>
#include "vector3d.pb.h"
#include <cstdlib>

gazebo::msgs::Vector3d req;
gazebo::transport::NodePtr node;
gazebo::transport::PublisherPtr pub;
static bool connected;

void ball_plugin_init(void) {
  //gazebo::client::setup();
  gazebo::transport::init();
  gazebo::transport::run();
  node = gazebo::transport::NodePtr(new gazebo::transport::Node());
  node->Init("default");

  std::string topic = "~/ball_plugin/set_ball_pose";
  pub = node->Advertise<gazebo::msgs::Vector3d>(topic);
  pub->WaitForConnection();
  connected = true;
}

void ball_plugin_destroy(void) {
  //gazebo::client::shutdown();
  gazebo::transport::fini();
  connected = false;
}

void ball_plugin_setPosition(double x, double y, double z) {
  gazebo::msgs::Set((gazebo::msgs::Vector3d *)&req,
      ignition::math::Vector3d(x, y, z));
  if (connected) {
    pub->Publish(req);
  }
}
