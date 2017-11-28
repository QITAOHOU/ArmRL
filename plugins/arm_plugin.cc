#include "arm_plugin.h"
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <sdf/sdf.hh>
#include "arm_pose_request.pb.h"
#include "vector3d.pb.h"
#include <cstdlib>

arm_msgs::msgs::ArmPoseRequest req;
gazebo::transport::NodePtr node;
gazebo::transport::PublisherPtr pub;
static bool connected;

void arm_plugin_init(void) {
  //gazebo::client::setup();
  gazebo::transport::init();
  gazebo::transport::run();
  node = gazebo::transport::NodePtr(new gazebo::transport::Node());
  node->Init("default");

  std::string topic = "~/arm_plugin/set_arm_joint_pose";
  pub = node->Advertise<arm_msgs::msgs::ArmPoseRequest>(topic);
  pub->WaitForConnection();
  connected = true;
}

void arm_plugin_destroy(void) {
  //gazebo::client::shutdown();
  gazebo::transport::fini();
  connected = false;
}

void arm_plugin_setPositions(
    double x1, double y1, double z1,
    double x2, double y2, double z2,
    double x3, double y3, double z3,
    double x4, double y4, double z4,
    double x5, double y5, double z5,
    double x6, double y6, double z6,
    double x7, double y7, double z7) {

  arma::mat pos = arma::mat({
      { x1, y1, z1 },
      { x2, y2, z2 },
      { x3, y3, z3 },
      { x4, y4, z4 },
      { x5, y5, z5 },
      { x6, y6, z6 },
      { x7, y7, z7 } });

  std::vector<gazebo::msgs::Vector3d *> jointPos({
      req.mutable_joint1(), req.mutable_joint2(), req.mutable_joint3(),
      req.mutable_joint4(), req.mutable_joint5(), req.mutable_joint6(),
      req.mutable_joint7() });

  for (size_t i = 0; i < jointPos.size(); i++) {
    jointPos[i]->set_x(pos(i, 0));
    jointPos[i]->set_y(pos(i, 1));
    jointPos[i]->set_z(pos(i, 2));
  }

  if (connected) {
    pub->Publish(req);
  }
}
