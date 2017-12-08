#include "arm_gz.h"
#include <vector>
#include <armadillo>
#include <iostream>
#include <cmath>

using namespace gazebo;

ArmPlugin::ArmPlugin(void) : WorldPlugin() {
}

ArmPlugin::~ArmPlugin(void) {
}

void ArmPlugin::Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf) {
  this->model = _parent;
  this->node = transport::NodePtr(new transport::Node());
  this->node->Init(this->model->Name());

  std::string topic = "/gazebo/default/arm_plugin/set_arm_joint_pose";
  this->sub = this->node->Subscribe(topic, &ArmPlugin::OnMsg, this);

  this->CreateModel();
}

void ArmPlugin::Update(void) {
}

void ArmPlugin::CreateModel(void) {
  for (int i = 0; i < 7; i++) {
    std::string modelStr =
      "<sdf version='1.6'>\
        <model name='armlink" + std::to_string(i + 1) + "'>\
          <pose frame=''>0 0 0 0 0 0</pose>\
          <link name='link'>\
            <pose>0 0 0 0 0 0</pose>\
            <visual name='visual'>\
              <geometry>\
                <cylinder>\
                  <radius>0.1</radius>\
                  <length>0.001</length>\
                </cylinder>\
              </geometry>\
              <material>\
                <script>\
                  <name>Gazebo/Grey</name>\
                  <uri>file://media/materials/scripts/gazebo.material</uri>\
                </script>\
              </material>\
              <transparency>0</transparency>\
              <cast_shadows>1</cast_shadows>\
            </visual>\
          </link>\
          <static>1</static>\
        </model>\
      </sdf>";
    sdf::SDF linkSDF;
    linkSDF.SetFromString(modelStr);
    this->model->InsertModelSDF(linkSDF);
    modelStr =
      "<sdf version='1.6'>\
        <model name='armjoint" + std::to_string(i + 1) + "'>\
          <pose frame=''>0 0 0 0 0 0</pose>\
          <link name='link'>\
            <pose>0 0 0 0 0 0</pose>\
            <visual name='visual'>\
              <geometry>\
                <sphere>\
                  <radius>0.15</radius>\
                </sphere>\
              </geometry>\
              <material>\
                <script>\
                  <name>Gazebo/SkyBlue</name>\
                  <uri>file://media/materials/scripts/gazebo.material</uri>\
                </script>\
              </material>\
              <transparency>0</transparency>\
              <cast_shadows>1</cast_shadows>\
            </visual>\
          </link>\
          <static>1</static>\
        </model>\
      </sdf>";
    sdf::SDF jointSDF;
    jointSDF.SetFromString(modelStr);
    this->model->InsertModelSDF(jointSDF);
  }
}

void ArmPlugin::OnMsg(ConstArmPoseRequestPtr &_msg) {
  // grab the poses from the message
  std::vector<arma::vec> pose(8);
  pose[0] = arma::vec({ 0, 0, 0 });
  pose[1] = arma::vec({ _msg->joint1().x(), _msg->joint1().y(), _msg->joint1().z() });
  pose[2] = arma::vec({ _msg->joint2().x(), _msg->joint2().y(), _msg->joint2().z() });
  pose[3] = arma::vec({ _msg->joint3().x(), _msg->joint3().y(), _msg->joint3().z() });
  pose[4] = arma::vec({ _msg->joint4().x(), _msg->joint4().y(), _msg->joint4().z() });
  pose[5] = arma::vec({ _msg->joint5().x(), _msg->joint5().y(), _msg->joint5().z() });
  pose[6] = arma::vec({ _msg->joint6().x(), _msg->joint6().y(), _msg->joint6().z() });
  pose[7] = arma::vec({ _msg->joint7().x(), _msg->joint7().y(), _msg->joint7().z() });

  for (size_t i = 0; i < pose.size() - 1; i++) {
    arma::vec mid = (pose[i] + pose[i + 1]) / 2.0;
    arma::vec end = pose[i + 1];
    double len = arma::norm(pose[i + 1] - pose[i]);
    arma::vec dir = (pose[i + 1] - pose[i]) / len;

    double yaw = atan2(dir(1), dir(0));
    double pitch = atan2(sqrt(dir(0) * dir(0) + dir(1) * dir(1)), dir(2));
    double roll = 0; // we dont actually care about the roll since they are cylinders

    this->model->ModelByName("armlink" + std::to_string(i + 1))->
      SetWorldPose(ignition::math::Pose3d(
          ignition::math::Vector3d(mid(0), mid(1), mid(2)),
          ignition::math::Quaterniond(roll, pitch, yaw)));
    this->model->ModelByName("armlink" + std::to_string(i + 1))->
      SetScale(ignition::math::Vector3d({ 1, 1, len / 0.001 }), true);
    this->model->ModelByName("armjoint" + std::to_string(i + 1))->
      SetWorldPose(ignition::math::Pose3d(
          ignition::math::Vector3d(end(0), end(1), end(2)),
          ignition::math::Quaterniond(0, 0, 0)));
  }
}
