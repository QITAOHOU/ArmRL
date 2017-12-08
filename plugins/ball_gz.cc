#include "ball_gz.h"
#include <vector>
#include <armadillo>
#include <iostream>
#include <cmath>

using namespace gazebo;

BallPlugin::BallPlugin(void) : WorldPlugin() {
}

BallPlugin::~BallPlugin(void) {
}

void BallPlugin::Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf) {
  this->model = _parent;
  this->node = transport::NodePtr(new transport::Node());
  this->node->Init(this->model->Name());

  std::string topic = "/gazebo/default/ball_plugin/set_ball_pose";
  this->sub = this->node->Subscribe(topic, &BallPlugin::OnMsg, this);

  this->CreateModel();
}

void BallPlugin::Update(void) {
}

void BallPlugin::CreateModel(void) {
  std::string modelStr =
    "<sdf version='1.6'>\
      <model name='ballmodel'>\
        <pose frame=''>0 0 0 0 0 0</pose>\
        <link name='link'>\
          <pose>0 0 0 0 0 0</pose>\
          <visual name='visual'>\
            <geometry>\
              <sphere>\
                <radius>0.241</radius>\
              </cylinder>\
            </geometry>\
            <material>\
              <script>\
                <name>Gazebo/Orange</name>\
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
}

void BallPlugin::OnMsg(ConstVector3dPtr &_msg) {
  this->model->ModelByName("ballmodel")->
    SetWorldPose(ignition::math::Pose3d(
          ignition::math::Vector3d(_msg->x(), _msg->y(), _msg->z()),
          ignition::math::Quaterniond(0, 0, 0)));
}
