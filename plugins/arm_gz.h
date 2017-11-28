#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <boost/shared_ptr.hpp>
#include "arm_pose_request.pb.h"

namespace gazebo {

  typedef const boost::shared_ptr<
    const arm_msgs::msgs::ArmPoseRequest>
      ArmPoseRequestPtr;

  class ArmPlugin : public WorldPlugin {
    public:
      ArmPlugin(void);
      ~ArmPlugin(void);

      void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf);

    private:
      physics::WorldPtr model;
      std::vector<event::ConnectionPtr> connections;

      transport::NodePtr node;
      transport::SubscriberPtr sub;

      void Update(void);
      void OnMsg(ArmPoseRequestPtr &_msg);
      void CreateModel(void);
  };

  GZ_REGISTER_WORLD_PLUGIN(ArmPlugin);
}
