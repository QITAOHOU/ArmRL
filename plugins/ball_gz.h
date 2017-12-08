#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <boost/shared_ptr.hpp>

namespace gazebo {
  class BallPlugin : public WorldPlugin {
    public:
      BallPlugin(void);
      ~BallPlugin(void);

      void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf);

    private:
      physics::WorldPtr model;
      std::vector<event::ConnectionPtr> connections;

      transport::NodePtr node;
      transport::SubscriberPtr sub;

      void Update(void);
      void OnMsg(ConstVector3dPtr &_msg);
      void CreateModel(void);
  };

  GZ_REGISTER_WORLD_PLUGIN(BallPlugin);
}
