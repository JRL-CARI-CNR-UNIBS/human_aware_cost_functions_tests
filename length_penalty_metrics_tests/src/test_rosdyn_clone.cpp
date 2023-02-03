#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <rosdyn_core/primitives.h>
#include <graph_core/informed_sampler.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_rosdyn_clone");
  ros::NodeHandle nh;

  ros::AsyncSpinner aspin(4);
  aspin.start();

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string group_name;
  nh.getParam("group_name",group_name);

  int n_tests;
  nh.getParam("n_tests",n_tests);

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();

  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();

  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);

  Eigen::VectorXd dq_max(dof);
  Eigen::VectorXd dq_min(dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }

    if(bounds.velocity_bounded_)
    {
      dq_max(idx) = bounds.max_velocity_;
      dq_min(idx) = bounds.min_velocity_;
    }
    else
    {
      dq_max(idx) = +10.0;
      dq_min(idx) = -10.0;
    }
  }

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);
  rosdyn::ChainPtr cloned_chain = chain->clone();

  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);
  pathplan::SamplerPtr dq_sampler = std::make_shared<pathplan::InformedSampler>(dq_min, dq_max, dq_min, dq_max);


  bool progress_bar_full = false;
  unsigned int progress = 0;
  Eigen::VectorXd q, dq;

  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> poses, cloned_poses;
  std::vector< Eigen::Vector6d, Eigen::aligned_allocator<Eigen::Vector6d>>twists, cloned_twists;

  ros::Duration(5).sleep();

  for(unsigned int i=0;i<n_tests;i++)
  {
    q = sampler->sample();
    dq = dq_sampler->sample();

    poses = chain->getTransformations(q);
    cloned_poses = cloned_chain->getTransformations(q);

    for(unsigned int j=0;j<poses.size();j++)
    {
      if(poses[j].matrix() != cloned_poses[j].matrix())
      {
        ROS_ERROR_STREAM("Poses matrix \n "<<poses[j].matrix());
        ROS_ERROR_STREAM("Cloned poses matrix \n "<<cloned_poses[j].matrix());

        throw std::runtime_error("cloned and poses matrix different");
      }
    }

    twists = chain->getTwist(q,dq);
    cloned_twists = cloned_chain->getTwist(q,dq);

    for(unsigned int j=0;j<twists.size();j++)
    {
      if(twists[j] != cloned_twists[j])
      {
        ROS_ERROR_STREAM("Twist matrix \n "<<twists[j].transpose());
        ROS_ERROR_STREAM("Cloned twist matrix \n "<<cloned_twists[j].transpose());

        throw std::runtime_error("cloned and twist matrix different");
      }
    }

    progress = std::ceil(((double)(i+1.0))/((double)n_tests)*100.0);
    if(progress%5 == 0 && not progress_bar_full)
    {
      std::string output = "\r[";

      for(unsigned int j=0;j<progress/5.0;j++)
        output = output+"=";

      output = output+">] ";
      output = "\033[1;37;42m"+output+std::to_string(progress)+"%\033[0m";  //1->bold, 37->foreground white, 42->background green

      if(progress == 100)
      {
        progress_bar_full = true;
        output = output+"\033[1;5;32m Succesfully completed!\033[0m\n";
      }

      std::cout<<output;
    }
  }
  return 0;
}
