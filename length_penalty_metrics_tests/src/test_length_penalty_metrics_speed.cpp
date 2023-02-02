#include <ros/ros.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <graph_core/graph/graph_display.h>
#include <graph_core/solvers/rrt_star.h>
#include <length_penalty_metrics.h>
#include <object_loader_msgs/AddObjects.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_length_penalty_metrics");
  ros::NodeHandle nh;

  ros::AsyncSpinner aspin(4);
  aspin.start();

  ros::ServiceClient ps_client = nh.serviceClient<moveit_msgs::GetPlanningScene> ("/get_planning_scene");
  ros::ServiceClient add_obj   = nh.serviceClient<object_loader_msgs::AddObjects>("add_object_to_scene");

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string object_type;
  nh.getParam("object_type",object_type);

  double max_step_size;
  nh.getParam("max_step_size",max_step_size);

  double max_distance;
  nh.getParam("max_distance",max_distance);

  int n_tests;
  nh.getParam("n_tests",n_tests);

  int n_object;
  nh.getParam("n_object",n_object);

  int n_threads;
  nh.getParam("n_threads",n_threads);

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);

  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();

  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }
  }

  ros::Duration(3).sleep();

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);

  ssm15066_estimator::SSM15066EstimatorPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator>(robot_model_loader.getURDF(),base_frame,tool_frame,max_step_size);
  ssm15066_estimator::ParallelSSM15066EstimatorPtr parallel_ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator>(robot_model_loader.getURDF(),base_frame,tool_frame,max_step_size,n_threads);

//  ssm15066_estimator::SSM15066EstimatorPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator>(chain,max_step_size);
//  ssm15066_estimator::ParallelSSM15066EstimatorPtr parallel_ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator>(chain,max_step_size,n_threads);

  if(not add_obj.waitForExistence(ros::Duration(10)))
  {
    ROS_FATAL("srv not found");
    return 1;
  }

  object_loader_msgs::Object obj;
  Eigen::Vector3d obstacle_position;
  object_loader_msgs::AddObjects add_srv;

  obj.object_type=object_type;
  obj.pose.header.frame_id=base_frame;

  obj.pose.pose.orientation.w = 1.0;
  obj.pose.pose.orientation.x = 0.0;
  obj.pose.pose.orientation.y = 0.0;
  obj.pose.pose.orientation.z = 0.0;

  for(unsigned int i=0;i<n_object;i++)
  {
    obj.pose.pose.position.x = double (((float)rand()/(float)RAND_MAX)*(ub[0]-lb[0])+lb[0]);
    obj.pose.pose.position.y = double (((float)rand()/(float)RAND_MAX)*(ub[1]-lb[1])+lb[1]);
    obj.pose.pose.position.z = double (((float)rand()/(float)RAND_MAX)*(ub[2]-lb[2])+lb[2]);

    add_srv.request.objects.push_back(obj);

    obstacle_position[0] = obj.pose.pose.position.x;
    obstacle_position[1] = obj.pose.pose.position.y;
    obstacle_position[2] = obj.pose.pose.position.z;

    ssm->addObstaclePosition(obstacle_position);
    parallel_ssm->addObstaclePosition(obstacle_position);
  }

  if(not add_obj.call(add_srv))
  {
    ROS_ERROR("call to srv not ok");
    return 1;
  }

  if(not add_srv.response.success)
  {
    ROS_ERROR("srv error");
    return 1;
  }

  if(not ps_client.waitForExistence(ros::Duration(10)))
  {
    ROS_ERROR("unable to connect to /get_planning_scene");
    return 1;
  }

  moveit_msgs::GetPlanningScene ps_srv;
  if(not ps_client.call(ps_srv))
  {
    ROS_ERROR("call to srv not ok");
    return 1;
  }

  if(not planning_scene->setPlanningSceneMsg(ps_srv.response.scene))
  {
    ROS_ERROR("unable to update planning scene");
    return 0;
  }

  pathplan::MetricsPtr metrics=std::make_shared<pathplan::Metrics>();
  pathplan::MetricsPtr metrics_ha=std::make_shared<pathplan::LengthPenaltyMetrics>(ssm);
  pathplan::MetricsPtr metrics_parallel_ha=std::make_shared<pathplan::LengthPenaltyMetrics>(parallel_ssm);

  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);

  ros::WallTime tic;
  Eigen::VectorXd parent, child;
  double distance, cost_ha, cost_parallel_ha;
  std::vector<double> time_euclidean, time_ha, time_ha_parallel;

  for(unsigned int i=0;i<n_tests;i++)
  {
    parent = sampler->sample();

    while(true)
    {
      child  = sampler->sample();
      distance = (child-parent).norm();

      if(distance>0.1)
        break;
    }

    if(distance>max_distance)
      child = parent+(child-parent)*max_distance/distance;


    tic = ros::WallTime::now();
    metrics->cost(parent,child);
    time_euclidean.push_back((ros::WallTime::now()-tic).toSec());

    tic = ros::WallTime::now();
    cost_ha = metrics_ha->cost(parent,child);
    time_ha.push_back((ros::WallTime::now()-tic).toSec());

    tic = ros::WallTime::now();
    cost_parallel_ha = metrics_parallel_ha->cost(parent,child);
    time_ha_parallel.push_back((ros::WallTime::now()-tic).toSec());


    if(std::abs((cost_ha-cost_parallel_ha))>1e-06)
    {
      ROS_ERROR_STREAM("Iter "<<i<<" NOT OK -> connection length "<<(parent-child).norm());

      ROS_ERROR_STREAM("cost ha "<<cost_ha<<" cost parallel ha "<<cost_parallel_ha);
      ROS_WARN("Repeating the computation with verbose set 1 -> find the q for which cost is different");

      ssm->setVerbose(1);
      parallel_ssm->setVerbose(1);

      ROS_ERROR_STREAM("cost ha "<<metrics_ha->cost(parent,child));
      ros::Duration(1.0).sleep();
      ROS_ERROR_STREAM("cost // ha "<<metrics_parallel_ha->cost(parent,child));

      ROS_WARN("Now check poses and twists of those q");

      ssm->setVerbose(2);
      parallel_ssm->setVerbose(2);

      ROS_ERROR_STREAM("cost ha "<<metrics_ha->cost(parent,child));
      ros::Duration(1.0).sleep();
      ROS_ERROR_STREAM("cost // ha "<<metrics_parallel_ha->cost(parent,child));

      throw std::runtime_error("cost should be equal");
    }
    else
      ROS_INFO_STREAM("Iter "<<i<<" OK -> connection length "<<(parent-child).norm());
  }

  double average_time_ha          = std::accumulate(time_ha         .begin(),time_ha         .end(),0.0)/time_ha         .size();
  double average_time_euclidean   = std::accumulate(time_euclidean  .begin(),time_euclidean  .end(),0.0)/time_euclidean  .size();
  double average_time_ha_parallel = std::accumulate(time_ha_parallel.begin(),time_ha_parallel.end(),0.0)/time_ha_parallel.size();

  ROS_BOLDWHITE_STREAM("Average time to compute Euclidean metric: "  <<average_time_euclidean  <<" s");
  ROS_BOLDCYAN_STREAM ("Average time to compute HA metric: "         <<average_time_ha         <<" s");
  ROS_BOLDGREEN_STREAM("Average time to compute HA-parallel metric: "<<average_time_ha_parallel<<" s");

  ROS_BOLDRED_STREAM("HA is "         <<(average_time_ha/average_time_euclidean)          <<" time slower than Euclidean metrics");
  ROS_BOLDRED_STREAM("HA-parallel is "<<(average_time_ha_parallel/average_time_euclidean)<<" time slower than Euclidean metrics" );
  ROS_BOLDRED_STREAM("HA-parallel is "<<(average_time_ha_parallel/average_time_ha)       <<" time slower than HA metrics"        );

  return 0;
}
