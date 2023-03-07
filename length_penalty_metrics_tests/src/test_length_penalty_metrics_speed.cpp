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
#include <ssm15066_estimators/parallel_ssm15066_estimator2D.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_length_penalty_metrics");
  ros::NodeHandle nh;

  ros::AsyncSpinner aspin(4);
  aspin.start();

  srand((unsigned int)time(NULL));

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

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double tr;
  nh.getParam("Tr",tr);

  double min_distance;
  nh.getParam("min_distance",min_distance);

  double v_h;
  nh.getParam("v_h",v_h);

  double max_step_size;
  nh.getParam("ssm_max_step_size",max_step_size);

  double max_distance;
  nh.getParam("max_distance",max_distance);

  int n_tests;
  nh.getParam("n_tests",n_tests);

  int n_object;
  nh.getParam("n_object",n_object);

  int n_threads;
  nh.getParam("ssm_n_threads",n_threads);

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

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);
  ssm15066_estimator::SSM15066Estimator2DPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,max_step_size);
  ssm15066_estimator::ParallelSSM15066Estimator2DPtr parallel_ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator2D>(chain,max_step_size,n_threads);

  ssm->setHumanVelocity(v_h,false);
  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(tr,false);
  ssm->setMinDistance(min_distance,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

  parallel_ssm->setHumanVelocity(v_h,false);
  parallel_ssm->setMaxCartAcc(max_cart_acc,false);
  parallel_ssm->setReactionTime(tr,false);
  parallel_ssm->setMinDistance(min_distance,false);
  parallel_ssm->setPoiNames(poi_names);
  parallel_ssm->updateMembers();

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

  bool progress_bar_full = false;
  unsigned int progress = 0;

  ros::Duration(5).sleep();

  for(unsigned int i=0;i<n_tests;i++)
  {
    parent = sampler->sample();
    child  = sampler->sample();

    distance = (child-parent).norm();

    if(distance>max_distance || distance<0.1)
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

    if(std::abs(cost_ha-cost_parallel_ha)>1e-04)
    {
      ROS_INFO_STREAM("cost ha "<<cost_ha<<" cost parallel ha "<<cost_parallel_ha);
      throw std::runtime_error("costs are different");
    }

    progress = std::ceil(((double)(i+1.0))/((double)n_tests)*100.0);
    if(progress%5 == 0 && not progress_bar_full)
    {
      std::string output = "\r[";

      for(unsigned int j=0;j<progress/5.0;j++)
        output = output+"=";

      output = output+">] ";
      output = "\033[1;41;42m"+output+std::to_string(progress)+"%\033[0m";  //1->bold, 37->foreground white, 42->background green

      if(progress == 100)
      {
        progress_bar_full = true;
        output = output+"\033[1;5;32m Succesfully completed!\033[0m\n";
      }

      std::cout<<output;
    }
  }

  //Mean
  double sum_time_ha          = std::accumulate(time_ha         .begin(),time_ha         .end(),0.0);
  double sum_time_euclidean   = std::accumulate(time_euclidean  .begin(),time_euclidean  .end(),0.0);
  double sum_time_ha_parallel = std::accumulate(time_ha_parallel.begin(),time_ha_parallel.end(),0.0);

  double mean_time_ha          = sum_time_ha         /time_ha         .size();
  double mean_time_euclidean   = sum_time_euclidean  /time_euclidean  .size();
  double mean_time_ha_parallel = sum_time_ha_parallel/time_ha_parallel.size();

  //Stdev
  double accum = 0.0;
  std::for_each (std::begin(time_ha), std::end(time_ha), [&](const double d) {
      accum += (d - mean_time_ha) * (d - mean_time_ha);
  });

  double stdev_time_ha = sqrt(accum/(time_ha.size()-1));

  accum = 0.0;
  std::for_each (std::begin(time_euclidean), std::end(time_euclidean), [&](const double d) {
      accum += (d - mean_time_euclidean) * (d - mean_time_euclidean);
  });

  double stdev_time_euclidean = sqrt(accum/(time_euclidean.size()-1));

  accum = 0.0;
  std::for_each (std::begin(time_ha_parallel), std::end(time_ha_parallel), [&](const double d) {
      accum += (d - mean_time_ha_parallel) * (d - mean_time_ha_parallel);
  });

  double stdev_time_ha_parallel = sqrt(accum/(time_ha_parallel.size()-1));

  ROS_BOLDWHITE_STREAM("Mean Euclidean metric: "  <<mean_time_euclidean  <<" s"<<" stdev "<<stdev_time_euclidean  );
  ROS_BOLDCYAN_STREAM ("Mean HA metric: "         <<mean_time_ha         <<" s"<<" stdev "<<stdev_time_ha         );
  ROS_BOLDGREEN_STREAM("Mean HA-parallel metric: "<<mean_time_ha_parallel<<" s"<<" stdev "<<stdev_time_ha_parallel);

  ROS_BOLDRED_STREAM("HA is "         <<(mean_time_ha/mean_time_euclidean)          <<" time slower than Euclidean metrics");
  ROS_BOLDRED_STREAM("HA-parallel is "<<(mean_time_ha_parallel/mean_time_euclidean)<<" time slower than Euclidean metrics" );
  ROS_BOLDRED_STREAM("HA-parallel is "<<(mean_time_ha_parallel/mean_time_ha)       <<" time slower than HA metrics"        );

  return 0;
}
