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

  ros::ServiceClient ps_client=nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
  ros::ServiceClient add_obj=nh.serviceClient<object_loader_msgs::AddObjects>("add_object_to_scene");

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

  int n_object;
  nh.getParam("n_object",n_object);

  int n_threads;
  nh.getParam("n_threads",n_threads);

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);
  ssm15066_estimator::ParallelSSM15066EstimatorPtr ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator>(chain,max_step_size,n_threads);

  if(not add_obj.waitForExistence(ros::Duration(10)))
  {
    ROS_FATAL("srv not found");
    return 1;
  }

  object_loader_msgs::AddObjects add_srv;

  for(unsigned int i=0;i<n_object;i++)
  {
    object_loader_msgs::Object obj;
    obj.object_type=object_type;
    obj.pose.header.frame_id=base_frame;
    obj.pose.pose.position.x = (rand()%(8 + 1))/10;
    obj.pose.pose.position.y = (rand()%(8 + 1))/10;
    obj.pose.pose.position.z = (rand()%(8 + 1))/10;

    add_srv.request.objects.push_back(obj);

    Eigen::Vector3d obstacle_position;
    obstacle_position[0] = obj.pose.pose.position.x;
    obstacle_position[1] = obj.pose.pose.position.y;
    obstacle_position[2] = obj.pose.pose.position.z;

    ssm->addObstaclePosition(obstacle_position);
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

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name, n_threads);

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
  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);
  pathplan::Display display(planning_scene,group_name);
  display.clearMarkers();

  /* EUCLIDEAN PATH */

  pathplan::TreeSolverPtr solver = std::make_shared<pathplan::RRTStar>(metrics,checker,sampler);

  Eigen::Vector3d start_conf;
  Eigen::Vector3d goal_conf;
  start_conf = {0.0,0.0,0.0};
  goal_conf = {0.8,0.8,0.8};

  pathplan::NodePtr start_node = std::make_shared<pathplan::Node>(start_conf);
  pathplan::NodePtr goal_node  = std::make_shared<pathplan::Node>(goal_conf);

  pathplan::PathPtr solution;
  bool success = solver->computePath(start_node, goal_node, nh, solution);

  if(success)
  {
    ROS_INFO_STREAM("Euclidean metric -> Path found!\n"<<*solution);

    display.changeConnectionSize();
    display.changeNodeSize();
    display.displayPathAndWaypoints(solution,"pathplan",{0,0,1,1});
  }
  else
    ROS_INFO("Euclidean metric -> No path found!");

  /* HUMAN-AWARE PATH */

  solver->resetProblem();
  start_node->disconnect();
  goal_node->disconnect();

  solver->setMetrics(metrics_ha);

  success = solver->computePath(start_node,goal_node,nh,solution);

  if(success)
  {
    ROS_INFO_STREAM("Human-aware metric -> Path found!\n"<<*solution);
    display.displayTree(solution->getTree());

    display.changeConnectionSize();
    display.changeNodeSize();
    display.displayPathAndWaypoints(solution,"pathplan",{1,0,0,1});
  }
  else
    ROS_INFO("Human-aware -> No path found!");

  return 0;
}
