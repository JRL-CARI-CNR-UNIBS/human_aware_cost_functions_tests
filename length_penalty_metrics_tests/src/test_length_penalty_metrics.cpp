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

  double solver_time;
  nh.getParam("solver_time",solver_time);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double t_r;
  nh.getParam("Tr",t_r);

  double min_distance;
  nh.getParam("min_distance",min_distance);

  double v_h;
  nh.getParam("v_h",v_h);

  int n_object;
  nh.getParam("n_object",n_object);

  int n_threads;
  nh.getParam("n_threads",n_threads);

  std::vector<double> start_configuration;
  nh.getParam("start_configuration",start_configuration);

  std::vector<double> stop_configuration;
  nh.getParam("stop_configuration",stop_configuration);

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  bool parallel_ssm;
  nh.getParam("parallel_ssm",parallel_ssm);

  bool display_tree;
  nh.getParam("display_tree",display_tree);

  bool use_vh;
  nh.getParam("use_vh",use_vh);

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);

  ssm15066_estimator::SSM15066EstimatorPtr ssm;
  if(parallel_ssm)
    ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator2D>(chain,max_step_size,n_threads);
  else
    ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,max_step_size);

  if(use_vh)
    ssm->setHumanVelocity(v_h,false);
  else
    ssm->setHumanVelocity(0.0,false);

  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(t_r,false);
  ssm->setMinDistance(min_distance,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

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
    obj.pose.pose.position.x = double (((float)rand()/(float)RAND_MAX)*(stop_configuration[0]-start_configuration[0])+start_configuration[0]);
    obj.pose.pose.position.y = double (((float)rand()/(float)RAND_MAX)*(stop_configuration[1]-start_configuration[1])+start_configuration[1]);
    obj.pose.pose.position.z = double (((float)rand()/(float)RAND_MAX)*(stop_configuration[2]-start_configuration[2])+start_configuration[2]);
    ROS_INFO_STREAM("obj x "<<obj.pose.pose.position.x <<" obj y "<<obj.pose.pose.position.y<<" obj z "<<obj.pose.pose.position.z);

    obj.pose.pose.orientation.w = 1.0;
    obj.pose.pose.orientation.x = 0.0;
    obj.pose.pose.orientation.y = 0.0;
    obj.pose.pose.orientation.z = 0.0;

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
  ros::Duration(1.0).sleep();
  display.clearMarkers();

  /* EUCLIDEAN PATH */

  pathplan::TreeSolverPtr solver = std::make_shared<pathplan::RRTStar>(metrics,checker,sampler);

  Eigen::VectorXd start_conf = Eigen::Map<Eigen::VectorXd>(start_configuration.data(), start_configuration.size());
  Eigen::VectorXd goal_conf  = Eigen::Map<Eigen::VectorXd>(stop_configuration .data(), stop_configuration .size());

  pathplan::NodePtr start_node = std::make_shared<pathplan::Node>(start_conf);
  pathplan::NodePtr goal_node  = std::make_shared<pathplan::Node>(goal_conf);

  pathplan::PathPtr solution;
  bool success = solver->computePath(start_node,goal_node,nh,solution,solver_time);

  if(success)
  {
    ROS_BOLDWHITE_STREAM("Euclidean metric -> Path found! Cost: "<<solution->cost());

    display.changeConnectionSize();
    display.changeNodeSize();
    display.displayPathAndWaypoints(solution,"pathplan",{0,0,1,1});

    int idx = 1;
    double human_aware_cost;
    double human_aware_path_cost = 0;
    for(pathplan::ConnectionPtr& c:solution->getConnections())
    {
      human_aware_cost = metrics_ha->cost(c->getParent(),c->getChild());
      human_aware_path_cost += human_aware_cost;
      ROS_BOLDWHITE_STREAM("conn "<<idx<<" length "<<c->norm()<<" human-aware cost "<<human_aware_cost);

      idx++;
    }
    ROS_BOLDWHITE_STREAM("Human-aware path cost -> "<<human_aware_path_cost);
  }
  else
    ROS_BOLDCYAN_STREAM("Euclidean metric -> No path found!");

  /* HUMAN-AWARE PATH */

  start_node->disconnect();
  goal_node->disconnect();

  solver = std::make_shared<pathplan::RRTStar>(metrics_ha,checker,sampler);
  success = solver->computePath(start_node,goal_node,nh,solution,solver_time);

  if(success)
  {
    ROS_BOLDCYAN_STREAM("Human-aware metric -> Path found! Cost: "<<solution->cost());
    if(display_tree)
      display.displayTree(solution->getTree(),"pathplan",{0,1,0,0.1});

    display.changeConnectionSize();
    display.changeNodeSize();
    display.displayPathAndWaypoints(solution,"pathplan",{0,1,0,1});

    int idx = 1;
    double human_aware_cost;
    for(pathplan::ConnectionPtr c: solution->getConnections())
    {
      human_aware_cost = metrics_ha->cost(c->getParent(),c->getChild());
      if(std::abs(c->getCost()-human_aware_cost)>1e-06)
        throw std::runtime_error("costs are different");

      ROS_BOLDCYAN_STREAM("conn "<<idx<<" length "<<c->norm()<<" human-aware cost "<<human_aware_cost<<" lambda "<<(human_aware_cost/c->norm()));
      idx++;
    }

    ROS_BOLDCYAN_STREAM("Euclidean metric "<<solution->computeEuclideanNorm());

    if(not use_vh)
    {
      ROS_BOLDYELLOW_STREAM("Revaluate path cost with human velocity set to param value");
      ssm->setHumanVelocity(v_h);

      idx = 1;
      for(pathplan::ConnectionPtr c: solution->getConnections())
      {
        human_aware_cost = metrics_ha->cost(c->getParent(),c->getChild());
        c->setCost(human_aware_cost);

        ROS_BOLDYELLOW_STREAM("conn "<<idx<<" length "<<c->norm()<<" VH human-aware cost "<<human_aware_cost<<" lambda "<<(human_aware_cost/c->norm()));
        idx++;
      }

      ROS_BOLDYELLOW_STREAM("Path cost considering human velocity -> "<<solution->cost());
    }
  }
  else
    ROS_BOLDCYAN_STREAM("Human-aware -> No path found!");

  return 0;
}
