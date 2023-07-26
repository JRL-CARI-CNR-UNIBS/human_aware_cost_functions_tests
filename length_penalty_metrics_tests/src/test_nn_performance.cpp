#include <nn_ssm/neural_networks/neural_network.h>
#include <nn_ssm/ssm15066_estimators/ssm15066_estimatorNN.h>
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
#include <object_loader_msgs/RemoveObjects.h>
#include <ssm15066_estimators/parallel_ssm15066_estimator2D.h>
#include <random>
#include <graph_core/solvers/birrt.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "crash_test_replanner");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  ros::NodeHandle nh;

  ros::ServiceClient ps_client=nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
  ros::ServiceClient add_obj=nh.serviceClient<object_loader_msgs::AddObjects>("add_object_to_scene");
  ros::ServiceClient remove_obj=nh.serviceClient<object_loader_msgs::RemoveObjects>("remove_object_from_scene");

  //  ////////////////////////////////////////// GET ROS PARAM ///////////////////////////////////////////////
  int n_iter, n_objects;
  std::vector<std::string> poi_names;
  std::vector<double> init_start_configuration, init_stop_configuration, end_start_configuration, end_stop_configuration, obs_max_range, obs_min_range;
  std::string group_name, base_frame, tool_frame, object_type, nn_namespace;
  double max_step_size, max_time, max_distance, max_cart_acc, tr, min_distance, v_h;

  nh.getParam("n_iter",n_iter);
  nh.getParam("max_time",max_time);
  nh.getParam("group_name",group_name);
  nh.getParam("init_start_configuration",init_start_configuration);
  nh.getParam("init_stop_configuration",init_stop_configuration);
  nh.getParam("end_start_configuration",end_start_configuration);
  nh.getParam("end_stop_configuration",end_stop_configuration);
  nh.getParam("max_distance",max_distance);
  nh.getParam("base_frame",base_frame);
  nh.getParam("tool_frame",tool_frame);
  nh.getParam("object_type",object_type);
  nh.getParam("max_step_size",max_step_size);
  nh.getParam("max_cart_acc",max_cart_acc);
  nh.getParam("Tr",tr);
  nh.getParam("min_distance",min_distance);
  nh.getParam("v_h",v_h);
  nh.getParam("n_object",n_objects);
  nh.getParam("poi_names",poi_names);
  nh.getParam("obj_max_range",obs_max_range);
  nh.getParam("obj_min_range",obs_min_range);
  nh.getParam("namespace",nn_namespace);

  //  ///////////////////////////////////UPLOAD THE ROBOT ARM/////////////////////////////////////////////////////////////
  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);

  const robot_state::JointModelGroup* joint_model_group = move_group.getCurrentState()->getJointModelGroup(group_name);
  std::vector<std::string> joint_names = joint_model_group->getActiveJointModelNames();

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

  //  /////////////////////////////////////PLANNING SCENE SERVICES////////////////////////////////////
  moveit_msgs::GetPlanningScene ps_srv;
  if (!ps_client.waitForExistence(ros::Duration(10)))
  {
    ROS_ERROR("unable to connect to /get_planning_scene");
    return 1;
  }

  if(not add_obj.waitForExistence(ros::Duration(10)))
  {
    ROS_FATAL("srv not found");
    return 1;
  }

  if(not remove_obj.waitForExistence(ros::Duration(10)))
  {
    ROS_FATAL("srv not found");
    return 1;
  }

  if (!ps_client.call(ps_srv))
  {
    ROS_ERROR("call to srv not ok");
    return 1;
  }

  if (!planning_scene->setPlanningSceneMsg(ps_srv.response.scene))
  {
    ROS_ERROR("unable to update planning scene");
    return 1;
  }

  // ///////////////////////////////////////////////PLAN//////////////////////////////////////////////////////////
  std::string last_link=planning_scene->getRobotModel()->getJointModelGroup(group_name)->getLinkModelNames().back();
  pathplan::DisplayPtr disp = std::make_shared<pathplan::Display>(planning_scene,group_name,last_link);


  ssm15066_estimator::SSM15066EstimatorPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,max_step_size);
  ssm->setHumanVelocity(v_h,false);
  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(tr,false);
  ssm->setMinDistance(min_distance,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

  neural_network::NeuralNetworkPtr nn = std::make_shared<neural_network::NeuralNetwork>();
  ROS_INFO("Importing neural network");
  nn->importFromParam(nh,nn_namespace);

  Eigen::Vector3d obs_eigen_min_range = Eigen::Map<Eigen::Vector3d>(obs_min_range.data(), obs_min_range.size());
  Eigen::Vector3d obs_eigen_max_range = Eigen::Map<Eigen::Vector3d>(obs_max_range.data(), obs_max_range.size());
  ssm15066_estimator::SSM15066EstimatorNNPtr ssm_nn = std::make_shared<ssm15066_estimator::SSM15066EstimatorNN>(chain,nn,obs_eigen_min_range,obs_eigen_max_range);

  Eigen::VectorXd scale; scale.setOnes(lb.rows(),1);
  pathplan::MetricsPtr metrics_ssm = std::make_shared<pathplan::LengthPenaltyMetrics>(ssm,scale);
  pathplan::MetricsPtr metrics_nn = std::make_shared<pathplan::LengthPenaltyMetrics>(ssm_nn,scale);
  pathplan::MetricsPtr metrics = std::make_shared<pathplan::Metrics>();

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name);
  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb,ub,lb,ub);

  Eigen::VectorXd init_start_conf = Eigen::Map<Eigen::VectorXd>(init_start_configuration.data(), init_start_configuration.size());
  Eigen::VectorXd init_goal_conf  = Eigen::Map<Eigen::VectorXd>(init_stop_configuration .data(), init_stop_configuration .size());

  Eigen::VectorXd end_start_conf = Eigen::Map<Eigen::VectorXd>(end_start_configuration.data(), end_start_configuration.size());
  Eigen::VectorXd end_goal_conf  = Eigen::Map<Eigen::VectorXd>(end_stop_configuration .data(), end_stop_configuration .size());

  Eigen::VectorXd delta_start = (end_start_conf - init_start_conf)/(std::max(n_iter-1,1));
  Eigen::VectorXd delta_goal  = (end_goal_conf  - init_goal_conf )/(std::max(n_iter-1,1));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  double distance,x,y,z;
  Eigen::VectorXd start_conf, goal_conf;
  pathplan::PathPtr path;
  pathplan::NodePtr start_node, goal_node;
  pathplan::BiRRTPtr solver = std::make_shared<pathplan::BiRRT>(metrics,checker,sampler);
  solver->setMaxDistance(max_distance);

  ros::WallDuration(5).sleep();

  std::vector<double> err;

  for(int i=0; i<n_iter; i++)
  {
    start_conf = init_start_conf+i*delta_start;
    goal_conf  = init_goal_conf +i*delta_goal ;

    ROS_INFO("---------------------------------------------------------------------------------------------------------");
    distance = (goal_conf-start_conf).norm();
    ROS_INFO_STREAM("Iter n: "<<std::to_string(i)<<" start: "<<start_conf.transpose()<< " goal: "<<goal_conf.transpose()<< " distance: "<<distance);


    // ADDING OBSTACLES
    object_loader_msgs::AddObjects add_srv;
    object_loader_msgs::RemoveObjects remove_srv;

    ssm->clearObstaclesPositions();
    for(unsigned int i=0;i<n_objects;i++)
    {
      Eigen::Vector3d obs_location;
      x = dist(gen); obs_location[0] = obs_min_range.at(0)+(obs_max_range.at(0)-obs_min_range.at(0))*x;
      y = dist(gen); obs_location[1] = obs_min_range.at(1)+(obs_max_range.at(1)-obs_min_range.at(1))*y;
      z = dist(gen); obs_location[2] = obs_min_range.at(2)+(obs_max_range.at(2)-obs_min_range.at(2))*z;

      object_loader_msgs::Object obj;
      obj.object_type=object_type;
      obj.pose.header.frame_id=base_frame;
      obj.pose.pose.position.x = obs_location[0];
      obj.pose.pose.position.y = obs_location[1];
      obj.pose.pose.position.z = obs_location[2];

      obj.pose.pose.orientation.w = 1.0;
      obj.pose.pose.orientation.x = 0.0;
      obj.pose.pose.orientation.y = 0.0;
      obj.pose.pose.orientation.z = 0.0;

      add_srv.request.objects.push_back(obj);

      ssm->addObstaclePosition(obs_location);
      ssm_nn->addObstaclePosition(obs_location);
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
    else
    {
      for(const std::string& str:add_srv.response.ids)
        remove_srv.request.obj_ids.push_back(str);
    }

    if(not ps_client.call(ps_srv))
    {
      ROS_ERROR("call to srv not ok");
      return 1;
    }

    if(not planning_scene->setPlanningSceneMsg(ps_srv.response.scene))
    {
      ROS_ERROR("unable to update planning scene");
      return 1;
    }
    checker->setPlanningSceneMsg(ps_srv.response.scene);

    // PLAN
    start_node = std::make_shared<pathplan::Node>(start_conf);
    goal_node  = std::make_shared<pathplan::Node>(goal_conf );

    disp->changeNodeSize();
    disp->displayNode(start_node,"pathplan",{0.0,1.0,0.0,1.0});
    disp->displayNode(goal_node,"pathplan",{0.0,1.0,0.0,1.0});
    disp->defaultNodeSize();

    sampler = std::make_shared<pathplan::InformedSampler>(start_conf,goal_conf,lb,ub);
    solver->setSampler(sampler);

    unsigned int max_iter = std::numeric_limits<double>::infinity();

    if(solver->computePath(start_node,goal_node,nh,path, max_time, max_iter))
    {
      disp->displayPathAndWaypoints(path,"pathplan");

      double cost_ssm, cost_nn, conn_cost_ssm, conn_cost_nn, lambda, lambda_nn;
      cost_ssm = cost_nn = 0.0;

      unsigned int t = 0;
      for(const pathplan::ConnectionPtr& c:path->getConnections())
      {
        conn_cost_ssm = metrics_ssm->cost(c->getParent(),c->getChild());
        conn_cost_nn = metrics_nn->cost(c->getParent(),c->getChild());
        cost_ssm += conn_cost_ssm;
        cost_nn += conn_cost_nn;
        lambda = conn_cost_ssm/c->norm();
        lambda_nn = conn_cost_nn/c->norm();

        err.push_back(std::abs(conn_cost_ssm-conn_cost_nn));
        ROS_INFO_STREAM("Connection "<<t<<" | length "<<c->norm()<<" | lambda ssm "<<lambda<<" | lambda nn "<<lambda_nn<<" | err "<<err.back());
        t++;
      }

      ROS_INFO_STREAM("Path length "<<path->computeEuclideanNorm());
      ROS_INFO_STREAM("Path cost ssm "<<cost_ssm);
      ROS_INFO_STREAM("Path cost nn "<<cost_nn);
    }

    ROS_INFO("PRESS A KEY");
    std::cin.ignore();
    disp->clearMarkers();

    if (not remove_obj.call(remove_srv))
      ROS_ERROR("call to remove obj srv not ok");
    if(not remove_srv.response.success)
      ROS_ERROR("remove obj srv error");
  }

  double mean = std::accumulate(err.begin(),err.end(),0.0)/err.size();

  double std_dev = 0.0;
  for(const double& d:err)
    std_dev+= std::pow((d-mean),2);

  std_dev = std::sqrt(std_dev/err.size());

  ROS_WARN_STREAM("mean -> "<<mean<<" std dev -> "<<std_dev);

  return 0;
}
