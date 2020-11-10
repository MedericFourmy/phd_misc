filters.py: implementation of Position/Velocity KF and Complementary Filter fusing inertial frame 
acceleration and leg odometry

solo_filters_real_data.py: run KF and CF filters and store the results in a .npz file   
Change `DATA_FOLDER_RESULTS`, `DATA_FOLDER` and `data_file` variables to suit your needs

post_process_npz_traj.py: align position/orientation trajectories between the mocap/estimation using https://github.com/uzh-rpg/rpg_trajectory_evaluation.git, 
compute mocap velocity using a centered window filter (Savitzky-Golay) and save some plots
