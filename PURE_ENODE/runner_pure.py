import torch, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'
torch.set_default_dtype(torch.float32)

import envs
import ctrl.ctrl as base
from ctrl import utils_pure
import wandb


################## environment and dataset ##################
dt      = 0.1 		# mean-time difference between observations
noise   = 0.0 		# observation noise std
ts_grid = 'fixed' 	# the distribution for the observation time differences: ['fixed','uniform','exp']
ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
env = ENV_CLS(dt=dt, obs_trans=True, device=device, obs_noise=noise, ts_grid=ts_grid, solver='dopri5')
D = utils_pure.collect_data(env, H=5.0, N=env.N0)


################## model ##################
dynamics = 'enode'   		# ensemble of neural ODEs
# dynamics = 'benode' 		# batch ensemble of neural ODEs
# dynamics = 'ibnode'		# implicit BNN ODEs
# dynamics = 'pets'	   		# PETS
# dynamics = 'deep_pilco' 	# deep PILCO
n_ens       = 10			# ensemble size
nl_f        = 3				# number of hidden layers in the differential function
nn_f        = 200			# number of hidden neurons in each hidden layer of f
act_f       = 'elu'			# activation of f (should be smooth)
dropout_f   = 0.05			# dropout parameter (needed only for deep pilco)
learn_sigma = False			# whether to learn the observation noise or keep it fixed
nl_g        = 2				# number of hidden layers in the policy function
nn_g        = 200			# number of hidden neurons in each hidden layer of g
act_g       = 'relu'		# activation of g
nl_V        = 2				# number of hidden layers in the state-value function
nn_V        = 200			# number of hidden neurons in each hidden layer of V
act_V       = 'tanh'		# activation of V (should be smooth)

ctrl = base.CTRL(env, dynamics, n_ens=n_ens, learn_sigma=learn_sigma,
                 nl_f=nl_f, nn_f=nn_f, act_f=act_f, dropout_f=dropout_f,
                 nl_g=nl_g, nn_g=nn_g, act_g=act_g, 
                 nl_V=nl_V, nn_V=nn_V, act_V=act_V).to(device)

print('Env dt={:.3f}\nObservation noise={:.3f}\nTime increments={:s}'.\
      format(env.dt,env.obs_noise,str(env.ts_grid)))

################## wandb log ##################
# Create a formatted run name
run_name = (f"dt_{dt}_noise_{noise}_ts_{ts_grid}_env_{ENV_CLS.__name__}_dynamics_{dynamics}_n_ens_{n_ens}_doubled")
wandb_run = wandb.init(project="oderl", name=run_name)

################## learning ##################
utils_pure.plot_model(ctrl, D, L=30, H=2.0, rep_buf=10, fname=ctrl.name+'-train.png',wandb_run=wandb_run)
utils_pure.plot_test(ctrl, D, L=30, H=2.0, N=5, fname=ctrl.name+'-test.png', wandb_run=wandb_run)

utils_pure.train_loop(ctrl, D, ctrl.name, 30, wandb_run=wandb_run, L=30, H=2.0)
wandb_run.finish()

# ctrl.save(D=D,fname=ctrl.name)				# save the model & dataset
# ctrl_,D_ = base.CTRL.load(env, f'{fname}')	# load the model & dataset



















