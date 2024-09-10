Run file `youbots.ttt` in CoppeliaSim, and then run file `ThesisQPPINVJ.py` or `ThesisQPZNN1.py`.

`Coppelia.py` file contains the required code to connect Coppelia and also the two controller (classic and neural). 

`robotics.py` file has the youbot info, and the ZNN-TVQPEI neural controller class definition.

Once the simulation has finished, the logs are stored in a sqlite database.
The file `plotlog.py` can be used to plot the values stored, but it is just a helper script, there are a lot of considerations to take before use. 