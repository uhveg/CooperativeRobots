Run file `FourYoubot.ttt` in CoppeliaSim, and then run file `MainController.py`.

`Coppelia.py` file contains the required code to connect Coppelia and also the controller. 

`robotics.py` file has the youbot info, and the ZNN-TVQPEI neural controller class definition.

Once the simulation has finished, the logs are stored in a sqlite database, that can be latter plot with the script `plotlogQP.py`, like:

- `python.exe plotlogQP.py logs.db images test`

the first argument is the database, the second is the folder used to store the results and the last one is the tablename (this is choosen in `MainController.py`)