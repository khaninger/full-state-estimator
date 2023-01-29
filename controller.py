import pickle
from constraint import *

class Controller():
    def __init__(self, c_set = 'constraints_set.pkl'):
        self.load_constraints(c_set)
        self.init_robot()
        self.loop()

        self.tcp_to_obj = None # Pose of object in TCP frame
        
    def load_constraints(self, c_file):
        with f = open(c_file):
            self.c_set = pickle.load(f)

    def init_robot(self):
        # Initialize robot, throwing error if failed
        
    def get_robot_data(self):
        # Get the TCP pose and forces from robot
        return x_tcp, f

    def get_object_data(self):
        # Get the object coordinates
        x_tcp, f = self.get_robot_data()
        
        return x_obj, f
    
    def loop(self):
        # Control loop, runs til kill
        while(True):
            x_tcp,f = self.get_robot_data()
            constraint_mode = self.c_set.id_constraint(x, f)
            

            
        



if __name__ == 'main':
    print("starting controller")
