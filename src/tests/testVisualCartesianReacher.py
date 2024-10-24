import os
import sys
sys.path.append("src/") 
sys.path.append('../')
sys.path.append('.')

from src.bendRL_env.VisualCartesianReacherFiveJointsGoal import VisualReacherFiveJoints

env = VisualReacherFiveJoints()

env.reset()
env.render()

#show move to goal
env.control.moveJ(env.GOAL_COORD)

#move back to start
env.reset()
env.render()


for i in range(5):
    env.step(0)
    env.render()

for i in range(20):
    env.step(env.action_space.sample())
    env.render()

env.close()