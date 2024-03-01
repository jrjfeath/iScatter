#!/usr/bin/env python3

def analyse_output(sc,trajectories):
   for i in range(trajectories):
      #Reads all xyz and velocities from file and analyses them:
      xf = 'outputs/out_'+str(i)+'.md.xyz'
      vf = 'outputs/out_'+str(i)+'.md.vel'
      sc.ReadSamples(xf,vf)
      # write sample log for each dynamics separately
      open("outputs/dynamics"+str(i)+".analinfo", "w").writelines(sc.slog)
      sc.slog = []

