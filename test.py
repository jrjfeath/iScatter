#!/usr/bin/env python3
def generate_conditions(sc, trajectories=10):
    # generate Distribuitions
    sc.InitialDist()
    # if already generated distribuitions previously (takes a few minutes), then just load 
    #sc.loaddists()
    # generate samples
    sc.GenSamples(N=trajectories)
    #write log into some file
    open(sc.filename + '.logfile','w').writelines(sc.log)
    return sc.fileout

