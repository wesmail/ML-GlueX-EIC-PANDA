#!/usr/bin/python
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_event():
    # z values are straw layer positions as in PANDA FTS
    z = np.array([294.895, 295.77 , 309.39 , 310.265, 326.895, 327.77 , 341.39 ,
        342.265, 393.995, 394.87 , 422.965, 423.84 , 437.49 , 438.365,
        466.965, 467.84 , 606.995, 607.87 , 621.49 , 622.365, 638.995,
        639.87 , 653.49 , 654.365])
    # six chambers   
    chamber_id = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6])  
    # generate random number of tracks per event
    # at least 2 and maximum 8
    ntracks = np.random.randint(2, 8, 1)[0]

    # generate slope and intercept from gaus distribution
    m = np.random.normal(loc=0, scale=0.1, size=ntracks)
    b = np.random.normal(loc=1, scale=0.1, size=ntracks)
    
    df = []
    for i in range(0,ntracks):
        # straight line tracks
        x = (m[i]*z + b[i])

        particle = pd.DataFrame()
        particle['x'] = np.round(x,3)
        particle['z'] = z
        particle['chamber_id']  = chamber_id
        particle['layer_id']    = np.arange(1,25)
        particle['particle_id'] = i

        df.append(particle)

    df = pd.concat(df, ignore_index=True)
    df['hit_id'] = np.arange(0,24*ntracks)

    # rearrange
    df = df[['hit_id','x','z','chamber_id','layer_id','particle_id']]

    return df

# main function
def main(argv):   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', type=int, required=True,  help="number of events")
    parser.add_argument('-f', type=str, required=False, help="output file", default='dummy_events.csv')    
    args = parser.parse_args()

    start_time = time.time()

    out = []
    for i in tqdm(range(int(args.n))):
        # generate one event
        event = generate_event()
        event['event_id'] = i
        event = event[['event_id','hit_id','x','z','chamber_id','layer_id','particle_id']]
        
        out.append(event)

    out = pd.concat(out, ignore_index=True)
    out.to_csv(args.f)

    end_time = time.time()
    print ('time :%.2f'%(end_time-start_time)+' sec')

# run main function
if __name__== '__main__':
    warnings.filterwarnings('ignore') 
    main(sys.argv[1:]) 