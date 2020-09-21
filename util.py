
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_events(path, chamber_id=None, start=0, end=-1):
    '''
    Load events from a csv file
    Arguments :
    1. path (str) : input .csv file name
    2. chamber_id (str) : which chambers to selec e.g. '1,2'
    3. start (int) : start event index
    4. end (int) : end event index
    '''
    evs = pd.read_csv(path, index_col=0)

    if chamber_id is not None:
        id1, id2 = chamber_id.split(',')
        evs = evs.loc[(evs.chamber_id==int(id1)) | (evs.chamber_id==int(id2))]
    print ('Stations selected are: ', np.unique(evs.chamber_id))
    
    evs = evs.assign( a = np.arctan2(evs.x,evs.z))
    evs = evs[['event_id','hit_id','x','z','a','chamber_id','layer_id','particle_id']]

    gb = evs.groupby('event_id')
    dfs = [gb.get_group(x) for x in gb.groups]        
    return dfs[start:end]

def plot_event(event, para='z,x'):
    '''
    Plot an event as a scatter plot
    Arguments :
    1. event (dataframe) : An event dataframe
    2. para  (str) : Which observables to plot, seperated by comma
    '''    
    pids = np.unique(event.particle_id)
    x, y = para.split(',')
    for pid in pids:
        df = event.loc[event.particle_id==pid]
        plt.scatter(df[str(x)], df[str(y)], s=50)
    plt.xlabel(str(x), fontsize=15)
    plt.ylabel(str(y), fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


# a function to create hit_pairs
def create_pairs(events):
    '''
    Create Hit Pairs
    Arguments :
    1. events (list ) : A list of events
    '''
    X = []
    y = []
    print ('Creating hit pairs ...')
    for event in tqdm(range(len(events))):          
        df = events[event]
        if df.empty: continue
        df = df[['hit_id','x','z','a','layer_id','particle_id']]
        data = np.array(df)

        for h_1 in data:
            for h_2 in data:
                if (h_1[0]==h_2[0]): continue
                if (abs(h_1[4]-h_2[4])!=1): continue

                X.append([h_1[1]/10., h_1[2]/100., h_1[3]*10,
                          h_2[1]/10., h_2[2]/100., h_2[3]*10])
                y.append([int(h_1[5]==h_2[5])])


    X = np.vstack(X)
    y = np.vstack(y)

    return X, y

# predict function for hit_pairs
def predict_pairs(data, model, cut=0.5):
    '''
    A function used to build hit-pairs and use the model to predict
    Arguments :
    1. data (dataframe) : An input event dataframe
    2. model (keras model) : A trained model for hit_pairs
    3. cut (float) : A threshold applied to the network output
    '''
    df = data.copy()
    df = df[['hit_id','x','z','a','layer_id','particle_id']]
    data = np.array(df)
    
    X   = []
    ids = []

    for i in data:
        for j in data:
            if (i[0]==j[0]): continue
            if (abs(i[4]-j[4])!=1): continue

            X.append([i[1]/10., i[2]/100., i[3]*10,
                      j[1]/10., j[2]/100., j[3]*10])
            ids.append([i[0],j[0]])
            
    X   = np.vstack(X)
    ids = np.vstack(ids)
    
    yhat = model.predict(X)
    # building probability table
    p_t  = np.hstack([ids, yhat])
    
    
    List = []
    hit_list = df.hit_id.values
    remove_list = set()
    
    for hit_id in df.hit_id:
        lists = []
        if hit_id in remove_list: continue
        lists.append(hit_id)
        remove_list.add(hit_id)
        
        for h1, h2, prob in p_t:
            if prob>=cut:
                if len(lists)==1:
                    if not h1==hit_id or h2==hit_id: continue
                    lists.append(h1)
                    lists.append(h2)
                
                    remove_list.add(h1)
                    remove_list.add(h2)
                
                elif (h1 in lists) | (h2 in lists):
                    lists.append(h1)
                    lists.append(h2)
                
                    remove_list.add(h1)
                    remove_list.add(h2)                
                
                else:
                    continue
    
        List.append(set(lists))
    return List

# a function to create hit triplets
def create_triplets(events):
    '''
    Create Hit Triplets
    Arguments :
    1. events (list ) : A list of events    
    '''
    X = []
    y = []
    print ('Creating hit pairs ...')
    for event in tqdm(range(len(events))):          
        df = events[event]
        if df.empty: continue
        df = df[['hit_id','x','z','a','layer_id','particle_id']]
        data = np.array(df)

        for h_1 in data:
            for h_2 in data:
                if (h_1[0]==h_2[0]): continue
                if (abs(h_1[4]-h_2[4])!=1): continue
                for h_3 in data:
                    if (h_2[0]==h_3[0]): continue
                    if (abs(h_2[4]-h_3[4])!=1 and abs(h_1[4]-h_3[4])!=2): continue
                
                    X.append([h_1[1]/10., h_1[2]/100., h_1[3]*10,
                              h_2[1]/10., h_2[2]/100., h_2[3]*10,
                              h_3[1]/10., h_3[2]/100., h_3[3]*10])
                    y.append([int(h_1[5]==h_2[5]==h_3[5])])


    X = np.vstack(X)
    y = np.vstack(y)

    return X, y

# -----------------------------
# Recurrent Neural Network Part
# -----------------------------
# a padding function
# padd array with zeros with given dim
def pad(arr, off=0):
    '''
    Pad the seqence with zeros to be of required length
    Arguments:
    1. arr (nd.array) : A given numpy array
    2. off (int) : offset, how many entries should be padded ?
    '''
    if off>0:
        updated = np.vstack([arr, np.zeros((off,3))])
        return updated
    else:
        return arr

# a function to create track-lets for LSTM classification
def create_tracklets(events):
    '''
    A function to create tracklets
    Arguments:
    1. events (list) : A list of events
    '''
    X = []
    y = []
    for event in tqdm(range(len(events))):
        df = events[event]
        if df.empty: continue      
        particle_ids = np.unique(df.particle_id.values)
        df = df[['x','z','a']+['chamber_id','particle_id']]

        df_1 = df.loc[(df.chamber_id==1) | (df.chamber_id==2)]
        df_2 = df.loc[(df.chamber_id==3) | (df.chamber_id==4)]

        list_df_1 = []
        list_df_2 = []
        
        for j in particle_ids:
            track_df_1 = df_1.loc[df_1.particle_id==j, df_1.columns.values]
            track_df_2 = df_2.loc[df_2.particle_id==j, df_2.columns.values]

            track_df_1 = track_df_1.drop(['particle_id','chamber_id'], axis=1)
            list_df_1.append((track_df_1,j))
            track_df_2 = track_df_2.drop(['particle_id','chamber_id'], axis=1)
            list_df_2.append((track_df_2,j))             

            scaler = StandardScaler()
            for df1, pid1 in list_df_1:
                for df2, pid2 in list_df_2:

                    data = pd.concat([df1,df2], axis=0)
                    offset = 24-data.shape[0] # 24 is a perfect track with the correct number of hits
                    if offset<0: continue
                    data = pad(data, offset)
                    data = scaler.fit_transform(data)
                    X.append(data)
                    y.append([int(pid1==pid2)])                

    X = np.stack(X)
    y = np.stack(y)
    return X, y

# predict on single event
def predict_tracklets(df, model):
    '''
    A function used to build track-lets and use the model to predict
    Arguments :
    1. data (dataframe) : An input event dataframe
    2. model (keras model) : A trained model for track-lets
    '''
    X       = []
    hit_ids = []

    particle_ids = np.unique(df.particle_id.values)
    df = df[['hit_id','x','z','a']+['chamber_id','particle_id']]

    df_1 = df.loc[(df.chamber_id==1) | (df.chamber_id==2)]
    df_2 = df.loc[(df.chamber_id==3) | (df.chamber_id==4)]

    list_df_1 = []
    list_df_2 = []

    for j in particle_ids:
        track_df_1 = df_1.loc[df_1.particle_id==j, df_1.columns.values]
        track_df_2 = df_2.loc[df_2.particle_id==j, df_2.columns.values]

        track_df_1 = track_df_1.drop(['particle_id','chamber_id'], axis=1)
        list_df_1.append((track_df_1,j))
        track_df_2 = track_df_2.drop(['particle_id','chamber_id'], axis=1)
        list_df_2.append((track_df_2,j))             

    scaler = StandardScaler()
    for df1, _ in list_df_1:
        for df2, _ in list_df_2:
            hit_ids.append(np.hstack([df1.hit_id.values,df2.hit_id.values]))
            
            data = pd.concat([df1[['x','z','a']],df2[['x','z','a']]], axis=0)
            offset = 24-data.shape[0] # 24 is a perfect track with the correct number of hits
            if offset<0: continue
            data = pad(data, offset)
            data = scaler.fit_transform(data)
            X.append(data)

    
    X = np.array(X)
    hit_ids = np.array(hit_ids)

    return X, hit_ids, model.predict(X)

# a function to create seeds of hits
# as a training sample for LSTM to use for prediction
def create_seeds(events):
    '''
    A function used to create seeds (a number of hits)
    Arguments:
    1. events (list) : A list of events
    '''
    X = []
    y = []
    for i in tqdm(range(len(events))):
        event = events[i]
        event = event.loc[event.chamber_id<=2]
        for pid in np.unique(event.particle_id):
            df = event.loc[event.particle_id==pid]
            df = df[['x','z','a']]
            df = df.sort_values('z')

            seed   = 4
            data   = df.iloc[0:seed,:].to_numpy()/np.array([10., 100., 0.01])
            target = df.iloc[seed:seed*2,:].to_numpy()/np.array([10., 100., 0.01])           
                
            X.append([data])
            y.append([target])
                
    return np.vstack(X), np.vstack(y)

# -----------------------------
# Graph Neural Network Part
# -----------------------------
def create_graph_list(events):
    '''
    Create a list of graphs
    Arguments:
    1. events (list) : A list of events
    '''
    nevts = len(events)

    print ('\n\nCreating Graph List. This may take a while, please be patient ')
    print ('Processing %.0f'%nevts, 'events ')
    print ()

    graphs = []
    Graph  = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y', 'hit_ids'])
    for idx in tqdm(range(nevts)):
        df = events[idx]
        if df.empty: continue      
        df = df.reset_index()
        df = df.assign(a = np.arctan2(df.x, df.z))
        feature_names = ['x','z']
        feature_scale = np.array([100., 100.])
        hits = df[feature_names]/feature_scale

        segments = []
        y        = []
        hit_ids  = []
        data     = df[['hit_id','layer_id','particle_id']].to_numpy()
        for i, ii in zip(data, range(data.shape[0])):
            for j, jj in zip(data, range(data.shape[0])):
                if (jj>=ii): continue
                if (abs(i[1]-j[1])!=1): continue

                segments.append([ii, jj])
                y.append([int(i[2]==j[2])])
                hit_ids.append([i[0],j[0]])

        if len(y)==0: continue
        segments = pd.DataFrame(segments, columns=['index_1', 'index_2'])
        y        = np.vstack(y)
        hit_ids  = np.vstack(hit_ids)

        n_hits = hits.shape[0]
        n_edges = segments.shape[0]
        Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
        Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
        hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
        seg_start = hit_idx.loc[segments.index_1].values
        seg_end = hit_idx.loc[segments.index_2].values

        Ri[seg_end, np.arange(n_edges)] = 1
        Ro[seg_start, np.arange(n_edges)] = 1
        X = hits.values
        G = Graph(X, Ri, Ro, y, hit_ids)
        graphs.append(G)
        
    return graphs

def graph_generator(in_graphs, start=0, end=100):
    '''
    A function to create batches of graphs
    '''
    # get the matrix maximum sizes
    n_features = in_graphs[0].X.shape[1]
    n_nodes    = np.array([g.X.shape[0] for g in in_graphs])
    n_edges    = np.array([g.y.shape[0] for g in in_graphs])
    max_nodes  = n_nodes.max()
    max_edges  = n_edges.max()
    
    # Allocate the tensors for this batch
    X_  = np.zeros((max_nodes, n_features), dtype=np.float32)
    Ri_ = np.zeros((max_nodes, max_edges), dtype=np.float32)
    Ro_ = np.zeros((max_nodes, max_edges), dtype=np.float32)
    y_  = np.zeros((max_edges, 1), dtype=np.float32)
    
    X_padded = []
    Ri_padded = []
    Ro_padded = []
    y_padded = []
    
    for  i in tqdm(range(start, end)):
        g = in_graphs[i]
        X, Ri, Ro, y = g.X, g.Ri, g.Ro, g.y
        X_[:X.shape[0],:X.shape[1]] = X
        Ri_[:Ri.shape[0],:Ri.shape[1]] = Ri
        Ro_[:Ro.shape[0],:Ro.shape[1]] = Ro
        y_[:y.shape[0],:y.shape[1]] = y
        
        X_padded.append(X_)
        Ri_padded.append(Ri_)
        Ro_padded.append(Ro_)
        y_padded.append(y_)
    
    return [np.stack(X_padded), np.stack(Ri_padded), np.stack(Ro_padded)], np.stack(y_padded)