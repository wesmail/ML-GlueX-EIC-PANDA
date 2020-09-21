import tensorflow as tf

class EdgeNetwork(tf.keras.Model):
    '''
    A Network which computes weights for edges of the graph.
    '''
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=activation),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')])
        
    def call(self, inputs):
        X, Ri, Ro = inputs[0], inputs[1], inputs[2]
        bo = tf.matmul(tf.transpose(X, perm=[0, 2, 1]), tf.transpose(Ro, perm=[0, 2, 1]))
        bi = tf.matmul(tf.transpose(X, perm=[0, 2, 1]), tf.transpose(Ri, perm=[0, 2, 1]))
        B  = tf.keras.layers.concatenate([bo, bi],axis=1)
        B  = tf.transpose(B, perm=[0, 2, 1])
        hidden = self.network(B)
        return hidden
    
class NodeNetwork(tf.keras.Model):
    '''
    A Network which computes new node features on the graph.
    '''
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=activation),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units, activation=activation)])
        
    def call(self, inputs):
        X, Ri, Ro, e = inputs[0], inputs[1], inputs[2], inputs[3]
        bo  = tf.matmul(tf.transpose(X, perm=[0, 2, 1]), tf.transpose(Ro, perm=[0, 2, 1]))
        bi  = tf.matmul(tf.transpose(X, perm=[0, 2, 1]), tf.transpose(Ri, perm=[0, 2, 1]))
        Rwo = tf.transpose(Ro, perm=[0, 2, 1]) * tf.transpose(e, perm=[0, 2, 1])
        Rwi = tf.transpose(Ri, perm=[0, 2, 1]) * tf.transpose(e, perm=[0, 2, 1])
        mi  = tf.matmul(Rwi, tf.transpose(bi, perm=[0, 2, 1]))
        mo  = tf.matmul(Rwo, tf.transpose(bo, perm=[0, 2, 1]))
        M   = tf.keras.layers.concatenate([mi, mo], axis=2)
        hidden = self.network(M)
        return hidden
    
    
class GNN(tf.keras.Model):
    '''
    Edge classification graph neural network model..
    '''
    def __init__(self, units, n_iters=2, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.n_iters = n_iters
        # input network
        self.input_network = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=activation),
            tf.keras.layers.Dropout(0.5)])
        
        # edge network
        self.edge_network = EdgeNetwork(units, activation)
        # node network
        self.node_network = NodeNetwork(units, activation)        
        
    def call(self, inputs):
        X, Ri, Ro = inputs[0], inputs[1], inputs[2]
        Ri = tf.transpose(Ri, perm=[0, 2, 1])
        Ro = tf.transpose(Ro, perm=[0, 2, 1])
        # get the hidden representation
        H = self.input_network(X)
        # shortcut connect the inputs onto the hidden representation
        H = tf.keras.layers.concatenate([H, X], axis=2)
        
        # Loop over iterations of edge and node networks (message-passing phase)
        for i in range(self.n_iters):
            # apply edge network
            e = self.edge_network([H, Ri, Ro])
            # apply node network
            H = self.node_network([H, Ri, Ro, e])
            # shortcut connect the inputs onto the hidden representation
            H = tf.keras.layers.concatenate([H, X], axis=2)
        # apply final edge network (read-out phase)
        return self.edge_network([H, Ri, Ro])    