import caffe
import os
import numpy as np
import yaml
import json
from fast_rcnn.config import cfg

class RoutingLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.routing_idx = int(layer_params['routing_idx'])
        with open(os.path.join(cfg.CACHE_DIR, 'tree.json')) as f:
            tree_data = json.load(f)
            self.object_list = tree_data['object_list']
            self.all_children_list = tree_data['all_children_list']
            self.immediate_children_list = tree_data['immediate_children_list']
        assert self.routing_idx >= 0 and self.routing_idx < len(self.object_list)
        self.children = self.all_children_list[self.routing_idx]
        top[0].reshape(*(bottom[0].shape))

    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        top[0].reshape(*(bottom[0].shape))
        top[0].data[...] = bottom[0].data 

    def backward(self, top, propagate_down, bottom):
        labels_idxs = bottom[1].data
        for i in xrange(len(labels_idxs)):
            labels_idx = int(labels_idxs[i])
            if labels_idx in self.children:
                bottom[0].diff[i,:] = top[0].diff[i,:] 
            else:
                bottom[0].diff[i,:] = 0
