import caffe
import numpy as np
import yaml
import json

class RoutingLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.routing_idx = layer_params.routing_idx
        with open('tree.json') as f:
            tree_data = json.load(f)
            self.object_list = tree_data['object_list']
            self.all_children_list = tree_data['all_children_list']
            self.immediate_children_list = tree_data['immediate_children_list']
        assert self.routing_idx >= 0 and self.routing_idx < len(self.object_list)
        self.children = self.children_list[self.routing_idx]

    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        top[0] = bottom[0]

    def backward(self, top, propagate_down, bottom):
        labels_idxs = bottom[1] 
        for i in xrange(len(labels_idxs)):
            labels_idx = labels_idxs[i]
            if labels_idx in self.all_children_list:
                bottom[0].diff[i,:] = 1

