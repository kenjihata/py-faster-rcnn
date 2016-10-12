import caffe
import numpy as np
import yaml
import json

class SoftmaxRoutingLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.routing_idx = layer_params.routing_idx
        with open('tree.json') as f:
            tree_data = json.load(f)
            self.object_list = tree_data['object_list']
            self.children_list = tree_data['children_list']
        assert self.routing_idx >= 0 and self.routing_idx < len(self.object_list)
        self.children = self.children_list[self.routing_idx]

    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        top[0] = bottom[0]

    def backward(self, top, propagate_down, bottom):
        softmax_level = bottom[1]
        if softmax_level in self.children:
            TODO

