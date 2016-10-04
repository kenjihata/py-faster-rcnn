# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from corloc_eval import voc_eval
from fast_rcnn.config import cfg
import math
class vg(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'vg_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VisualGenome' + self._year)
        self._classes = ('__background__', 
                "man", "person", "window", "building", "tree", "woman", "shirt", "wall", "sign", "table", "water", "pole", "car", "head", "hand", "plate", "leg", "train", "ear", "fence", "door", "chair", "pants", "road", "street", "bus", "eye", "hat", "snow", "giraffe", "boy", "jacket", "wheel", "plane", "elephant", "zebra", "clock", "dog", "boat", "girl", "horse", "sidewalk", "nose", "cat", "bird", "cloud", "shoe", "tail", "umbrella", "helmet", "flower", "shorts", "leaf", "cow", "bench", "sheep", "arm", "glass", "face", "bag", "pizza", "truck", "bear", "food", "bottle", "kite", "motorcycle", "rock", "tile", "tire", "post", "bowl", "player", "surfboard", "skateboard", "bike", "roof", "foot", "house", "tracks", "shelf", "cap", "pillow", "box", "bed", "jeans", "mouth", "banana", "cup", "counter", "beach", "sink", "lamp", "vase", "toilet", "laptop", "sand", "wave", "ball", "bush", "flag", "child", "airplane", "neck", "cake", "paper", "desk", "coat", "seat", "vehicle", "phone", "wing", "book", "glove", "tower", "lady", "frisbee", "hill", "cabinet", "mountain", "tie", "headlight", "ocean", "skier", "container", "keyboard", "racket", "towel", "track", "orange", "guy", "windshield", "couch", "pot", "basket", "fork", "fruit", "knife", "bat", "engine", "suitcase", "finger", "broccoli", "donut", "bicycle", "surfer", "backpack", "apple", "sandwich", "collar", "sock", "men", "blanket", "bread", "cheese", "tray", "paw", "van", "dress", "watch", "curtain", "lid", "court", "bridge", "wood", "sweater", "traffic light", "uniform", "pavement", "napkin", "stone", "computer", "kid", "bathroom", "camera", "carrot", "suit", "stem", "faucet", "faucet", "drawer", "platform", "ski", "hydrant", "spoon", "fur", "wrist", "stove", "meat", "luggage", "cone", "vest", "snowboard", "teddy bear", "strap", "key", "carpet", "statue", "tv", "mouse", "license plate", "monitor", "horn", "candle", "dish", "rug", "street sign", "tomato", "sleeve", "can", "goggles", "catcher", "street light", "kitchen", "beak", "belt", "oven", "ring", "banner", "cloth", "pipe", "boot", "vegetable", "remote", "jersey", "cell phone", "tennis racket", "baby", "ramp", "hot dog", "stick", "purse", "chain", "park", "gravel", "sneaker", "skirt", "television", "ski pole", "skateboarder", "pillar", "wetsuit", "balcony", "jet", "tennis court", "refrigerator", "batter", "gate", "knee", "scarf", "graffiti", "river", "tree trunk", "shore", "umpire", "awning", "microwave", "walkway", "bun", "sofa", "cover", "pen", "cart", "pan", "fire hydrant", "net", "sun", "bracelet", "band", "wine", "runway", "path", "cellphone", "tip", "train car", "necklace", "tusk", "clothes", "pond", "concrete", "metal", "baseball", "beard", "crust", "doughnut", "jar", "pocket", "painting", "mug", "fridge", "city", "holder", "doorway", "scissors", "bucket", "bolt", "tent", "column", "poster", "snowboarder", "stop sign", "outlet", "shoulder", "speaker", "onion", "log", "tablecloth", "spectator", "duck", "green", "panel", "controller", "lettuce", "forest", "advertisement", "hood", "cushion", "store", "tennis ball", "parking lot", "chimney", "pitcher", "baseball player", "teeth", "toothbrush", "slice", "pepper", "star", "pepperoni", "tennis player", "slope", "outfit", "case", "suv", "toy", "tank", "palm tree", "frosting", "crowd", "vent", "crack", "symbol", "sheet", "wristband", "white shirt", "propeller", "liquid", "toilet paper", "straw", "ribbon", "rider", "display", "clothing", "thumb", "smoke", "lake", "step", "headboard", "smiling woman", "hay", "fan", "hotdog", "dock", "elbow", "cockpit", "ladder", "rim", "earring", "ripple", "chest", "tub", "bumper", "tongue", "pane", "chicken", "grill", "hoodie", "station", "white line", "signal", "brush", "stool", "goat", "wine glass", "hillside", "mat", "topping", "countertop", "tarmac", "airport", "decoration", "blue shirt", "game", "shrub", "headband", "meter", "tank top", "fabric", "salad", "utensil", "drink", "scooter", "bin", "motorbike", "mask", "foam", "skater", "sweatshirt", "block", "sunlight", "pile", "planter", "fireplace", "ship", "trash", "crosswalk", "petal", "mitt", "border", "canopy", "cable", "screw", "cupcake", "cement", "cross", "ponytail", "egg", "bathtub", "drain", "ledge", "wet suit", "right hand", "leash", "nail", "sail", "pastry", "eyebrow", "barrier", "print", "wiper", "lot", "beer", "zipper", "mushroom", "visor", "shower", "left hand", "landing gear", "enclosure", "lamp post", "beam", "tail light", "home plate", "icing", "crate", "bow", "parking meter", "urinal", "toilet seat", "burner", "dresser", "slat", "arch", "rice", "counter top", "apron", "pad", "tarp", "church", "streetlight", "potato", "tennis", "sandal", "tennis shoe", "soap", "strawberry", "chin", "fencing", "tape", "traffic signal", "hinge", "light pole", "rose", "roll", "restaurant", "olive", "snout", "comforter", "male", "sea", "machine", "emblem", "lemon", "trailer", "fish", "coffee table", "tattoo", "carriage", "stuffed animal", "plastic", "saucer", "clock tower", "ketchup", "lamb", "dispenser", "mound", "driver", "smile", "water bottle", "street lamp", "dessert", "balloon", "sausage", "front wheel", "bunch", "black shirt", "coffee", "nostril", "cutting board", "bookshelf", "table cloth", "heart", "light fixture", "switch", "white plate", "pine tree", "saddle", "baseball field", "telephone", "red shirt", "steeple", "boulder", "stroller", "deck", "pier", "passenger", "racquet", "billboard", "sailboat", "stack", "doll", "surf board", "baseball bat", "magnet", "hook", "bell", "soup", "mustache", "trail", "nightstand", "stain", "hose", "belly", "moss", "baseball cap", "asphalt", "ice", "bull", "entrance", "zoo", "crane", "aircraft", "polar bear", "menu", "silverware", "traffic", "trashcan", "bacon", "remote control", "cage", "gear", "claw", "harness", "weed", "left ear", "meal", "cab", "marking", "shutter", "puddle", "blender", "mustard", "bark", "cupboard", "seagull", "clock face", "white cloud", "intersection", "barrel", "grate", "front leg", "mud", "antenna", "human", "wrist band", "fingernail", "glass window", "right ear", "cooler", "art", "dugout", "lock", "magazine", "newspaper", "female", "archway", "fixture", "drawing", "taxi", "short", "tshirt", "train track", "outdoors", "train station", "microphone", "pigeon", "staircase", "whisker", "mast", "grape", "red light", "left eye", "card", "brick wall", "plug", "pool", "platter", "pedestrian", "porch", "blouse", "home", "soccer ball", "traffic sign", "paddle", "spinach", "toilet bowl", "power line", "reflector", "stop", "footprint", "forehead", "light post", "opening", "toddler", "device", "marker", "computer monitor", "furniture", "spoke", "shop", "plank", "light switch", "knot", "fin", "chocolate", "tee shirt", "cucumber", "placemat", "french fry", "metal pole", "wool", "dishwasher", "hedge", "right eye", "kitten", "debris", "cleat", "wagon", "windshield wiper", "juice", "wii", "lap", "mattress", "fence post", "sprinkle", "wine bottle", "tube", "tissue", "support", "bedspread", "sculpture", "island", "dome", "roadway", "tooth", "front tire", "equipment", "buoy", "ski lift", "kettle", "pickle", "fender", "adult", "stop light", "someone", "cookie", "table top", "officer", "center", "shoreline", "barn", "wrinkle", "kickstand", "package", "soda", "moped", "lamp shade", "garage", "middle", "pant", "spire", "skyscraper", "corn", "time", "ad", "highway", "handbag", "bulb", "shaker", "pie", "calf", "biker", "cattle", "crumb", "splash", "triangle", "cream", "foil", "left arm", "cabinet door", "lighting", "candy", "driveway", "artwork", "ham", "wrapper", "bookcase", "coffee cup", "sugar", "waist", "stoplight", "lampshade", "stairway", "bandana", "pedal", "diamond", "bus stop", "fry", "spatula", "pizza slice", "globe", "cabin", "buckle", "green shirt", "plaque", "bank", "bracket", "jug", "milk", "sedan", "beanie", "right arm", "bridle", "tunnel", "telephone pole", "side mirror", "worker", "jeep", "trouser", "wallpaper", "pineapple", "pasta", "green leaf", "bean", "rust", "chandelier", "pathway", "notebook", "beverage", "toaster", "guitar", "cliff", "tool", "canoe", "stair", "lighthouse", "muffin", "hanger", "bikini", "american flag", "right leg", "armrest", "herd", "left leg", "goose", "plastic bag", "pumpkin", "door handle", "brick building", "picture frame", "ornament", "stump", "green tree", "costume", "window sill", "mouse pad", "steam", "black jacket", "control", "lip", "butter", "ram", "wii remote", "white sign", "train engine", "shower curtain", "motor", "covering", "rain", "ottoman", "stadium", "cheek", "necktie", "socket", "baseball glove", "fountain", "mantle", "shed", "carton", "material", "dial", "puppy", "freezer", "garbage", "skiis", "watermelon", "steering wheel", "pilot", "window pane", "shower head", "toe", "railroad", "pony tail", "note", "flamingo", "appliance", "wooden", "market", "trolley", "ski slope", "booth", "yellow shirt", "thigh", "knee pad", "icon", "brake light", "divider", "latch", "character", "oar", "sole", "skiier", "facial hair", "policeman", "palm", "muzzle", "skate park", "wooden table", "baby elephant", "teapot", "nut", "clip", "overpass", "lift", "pack", "little girl", "picnic table", "left foot", "parasail", "cardboard", "pear", "mother", "lanyard", "cauliflower", "white wall", "flip flop", "manhole", "pony", "coffee mug", "pencil", "air conditioner", "limb", "butterfly", "rear", "cloudy sky", "yellow flower", "toothpick", "feeder", "cabbage", "radiator", "spray", "toilet lid", "stalk", "bouquet", "seed", "toast", "towel rack", "produce", "netting", "shrimp", "watermark", "locomotive", "berry", "taillight", "salt shaker", "cigarette", "right foot", "cardboard box", "cuff", "wii controller", "head light", "slab", "panda", "back wheel", "mailbox", "computer mouse", "guard", "mannequin", "peel", "parrot", "wristwatch", "map", "monkey", "handrail", "median", "baseboard", "wake", "mountain range", "bow tie", "coaster", "coffee maker", "ice cream", "shelter", "audience", "night stand", "hubcap", "cd", "burger", "skate board", "sunset", "bubble", "soccer player", "lace", "doorknob", "striped shirt", "stone wall", "red sign", "white car", "door knob", "business", "raft", "badge", "pedestal", "deer", "pink shirt", "computer screen", "quilt", "stove top", "robe", "desert", "sweatband", "young man", "brocolli", "game controller", "police officer", "chip", "motorcyclist", "office", "tap", "minivan", "range", "seaweed", "green light", "printer", "fire", "stall", "flower pot", "basin", "traffic cone", "red jacket", "blueberry", "harbor", "blue sign", "back leg", "swan", "back tire", "vanity", "lamppost", "grout", "storefront", "arm rest", "photographer", "front window", "white snow", "black hat", "piano", "fire truck", "sill", "stream", "dough", "town", "soap dispenser", "paper plate", "end table", "trash bin", "tennis net", "metal fence", "mousepad", "lens", "lantern", "utility pole", "black car", "stage", "lime", "celery", "leave", "briefcase", "basil", "toothpaste", "hour hand", "ankle", "brown horse", "jockey", "wrap", "black nose", "peak", "police", "bib", "tongs", "barricade", "squash", "minute hand", "hotel", "scale", "bill", "castle", "canister", "bride", "set", "tractor", "lane", "grip", "road sign", "calendar", "flame", "disc", "skull", "roadside", "white tile", "heel", "goatee", "fox", "hallway", "blue jacket", "white building", "cherry", "tennis racquet", "pin", "scoreboard", "windowsill", "pea", "avocado", "mulch", "desktop", "ridge", "ski boot", "hen", "vine", "brace", "ipod", "seam", "moon", "bagel", "blazer", "sign post", "red car", "paper towel", "trick", "old woman", "park bench", "back pack", "noodle", "chef", "wallet", "toilet tank", "speck", "tablet", "gas tank", "cloudy", "tabletop", "side walk", "video game", "compartment", "crown", "pebble", "wooden fence", "fog", "side window", "window frame", "skiing", "advertising", "snowsuit", "cereal", "lion", "butt", "jean", "bead", "rectangle", "garage door", "turkey", "black bag", "ceiling light", "soldier", "antelope", "candle holder", "guard rail", "orange shirt", "undershirt", "hind leg", "white hat", "salt", "swimsuit", "owl", "torso", "black shoe", "ocean water", "groom", "seasoning", "paneling", "radio", "closet", "steak", "snow board", "hut", "beef", "parachute", "subway", "teddy", "backsplash", "peach", "side table", "laptop computer", "orange juice", "pancake", "veggie", "restroom", "gun", "door frame", "valve", "name tag", "yellow sign", "kayak", "manhole cover", "exhaust pipe", "gutter", "tin", "binder", "oven door", "wrist watch", "dust", "cactus", "tv stand", "monument", "soccer", "packet", "clay", "wooden post", "rear wheel", "hand rail", "moustache", "mixer", "crest", "glaze", "tea", "electrical outlet", "plain", "chalk", "parsley", "tall building", "pink flower", "ivy", "heater", "folder", "wooden bench", "stirrup", "black tire", "rooftop", "face mask", "charger", "shield", "dryer", "parasol", "stuffed bear", "sunshine", "portrait", "hamburger", "dumpster", "green pepper", "disk", "cop", "hip", "life preserver", "floor tile", "shin guard", "sponge", "bathing suit", "pottery", "grafitti", "dinner", "match", "canal", "christmas tree", "buggy", "armchair", "pail", "train platform", "white shoe", "shingle", "skatepark", "silhouette", "book shelf", "left wing", "soap dish", "light bulb", "outfield", "ostrich", "pickup truck", "helicopter", "rag", "thread", "oil", "evergreen tree", "shopping bag", "right wing", "ceiling fan", "lunch", "bud", "muffler", "bike rack", "garnish", "cane", "silver car", "coffee pot", "sack", "signal light", "elbow pad", "beer bottle", "pepper shaker", "rock wall", "green bush", "sandwhich", "scissor", "tall tree", "tulip", "ski jacket", "orange slice", "mural", "dvd", "tangerine", "boardwalk", "life jacket", "playground", "asparagus", "lipstick", "kneepad", "tissue box", "referee", "weather vane", "green sign", "cowboy hat", "tail fin", "conductor", "keypad", "runner", "streetlamp", "head band", "gold", "safety cone", "garlic", "signboard", "flag pole", "school bus", "hill side", "jet engine", "condiment", "grain", "surf", "skillet", "raspberry", "bowtie", "gravy", "blue water", "banana bunch", "tea kettle", "tomato slice", "smoke stack", "pew", "rainbow", "sunflower", "tile floor", "floor lamp", "surfing", "tomato sauce", "pump", "young girl", "glass door", "throw pillow", "page", "gap", "spout", "breast", "cargo", "shoelace", "bath tub", "breakfast", "melon", "buffalo", "gazelle", "tail wing", "braid", "syrup", "plunger", "little boy", "elephant trunk", "snowflake", "white paper", "red pepper", "place mat", "ox", "drum", "slide", "goal", "nike logo", "young boy", "wooden chair", "firetruck", "drape", "fire extinguisher", "ashtray", "white flower", "beach chair", "tomatoe", "cyclist", "potted plant", "spice", "link", "chalkboard", "tripod", "yacht", "pizza crust", "baseball uniform", "skylight", "dandelion", "podium", "powerlines", "tights", "city street", "clock hand", "urn", "stovetop", "evergreen", "baseball hat", "black coat", "bed frame", "rearview mirror", "ski suit", "carrier", "purple flower", "carving", "pillowcase", "tanktop", "red flower", "birthday cake", "framed picture", "wreath", "stitching", "flagpole", "blue car", "envelope", "gas station", "signpost", "steel", "pine", "flyer", "cape", "toilet brush", "flour", "small window", "sail boat", "brown cow", "white pillow", "comb", "padding", "tennis match", "broom", "police car", "white truck", "white surfboard", "gown", "cracker", "tassel", "shell", "gray shirt", "white cap", "black cow", "bell pepper", "iron", "wood table", "passenger car", "food truck", "donkey", "canvas", "automobile", "black eye", "office chair", "rear tire", "zucchini", "shower door", "fern", "orange cone", "white house", "mountain top", "white toilet", "black suit", "white boat", "passenger window", "white sheep", "pillow case", "yarn", "store front", "pylon", "black pole", "white door", "lounge chair", "pizza box", "mitten", "needle", "red hat", "paper bag", "white sock", "radish", "pallet", "kiwi", "red umbrella", "cobblestone", "turbine", "wooden pole", "trash bag", "croissant")
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VisualGenome/VisualGenome2016/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VisualGenome')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
                'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        #filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filename = self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VisualGenome' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VisualGenome' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VisualGenome' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap, ap_true, num_gt = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            if num_gt > 0:
                aps += [ap]
                print('AP for {} = {:.4f}'.format(cls, ap))
                print('Number of detections for {} = {}'.format(cls, ap_true))
                print('Number of gt labels for {} = {}'.format(cls, num_gt))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean([a for a in aps if not math.isnan(a)])))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
                .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                #os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.vg import vg 
    d = vg('trainval', '2016')
    res = d.roidb
    from IPython import embed; embed()
