#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import matplotlib

CLASSES = ('__background__',
            "man", "person", "window", "building", "tree", "woman", "shirt", "wall", "sign", "table", "water", "pole", "car", "head", "hand", "plate", "leg", "train", "ear", "fence", "door", "chair", "pants", "road", "street", "bus", "eye", "hat", "snow", "giraffe", "boy", "jacket", "wheel", "plane", "elephant", "zebra", "clock", "dog", "boat", "girl", "horse", "sidewalk", "nose", "cat", "bird", "cloud", "shoe", "tail", "umbrella", "helmet", "flower", "shorts", "leaf", "cow", "bench", "sheep", "arm", "glass", "face", "bag", "pizza", "truck", "bear", "food", "bottle", "kite", "motorcycle", "rock", "tile", "tire", "post", "bowl", "player", "surfboard", "skateboard", "bike", "roof", "foot", "house", "tracks", "shelf", "cap", "pillow", "box", "bed", "jeans", "mouth", "banana", "cup", "counter", "beach", "sink", "lamp", "vase", "toilet", "laptop", "sand", "wave", "ball", "bush", "flag", "child", "airplane", "neck", "cake", "paper", "desk", "coat", "seat", "vehicle", "phone", "wing", "book", "glove", "tower", "lady", "frisbee", "hill", "cabinet", "mountain", "tie", "headlight", "ocean", "skier", "container", "keyboard", "racket", "towel", "track", "orange", "guy", "windshield", "couch", "pot", "basket", "fork", "fruit", "knife", "bat", "engine", "suitcase", "finger", "broccoli", "donut", "bicycle", "surfer", "backpack", "apple", "sandwich", "collar", "sock", "men", "blanket", "bread", "cheese", "tray", "paw", "van", "dress", "watch", "curtain", "lid", "court", "bridge", "wood", "sweater", "traffic light", "uniform", "pavement", "napkin", "stone", "computer", "kid", "bathroom", "camera", "carrot", "suit", "stem", "faucet", "faucet", "drawer", "platform", "ski", "hydrant", "spoon", "fur", "wrist", "stove", "meat", "luggage", "cone", "vest", "snowboard", "teddy bear", "strap", "key", "carpet", "statue", "tv", "mouse", "license plate", "monitor", "horn", "candle", "dish", "rug", "street sign", "tomato", "sleeve", "can", "goggles", "catcher", "street light", "kitchen", "beak", "belt", "oven", "ring", "banner", "cloth", "pipe", "boot", "vegetable", "remote", "jersey", "cell phone", "tennis racket", "baby", "ramp", "hot dog", "stick", "purse", "chain", "park", "gravel", "sneaker", "skirt", "television", "ski pole", "skateboarder", "pillar", "wetsuit", "balcony", "jet", "tennis court", "refrigerator", "batter", "gate", "knee", "scarf", "graffiti", "river", "tree trunk", "shore", "umpire", "awning", "microwave", "walkway", "bun", "sofa", "cover", "pen", "cart", "pan", "fire hydrant", "net", "sun", "bracelet", "band", "wine", "runway", "path", "cellphone", "tip", "train car", "necklace", "tusk", "clothes", "pond", "concrete", "metal", "baseball", "beard", "crust", "doughnut", "jar", "pocket", "painting", "mug", "fridge", "city", "holder", "doorway", "scissors", "bucket", "bolt", "tent", "column", "poster", "snowboarder", "stop sign", "outlet", "shoulder", "speaker", "onion", "log", "tablecloth", "spectator", "duck", "green", "panel", "controller", "lettuce", "forest", "advertisement", "hood", "cushion", "store", "tennis ball", "parking lot", "chimney", "pitcher", "baseball player", "teeth", "toothbrush", "slice", "pepper", "star", "pepperoni", "tennis player", "slope", "outfit", "case", "suv", "toy", "tank", "palm tree", "frosting", "crowd", "vent", "crack", "symbol", "sheet", "wristband", "white shirt", "propeller", "liquid", "toilet paper", "straw", "ribbon", "rider", "display", "clothing", "thumb", "smoke", "lake", "step", "headboard", "smiling woman", "hay", "fan", "hotdog", "dock", "elbow", "cockpit", "ladder", "rim", "earring", "ripple", "chest", "tub", "bumper", "tongue", "pane", "chicken", "grill", "hoodie", "station", "white line", "signal", "brush", "stool", "goat", "wine glass", "hillside", "mat", "topping", "countertop", "tarmac", "airport", "decoration", "blue shirt", "game", "shrub", "headband", "meter", "tank top", "fabric", "salad", "utensil", "drink", "scooter", "bin", "motorbike", "mask", "foam", "skater", "sweatshirt", "block", "sunlight", "pile", "planter", "fireplace", "ship", "trash", "crosswalk", "petal", "mitt", "border", "canopy", "cable", "screw", "cupcake", "cement", "cross", "ponytail", "egg", "bathtub", "drain", "ledge", "wet suit", "right hand", "leash", "nail", "sail", "pastry", "eyebrow", "barrier", "print", "wiper", "lot", "beer", "zipper", "mushroom", "visor", "shower", "left hand", "landing gear", "enclosure", "lamp post", "beam", "tail light", "home plate", "icing", "crate", "bow", "parking meter", "urinal", "toilet seat", "burner", "dresser", "slat", "arch", "rice", "counter top", "apron", "pad", "tarp", "church", "streetlight", "potato", "tennis", "sandal", "tennis shoe", "soap", "strawberry", "chin", "fencing", "tape", "traffic signal", "hinge", "light pole", "rose", "roll", "restaurant", "olive", "snout", "comforter", "male", "sea", "machine", "emblem", "lemon", "trailer", "fish", "coffee table", "tattoo", "carriage", "stuffed animal", "plastic", "saucer", "clock tower", "ketchup", "lamb", "dispenser", "mound", "driver", "smile", "water bottle", "street lamp", "dessert", "balloon", "sausage", "front wheel", "bunch", "black shirt", "coffee", "nostril", "cutting board", "bookshelf", "table cloth", "heart", "light fixture", "switch", "white plate", "pine tree", "saddle", "baseball field", "telephone", "red shirt", "steeple", "boulder", "stroller", "deck", "pier", "passenger", "racquet", "billboard", "sailboat", "stack", "doll", "surf board", "baseball bat", "magnet", "hook", "bell", "soup", "mustache", "trail", "nightstand", "stain", "hose", "belly", "moss", "baseball cap", "asphalt", "ice", "bull", "entrance", "zoo", "crane", "aircraft", "polar bear", "menu", "silverware", "traffic", "trashcan", "bacon", "remote control", "cage", "gear", "claw", "harness", "weed", "left ear", "meal", "cab", "marking", "shutter", "puddle", "blender", "mustard", "bark", "cupboard", "seagull", "clock face", "white cloud", "intersection", "barrel", "grate", "front leg", "mud", "antenna", "human", "wrist band", "fingernail", "glass window", "right ear", "cooler", "art", "dugout", "lock", "magazine", "newspaper", "female", "archway", "fixture", "drawing", "taxi", "short", "tshirt", "train track", "outdoors", "train station", "microphone", "pigeon", "staircase", "whisker", "mast", "grape", "red light", "left eye", "card", "brick wall", "plug", "pool", "platter", "pedestrian", "porch", "blouse", "home", "soccer ball", "traffic sign", "paddle", "spinach", "toilet bowl", "power line", "reflector", "stop", "footprint", "forehead", "light post", "opening", "toddler", "device", "marker", "computer monitor", "furniture", "spoke", "shop", "plank", "light switch", "knot", "fin", "chocolate", "tee shirt", "cucumber", "placemat", "french fry", "metal pole", "wool", "dishwasher", "hedge", "right eye", "kitten", "debris", "cleat", "wagon", "windshield wiper", "juice", "wii", "lap", "mattress", "fence post", "sprinkle", "wine bottle", "tube", "tissue", "support", "bedspread", "sculpture", "island", "dome", "roadway", "tooth", "front tire", "equipment", "buoy", "ski lift", "kettle", "pickle", "fender", "adult", "stop light", "someone", "cookie", "table top", "officer", "center", "shoreline", "barn", "wrinkle", "kickstand", "package", "soda", "moped", "lamp shade", "garage", "middle", "pant", "spire", "skyscraper", "corn", "time", "ad", "highway", "handbag", "bulb", "shaker", "pie", "calf", "biker", "cattle", "crumb", "splash", "triangle", "cream", "foil", "left arm", "cabinet door", "lighting", "candy", "driveway", "artwork", "ham", "wrapper", "bookcase", "coffee cup", "sugar", "waist", "stoplight", "lampshade", "stairway", "bandana", "pedal", "diamond", "bus stop", "fry", "spatula", "pizza slice", "globe", "cabin", "buckle", "green shirt", "plaque", "bank", "bracket", "jug", "milk", "sedan", "beanie", "right arm", "bridle", "tunnel", "telephone pole", "side mirror", "worker", "jeep", "trouser", "wallpaper", "pineapple", "pasta", "green leaf", "bean", "rust", "chandelier", "pathway", "notebook", "beverage", "toaster", "guitar", "cliff", "tool", "canoe", "stair", "lighthouse", "muffin", "hanger", "bikini", "american flag", "right leg", "armrest", "herd", "left leg", "goose", "plastic bag", "pumpkin", "door handle", "brick building", "picture frame", "ornament", "stump", "green tree", "costume", "window sill", "mouse pad", "steam", "black jacket", "control", "lip", "butter", "ram", "wii remote", "white sign", "train engine", "shower curtain", "motor", "covering", "rain", "ottoman", "stadium", "cheek", "necktie", "socket", "baseball glove", "fountain", "mantle", "shed", "carton", "material", "dial", "puppy", "freezer", "garbage", "skiis", "watermelon", "steering wheel", "pilot", "window pane", "shower head", "toe", "railroad", "pony tail", "note", "flamingo", "appliance", "wooden", "market", "trolley", "ski slope", "booth", "yellow shirt", "thigh", "knee pad", "icon", "brake light", "divider", "latch", "character", "oar", "sole", "skiier", "facial hair", "policeman", "palm", "muzzle", "skate park", "wooden table", "baby elephant", "teapot", "nut", "clip", "overpass", "lift", "pack", "little girl", "picnic table", "left foot", "parasail", "cardboard", "pear", "mother", "lanyard", "cauliflower", "white wall", "flip flop", "manhole", "pony", "coffee mug", "pencil", "air conditioner", "limb", "butterfly", "rear", "cloudy sky", "yellow flower", "toothpick", "feeder", "cabbage", "radiator", "spray", "toilet lid", "stalk", "bouquet", "seed", "toast", "towel rack", "produce", "netting", "shrimp", "watermark", "locomotive", "berry", "taillight", "salt shaker", "cigarette", "right foot", "cardboard box", "cuff", "wii controller", "head light", "slab", "panda", "back wheel", "mailbox", "computer mouse", "guard", "mannequin", "peel", "parrot", "wristwatch", "map", "monkey", "handrail", "median", "baseboard", "wake", "mountain range", "bow tie", "coaster", "coffee maker", "ice cream", "shelter", "audience", "night stand", "hubcap", "cd", "burger", "skate board", "sunset", "bubble", "soccer player", "lace", "doorknob", "striped shirt", "stone wall", "red sign", "white car", "door knob", "business", "raft", "badge", "pedestal", "deer", "pink shirt", "computer screen", "quilt", "stove top", "robe", "desert", "sweatband", "young man", "brocolli", "game controller", "police officer", "chip", "motorcyclist", "office", "tap", "minivan", "range", "seaweed", "green light", "printer", "fire", "stall", "flower pot", "basin", "traffic cone", "red jacket", "blueberry", "harbor", "blue sign", "back leg", "swan", "back tire", "vanity", "lamppost", "grout", "storefront", "arm rest", "photographer", "front window", "white snow", "black hat", "piano", "fire truck", "sill", "stream", "dough", "town", "soap dispenser", "paper plate", "end table", "trash bin", "tennis net", "metal fence", "mousepad", "lens", "lantern", "utility pole", "black car", "stage", "lime", "celery", "leave", "briefcase", "basil", "toothpaste", "hour hand", "ankle", "brown horse", "jockey", "wrap", "black nose", "peak", "police", "bib", "tongs", "barricade", "squash", "minute hand", "hotel", "scale", "bill", "castle", "canister", "bride", "set", "tractor", "lane", "grip", "road sign", "calendar", "flame", "disc", "skull", "roadside", "white tile", "heel", "goatee", "fox", "hallway", "blue jacket", "white building", "cherry", "tennis racquet", "pin", "scoreboard", "windowsill", "pea", "avocado", "mulch", "desktop", "ridge", "ski boot", "hen", "vine", "brace", "ipod", "seam", "moon", "bagel", "blazer", "sign post", "red car", "paper towel", "trick", "old woman", "park bench", "back pack", "noodle", "chef", "wallet", "toilet tank", "speck", "tablet", "gas tank", "cloudy", "tabletop", "side walk", "video game", "compartment", "crown", "pebble", "wooden fence", "fog", "side window", "window frame", "skiing", "advertising", "snowsuit", "cereal", "lion", "butt", "jean", "bead", "rectangle", "garage door", "turkey", "black bag", "ceiling light", "soldier", "antelope", "candle holder", "guard rail", "orange shirt", "undershirt", "hind leg", "white hat", "salt", "swimsuit", "owl", "torso", "black shoe", "ocean water", "groom", "seasoning", "paneling", "radio", "closet", "steak", "snow board", "hut", "beef", "parachute", "subway", "teddy", "backsplash", "peach", "side table", "laptop computer", "orange juice", "pancake", "veggie", "restroom", "gun", "door frame", "valve", "name tag", "yellow sign", "kayak", "manhole cover", "exhaust pipe", "gutter", "tin", "binder", "oven door", "wrist watch", "dust", "cactus", "tv stand", "monument", "soccer", "packet", "clay", "wooden post", "rear wheel", "hand rail", "moustache", "mixer", "crest", "glaze", "tea", "electrical outlet", "plain", "chalk", "parsley", "tall building", "pink flower", "ivy", "heater", "folder", "wooden bench", "stirrup", "black tire", "rooftop", "face mask", "charger", "shield", "dryer", "parasol", "stuffed bear", "sunshine", "portrait", "hamburger", "dumpster", "green pepper", "disk", "cop", "hip", "life preserver", "floor tile", "shin guard", "sponge", "bathing suit", "pottery", "grafitti", "dinner", "match", "canal", "christmas tree", "buggy", "armchair", "pail", "train platform", "white shoe", "shingle", "skatepark", "silhouette", "book shelf", "left wing", "soap dish", "light bulb", "outfield", "ostrich", "pickup truck", "helicopter", "rag", "thread", "oil", "evergreen tree", "shopping bag", "right wing", "ceiling fan", "lunch", "bud", "muffler", "bike rack", "garnish", "cane", "silver car", "coffee pot", "sack", "signal light", "elbow pad", "beer bottle", "pepper shaker", "rock wall", "green bush", "sandwhich", "scissor", "tall tree", "tulip", "ski jacket", "orange slice", "mural", "dvd", "tangerine", "boardwalk", "life jacket", "playground", "asparagus", "lipstick", "kneepad", "tissue box", "referee", "weather vane", "green sign", "cowboy hat", "tail fin", "conductor", "keypad", "runner", "streetlamp", "head band", "gold", "safety cone", "garlic", "signboard", "flag pole", "school bus", "hill side", "jet engine", "condiment", "grain", "surf", "skillet", "raspberry", "bowtie", "gravy", "blue water", "banana bunch", "tea kettle", "tomato slice", "smoke stack", "pew", "rainbow", "sunflower", "tile floor", "floor lamp", "surfing", "tomato sauce", "pump", "young girl", "glass door", "throw pillow", "page", "gap", "spout", "breast", "cargo", "shoelace", "bath tub", "breakfast", "melon", "buffalo", "gazelle", "tail wing", "braid", "syrup", "plunger", "little boy", "elephant trunk", "snowflake", "white paper", "red pepper", "place mat", "ox", "drum", "slide", "goal", "nike logo", "young boy", "wooden chair", "firetruck", "drape", "fire extinguisher", "ashtray", "white flower", "beach chair", "tomatoe", "cyclist", "potted plant", "spice", "link", "chalkboard", "tripod", "yacht", "pizza crust", "baseball uniform", "skylight", "dandelion", "podium", "powerlines", "tights", "city street", "clock hand", "urn", "stovetop", "evergreen", "baseball hat", "black coat", "bed frame", "rearview mirror", "ski suit", "carrier", "purple flower", "carving", "pillowcase", "tanktop", "red flower", "birthday cake", "framed picture", "wreath", "stitching", "flagpole", "blue car", "envelope", "gas station", "signpost", "steel", "pine", "flyer", "cape", "toilet brush", "flour", "small window", "sail boat", "brown cow", "white pillow", "comb", "padding", "tennis match", "broom", "police car", "white truck", "white surfboard", "gown", "cracker", "tassel", "shell", "gray shirt", "white cap", "black cow", "bell pepper", "iron", "wood table", "passenger car", "food truck", "donkey", "canvas", "automobile", "black eye", "office chair", "rear tire", "zucchini", "shower door", "fern", "orange cone", "white house", "mountain top", "white toilet", "black suit", "white boat", "passenger window", "white sheep", "pillow case", "yarn", "store front", "pylon", "black pole", "white door", "lounge chair", "pizza box", "mitten", "needle", "red hat", "paper bag", "white sock", "radish", "pallet", "kiwi", "red umbrella", "cobblestone", "turbine", "wooden pole", "trash bag", "croissant")
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF.caffemodel')}


def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("demo"+image_name)
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.3
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)
        thresh = CONF_THRESH
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
           continue 

        im = im[:, :, (2, 1, 0)]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig("demo/"+image_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
    print caffemodel
    print prototxt
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = [#'000456.jpg', '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg', 
                '2350796.jpg', '2374454.jpg', '2346864.jpg',
                '2356076.jpg', '1159449.jpg', '2349840.jpg', 
                '2408794.jpg','1159568.jpg','1160228.jpg', 
                '2316207.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

