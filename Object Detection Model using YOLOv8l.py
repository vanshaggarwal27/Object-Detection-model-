from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#cap=cv2.VideoCapture("cycles.mp4")
model = YOLO("yolov8l.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", 
    "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush", "tree", 
    "grass", "cloud", "moon", "sun", "flower", "rock", "fence", "building", "door", "window", "stairs", "carpet", 
    "pillow", "blanket", "lamp", "mirror", "painting", "photo frame", "candle", "light bulb", "fan", "air conditioner", 
    "heater", "desk", "shelf", "wardrobe", "drawer", "washing machine", "dryer", "dishwasher", "blender", "coffee maker", 
    "toaster oven", "food processor", "kettle", "microwave oven", "stove", "grill", "pan", "pot", "cutting board", 
    "measuring cup", "whisk", "spatula", "peeler", "knife sharpener", "thermometer", "ice cream scoop", "rolling pin", 
    "can opener", "plate", "bowl", "napkin", "paper towel", "tray", "shopping bag", "cushion", "rug", "curtain", 
    "blinds", "fireplace", "ceiling fan", "shower", "bath tub", "razor", "shaving cream", "toothpaste", "mouthwash", 
    "hairbrush", "comb", "shampoo", "conditioner", "soap", "sponge", "towel", "toilet paper", "plunger", "trash can", 
    "broom", "mop", "vacuum cleaner", "bucket", "dustpan", "detergent", "spray bottle", "cleaning cloth", "laundry basket", 
    "iron", "ironing board", "hanger", "shoe", "boot", "slipper", "sandal", "watch", "bracelet", "necklace", "earrings", 
    "ring", "glasses", "hat", "cap", "scarf", "belt", "wallet", "credit card", "passport", "keys", "pen", "pencil", 
    "eraser", "notebook", "journal", "calculator", "calendar", "ruler", "sticky notes", "paperclip", "binder", "folder", 
    "envelope", "stamp", "glue", "tape", "stapler", "highlighter", "marker", "crayon", "paintbrush", "canvas", "easel", 
    "sculpture", "statue", "trophy", "medal", "badge", "ticket", "map", "compass", "telescope", "binoculars", "flashlight", 
    "battery", "extension cord", "power outlet", "remote control", "joystick", "game controller", "video game console", 
    "headphones", "earbuds", "microphone", "speaker", "camera", "video camera", "tripod", "lens", "memory card", 
    "smartwatch", "tablet", "drone", "robot", "smart home device", "thermostat", "light switch", "doorbell", 
    "security camera", "alarm clock"
]
while True:
    success, img= cap.read()
    results=model(img,stream=True)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0] 
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)                                                   #for bounding box
            #print(x1,y1,x2,y2)
            w,h= x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            
            
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            conf=math.ceil(box.conf[0]*100)/100                                                             #for confidence level
            print(conf)
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1),max(20,y1)),scale=1,thickness=2)   #confidence level inside box
    cv2.imshow("Image",img)
    cv2.waitKey(1)                                                                                          #default camera
