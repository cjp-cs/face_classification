import cv2
import numpy as np
import os
import tensorflow as tf
import time
import label_map_util
import threading

os.environ["KMP_BLOCKTIME"] = '0'
os.environ["KMP_SETTINGS"] = '0'
os.environ["KMP_AFFINITY"]= 'granularity=fine,verbose,compact,1,0'
os.environ["OMP_NUM_THREADS"]= '4'

CAP_SOURCE = 'cartest.mp4'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

# What model to download.
class FaceDetector:
    def __init__(self, pb_path, label_path, num_classes):
        self.pb_path = pb_path
        self.label_path = label_path
        self.detection_graph = tf.Graph()
        self.num_classes = num_classes
        self.label_map = None
        self.categries = None
        self.category_index = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.draw_flag = False
        self.faces_lock = threading.Lock()
        self.faces = []
        self.draw_flag_lock = threading.Lock()
        self.count = 0
        self.start_time = time.time()

    def update_frame(self, frame):
        self.frame_lock.acquire()
        self.frame = frame
        self.frame_lock.release()

    def read_frame(self):
        self.frame_lock.acquire()
        frame = self.frame
        self.frame_lock.release()
        return frame

    def read_flag(self):
        self.draw_flag_lock.acquire()
        flag = self.draw_flag
        self.draw_flag_lock.release()
        return flag

    def update_flag(self, flag):
        self.draw_flag_lock.acquire()
        self.draw_flag = flag
        self.draw_flag_lock.release()

    def update_faces(self, faces):
        self.faces_lock.acquire()
        self.faces = faces
        self.faces_lock.release()

    def get_face(self):
        self.faces_lock.acquire()
        faces = self.faces
        self.faces_lock.release()
        return faces

    def draw_bbox(self, frame, sess, min_score_thresh=0.5, max_boxes_to_draw=20):
        faces = []
        (im_width, im_height) = (frame.shape[1], frame.shape[0])
        image_np = np.array(frame).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        in_dict = {image_tensor: image_np_expanded}

        # Actual detection.
        start_time1 = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        self.count = self.count + 1
        if (self.count == 10):
            elapsed_time = time.time() - self.start_time
            fps = 10.0/elapsed_time
            #print("fps : {}".format(fps))
            self.count = 0
            self.start_time = time.time()

        cost_time1 = time.time() - start_time1
        res_boxes = np.squeeze(boxes)
        res_scores = np.squeeze(scores)

        if not max_boxes_to_draw:
            max_boxes_to_draw = res_boxes.shape[0]
        for i in range(min(max_boxes_to_draw, res_boxes.shape[0])):
            if scores is None or res_scores[i] > min_score_thresh:
                face = FaceRect(res_boxes[i][0] * im_height, res_boxes[i][1] * im_width, res_boxes[i][2] * im_height,
                                res_boxes[i][3] * im_width, res_scores[i])
                faces.append(face)
        self.update_faces(faces)

    def draw_rect(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(self.label_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if (self.read_flag() == False or np.all(self.read_frame()) == None):
                    time.sleep(2.5)
                while (self.read_flag() and np.all(self.read_frame()) != None):
                    self.frame_lock.acquire()
                    src_frame = self.frame
                    self.frame_lock.release()
                    self.draw_bbox(src_frame, sess, 0.7)


class FaceRect:
    def __init__(self, ymin, xmin, ymax, xmax, score):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax
        self.score = score


class DetectThread(threading.Thread):
    def __init__(self, detector):
        threading.Thread.__init__(self)
        self.detector = detector

    def run(self):
        self.detector.draw_rect()


if __name__ == '__main__':

    # Dimensions of mouth input
    MOUTH_IMG_H = 32
    MOUTH_IMG_W = 32
    DIM = 1

    # What face detection model to download
    MODEL_DIR = 'face_detect/tf_model'
    PATH_TO_FACE_PB = MODEL_DIR + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(MODEL_DIR, 'wider_face_label_map.pbtxt')

    PATH_TO_MOUTH_PB = 'mouth_graph.pb'
    PATH_TO_EYES_PB = 'eyes_graph.pb'

    # Load mouth detection graph
    graph_mouth = load_graph(PATH_TO_MOUTH_PB)

    # Load eyes detection graph
    graph_eyes = load_graph(PATH_TO_EYES_PB)


    # We can verify that we can access the list of operations in the graph
    for op in graph_mouth.get_operations():
        print(op.name)
    for op in graph_eyes.get_operations():
        print(op.name)

    # Get input and output tensors from frozen graphs
    x_mouth = graph_mouth.get_tensor_by_name('prefix/input_layer:0')
    y_mouth = graph_mouth.get_tensor_by_name('prefix/softmax_tensor:0')
    x_eyes = graph_eyes.get_tensor_by_name('prefix/input:0')
    y_eyes = graph_eyes.get_tensor_by_name('prefix/output:0')

    # Face detection
    NUM_CLASSES = 3
    face_detector = FaceDetector(PATH_TO_FACE_PB, PATH_TO_LABELS, NUM_CLASSES)
    detect_thread = DetectThread(face_detector)
    detect_thread.start()
    cap = cv2.VideoCapture(CAP_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    font = cv2.FONT_HERSHEY_TRIPLEX
    faces = []

    SAVE_FOLDER = 'face_detect'
    NAME = 'tf_model'

    # Make file
    if os.path.exists(SAVE_FOLDER):
	    print("Folder data has existed")
    else:
	    print("make directory ")
	    os.mkdir(SAVE_FOLDER) 
	    os.mkdir(SAVE_FOLDER+NAME+"_face/") 

    frames = 0    
    total_time = 0
    open_count = 0
    smoke_count = 0
    closed_count = 0

    with tf.Session(graph=graph_mouth) as sess_mouth:
        with tf.Session(graph=graph_eyes) as sess_eyes:
            while (1):
                # Get a frame
                ret, frame = cap.read()

                # Show a frame
                face_detector.update_frame(frame)
                if face_detector.read_flag() == False:
                    face_detector.update_flag(True)
                faces = face_detector.get_face()
                
                if (len(faces) > 0) :
                    cropstatus = True

                # Identify face with largest area
                area = 0

                for face in faces:
                    w = face.xmax-face.xmin
                    h = face.ymax-face.ymin

                    tlx = face.xmin
                    tly = face.ymin
                    brx = face.xmax
                    bry = face.ymax

                    # Record coordinates of face with largest area
                    if (brx - tlx) * (bry - tly) > area:
                        area = (brx - tlx) * (bry - tly)
                        xmin = tlx
                        xmax = brx
                        ymin = tly
                        ymax = bry
                
                if len(faces) > 0:

                    # Crop and show face
                    face_crop = frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                    face_crop = cv2.resize(face_crop, (128, 128))
                    cv2.imshow("face", face_crop)

                    # Crop and show mouth
                    mouth_crop = face_crop[70:120]
                    cv2.imshow("mouth", mouth_crop)
                    mouth_crop = cv2.cvtColor(mouth_crop,cv2.COLOR_BGR2GRAY)
                    mouth= cv2.resize(mouth_crop,(MOUTH_IMG_H, MOUTH_IMG_W),interpolation=cv2.INTER_AREA)	
                    mouth2 = np.reshape(mouth,(1, MOUTH_IMG_H, MOUTH_IMG_W,DIM))
                    mouth2 = mouth2/255.0

                    # Crop and show eyes
                    cropped_eyes = face_crop[int(0.2*face_crop.shape[0]):int(0.6*face_crop.shape[0]),:]
                    cv2.imshow("cropped eyes", cropped_eyes)
                    resized = cv2.resize(cropped_eyes, (128, 128))
                    cv2.imshow("resized eyes", cropped_eyes)

                    # Convert to grayscale
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    gray = gray/255.0

                    # Use numpy to reshape
                    reshaped = np.reshape(gray, (1, 128, 128, 1))

                    # Feed final images into sess.run()
                    start = time.time()
                    result_mouth = sess_mouth.run(y_mouth, feed_dict={x_mouth:mouth2})
                    result_eyes = sess_eyes.run(y_eyes, feed_dict={x_eyes:reshaped})
                    end = time.time()

                    # Record inference time, disregarding GPU initialization time from first frame
                    frames += 1
                    if frames != 1:
                        total_time += (end - start)

                    # Get scores and predictions
                    maxId_mouth = np.argmax(result_mouth)
                    predScore_mouth = result_mouth[0][maxId_mouth]
                    maxId_eyes = np.argmax(result_eyes)
                    predScore_eyes = result_eyes[0][maxId_eyes]

                    # Display class, score, and warning if applicable - need 5 consecutive frames of mouth open for warning to appear
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (40, 80)
                    fontScale              = 3
                    fontColor              = (0, 0, 255)
                    lineType               = 2

                    if str(maxId_mouth) == '0':
                        status_mouth = 'closed'
                        smoke_count = 0
                        open_count = 0
                    elif str(maxId_mouth) == '1':
                        status_mouth = 'open'
                        open_count += 1
                        if open_count >= 5:
                            cv2.putText(frame,'WARNING!', 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
                    else:
                        status_mouth = 'smoking'
                        smoke_count += 1
                        if smoke_count >= 5:
                            cv2.putText(frame,'WARNING!', 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

                    if str(maxId_eyes) == '0':
                        print("Eyes: eyes open")
                        closed_count = 0

                    elif str(maxId_eyes) == '1':
                        print("Eyes: eyes closed")
                        closed_count += 1
                        if closed_count >= 8:
                            cv2.putText(frame,'WARNING!', 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
                    elif str(maxId_eyes) == '2':
                        cv2.putText(frame,'WARNING!', 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
                        print("Eyes: no eyes detected")
                        
                    print("Mouth: " + status_mouth)

                cv2.imshow("capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()
        
        print('Average speed per image: {} seconds'.format(total_time / frames))
