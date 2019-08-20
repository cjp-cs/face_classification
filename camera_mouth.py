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


#CAP_SOURCE ="../collect/via_member/darkroom/zimdyvideo5.h264"
CAP_SOURCE = 0

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

    # modify according to input size
    IMG_H = 32
    IMG_W = 32
    DIM = 1

    # What model to download.
    MODEL_DIR = 'face_detect/tf_model'
    PATH_TO_FACE_PB = MODEL_DIR + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(MODEL_DIR, 'wider_face_label_map.pbtxt')

    MOUTH_DIR = 'model20'
    PATH_TO_MOUTH_PB = MOUTH_DIR + '/frozen_model.pb'

    # load mouth detection graph
    graph = load_graph(PATH_TO_MOUTH_PB)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/input_layer:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

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

    count = 0    
    t = 0
    op = 0
    smoke = 0

    with tf.Session(graph=graph) as sess:
        while (1):
            # get a frame
            ret, frame = cap.read()
            # show a frame
            face_detector.update_frame(frame)
            if face_detector.read_flag() == False:
                face_detector.update_flag(True)
            faces = face_detector.get_face()
            
            if (len(faces) > 0) :
                cropstatus = True

            area = 0

            for face in faces:
                w = face.xmax-face.xmin
                h = face.ymax-face.ymin

                tlx = face.xmin
                tly = face.ymin
                brx = face.xmax
                bry = face.ymax

                # record coordinates of face with largest area
                if (brx - tlx) * (bry - tly) > area:
                    area = (brx - tlx) * (bry - tly)
                    xmin = tlx
                    xmax = brx
                    ymin = tly
                    ymax = bry
            
            if len(faces) > 0:

                # show face
                cropI = frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                cropI = cv2.resize(cropI, (128, 128))
                cv2.imshow("face", cropI)

                # crop and show mouth
                cropII = cropI[70:120]

                cv2.imshow("mouth", cropII)
                cropII = cv2.cvtColor(cropII,cv2.COLOR_BGR2GRAY)
                mouth= cv2.resize(cropII,(IMG_H,IMG_W),interpolation=cv2.INTER_AREA)	
                mouth2 = np.reshape(mouth,(1,IMG_H,IMG_W,DIM))

                #mouth = cv2.resize(cropII, (128, 128))
                mouth2 = mouth2/255.0

                count += 1

                # run prediction and record computation time - disregard first image/gpu initialization time
                start = time.time()
                pred = sess.run(y,feed_dict={x:mouth2})
                end = time.time()
                if count != 1:
                    t += (end - start)

                maxId = np.argmax(pred)
                predScore = pred[0][maxId]

                # Display class, score, and warning if applicable - need 5 consecutive frames of mouth open for warning to appear
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (40, 80)
                fontScale              = 3
                fontColor              = (0, 0, 255)
                lineType               = 2

                if str(maxId) == '0':
                    status = 'closed'
                    smoke = 0
                    op = 0
                elif str(maxId) == '1':
                    status = 'open'
                    op += 1
                    if op >= 5:
                        cv2.putText(frame,'WARNING!', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
                else:
                    status = 'smoking'
                    smoke += 1
                    if smoke >= 5:
                        cv2.putText(frame,'WARNING!', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
                
                print("Class: " + status + "  Score : " + str(predScore))

            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                face_detector.draw_flag = False
                break
        detect_thread.join()
        cap.release()
        cv2.destroyAllWindows()
    
    print('Average speed per image: {} seconds'.format(t / count))

    # write speed results to text file
    file1 = open(MOUTH_DIR + "/{}_results.txt".format(MOUTH_DIR),"a") 
    file1.write('Average speed per image: {} seconds\n'.format(t / count))
    file1.close()