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
            print("fps : {}".format(fps))
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
    # What model to download.
    MODEL_DIR = 'tf_model'

    PATH_TO_FACE_PB = MODEL_DIR + '/frozen_inference_graph.pb'

    #  = "../pb/facial_tight.pb"

    PATH_TO_LABELS = os.path.join(MODEL_DIR, 'wider_face_label_map.pbtxt')

    PATH_TO_FROZEN_GRAPH = '/home/chelsea/Documents/resnet_frozen_ckpt/v1.pb'

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

    frozen_graph = load_graph(PATH_TO_FROZEN_GRAPH)

    x = frozen_graph.get_tensor_by_name('prefix/input:0')
    y = frozen_graph.get_tensor_by_name('prefix/output:0')

    font = cv2.FONT_HERSHEY_SIMPLEX

    closed_count = 0
    face_area = 0
    count = 1
    total_time = 0

    with tf.Session(graph=frozen_graph) as sess:
        while (1):
            # get a frame
            ret, frame = cap.read() # get images from webcam
            # show a frame
            face_detector.update_frame(frame)
            if face_detector.read_flag() == False:
                face_detector.update_flag(True)
            faces = face_detector.get_face()

            face_area = 0

            if (len(faces) > 0) :
                cropstatus = True

            for face in faces:
                w = face.xmax-face.xmin
                h = face.ymax-face.ymin

                tlx = face.xmin
                tly = face.ymin
                brx = face.xmax
                bry = face.ymax
                print(tlx)
                print(tly)

                # Calculate areas and set the face crop to the largest one
                if (w * h) > face_area:
                    face_area = w * h
                    face_crop = frame[int(tly):int(bry), int(tlx):int(brx)]
                    cv2.imshow("initial face crop", face_crop)

            if len(faces) > 0:

                # Crop eyes
                cropped_eyes = face_crop[int(0.2*face_crop.shape[0]):int(0.6*face_crop.shape[0]),:]
                cv2.imshow("cropped eyes", cropped_eyes)

                # Resize
                resized = cv2.resize(cropped_eyes, (128, 128))
                cv2.imshow("resized eyes", cropped_eyes)

                # Convert to grayscale
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                gray = gray/255.0

                # Use numpy to reshape
                reshaped = np.reshape(gray, (1, 128, 128, 1))

                # Feed final image into sess.run()
                start = time.time()
                result = sess.run(y,feed_dict={x:reshaped})
                end = time.time()
                count += 1

                prediction_time = end - start
                total_time += prediction_time

                prediction = np.argmax(result)

                if str(prediction) == '0':
                    print("Prediction: eyes open")
                    closed_count = 0

                elif str(prediction) == '1':
                    print("Prediction: eyes closed")
                    closed_count += 1
                    if closed_count >= 5:
                        cv2.putText(face_crop, "WARNING", (10, 100), font, 1, (0, 0, 255), 2)
                        cv2.imshow("crop1", face_crop)

                elif str(prediction) == '2':
                    cv2.putText(face_crop, "WARNING", (10, 100), font, 1, (0, 0, 255), 2)
                    cv2.imshow("initial face crop", face_crop)
                    print("Prediction: no eyes detected")

                # print(type(prediction))
                # print('result: ' + str(result))
                # print('prediction: ' + str(prediction))

            cv2.imshow("capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                face_detector.draw_flag = False
                break

        print("Time: " + str(total_time / count))
            
        detect_thread.join()
        cap.release()
        cv2.destroyAllWindows()

