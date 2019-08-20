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

IMG_H = 128
IMG_W = 128
DIM = 1
prethres = 0.9

#PB_DIR = "../pb/v15/v15_4class_70%.pb"
#PB_DIR = "../pb/mobilenet/+mobilenetv3_2.pb"
#SAVE_FOLDER = "../result/John_20190710_084730_v15+noclahe/"
CAP_SOURCE = 0
#CAP_SOURCE = "../dataset/John/record-20190710_084730-1280x720.mp4"
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

    #PATH_TO_LM_PB = PB_DIR

    PATH_TO_LABELS = os.path.join(MODEL_DIR, 'wider_face_label_map.pbtxt')



    # We use our "load_graph" function
    graph = load_graph(PATH_TO_FACE_PB)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/input:0')
    #x = graph.get_tensor_by_name('prefix/Reshape:0')
    y = graph.get_tensor_by_name('prefix/output:0')
    #y = graph.get_tensor_by_name('prefix/mobilenetv3_small/prob:0')
    NUM_CLASSES = 7
    face_detector = FaceDetector(PATH_TO_FACE_PB, PATH_TO_LABELS, NUM_CLASSES)
    detect_thread = DetectThread(face_detector)
    detect_thread.start()
    cap = cv2.VideoCapture(CAP_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    font = cv2.FONT_HERSHEY_TRIPLEX
    faces = []

    addpixRatio = 0.0
    cropstatus = False
    pretlx = 0
    pretly = 0
    prebrx = 0 
    prebry = 0

    noface_count = 0
    grayt = np.zeros((1,IMG_H,IMG_W,1))

    with tf.Session(graph=graph) as sess:

        distract = False
        warn = []
        warnStatus = 0

        num = 0
        while (1):
            # get a frame
            ret, frame = cap.read()
            # Test gray
            framegray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	        # show a frame
            face_detector.update_frame(frame)
            if face_detector.read_flag() == False:
                face_detector.update_flag(True)
            faces = face_detector.get_face()
	    #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
            if len(faces) > 0:
                cropstatus = True
                for face in faces:
                    w = face.xmax-face.xmin
                    h = face.ymax-face.ymin
                    tlx = face.xmin - w*addpixRatio
                    tly = face.ymin 
                    brx = face.xmax + w*addpixRatio
                    bry = face.ymax + h*addpixRatio

            if tlx<0:
                tlx = 0
            if tly<0:
                tly = 0
            if brx>frame.shape[1]:
                brx = frame.shape[1]
            if bry>frame.shape[0]:
                bry = frame.shape[0]


                pretlx = tlx
                pretly = tly
                prebrx = brx 
                prebry = bry

                cropI = frame[int(tly):int(bry),int(tlx):int(brx)]

                cropI = cv2.cvtColor(cropI,cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cropI = clahe.apply(cropI)
		   
    		    #cropI = cv2.equalizeHist(cropI)

                    #cv2.imshow("cropI", cropI)
	    	    #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",cropI)
                gray= cv2.resize(cropI,(IMG_H,IMG_W),interpolation=cv2.INTER_AREA)	
                grayt = np.reshape(gray,(1,IMG_H,IMG_W,DIM))

                noface_count = 0

                # show image
                cv2.rectangle(frame, (int(tlx), int(tly)), (int(brx), int(bry)), (0, 255, 0), 1)
                score = str(face.score)[:5]
                ((text_width, text_height), text_baseline) = cv2.getTextSize('score: {}'.format(score), font, 1, 1)
                w = (brx - tlx - text_width) / 2.0
                cv2.putText(frame, 'score: {}'.format(score), (int(tlx) + int(w), int(tly) - 10),
                    font, 1, (0, 0, 255), 1, False)

 
            elif cropstatus == True:

                cropI = frame[int(pretly):int(prebry),int(pretlx):int(prebrx)]
                cropI = cv2.cvtColor(cropI,cv2.COLOR_BGR2GRAY)
                #cropI = cv2.equalizeHist(cropI)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cropI = clahe.apply(cropI)
                        #cv2.imshow("cropI", cropI)
                    #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",cropI)
                gray= cv2.resize(cropI,(IMG_H,IMG_W),interpolation=cv2.INTER_AREA)
                grayt = np.reshape(gray,(1,IMG_H,IMG_W,DIM))
                noface_count +=1
                cv2.rectangle(frame, (int(pretlx), int(pretly)), (int(prebrx), int(prebry)), (0, 255, 0), 1)
                score = str(face.score)[:5]
                ((text_width, text_height), text_baseline) = cv2.getTextSize('score: {}'.format(score), font, 1, 1)
                w = (brx - tlx - text_width) / 2.0
                cv2.putText(frame, 'score: {}'.format(score), (int(tlx) + int(w), int(tly) - 10),
                    font, 1, (0, 255, 0), 1, False)
                if noface_count >20:
                    cropstatus=False
                    warn = [] 
                    distract = False
                    warnStatus = 0

                    score = str(face.score)[:5]
                    ((text_width, text_height), text_baseline) = cv2.getTextSize('score: {}'.format(score), font, 1, 1)
                    w = (brx - tlx - text_width) / 2.0
                    cv2.putText(frame, 'score: {}'.format(score), (int(tlx) + int(w), int(tly) - 10),
                        font, 1, (0,255, 0), 1, False)	


            ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            #if (maxId == 1 or        ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()r maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 o       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            
            else:	
                warn.append(0)       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            if warnStatus >=10       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()um of frames detect
                if ((sum(warn)       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows())*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num       ## facial warning
            
            grayt = grayt/255.0
            pred = sess.run(y,feed_dict={x:grayt})
            maxId = np.argmax(pred)
            predScore = pred[0][maxId]

            if cropstatus == False:
                maxId = 0

            print("Classify Status : "+str(maxId)+"  Score : "+str(predScore))



            #Distract Status
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or  maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                distract = True


            if distract==True:
                warnStatus += 1
            if (maxId == 1 or maxId==3 or maxId == 4 or maxId == 5 or maxId == 7 or maxId==8 or maxId==9 or maxId==10) and predScore >prethres:
            #if ((maxId == 1 or maxId==2) or (maxId == 3 or maxId == 4) or (maxId==8 or maxId == 5)) and predScore >prethres:
                warn.append(1)
            
            else:	
                warn.append(0)
            if warnStatus >=10: #chose sum of frames detect
                if ((sum(warn)/(len(warn)*1.0))>0.5)and(num>30): #0.5 = distract ratio  in frames , after initial 30 frames start
                    #print num
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()
                    print("warning")
                    #if goal1 !=0:
                    cv2.putText(frame,"DISTRACT!!!",(100,100), font,3,(0,255,255),6,cv2.LINE_AA)
                    warnStatus -= 1
                    if len(warn)>0: #prevent if no appends pop null
                        warn.pop(0)
                    #print warnStatus 
                else:
                    warn = [] #2018/07/24 add this
                    distract = False
                    warnStatus = 0
                num+=1
                #cv2.imwrite(SAVE_FOLDER+str(num)+".jpg",frame)
                cv2.imshow("capture", frame)
                #cv2.imshow("framegray", framegray)       
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detector.draw_flag = False
                    break
            detect_thread.join()
            cap.release()
            cv2.destroyAllWindows()
