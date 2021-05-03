import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import YoloV3Tiny,YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    # yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    # yolo.load_weights('./weights/yolov3-tiny.tf')
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights('./weights/yolov3.tf')
    #yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    counter=[]
    gct=[]
    from _collections import deque
    foo_hist = [deque(maxlen=30) for _ in range(100)]
    while True:
        _, img = vid.read()

        if img is None:
            print('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        current_count = 0
        left_gate = False
        right_gate = False

        foo={}

        #foo_hist=[]

        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            if class_name=='person':
                counter.append(int(track.track_id))
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
            else:
                continue
            #print('Center:',center)
            cv2.circle(img,center,10,(0,255,255),3)
            height, width, _ = img.shape
            # #LL
            # cv2.line(img, (int(width/2-width/10),0), ( int(width/2-width/10),height), (0, 255, 0), thickness=2)
            # #LR
            # cv2.line(img, (int(width/2-width/30),0), ( int(width/2-width/30),height), (0, 255, 0), thickness=2)
            # #RR
            # cv2.line(img, (int((width/2+width/10)),0), ( int(width/2+width/10),height), (0, 255, 0), thickness=2)
            # #RL
            # cv2.line(img, (int((width/2+width/30)),0), ( int(width/2+width/30),height), (0,255, 0), thickness=2)
######################TRIAL
            cv2.line(img, (int(width/2),0), (int(width/2),height), (0,255, 0), thickness=2)


            foo_hist[track.track_id].append(center)
            for j in range(1, len(foo_hist[track.track_id])):
                if foo_hist[track.track_id][j-1] is None or foo_hist[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64/float(j+1))*2)
                cv2.line(img, (foo_hist[track.track_id][j-1]), (foo_hist[track.track_id][j]), color, thickness)

                print('ID:',track.track_id,'foo_hist:',foo_hist[track.track_id])
            center_x = int((bbox[0]+bbox[2])/2)
            # foo[track.track_id]=center_x
            # foo_hist.append(foo)
            # # print('Track ID:',track.track_id)
            # # print('center_x:',center_x)
            # #print(foo_hist[-1])
            # #print(foo)
            # come_from_left=[]
            # come_from_right=[]
            # lc=0
            # for key,val in foo_hist[-1].items():
            #      #print('ID:',key,'Position:',val)
            #      if val<500:
            #          #print('ID',key,'is on left')
            #          come_from_left.append(key)
            #          lc+=1
            #          print('LeftCounter:',lc)
            # print('LEFT:',come_from_left)
            # # for x in come_from_left:
            # #     # if foo_hist[x].values()>500:
            # #     #     print('YAAY')
            # #     print(x)
            # rc=0
            # for key,val in foo_hist[-1].items():
            #     if val>500 :
            #         #print('ID',key,'is on right')
            #         come_from_right.append(key)
            #         rc+=1
            #         print('RightCounter:',rc)
            # print('RIGHT:',come_from_right)
            # # for x in come_from_left:
            # #     if x in come_from_right:
            # #         print(x,'has moved fro  left to right')

#####################################################################################
            # if center_x <= (int(width/2-width/30)) and center_x >= (int(width/2-width/10)):
            #     #cv2.putText(img, "LEFT Passed ", (1000,250), 0, 1, (255,0,255), 2)
            #     left_gate=True
            # if center_x <= (int(width/2+width/10)) and center_x >= (int(width/2+width/30)):
            #     #cv2.putText(img, "RIGHT Passed " , (1000,500), 0, 1, (255,0,255), 2)
            #     right_gate=True
            # if center_x <= (int(width/2+width/10)) and center_x >= (int(width/2+width/30)) and left_gate==True:
            #
            #     gct.append(int(track.track_id))
            #     cv2.putText(img, 'ID who crossed:'+str(track.track_id) , (900,300), 0, 1, (0,0,255), 2)
#####################################################################################
            if center_x <= (int(width/2-width/30)) and center_x >= (int(width/2-width/10)):
                #cv2.putText(img, "LEFT Passed ", (1000,250), 0, 1, (255,0,255), 2)
                left_gate=True
            if center_x <= (int(width/2+width/10)) and center_x >= (int(width/2+width/30)):
                #cv2.putText(img, "RIGHT Passed " , (1000,500), 0, 1, (255,0,255), 2)
                right_gate=True
            if center_x <= (int(width/2+width/10)) and center_x >= (int(width/2+width/30)) and left_gate==True:

                gct.append(int(track.track_id))
                cv2.putText(img, 'ID who crossed:'+str(track.track_id) , (900,300), 0, 1, (0,0,255), 2)
######################################################
            #print('ID:',track.track_id,'\nPosition:',foo_hist[track.track_id])

        cv2.putText(img, "Moved Left2Right: " + str(len(set(gct))), (900,100), 0, 1, (0,0,255), 2)
        # print fps on screen
        total_count = len(set(counter))
        #cv2.putText(img, "Current Person Count: " + str(current_count), (0,80), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Total Person Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)
        #cv2.putText(img, "Total Person Left2Right: " + str(counter_l2r), (0,60), 0, 1, (0,0,255), 2)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
