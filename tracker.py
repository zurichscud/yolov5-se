'''
iou追踪示例
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time
import random
import os
from shapely.geometry import Polygon, LineString
import json

class IouTracker:
    def __init__(self):

        # 加载检测模型
        self.detection_model = YOLO("./best.pt")
        # 获取类别
        self.objs_labels = self.detection_model.names
        # 打印类别
        print(self.objs_labels)
        # 只处理person
        self.track_classes = {0: 'fish'}

        # 追踪的IOU阈值
        self.sigma_iou = 0.2
        # detection threshold
        self.conf_thresh = 0.3
        self.track_history = {}
        # 速度阈值
        self.speed_threshold = 2  # 像素/秒

    def iou(sel,bbox1, bbox2):
        """
        计算两个bounding box的IOU
        """

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # 计算重叠的矩形的坐标
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # 检查是否有重叠
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # 计算重叠矩形的面积以及两个矩形的面积
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        # 计算IOU
        return size_intersection / size_union


    def predict(self, frame):
        '''
        检测
        '''
        result = list(self.detection_model(frame, stream=True, conf=self.conf_thresh))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array

        dets = [] # 检测结果
        # 遍历每个框
        for box in boxes.data:
            l,t,r,b = box[:4] # left, top, right, bottom
            conf, class_id = box[4:] # confidence, class
            # 排除不需要追踪的类别
            if class_id not in self.track_classes:
                continue
            dets.append({'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id })
        return dets


    def main(self):
        '''
        主函数
        '''
        # 读取视频
        cap = cv2.VideoCapture("test.mp4")

        # 获取视频帧率、宽、高
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"fps: {fps}, width: {width}, height: {height}")


        tracks_active = [] # 活跃的跟踪器
        frame_id = 1 # 帧ID
        track_idx = 1 # 跟踪器ID

        # writer
        out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

        while True:
            # 读取一帧
            start_time = time.time()
            ret, raw_frame = cap.read()
            if ret:
                # 检测
                frame = cv2.resize(raw_frame, (1280, 720))
                raw_frame = frame
                dets = self.predict(raw_frame)
                # 更新后的跟踪器
                updated_tracks = []
                # 遍历活跃的跟踪器
                for track in tracks_active:
                    if len(dets) > 0:
                        # 根据最大IOU更新跟踪器，先去explain.ipynb中看一下MAX用法
                        best_match = max(dets, key=lambda x: self.iou(track['bboxes'][-1], x['bbox'])) # 找出dets中与当前跟踪器（track['bboxes'][-1]）最匹配的检测框（IOU最大）
                        # 如果最大IOU大于阈值，则将本次检测结果加入跟踪器
                        if self.iou(track['bboxes'][-1], best_match['bbox']) > self.sigma_iou:
                            # 将本次检测结果加入跟踪器
                            track['bboxes'].append(best_match['bbox'])
                            track['max_score'] = max(track['max_score'], best_match['score'])
                            track['frame_ids'].append(frame_id)
                            # 更新跟踪器
                            updated_tracks.append(track)
                            # 删除已经匹配的检测框，避免后续重复匹配以及新建跟踪器
                            del dets[dets.index(best_match)]


                # 如有未分配的目标，创建新的跟踪器
                new_tracks = []
                for det in dets: # 未分配的目标，已经分配的目标已经从dets中删除
                    new_track = {
                        'bboxes': [det['bbox']], # 跟踪目标的矩形框
                        'max_score': det['score'], # 跟踪目标的最大score
                        'start_frame': frame_id,  # 目标出现的 帧id
                        'frame_ids': [frame_id],  # 目标出现的所有帧id
                        'track_id': track_idx,    # 跟踪标号
                        'class_id': det['class_id'], # 类别
                        'is_counted': False       # 是否已经计数
                    }
                    track_idx += 1
                    new_tracks.append(new_track)
                # 最终的跟踪器
                tracks_active = updated_tracks + new_tracks

                cross_line_color = (0,255,0) # 越界线的颜色

                # 绘制跟踪器
                for tracker in tracks_active:
                    # 计算实时速度
                    if len(tracker['bboxes']) > 1:
                        prev_bbox = tracker['bboxes'][-2]
                        curr_bbox = tracker['bboxes'][-1]
                        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                        curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
                        displacement = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
                        velocity = displacement * fps  # 计算速度
                        # 如果速度小于阈值，绘制跟踪器的矩形框
                        if 1 < velocity <= self.speed_threshold:
                            l,t,r,b = tracker['bboxes'][-1]
                            # 取整
                            l,t,r,b = int(l), int(t), int(r), int(b)
                            class_id = tracker['class_id']
                            cv2.rectangle(raw_frame, (l,t), (r,b), cross_line_color, 2)
                            #pixels/s
                            cv2.putText(raw_frame, f"v: {velocity:.2f}", (l, t - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)

                # 设置半透明
                color = (0,0,0)
                alpha = 0.2
                l,t = 0,0
                r,b = l+240,t+40
                raw_frame[t:b,l:r,0] = raw_frame[t:b,l:r,0] * alpha + color[0] * (1-alpha)
                raw_frame[t:b,l:r,1] = raw_frame[t:b,l:r,1] * alpha + color[1] * (1-alpha)
                raw_frame[t:b,l:r,2] = raw_frame[t:b,l:r,2] * alpha + color[2] * (1-alpha)

                # end time
                end_time = time.time()
                # FPS
                fps = 1 / (end_time - start_time)
                # 绘制FPS
                cv2.putText(raw_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



                # 显示
                cv2.imshow("frame", raw_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                out.write(raw_frame)
            else:
                break

        out.release()

    # 实例化
iou_tracker = IouTracker()
# 运行
iou_tracker.main()
