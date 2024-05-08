import gradio as gr
import cv2
import torch
theme3=gr.themes.Soft()
def detection(image,iou,conf):
    model.conf=conf
    model.iou=iou
    return model(image).render()[0]

def tracker(image,speed):
    pass

model = torch.hub.load('./', 'yolov5s', source='local')
# 布局
with gr.Blocks(theme=theme3,css="footer {visibility: hidden}") as demo:
    gr.Markdown('# 基于改进的YOLOv5的鱼病自动检测系统')
    gr.Markdown('# ')
    gr.Markdown('# ')
    gr.Markdown('# ')
    with gr.Tab(label='EUS鱼病检测'):
        gr.Markdown('# EUS鱼病检测')
        gr.Markdown('提交EUS鱼病图片即可进行检测')
        with gr.Row():
            image1=gr.Image(label='image')
            image2=gr.Image(label='result')
        with gr.Row():
            iou_thres=gr.Slider(value=0.45,label='IOU阈值',minimum=0,maximum=1,info='IOU阈值越高，检测的目标越多')
            conf_thres=gr.Slider(value=0.25,label='置信度阈值',minimum=0,maximum=1,info='置信度阈值越高，检检测的目标越少')
        with gr.Row():
            button5=gr.Button('检测',variant='primary')
            gr.ClearButton([image1,image2])
            button5.click(detection,inputs=[image1,iou_thres,conf_thres],outputs=image2)

    with gr.Tab(label='鱼类异常轨迹检测'):
        gr.Markdown('# 鱼类异常轨迹检测')
        gr.Markdown('提交鱼类视频或实时视频流即可开始检测')
        with gr.Row():
            image3=gr.Image(label='video')
            image4=gr.Image(label='result')
        with gr.Row():
            speed=gr.Slider(label='速度阈值',minimum=0,maximum=10,info='高于该阈值的鱼类将不会显示')
        with gr.Row():
            button5=gr.Button('检测',variant='primary')
            gr.ClearButton([image3,image4])
            button5.click(tracker,inputs=[image3,speed],outputs=image4)


demo.launch()