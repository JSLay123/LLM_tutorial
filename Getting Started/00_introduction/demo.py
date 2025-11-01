import gradio as gr
from transformers import *


# # 通过Interface加载pipeline并启动文本分类服务
# gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roborta-base-finetuned-dianping-chinese")).launch()

# 通过Interface加载pipeline并启动阅读理解服务
# 如果无法通过这种方式加载，可以采用离线加载的方式
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()