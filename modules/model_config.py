# 创建一个统一的模型路径管理文件 model_config.py
MODEL_PATHS = {
    "bert-base-uncased": "/home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased",
    "clip-vit-large-patch14": "/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14",
    "facebook/dinov2-large": "/home/ma-user/work/zhangWei/mtu3d/data/trans/dinov2-large",
    "t5-small": "/home/ma-user/work/zhangWei/mtu3d/data/trans/t5-small",
    "openai/clip-vit-large-patch14": "/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14"
}

# 在其他文件中使用
# from model_config import MODEL_PATHS
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["bert-base-uncased"])