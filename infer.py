import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from torchvision import transforms
import torchvision
from PIL import Image
import cv2 
import torch 
import numpy as np
import streamlit as st
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from glob import glob 

def list_file_in_folder(path, ext):
    return glob(path + f"/*.{ext}")

def read_image(img_path):
    img = torchvision.io.read_image(img_path)
    
fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
# num_entities, dim = 3000, 8
dim = 768

if __name__ == "__main__":
    #-------------------------Load model-------------------------
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/PGDS/configs/market/swin_tiny.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    device = "cpu"
    head_fusion = "max"
    discard_ratio = 0.9
    category_index = None
    w, h = 384, 128
    transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class= 77, camera_num=3, view_num = 3, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    print(model)
    model.eval()

    #------------------------End load model-------------------------------------------------
    
    #------------------------Generate test embedding vector--------------------------------
    # embedded_vectors = []
    # imgs = list_file_in_folder("/DukeMTMC-reID/test", "jpg")
    # for imagePath in imgs:
    #     img = Image.open(imagePath).convert('RGB')
    #     # img = img.resize((w, h))
    #     # print(img.shape)
    #     input_tensor = transform(img).unsqueeze(0)
    #     with torch.no_grad():
    #         model = model.to(device)
    #         input_tensor = input_tensor.to(device)
    #         out = model(input_tensor)
    #     embedded_vectors.append(out[0])
    #     # print(out[0].shape)
    # np.save("/DukeMTMC-reID/testOutput.npy", embedded_vectors)
    #------------------------Generate test embedding vector--------------------------------

    #------------------------Connect milvus--------------------------------
    print(fmt.format("start connecting to Milvus"))
    client = connections.connect("default", host="localhost", port="19530")

    has = utility.has_collection("pgds")
    print(f"Does collection pgds exist in Milvus: {has}")
    pgds_col = Collection("pgds")
    pgds_col.load()

    #------------------------ Connect milvus--------------------------------
    
    
    #------------------------Streamlit search app--------------------------------
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            model = model.to(device)
            input_tensor = input_tensor.to(device)
            out = model(input_tensor)
        print(fmt.format("Start searching"))
        search_params = {
            "metric_type": "COSINE",
            "params": {},
        }
        vector_to_search = np.array(out[0])
        result = pgds_col.search(vector_to_search, "embeddings", search_params, limit=20, output_fields=["filepath"])
        paths = [hit.entity.get('filepath') for hits in result for hit in hits]
        st.image(paths)