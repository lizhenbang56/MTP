# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
from loguru import logger
import os.path as osp
import random
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg
import os


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')
    parser.add_argument('--list_file', default='')
    parser.add_argument('--loop', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='3')
    return parser


def build_siamfcpp_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers


def build_sat_tester(task_cfg):
    # build model
    tracker_model = model_builder.build("track", task_cfg.tracker_model)
    tracker = pipeline_builder.build("track",
                                     task_cfg.tracker_pipeline,
                                     model=tracker_model)
    segmenter = model_builder.build('vos', task_cfg.segmenter)
    # build pipeline
    pipeline = pipeline_builder.build('vos',
                                      task_cfg.pipeline,
                                      segmenter=segmenter,
                                      tracker=tracker)
    # build tester
    testers = tester_builder('vos', task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.gpu

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    if task == 'track':
        testers = build_siamfcpp_tester(task_cfg)
    elif task == 'vos':
        testers = build_sat_tester(task_cfg)
    for tester in testers:
        list_file = parsed_args.list_file
        blur_param = random.uniform(0.0, 2.0)  # 模糊：0~2
        illum_param = random.uniform(0.5, 1.5)  # 光照：0.5~1.5
        #box_param = random.uniform(0.0, 0.13)  # 对应IoU：0.6~1
        box_param = random.uniform(0.0, 0.07)  # 对应IoU：0.76~1
        noise = {'blur': blur_param, 'illum': illum_param, 'box': box_param}

        #noise = {'blur': 0.7274792219167221, 'illum': 0.591336904517266, 'box': 0.09983950738438359}
        #noise = {'blur': 0.43215783927044993, 'illum': 0.8813870565498486, 'box': 0.06702118765679675}
        #noise = {'blur': 1.2803719499595194, 'illum': 1.45929789395201, 'box': 0.039839414937659696}
        #noise = {'blur': 1.268651985692494, 'illum': 1.1265446445617542, 'box': 0.020809575792020297}
        #noise = {'blur': 1.1234396768366015, 'illum': 0.7448875786003296, 'box': 0.11913520040610091}
        #noise = {'blur': 0.5408449180484376, 'illum': 1.042258692150103, 'box': 0.09756033450772304}
        # noise = {'blur': 0.05790033670648609, 'illum': 0.9896036770023554, 'box': 0.08350780543791268}
        # noise = {'blur': 1.0626950366986398, 'illum': 0.8185168638891629, 'box': 0.04969270026515132}
        # noise = {'blur': 1.597375502807567, 'illum': 0.5719884240427697, 'box': 0.018498100313157285}
        print(noise)
        if len(list_file) == 0:
            tester.test(noise=noise, loop=parsed_args.loop)
            tester.test(noise=noise, loop=0)
        else:
            tester.test(list_file)
