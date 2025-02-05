corner_detection_config = {
    'path_to_model': './CCCD_identify/model_backend/model_config/corner_detection_config/frozen_inference_graph.pb',
    'nms_ths': 0.3,
    'score_ths': 0.5
}

text_detection_config = {
    'path_to_model': './CCCD_identify/model_backend/model_config/text_detection_config_tflite/best.pt',
    'path_to_labels': './CCCD_identify/model_backend/model_config/text_detection_config_tflite/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.5
}

text_detection_config_full = {
    'path_to_model': './CCCD_identify/model_backend/model_config/text_detection_config/saved_model',
    'path_to_labels': './CCCD_identify/model_backend/model_config/text_detection_config/label_map.pbtxt',
    'nms_ths': 0.3,
    'score_ths': 0.9
}

text_recognition = {
    'base_config': './CCCD_identify/model_backend/model_config/text_recognition_config/config.yml',
    'vgg_config': './CCCD_identify/model_backend/model_config/text_recognition_config/config.yml',
    'model_weight': './CCCD_identify/model_backend/model_config/text_recognition_config/transformerocr.pth'
}
