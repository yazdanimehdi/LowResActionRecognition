def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'TinyVirat':
        cfg.data_folder = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/videos'
        cfg.train_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_train_v2.json'
        cfg.val_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_val_v2.json'
        cfg.test_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_test_v2_public.json'
        cfg.class_map = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/class_map.json'
        cfg.num_classes = 26
    elif dataset == 'TinyVirat-d':
        cfg.data_folder = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/videos'
        cfg.train_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_train_v2.json'
        cfg.val_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_val_v2.json'
        cfg.test_annotations = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/tiny_test_v2_public.json'
        cfg.class_map = '/home/arsalan/PycharmProjects/AdvanceCV/TinyVIRAT_V2/class_map.json'
        cfg.num_classes = 26
    #cfg.saved_models_dir = './results/saved_models'
    #cfg.tf_logs_dir = './results/logs'
    return cfg