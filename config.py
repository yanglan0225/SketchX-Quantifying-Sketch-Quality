
class Config:
    def __init__(self):
        self.comments = 'm_4'
        # dataset setting
        self.data_dir = '/home/yl/cvpr22/full_data/' ## the folder saving original quickdraw data, in .npz format
        self.cls_json_file = './quickdraw_all.json' ## saving the 345 categories' name in a json file
        self.num_classes = 345
        self.max_length = 100

        # training setting
        self.max_epoch = 100
        self.lr = 1e-3
        self.batch_size = 512
        self.device = 'cuda:0'
        self.anneal_step = 20
        self.min_lr = 0
        self.lambda_g = 150

        # file setting
        self.log_dir = './tensorboard/'
        self.save_dir = './trained_model/'
        #self.pretrain_file = ''
        self.pretrain_file = None

        # model setting
        self.GACL = 'm4'
        self.n_layers = 2
        self.num_worker = 8
        self.d_feed = 1024
        self.dropout = 0.2
        self.l_a = 10
        self.l_m = 0.3
        self.u_a = 100
        self.u_m = 0.6
        self.scale = 64
        self.tau = 1e-1



