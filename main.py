from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os
# from Auto_encoder import Auto_encoder
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from scipy.linalg import pinv
from scipy import spatial
from torchvision.utils import save_image
from scipy import spatial
import time

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = torch.cdist(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)
        return knn


def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst, file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    # copy_files('./', source_code_save_path,
    #            ['.git', '.vscode', '__pycache__', 'logs', 'README', 'samples', 'LICENSE'])  # copy source code
    return embeddings_path, sample_path, source_code_save_path


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

def reshape_mu_embedding(embedding):
    embedding_list = []
    for i in range(embedding.shape[2]):
        for j in range(embedding.shape[3]):
            embedding_list.append(embedding[:, :, i, j])
    return embedding_list


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        # load dataset
        self.img_paths, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
            img_tot_paths.extend(img_paths)
            tot_types.extend([defect_type] * len(img_paths))

        return img_tot_paths, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, img_type = self.img_paths[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, os.path.basename(img_path[:-4]), img_type

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)


class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)
        self.embedding_dir_path = ''
        self.Q_features = []
        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        # self.model = torch.hub.load('pytorch/vision:v1.10.2', 'wide_resnet50_2', pretrained=True)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()



        self.data_transforms = transforms.Compose([
            transforms.Resize((args.load_size[0], args.load_size[1]), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(args.input_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

        self.Sc = []
        self.mu = []
        self.covariance = []
        self.embedding_list_mu_cor = []


    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def hook_func(self, module, input, output):
        self.Q_features = []
        self.Q_features.append(output)

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def Cal_Mu_Cor(self, total_embedding):
        mu = torch.mean(total_embedding, 0)
        return mu

    def save_anomaly_map(self, anomaly_map, input_img, file_name, x_type):
        print('start save anomly_map picture:{}'.format(file_name))
        if anomaly_map.shape[0] != input_img.shape[0]:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[1], input_img.shape[0]))
            print('done')
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)


    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category),
                                      transform=self.data_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4)  # , pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category),
                                     transform=self.data_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False,
                                 num_workers=4)  # pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval()  # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []

    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)

    def training_step(self, batch, batch_idx):  # save locally aware patch features

        x, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_temp = embedding_concat(embeddings[0], embeddings[1])
        embedding = embedding_temp.permute(0, 2, 3, 1).contiguous()
        embedding = embedding.view(-1, embedding_temp.shape[1])
        # embedding = np.array(reshape_embedding(np.array(embedding_temp)))

        # ready for mu
        embedding_mu = embedding_temp.view(embedding_temp.shape[0], embedding_temp.shape[1], -1)
        self.embedding_list = embedding
        self.embedding_list_mu_cor = embedding_mu



    def training_epoch_end(self, outputs):
        # total_embeddings = np.array(self.embedding_list)
        total_embeddings = self.embedding_list
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
        self.Sc = total_embeddings[selected_idx]

        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.Sc.shape)
        # self.Sc = torch.tensor(self.embedding_coreset)
        print('Training process is done')
        self.mu = self.Cal_Mu_Cor(self.embedding_list_mu_cor)


    def test_step(self, batch, batch_idx):  # Nearest Neighbour Search
        print('start')
        x, file_name, x_type = batch
        x = x.to(device)
        features = self(x)
        embeddings = []
        m = torch.nn.AvgPool2d(3, 1, 1)
        for feature in features:
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding = embedding_.permute(0, 2, 3, 1).contiguous()
        Q = embedding.view(-1, embedding_.shape[1])
        print(Q.shape)
        # shape = int(np.ceil(args.input_size / 8))


        Sc = self.Sc
        mu = torch.t(self.mu)
        lamda = args.lambde
        print(Sc.shape, mu.shape)

        temp = (torch.mm(Sc, torch.t(Sc)) + lamda * torch.mm(Sc, torch.t(Sc)))
        temp2 = torch.mm(mu, torch.t(Sc))
        W = torch.mm((torch.mm(Q, torch.t(Sc)) + lamda * temp2), torch.linalg.inv(temp))
        Q_hat = torch.mm(W, Sc)


        # original
        score_patches = torch.abs(Q - Q_hat)
        score_patches_temp = torch.norm(score_patches, dim=1)
        score_patches = np.mat(score_patches_temp)

        # form heatmap
        score_patches = score_patches.T
        anomaly_map = score_patches.reshape((int(args.input_size[0] / 8), int(args.input_size[1] / 8)))

        # max value
        score = float(max(score_patches))  # Image-level score
        # gt_np = gt.cpu().numpy()[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size[0], args.input_size[1]))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)


        print('test picture {} of type: {} has max pix:{}'.format(file_name, x_type, score))

        # self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        # self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save image
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print('test_epoch_end')



def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='train')

    # /home/server/Fz/PatchCore/mvtec_anomaly_detection
    parser.add_argument('--dataset_path', default='./Data')
    # parser.add_argument('--dataset_path', default=r'/home/server/Fz/PatchCore/mvtec_anomaly_detection')

    parser.add_argument('--category', default='hazelnut ')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--load_size', default=[7200, 2400])  # 256/900
    parser.add_argument('--input_size', default=[7200, 2400])  # 224
    parser.add_argument('--coreset_sampling_ratio', default=0.001)  # 0.01
    parser.add_argument('--project_root_path',
                        default=r'./test')  # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test')
    parser.add_argument('--save_src_code', default=False)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=2)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--lambde', type=float, default=2)
    parser.add_argument('--error', type=float, default=0.1)
    parser.add_argument('--train_num', type=float, default=4)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    category = 'bottle'
    selection = 0.15
    args.category = category
    # args.k = k
    args.phase = 'train'
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path,
                                                                                args.category),
                                            max_epochs=args.num_epochs,
                                            gpus=1)  # check_val_evsshery_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    model.to(device)
    trainer.fit(model)
    trainer.test(model)























