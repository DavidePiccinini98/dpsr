import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio



def compare_mpsnr(x_true, x_pred, data_range = 1.):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)

def compare_sam(x_true, x_pred):

    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.minimum(1, np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred))))

                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg


def compare_mssim(x_true, x_pred, data_range = 1.):

    mssim = [structural_similarity(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)

def compare_rmse(x_true, x_pred):

    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))

def compare_ergas(x_true, x_pred, ratio = 4):

    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])

def img_2d_mat(x_true, x_pred):

    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat
