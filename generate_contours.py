from NoduleNet.config import config
from NoduleNet.utils.util import *
import nrrd
import os

if not os.path.exists("./3d_contours"):
    os.mkdir("./3d_contours")


def get_contours(img_res, nodule: pd.Series, pred: np.array):
    D, H, W = img_res
    z, y, x, d = nodule.coordZ, nodule.coordY, nodule.coordX, nodule.diameter_mm

    z_start = max(0, int(np.floor(z - d / 2.)))
    y_start = max(0, int(np.floor(y - d / 2.)))
    x_start = max(0, int(np.floor(x - d / 2.)))
    z_end = min(D, int(np.ceil(z + d / 2.)))
    y_end = min(H, int(np.ceil(y + d / 2.)))
    x_end = min(W, int(np.ceil(x + d / 2.)))

    mask = pred[z_start: z_end, y_start: y_end, x_start: x_end].copy()

    return mask


def load_img(data_dir, path_to_img):
    if path_to_img.startswith('LKDS'):
        img = np.load(os.path.join(data_dir, '%s_clean.npy' % (path_to_img)))
    else:
        img, _ = nrrd.read(os.path.join(data_dir, '%s_clean.nrrd' % (path_to_img)))
    img = img[np.newaxis, ...]

    return img


def load_mask(data_dir, filename):
    mask, _ = nrrd.read(os.path.join(data_dir, '%s_mask.nrrd' % (filename)))

    return mask


data_dir = config['DATA_DIR']
load_dir = os.path.join(config["ROOT_DIR"], "NoduleNet", "results", "cross_val_test", "res", str(200))
rpns = pd.read_csv(load_dir + "\\FROC\\submission_rpn.csv")

THRESHOLD = 0.99
for uid in rpns.seriesuid.unique():
    image = load_img(data_dir, uid)
    normalized_img = normalize(image[0])

    print("Generating Scan Animation... May take a moment.")
    # generate_image_anim(image[0], save_path=f"my_figs/{uid}_anim.mp4")

    pred = np.load(os.path.join(load_dir, uid + ".npy"))

    gt = load_mask(data_dir, uid)

    padded_img = pad2factor(normalized_img)

    # get the row of the highest prediction
    uid_nodules = rpns[(rpns.seriesuid == uid)]
    uid_max_prob = uid_nodules[uid_nodules.probability >= THRESHOLD]
    print(f"\nContours of {uid}")
    for idx, nodule in uid_max_prob.iterrows():
        # bbox must be of [coordZ, coordY, coordX]
        bbox = np.array([nodule.coordZ, nodule.coordY, nodule.coordZ], dtype=np.int16)

        diamter = int(nodule.diameter_mm / 2) + 1
        start = int(nodule.coordZ) - 1

        nodule_contour = get_contours(image[0].shape, nodule, pred)

        print(f"\tSaving contours {idx} of {uid}...")
        np.save(f"3d_contours/{uid}_{idx}.npy", nodule_contour)
    exit()
