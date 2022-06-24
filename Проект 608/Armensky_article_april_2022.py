from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import jaccard_score


def test_json_parse(img_dir, coco):
    """
    Тестовая функция для обработки COCO файла
    """
    image_id = 1  # для примера берем id картинки
    img = coco.imgs[image_id]  # обращаемся по id к картинке
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    plt.imshow(image)
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    mask = coco.annToMask(anns[0])  # создаем маску
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    mask[mask > 1] = 1  # приводим маски объектов к одному типу
    plt.imshow(mask, cmap='gray')
    plt.show()


def generate_real_masks_from_coco_json(coco, masks_dir):
    """
    Генерация и сохранение масок изображений по файлу COCO json
    :param coco: coco файл с метками
    :param masks_dir: диерктория с фото для обработки
    :return: список масок
    """
    real_masks_array = []
    for idx in np.arange(1, len(coco.imgs) + 1):
        img = coco.imgs[idx]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        mask[mask > 1] = 255
        real_masks_array.append(mask)
        cv2.imwrite(masks_dir + img['file_name'][:-4] + '_mask_real.png', mask)
    return real_masks_array


def get_rectangle_for_grab_cut(detail, empty_table, coeff=50):
    """
    Генерация предварительного прямоугольника для работы GrabCut
    :param detail: фото детали
    :param empty_table: фото пустого стола
    :param coeff: коэффициент для threshold
    :return: прямоугольник
    """
    abs_diff = cv2.absdiff(empty_table, detail)  # делаем разницу кадров
    _, diff_thresh = cv2.threshold(abs_diff, coeff, 255, cv2.THRESH_BINARY)
    mask_gray = cv2.cvtColor(diff_thresh, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(
        mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    min_area = 50  # минимальная площадь рисуемого прямоугольника
    x_0, y_0, w_0, h_0 = 80, 50, 1750, 1080  # рисуем границы рабочей области
    eps = 50  # делаем прямоугольник не вплотную (отступаем несколько пикселей)
    cv2.rectangle(detail, (x_0, y_0), (w_0, h_0), (0, 255, 0), 2)
    list_of_rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (cv2.contourArea(c) < min_area) or (x < x_0) or (x > w_0) or (y < y_0) or (y > h_0):
            continue
        x -= eps
        y -= eps
        w += 2 * eps
        h += 2 * eps
        cv2.rectangle(detail, (x, y), (x + w, y + h), (0, 255, 0), 2)
        list_of_rects.append((x, y, w, h))
    if list_of_rects:
        rect = list_of_rects[0]
    else:
        raise Exception('Empty list of rectangles')
    return rect


#  применение grab_cut
def apply_grab_cut(detail_def, rect_for_grab_cut, iters=5):
    """
    Применяем GrabCut к изображению в заданном прямоугольнике
    :param detail_def: фото детали
    :param rect_for_grab_cut: примерный прямоугольник в котором искать деталь
    :param iters: количество итераций GrabCut
    :return: итоговая черно-белая маска
    """
    mask = np.zeros(detail_def.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    grab_cut_mask_final, _, _ = cv2.grabCut(detail_def, mask, rect_for_grab_cut, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    black_and_white_mask = np.where(mask2 == 0, mask2, 255).astype('uint8')
    return black_and_white_mask


#  генерация итоговой маски для grab_cut_absdiff_thresh
def generate_grab_cut_and_absdiff_masks(img_dir, root, masks_dir_pred):
    """
    Генерация итоговой маски для grab_cut_absdiff_thresh
    :param img_dir: фото детали
    :param root: корень директории
    :param masks_dir_pred: папка для сохранения масок
    :return: список сгенерированных масок
    """
    #  ВАЖНО! имя дефолтного фото пустого стола должно быть 'empty_table.jpg'
    #  и лежать в той же папке с исходными масками
    if 'empty_table.jpg' not in os.listdir(img_dir):
        raise Exception('No empty table image found')
    empty_table_str = 'empty_table.jpg'
    empty_table = cv2.imread(empty_table_str)
    calculated_masks_array = []
    for filename in os.listdir(img_dir):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png'] and (filename != empty_table_str):
            absolute_path = os.path.join(root, filename)
            detail = cv2.imread(absolute_path)
            detail_def = detail.copy()
            rect_for_grab_cut = get_rectangle_for_grab_cut(detail, empty_table)
            mask_calculated_grab_cut = apply_grab_cut(detail_def, rect_for_grab_cut)
            cv2.imwrite(masks_dir_pred + filename[:-4] + '_mask_generated.png', mask_calculated_grab_cut)
            calculated_masks_array.append(mask_calculated_grab_cut)
    return calculated_masks_array


def calculate_jacard_score(root, ground_truth_dir, pred_dir):
    """
    Подсчет IoU сгенерированной и эталонной маски
    :param root: корень директории
    :param ground_truth_dir: директория с эталонными масками
    :param pred_dir: директория с предсказанными масками
    :return: (список предсказанных файлов, список метрик IoU для каждого фото детали)
    """
    absolute_path_ground_truth = os.path.join(root, ground_truth_dir)
    absolute_path_pred = os.path.join(root, pred_dir)
    list_of_files_ground_truth = []
    list_of_files_pred = []
    for file in os.listdir(absolute_path_ground_truth):
        if file.endswith('.png'):
            list_of_files_ground_truth.append(file)
    for file in os.listdir(absolute_path_pred):
        if file.endswith('.png'):
            list_of_files_pred.append(file)
    list_of_files_ground_truth.sort()
    list_of_files_pred.sort()
    jaccard_score_list = []
    for i in range(len(list_of_files_pred)):
        absolute_path_gt = os.path.join(absolute_path_ground_truth, list_of_files_ground_truth[i])
        absolute_path_pr = os.path.join(absolute_path_pred, list_of_files_pred[i])
        gt_ = cv2.imread(absolute_path_gt)
        pr_ = cv2.imread(absolute_path_pr)
        gt = cv2.cvtColor(gt_, cv2.COLOR_BGR2GRAY)
        pr = cv2.cvtColor(pr_, cv2.COLOR_BGR2GRAY)
        gt[gt > 1] = 1
        pr[pr > 1] = 1
        gt = np.array(gt).reshape(-1, 1)
        pr = np.array(pr).reshape(-1, 1)
        score = jaccard_score(gt, pr)
        jaccard_score_list.append(score)
    return list_of_files_pred, np.array(jaccard_score_list)


def pure_abs_diff_calculate(img_dir, root, masks_dir_pred_abs_diff, coeff=50):
    """
    Применение чистой разности кадров и искусственного threshold к изображению
    :param img_dir: фото детали
    :param root: корень директории
    :param masks_dir_pred_abs_diff: директория с предсказанными масками
    :param coeff: коэффициент для threshold
    :return: список сгенерированных масок
    """
    #  ВАЖНО! имя дефолтного фото пустого стола должно быть 'empty_table.jpg'
    #  и лежать в той же папке с исходными масками
    if 'empty_table.jpg' not in os.listdir(img_dir):
        raise Exception('No empty table image found')
    empty_table_str = 'empty_table.jpg'
    empty_table = cv2.imread(empty_table_str)
    calculated_masks_array = []
    x_0, y_0, w_0, h_0 = 80, 75, 1750, 1080  # границы рабочей области
    for filename in os.listdir(img_dir):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png'] and (filename != empty_table_str):
            absolute_path = os.path.join(root, filename)
            detail = cv2.imread(absolute_path)
            abs_diff = cv2.absdiff(empty_table, detail)  # делаем разницу кадров
            mask_gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
            black_and_white_mask = np.where(mask_gray <= coeff, mask_gray, 255).astype('uint8')
            black_and_white_mask[black_and_white_mask <= coeff] = 0
            black_and_white_mask[0:x_0, 0:1920] = 0  # закрашиваем то, что за пределами рабочей области
            black_and_white_mask[0:1080, w_0:1920] = 0
            black_and_white_mask[0:1080, 0:y_0] = 0
            cv2.imwrite(masks_dir_pred_abs_diff + filename[:-4] + '_mask_generated.png', black_and_white_mask)
            calculated_masks_array.append(black_and_white_mask)
    return calculated_masks_array


# чистый grab_cut область -- весь стол
def pure_grab_cut_full_table(img_dir, root, masks_dir_grab_cut):
    """
    Применение алгоритма GrabCut когда область поиска -- весь стол
    :param img_dir: фото детали
    :param root: корень директории
    :param masks_dir_grab_cut: директория с предсказанными масками
    :return: список сгенерированных масок
    """
    #  ВАЖНО! имя дефолтного фото пустого стола должно быть 'empty_table.jpg'
    # и лежать в той же папке с исходными масками
    if 'empty_table.jpg' not in os.listdir(img_dir):
        raise Exception('No empty table image found')
    empty_table_str = 'empty_table.jpg'
    calculated_masks_array = []
    x_0, y_0, w_0, h_0 = 80, 75, 1750, 1080  # границы рабочей области
    for filename in os.listdir(img_dir):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png'] and (filename != empty_table_str):
            absolute_path = os.path.join(root, filename)
            detail = cv2.imread(absolute_path)
            rect_for_grab_cut = x_0, y_0, w_0, h_0
            black_and_white_mask = apply_grab_cut(detail, rect_for_grab_cut, iters=20)
            black_and_white_mask_inv = np.where(black_and_white_mask > 0, 0, 255).astype('uint8')
            black_and_white_mask_inv[0:x_0, 0:1920] = 0
            black_and_white_mask_inv[0:1080, w_0:1920] = 0
            black_and_white_mask_inv[0:1080, 0:y_0] = 0
            cv2.imwrite(masks_dir_grab_cut + filename[:-4] + '_mask_generated.png', black_and_white_mask_inv)
            calculated_masks_array.append(black_and_white_mask_inv)
    return calculated_masks_array


def pure_threshold_calculate(img_dir, root, masks_dir_pred_threshold):
    """
    Применение чистого threshold
    :param img_dir: фото детали
    :param root: корень директории
    :param masks_dir_pred_threshold: директория с предсказанными масками
    :return: список сгенерированных масок
    """
    #  ВАЖНО! имя дефолтного фото пустого стола должно быть 'empty_table.jpg'
    # и лежать в той же папке с исходными масками
    if 'empty_table.jpg' not in os.listdir(img_dir):
        raise Exception('No empty table image found')
    empty_table_str = 'empty_table.jpg'
    calculated_masks_array = []
    x_0, y_0, w_0, h_0 = 80, 75, 1750, 1080  # границы рабочей области
    for filename in os.listdir(img_dir):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png'] and (filename != empty_table_str):
            absolute_path = os.path.join(root, filename)
            detail = cv2.imread(absolute_path)
            detail = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh[0:x_0, 0:1920] = 0
            thresh[0:1080, w_0:1920] = 0
            thresh[0:1080, 0:y_0] = 0
            cv2.imwrite(masks_dir_pred_threshold + filename[:-4] + '_mask_generated.png', thresh)
            calculated_masks_array.append(thresh)
    return calculated_masks_array


def threshold_kanny_calculate(img_dir, root, masks_dir_pred_threshold_kanny):
    """
    Применение алгоритма Canny и threshold
    :param img_dir: фото детали
    :param root: корень директории
    :param masks_dir_pred_threshold_kanny: директория с предсказанными масками
    :return: список сгенерированных масок
    """
    #  ВАЖНО! имя дефолтного фото пустого стола должно быть 'empty_table.jpg'
    #  и лежать в той же папке с исходными масками
    if 'empty_table.jpg' not in os.listdir(img_dir):
        raise Exception('No empty table image found')
    empty_table_str = 'empty_table.jpg'
    calculated_masks_array = []
    x_0, y_0, w_0, h_0 = 80, 75, 1750, 1080  # границы рабочей области
    for filename in os.listdir(img_dir):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png'] and (filename != empty_table_str):
            absolute_path = os.path.join(root, filename)
            detail = cv2.imread(absolute_path)
            detail = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            edges = cv2.Canny(thresh, 0, 255)
            thresh += edges
            mask_thresh_kanny = thresh
            mask_thresh_kanny[0:x_0, 0:1920] = 0
            mask_thresh_kanny[0:1080, w_0:1920] = 0
            mask_thresh_kanny[0:1080, 0:y_0] = 0
            cv2.imwrite(masks_dir_pred_threshold_kanny + filename[:-4] + '_mask_generated.png', mask_thresh_kanny)
            calculated_masks_array.append(mask_thresh_kanny)
    return calculated_masks_array


# Load the annotations for the coco dataset

coco = COCO('dataset_article/labels_article_names.json')
# папка с исходными фотографиями
img_dir = 'dataset_article/'
#  относительный путь до папки с масками исходных фото
masks_dir_real = 'dataset_article/masks_real/'
#  имя папки с масками исходных фото
ground_truth_dir = 'masks_real'
#  абсолютный путь до папки с фото
root = '/Users/alexsoldatov/PycharmProjects/608_project/Background_open_cv/Raznost_kadrov/dataset_article'
#  генерация реальных масок изображений по разметке
# real_masks_array = generate_real_masks_from_coco_json(coco, masks_dir_real)

# -------------------------------------------------------------------------------------------------------------
#  относительный путь до папки с предсказанными масками grab_cut_absdiff_thresh
masks_dir_pred_grab_cut_absdiff_thresh = 'dataset_article/masks_dir_pred_grab_cut_absdiff_thresh/'
#  имя папки с масками предсказанных фото grab_cut_absdiff_thresh
pred_dir_grab_cut_absdiff_thresh = 'masks_pred_grab_cut_absdiff_thresh'
pred_masks_array_grab_cut_absdiff_thresh = generate_grab_cut_and_absdiff_masks(img_dir, root, masks_dir_pred_grab_cut_absdiff_thresh)
# file_names, jaccard_grab_cut_absdiff_thresh = calculate_jacard_score(root,
#                                                                      ground_truth_dir,
#                                                                      pred_dir_grab_cut_absdiff_thresh)
# print(file_names)
# print(jaccard_grab_cut_absdiff_thresh)
# print(np.mean(jaccard_grab_cut_absdiff_thresh))
# ----------------------------------------------------------------------------------------
#  относительный путь до папки с предсказанными масками abs_diff
masks_dir_pred_abs_diff = 'dataset_article/masks_pred_abs_diff/'
#  имя папки с масками предсказанных фото abs_diff
pred_dir_abs_diff = 'masks_pred_abs_diff'
# pred_masks_array_absdiff = pure_abs_diff_calculate(img_dir, root, masks_dir_pred_abs_diff)
# file_names, jaccard_abs_diff = calculate_jacard_score(root, ground_truth_dir, pred_dir_abs_diff)
# print(file_names)
# print(jaccard_abs_diff)
# print(np.mean(jaccard_abs_diff))
# ------------------------------------------------------------------------------------------------
#  относительный путь до папки с предсказанными масками grab_cut
masks_dir_pred_grab_cut = 'dataset_article/masks_pred_grab_cut/'
#  имя папки с масками предсказанных фото grab_cut
pred_dir_grab_cut = 'masks_pred_grab_cut'
# pred_masks_array_grabcut = pure_grab_cut_full_table(img_dir, root, masks_dir_pred_grab_cut)
# file_names, jaccard_grab_cut = calculate_jacard_score(root, ground_truth_dir, pred_dir_grab_cut)
# print(file_names)
# print(jaccard_grab_cut)
# print(np.mean(jaccard_grab_cut))
# ---------------------------------------------------------------------------------------------------
#  относительный путь до папки с предсказанными масками threshold
masks_dir_pred_threshold = 'dataset_article/masks_pred_threshold/'
#  имя папки с масками предсказанных фото threshold
pred_dir_threshold = 'masks_pred_threshold'
# pred_masks_array_threshold = pure_threshold_calculate(img_dir, root, masks_dir_pred_threshold)
# file_names, jaccard_threshold = calculate_jacard_score(root, ground_truth_dir, pred_dir_threshold)
# print(file_names)
# print(jaccard_threshold)
# print(np.mean(jaccard_threshold))
# --------------------------------------------------------------------------------------------------------
#  относительный путь до папки с предсказанными масками threshold_kanny
masks_dir_pred_threshold_kanny = 'dataset_article/masks_pred_threshold_kanny/'
#  имя папки с масками предсказанных фото threshold_kanny
pred_dir_threshold_kanny = 'masks_pred_threshold_kanny'
# pred_masks_array_threshold_kanny = threshold_kanny_calculate(img_dir, root, masks_dir_pred_threshold_kanny)
# file_names, jaccard_threshold_kanny = calculate_jacard_score(root, ground_truth_dir, pred_dir_threshold_kanny)
# print(file_names)
# print(jaccard_threshold_kanny)
# print(np.mean(jaccard_threshold_kanny))
