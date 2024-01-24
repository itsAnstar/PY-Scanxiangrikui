import cv2
import numpy as np


def detect_features(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 定义按钮区域
    button_top = 175
    button_left = 14
    button_bottom = button_top + 10
    button_right = image.shape[1] - 180

    # 提取按钮区域
    button_roi = image[button_top:button_bottom, button_left:button_right]

    # 将按钮区域转换为灰度
    gray_button = cv2.cvtColor(button_roi, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges_button = cv2.Canny(gray_button, 50, 150)

    # 查找按钮区域中的轮廓
    contours_button, _ = cv2.findContours(edges_button, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对轮廓进行筛选，找到面积最大的轮廓
    max_contour_button = max(contours_button, key=cv2.contourArea)

    # 在按钮区域上绘制轮廓
    cv2.drawContours(button_roi, [max_contour_button], -1, (0, 255, 0), 2)

    # 在指定位置画黄色框
    cv2.rectangle(image, (button_left, button_top), (button_right, button_bottom), (0, 255, 255), 2)

    # 显示结果图像
    cv2.imshow('Detected Features', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 终止进程
    exit()


if __name__ == "__main__":
    # 图像路径，根据实际情况修改
    image_path = '1.png'

    detect_features(image_path)
