import cv2
import math


colors = {
    # BGR
    "blue":     (255, 0, 0),
    "green":    (0, 255, 0),
    "red":      (0, 0, 255),
    "yellow":   (0, 255, 255),
    "aqua":     (255, 255, 0),
    "magenta":  (255, 0, 255),
    "orange":   (0, 174, 255),
    "white":    (255, 255, 255),
    "black":    (0, 0, 0),

    "light-blue": (255, 193, 86),
    "light-green": (78, 250, 136),
    "light-yellow": (50, 217, 255),
    "light-red": (141, 150, 255)
}



def normalized_to_pixel_coordinates(point:(float, float), image_width: int, image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    # if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    #     return
    x_px = math.floor(point[0] * image_width)
    y_px = math.floor(point[1] * image_height)
    return (x_px, y_px)

def drawPoint(image, point, radius, c="white"):
    c = colors[c]
    cv2.circle(image, center=point, radius=radius, color=c, thickness=-1, lineType=0)


def drawLine(image, ps, pe, thickness=5, c="black"):
    c = colors[c]
    cv2.line(image, ps, pe, color=c, thickness=thickness, lineType=0)


def drawRect(image, p1, p2, p3, p4, c="white"):
    c = colors[c]
    cv2.line(image, p1, p2, color=c, thickness=5, lineType=0)
    cv2.line(image, p2, p3, color=c, thickness=5, lineType=0)
    cv2.line(image, p3, p4, color=c, thickness=5, lineType=0)
    cv2.line(image, p4, p1, color=c, thickness=5, lineType=0)

