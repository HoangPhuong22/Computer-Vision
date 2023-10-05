import cv2
def connerReact(image, bbox, l = 30, t = 5, rt = 1, colorR = (255, 0, 255), colorC = (0, 255, 0)):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    if rt != 0:
        cv2.rectangle(image, bbox, colorR, rt)

    #Top left x y
    cv2.line(image, (x1, y1) , (x1 + l, y1), colorC, t)
    cv2.line(image, (x1, y1), (x1 , y1 + l), colorC , t)
    #Top right x, y
    cv2.line(image, (x2 - l, y1), (x2, y1), colorC, t)
    cv2.line(image,(x2, y1), (x2, y1 + l), colorC, t)
    #Bootom left x, y
    cv2.line(image, (x1, y2 - l), (x1, y2), colorC, t)
    cv2.line(image, (x1 + l, y2), (x1, y2), colorC, t)
    #Bottom righ x, y
    cv2.line(image, (x2, y2), (x2 - l, y2), colorC, t)
    cv2.line(image, (x2, y2), (x2, y2 - l), colorC, t)

    return image