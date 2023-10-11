import cv2
def connerReact(image, bbox, l = 30, t = 5, rt = 1, colorR = (255, 0, 255), colorC = (0, 255, 0)):
    x1, y1, w, h = bbox
    x1 -= 20
    y1 -= 20
    w += 40
    h += 40
    x2, y2 = x1 + w, y1 + h
    if rt != 0:
        cv2.rectangle(image, (x1, y1, w, h), colorR, rt)

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
def putTexRect(image, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
               colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
               offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(image, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(image, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(image, text, (ox, oy), font, scale, colorT, thickness)

    return image
