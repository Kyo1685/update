import cv2, numpy as np, mss, config
from config import LAYOUT
img = np.asarray(mss.mss().grab(
    {"left": config.CAPTURE_ORIGIN[0], "top": config.CAPTURE_ORIGIN[1],
     "width": config.RES_W, "height": config.RES_H}))[:, :, :3]
img = np.ascontiguousarray(img)
for grp, col in ((LAYOUT.ally_picks,(60,230,120)),(LAYOUT.enemy_picks,(70,70,235)),
                 (LAYOUT.ally_bans,(160,160,160)),(LAYOUT.enemy_bans,(160,160,160))):
    for b in grp: cv2.rectangle(img,(b.x,b.y),(b.x2,b.y2),col,3)
cv2.imwrite("calib.png", img); print("wrote calib.png")