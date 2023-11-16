import cv2
#import cvlib as cv
import numpy as np
from utils.utils import check_keys, remove_prefix, load_model, get_screen, get_model, prior_box


# # @hydra.main(version_base=None, config_path="config", config_name="config")
# # def main(cfg: DictConfig) -> None:
# #     print(OmegaConf.to_yaml(cfg))
#     #cfg = OmegaConf.to_object(cfg)

#     #capture = get_screen(cfg)
#     #device = torch.device("cpu" if cfg["cpu"] else "cuda")

#     # _t = {'forward_pass': Timer(), 'misc': Timer()}
#     # while cv2.waitKey(33) < 0:
#     #     ret, frame = capture.read()
#     #     img = np.float32(frame)
#     #     # 얼굴 찾기
#     #     faces, confidences = cv.detect_face(img)

#     #     for (x, y, x2, y2)in faces:

#     #         # 얼굴 roi 지정
#     #         face_img = img[y:y2, x:x2]

#     #         # 성별 예측하기
#     #         label, confidence = cv.detect_gender(face_img)

#     #         cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

#     #         gender = np.argmax(confidence)
#     #         text = f'{label[gender]}:{confidence[gender]:.1%}'
#     #         cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

#     #     # 영상 출력
#     #     cv2.imshow('image', img)

#     #     key = cv2.waitKey(0)
    #     cv2.destroyAllWindows()