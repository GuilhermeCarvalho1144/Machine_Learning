import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)

resnet_model = InceptionResnetV1(pretrained='vggface2').eval()

test_img_path = '/mnt/1b0f3c80-0858-4ed5-a19d-c13144d4a615/Database/Face_detection/WIDER_test/images/3--Riot/3_Riot_Riot_3_1.jpg'
test_img = Image.fromarray(cv2.imread(test_img_path, cv2.COLOR_BGR2RGB))

boxes = mtcnn.detect(test_img)

img_result = test_img.copy()

img_result.save('test_image', 'JPEG')
print(img_result)





