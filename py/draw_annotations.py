import sys
import cv2
import pandas as pd
import os
import types
"""
Usage:
    python draw_annotations.py <annotations.txt> <images directory> <output drawn images> [<threshold>]]
    python draw_annotations.py ../datasets/open_images/testimg/sub-train-annotations-bbox.csv ../datasets/open_images/testimg/train /tmp/drawn2
"""

eval_result_file = sys.argv[1]
image_dir = sys.argv[2]
output_dir = sys.argv[3]
if (len(sys.argv)>4):
    threshold = float(sys.argv[4])
else:
    threshold = 0.5

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,ClassId,ClassName
r = pd.read_csv(eval_result_file, delimiter=",", 
    names=["ImageID", "Source","LabelName","Prob", "x1", "x2", "y1", "y2","f1","f2","f3","f4","f5","f6","f7"])

# print('r.shape = {}, r[ImageID][0]={}'.format(r.shape,r['ImageID'][0]))
if (r['ImageID'][0] == 'ImageID'): # skip header
    r = r.drop(0)
# print('r.shape = {}, r[ImageID][1]={}'.format(r.shape,r['ImageID'][1]))

r['x1'] = r['x1'].astype(float)
r['y1'] = r['y1'].astype(float)
r['x2'] = r['x2'].astype(float)
r['y2'] = r['y2'].astype(float)


for image_id, g in r.groupby('ImageID'):
    image = cv2.imread(os.path.join(image_dir, image_id + ".jpg"))
    if (type(image) == types.NoneType):
        continue
    h = image.shape[0]
    w = image.shape[1]
    for row in g.itertuples():
        if row.Prob < threshold:
            continue
        x1 = row.x1; x2 = row.x2; y1 = row.y1; y2 = row.y2;
        if (abs(x1)<=1 and abs(x2)<=1 and abs(y1)<=1 and abs(y2)<=1 ):  # normalized values
            x1 *= w; x2 *= w; y1 *= h; y2 *= h; 
        x1 = int(x1); x2 = int(x2); y1 = int(y1); y2 = int(y2);
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 4)
        label = "{} {:.2f}".format(row.LabelName,float(row.Prob))
        cv2.putText(image, label,
                    (x1 + 20, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imwrite(os.path.join(output_dir, image_id + ".jpg"), image)
print("Task Done. Processed {} bounding boxes, drawn over images in directory {}".format(r.shape[0],output_dir))