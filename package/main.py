from part1 import Part1
from part2 import Part2
import glob
import os
import cv2 as cv
part1 = Part1()
part2 = Part2()
images_list = []
try:  # creating results folders
    if not os.path.exists('../results/part1/'):
        os.makedirs('../results/part1/')
    if not os.path.exists('../results/part2/'):
        os.makedirs('../results/part2/')
except OSError:
    print('Error')

for filename in sorted(glob.glob('../images/*.jpg')):  # reading images
    im = cv.imread(filename)
    images_list.append(im)  # adding to list


part1.bilateral(images_list[0], images_list[1], 9, 10, 50)
# part2.experiments(images_list[0], images_list[1], 20, 5, 300)  # i have tried different values in report
part2.labcolorspace(images_list[0], images_list[1], 9, 10, 50)
part2.additionalimagepairs(images_list[2], images_list[3], 9, 10, 50, 1)
part2.additionalimagepairs(images_list[4], images_list[5], 9, 10, 50, 2)

