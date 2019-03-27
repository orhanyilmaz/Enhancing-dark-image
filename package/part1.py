import cv2 as cv
import numpy as np


class Part1(object):
    def bilateral(self, imf, imnf, kernels, sigmac, sigmas):

        grayimf = cv.cvtColor(imf, cv.COLOR_BGR2GRAY)
        grayimnf = cv.cvtColor(imnf, cv.COLOR_BGR2GRAY)
        r, g, b = cv.split(imf)
        rinfo = r.astype(float) / grayimf.astype(float)
        ginfo = g.astype(float) / grayimf.astype(float)
        binfo = b.astype(float) / grayimf.astype(float)
        largesl = cv.bilateralFilter(grayimnf, kernels, sigmac, sigmas)
        detaill = grayimf.astype(float)/cv.bilateralFilter(grayimf, kernels, sigmac, sigmas)
        intens = detaill.astype(float) * largesl.astype(float)
        ri = intens.astype(float) * rinfo.astype(float) * 2
        gi = intens.astype(float) * ginfo.astype(float) * 2
        bi = intens.astype(float) * binfo.astype(float) * 2
        cv.imwrite('../results/part1/Reconstructed.jpg', np.uint8(cv.merge((ri, gi, bi))))  # saving image

