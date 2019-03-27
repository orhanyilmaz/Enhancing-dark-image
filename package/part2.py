import cv2 as cv
import numpy as np


class Part2(object):
    def experiments(self, imf, imnf, kernels, sigmac, sigmas):

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
        img = np.uint8(cv.merge((ri, gi, bi)))  # reconstructed
        # cv.imshow('experimental', img)
        # cv.waitKey(0)

    def labcolorspace(self, imf, imnf, kernels, sigmac, sigmas):

        imf2 = cv.cvtColor(imf, cv.COLOR_BGR2LAB)
        imnf = cv.cvtColor(imnf, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(imf2)
        lnf, anf, bnf = cv.split(imnf)
        r, g, b = cv.split(imf)
        rinfo = r.astype(float) / l.astype(float)
        ginfo = g.astype(float) / l.astype(float)
        binfo = b.astype(float) / l.astype(float)
        largesl = cv.bilateralFilter(lnf, kernels, sigmac, sigmas)
        detaill = l.astype(float)/cv.bilateralFilter(l, kernels, sigmac, sigmas)
        intens = detaill.astype(float) * largesl.astype(float)
        ri = intens.astype(float) * rinfo.astype(float) * 2
        gi = intens.astype(float) * ginfo.astype(float) * 2
        bi = intens.astype(float) * binfo.astype(float) * 2
        cv.imwrite('../results/part2/ReconstructedLAB.jpg', np.uint8(cv.merge((ri, gi, bi))))  # saving image

    def additionalimagepairs(self, imf, imnf, kernels, sigmac, sigmas, decider):

        grayimf = cv.cvtColor(imf, cv.COLOR_BGR2GRAY)
        grayimnf = cv.cvtColor(imnf, cv.COLOR_BGR2GRAY)
        r, g, b = cv.split(imf)
        rinfo = r.astype(float) / grayimf.astype(float)
        ginfo = g.astype(float) / grayimf.astype(float)
        binfo = b.astype(float) / grayimf.astype(float)
        largesl = cv.bilateralFilter(grayimnf, kernels, sigmac, sigmas)
        detaill = grayimf.astype(float) / cv.bilateralFilter(grayimf, kernels, sigmac, sigmas)
        intens = detaill.astype(float) * largesl.astype(float)
        ri = intens.astype(float) * rinfo.astype(float) * 2
        gi = intens.astype(float) * ginfo.astype(float) * 2
        bi = intens.astype(float) * binfo.astype(float) * 2
        if decider == 1:
            cv.imwrite('../results/part2/ReconstructedShadowOnTeddy.jpg', np.uint8(cv.merge((ri, gi, bi))))  # saving image
        if decider == 2:
            cv.imwrite('../results/part2/ReconstructedTeddy.jpg', np.uint8(cv.merge((ri, gi, bi))))  # saving image



