# This file contains a class which is responsible for finding
# the orientation of a finger image, aligning it to be
# pointing upwards and then cropping the ROI (fingerprint).

# The two final methods call all the rest to perform the operations.
# "process" returns the final aligned, cropped image and
# "process_and_display" displays all the steps taken in order to get there.

class OrientationCrop:

    def __init__(self):
        pass

    def findAngle(self, img, display=False):
        '''Finds angle/orientation of finger image'''

        # Grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Threshold
        _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        h, w = img.shape[0], img.shape[1]

        # Contours - Find the real contour
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        blank = np.zeros((h, w, 3), dtype='uint8')  # Fow displaying purposes only
        cv.drawContours(blank, contours, -1, (0, 0, 255), 4)  # Fow displaying purposes only
        c = []
        for i, cnt in enumerate(contours):
            c.append(cv.contourArea(cnt))
        ind = c.index(max(c))
        cnt = contours[ind]

        # Fit Ellipse to get orientation
        ellipse = cv.fitEllipse(cnt)
        (x, y), (MA, ma), angle = cv.fitEllipse(cnt)
        x1, y1 = x, y  # Fow displaying purposes only
        x2 = x1 + 500 * math.cos((angle - 90) * 3.1415 / 180.)  # Fow displaying purposes only
        y2 = y1 + 500 * math.sin((angle - 90) * 3.1415 / 180.)  # Fow displaying purposes only

        if display:
            # Show images of all steps
            cv.namedWindow('Original', cv.WINDOW_NORMAL)
            cv.imshow('Original', img)

            cv.namedWindow('Step 1: Grayscale', cv.WINDOW_NORMAL)
            cv.imshow('Step 1: Grayscale', gray)

            cv.namedWindow('Step 2: Thresholded', cv.WINDOW_NORMAL)
            cv.imshow('Step 2: Thresholded', thresh)

            cv.namedWindow('Step 3: Contour', cv.WINDOW_NORMAL)
            cv.imshow('Step 3: Contour', blank)
            thresh2 = np.zeros(img.shape)
            thresh2[:, :, 0], thresh2[:, :, 1], thresh2[:, :, 2] = thresh.copy(), thresh.copy(), thresh.copy()
            cv.arrowedLine(thresh2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 8)
            cv.namedWindow('Step 4: Orientation', cv.WINDOW_NORMAL)
            cv.imshow('Step 4: Orientation', thresh2)

        return angle

    def rotate(self, img, angle, rotPoint=None, display=False):
        '''Returns the original image but rotated in the right direction'''

        (h, w) = img.shape[:2]

        if rotPoint == None:
            rotPoint = (w // 2, h // 2)

        rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (w, h)

        rotated = cv.warpAffine(img, rotMat, dimensions)

        if display:
            # display rotated image
            cv.namedWindow('Step 5: Rotation', cv.WINDOW_NORMAL)
            cv.imshow('Step 5: Rotation', rotated)

        return rotated

    def crop(self, rotated, display=False):
        '''
        Finds the center of the fingerprint and returns a crop of the original image
        to be passed to the next stage of preprocessing. First part is same as
        findAngle method because the contours are needed again for the new image.
        '''

        # Grayscale
        gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

        # Threshold
        _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        h, w = rotated.shape[0], rotated.shape[1]

        # Contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        blank = np.zeros((h, w, 3), dtype='uint8')
        cv.drawContours(blank, contours, -1, (0, 0, 255), 3)
        # Find the real contour
        c = []
        for i, cnt in enumerate(contours):
            c.append(cv.contourArea(cnt))
        ind = c.index(max(c))
        cnt = contours[ind]

        # Fit Rectangle/Get center
        x, y, w, h = cv.boundingRect(cnt)

        # Crop
        cropped = rotated[y:y + h, x:x + w, :]

        if display:
            # display cropped image
            cv.namedWindow('Step 6: Crop', cv.WINDOW_NORMAL)
            cv.imshow('Step 6: Crop', cropped)
            cv.waitKey(0)
        return cropped

    def process(self, img):
        '''Runs all the methods and returns final cropped BGR image'''
        angle = self.findAngle(img)
        rotated = self.rotate(img, angle)
        cropped = self.crop(rotated)
        return cropped

    # This method will plot all the steps of the process.
    def process_and_display(self, img):
        '''Runs all the methods BUT displays the result of each step'''
        angle = self.findAngle(img, display=True)
        rotated = self.rotate(img, angle, display=True)
        cropped = self.crop(rotated, display=True)