import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
import cv2 as cv


''' Image 1 processing for skeleton '''
img1 = cv.imread('./Images/fingerprint.jpg')  # 0 means load as grayscale
_, img1 = cv.threshold(img1, 100, 255, cv.THRESH_BINARY)
img1 = img1/255.
skeleton1 = skeletonize(img1)
skeleton1 = cv.cvtColor(skeleton1, cv.COLOR_BGR2GRAY)
#plt.imshow(skeleton1, cmap='gray')
#plt.show()
print("Skeleton1 shape: ", skeleton1.shape)


''' Image 2 processing for skeleton '''
img2 = cv.imread('./Images/fingerprint_flip.jpg')
_, img2 = cv.threshold(img2, 100, 255, cv.THRESH_BINARY)
img2 = img2/255.
skeleton2 = skeletonize(img2)
skeleton2 = cv.cvtColor(skeleton2, cv.COLOR_BGR2GRAY)
#plt.imshow(skeleton2, cmap='gray')
#plt.show()
print("Skeleton2 shape: ", skeleton2.shape)

''' SIFT creation for keypoints '''
sift = cv.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(skeleton1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(skeleton2, None)

matches = cv.FlannBasedMatcher(dict(algorithm=1, trees=10),
                               dict()).knnMatch(descriptors_1, descriptors_2, k=2)
match_points = []
for p, q in matches:
    if p.distance < 0.7 * q.distance:
        match_points.append(p)

keypoints = 0
if len(keypoints_1) <= len(keypoints_2):
    keypoints = len(keypoints_1)
else:
    keypoints = len(keypoints_2)

print(len(match_points))
print(keypoints)

if (len(match_points) / keypoints) > 0.95:
    print("% match: ", len(match_points) / keypoints * 100)
    result = cv.drawMatches(skeleton1, keypoints_1,
                            skeleton2, keypoints_2, match_points, None)

    result = cv.resize(result, None, fx=2.5, fy=2.5)

    cv.imshow("result", result)
    cv.waitKey(0)

else:
    print("Not enough matches.")


result = cv.drawMatches(skeleton1, keypoints_1,
                        skeleton2, keypoints_2, match_points, None)

result = cv.resize(result, None, fx=2.5, fy=2.5)

cv.imshow("result", result)
cv.waitKey(0)


"""
fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(skeleton1, cmap=plt.cm.gray)
ax[0].set_title('skeleton1')
ax[0].axis('off')

ax[1].imshow(skeleton2, cmap=plt.cm.gray)
ax[1].set_title('skeleton2')
ax[1].axis('off')

fig.tight_layout()
plt.show()
"""