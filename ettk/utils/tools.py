import cv2

def dhash(img, hash_size=32):

    # resize the input img, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(img, (hash_size + 1, hash_size))
	
    # compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	
    # convert the difference img to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
