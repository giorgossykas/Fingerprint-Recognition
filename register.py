def register_user_fingerprints(Return):
    '''
    This method will take the Full Name and a finger image
    of the corresponding person as input, process the image
    to turn it into a contact-based fingerprint image and
    save them into a database.
    '''

    # The variable "img" will be the image and "name" will be the Full Name
    # The image can either be uploaded or captured from the phone camera.
    # Here it is set to zero because of the pc and not phone operation.

    img = np.zeros((1000, 1000))
    name = "name"

    # Save img and name in database
    if Return:
        return img, name