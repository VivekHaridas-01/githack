import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect
import os
from werkzeug.utils import secure_filename
# 6 Filters : Brightness, Negative, Pencil Sketch, Color Extract, Color Focus, Cartoon


# Brightness Filter : Increase or decrease brightness
def brightness(frame, val):

    h, w = frame.shape[:2]
    # Convert image to HSV(Hue, Saturation, Value) colorspace and store as nupy array with datatype float64 to reduce loss during scaling.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)

    # User provides value between 0 and 150, which must be scaled down to be within range 0 to 1.5 so that pixel values can be appropriately scaled
    val = int(val)/100

    # To change brightness, saturation(Channel 1) and value(Channel 2) of image must be altered
    # Scale pixel values for channel 1 by multiplying with val. For values that exceed 255, change the value to the max, which is 255.
    hsv[:, :, 1] = hsv[:, :, 1] * val
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255

    # Scale pixel values for channel 2 by multiplying with val. For values that exceed 255, change the value to the max, which is 255.
    hsv[:, :, 2] = hsv[:, :, 2] * val
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    # Convert pixel values datatype back to integer 8 bit, and then convert image from HSV to BGR color space that OpenCV uses.
    hsv = np.array(hsv, dtype=np.uint8)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bright_img = cv2.resize(bright_img, (w, h), interpolation=cv2.INTER_LINEAR)
    return bright_img


# Pencil Sketch : Pencil sketch of image
def pencilSketch(frame, k_size):
    k_size = int(k_size)
    # Get the grayscale image of the original image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the one channelled grayscale image to a three channelled image
    gray_img = np.stack((gray_img,)*3, axis=-1)

    # Invert Image
    invert_img = cv2.bitwise_not(gray_img)

    # Blur image to smoothen, i.e., reduce some noise and amount of detail in final sketch image. Increasing kernel size k_size
    # creates more thinner lines in the resulting sketch
    blurred_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)

    # Invert Blurred Image
    invert_blurred_img = cv2.bitwise_not(blurred_img)

    # Sketch Image: dividing the image values by the inverted values of the smoothened image which accentuates the most prominent lines
    sketch_img = cv2.divide(gray_img, invert_blurred_img, scale=256.0)

    return sketch_img


# Negative Filter : Invert image to create a negative effect
def negative(frame):
    # To invert the image's pixel values, we must subtract pixel value from 255. This is done using cv2.bitwise_not().
    negative_img = cv2.bitwise_not(frame)

    return negative_img


# Color Extraction : Display all pixels within specific color range and make rest of the image black.
def colorExtract(frame, l, u):
    # Convert image to HSV(Hue, Saturation, Value) colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Save the user-specified lower and upper color bounds as numpy arrays. These will form the hue range for the color to be extracted.
    lower = np.array([int(l[0]), int(l[1]), int(l[2])])
    upper = np.array([int(u[0]), int(u[1]), int(u[2])])

    # Create a mask of image for given color range using inRange function, which takes all pixels that fall in the range of (lower, upper)
    mask = cv2.inRange(hsv, lower, upper)

    # Combine the mask and image and blacken all pixels not in color range. This is done by performing cv2.bitwise_and() on the original
    # image using the mask. This keeps the original colors where the mask value isn't 0, and black where the mask value is 0.
    extract_img = cv2.bitwise_and(frame, frame, mask=mask)

    return extract_img


# Color Focusing : Display all pixels within specific color range and make rest of the image grayscale.
def colorFocus(frame, l, u):
    # Convert image to HSV(Hue, Saturation, Value) colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the grayscale image of the original image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save the user-specified lower and upper color bounds as numpy arrays. These will form the hue range for the color to be focused.
    lower = np.array([int(l[0]), int(l[1]), int(l[2])])
    upper = np.array([int(u[0]), int(u[1]), int(u[2])])

    # Create a mask of image for given color range using inRange function, which takes all pixels that fall in the range of (lower, upper)
    mask = cv2.inRange(hsv, lower, upper)

    # Create inverted mask using bitwise_not(). This is to get all the pixels in the image which does not fall in the required range. This
    # mask is used to filter out the background pixels from the grayscale image
    invert_mask = cv2.bitwise_not(mask)

    # Filter only the specific colour from the original image using the mask(foreground)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Filter the regions containing colours other than red from the grayscale image(background)
    background = cv2.bitwise_and(gray_img, gray_img, mask=invert_mask)

    # Convert the one channelled grayscale background to a three channelled image
    background = np.stack((background,)*3, axis=-1)

    # Add the foreground and the background
    focus_img = cv2.add(foreground, background)

    return focus_img


def cartoon(frame, kernel, area_size):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise so that a less of the smaller details are picked up as edges. More blur results in weaker and less edges.
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # Gets the edges using adaptive threshold in which the user can specify pixel area size. Larger area means more details are picked up
    # to be used in the cartoonized image.
    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, area_size, 9)

    # Use bilateral filter to reduce noise in original image while preserving edges.
    img = cv2.bilateralFilter(frame, area_size, 150, 150)

    # Perform bitwise_and operation between mask of edges and blurred image to add edges onto image, resulting in cartoonish look.
    cartoon_img = cv2.bitwise_and(img, img, mask=edges)

    return cartoon_img


def valParse(d, img):

    d = request.form.getlist('selected-val')
    if d[0] == "Brightness":
        b = request.form.getlist('brightRangeInput')
        return brightness(img, b[0])
    elif d[0] == "Pencil Sketch":
        p = request.form.getlist('sketchRangeInput')
        return pencilSketch(img, p[0])
    elif d[0] == "Negative":
        return negative(img)
    elif d[0] == "Cartoon":
        c1 = request.form.getlist('cartoonRangeInput1')
        c2 = request.form.getlist('cartoonRangeInput2')
        return cartoon(img, c1[0], c2[0])
    elif d[0] == "Color Extract":
        ce1 = request.form.getlist('extractRangeInput1')
        ce2 = request.form.getlist('extractRangeInput2')
        ce3 = request.form.getlist('extractRangeInput3')
        ce4 = request.form.getlist('extractRangeInput4')
        ce5 = request.form.getlist('extractRangeInput5')
        ce6 = request.form.getlist('extractRangeInput6')
        l1 = [ce1[0], ce2[0], ce3[0]]
        u1 = [ce4[0], ce5[0], ce6[0]]
        return colorExtract(img, l1, u1)
    elif d[0] == "Color Focus":
        cf1 = request.form.getlist('focusRangeInput1')
        cf2 = request.form.getlist('focusRangeInput2')
        cf3 = request.form.getlist('focusRangeInput3')
        cf4 = request.form.getlist('focusRangeInput4')
        cf5 = request.form.getlist('focusRangeInput5')
        cf6 = request.form.getlist('focusRangeInput6')
        l2 = [cf1[0], cf2[0], cf3[0]]
        u2 = [cf4[0], cf5[0], cf6[0]]
        return colorFocus(img, l2, u2)
    return img


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploaded"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    EXT = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    if request.method == 'POST':
        d = request.form
        er = request.url+'/error'
        # check if the post request has the file part
        if 'inpFile' not in request.files:
            return redirect(request.url)
        file = request.files['inpFile']
        _, ext = os.path.splitext(file.filename)
        print(ext)
        print(file.filename)
        if ext not in EXT:
            print('ivde keri1')
            return redirect(er)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('ivde keri2')
            return redirect(er)
        if file and allowed_file(file.filename):
            print('ivde keri3')
            filename = secure_filename(file.filename)
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(f)
            img = cv2.imread(f)
            processedImg = valParse(d, img)
            cv2.imwrite(f, processedImg)
            return render_template("index.html", filename=file.filename)
        return render_template("index.html")
    return render_template("index.html")


@app.route('/error')
def error_page():
    return render_template("error.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
