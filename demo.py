from predict import *

result = predict_one_image('images/receipt.jpeg')
cv2.imwrite('./output.jpg', result)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
show(result)


