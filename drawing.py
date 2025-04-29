import cv2
import numpy as np
import tensorflow as tf

# Carrega o modelo treinado
model = tf.keras.models.load_model('model.h5')

# Área de desenho: 560x560 px
draw_area = np.zeros((560, 560), dtype=np.uint8)
drawing = False
last_point = None
font = cv2.FONT_HERSHEY_SIMPLEX

def draw(event, x, y, flags, param):
    global drawing, last_point
    if x >= 560:
        return  # fora da área de desenho
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_point is not None:
            cv2.line(draw_area, last_point, (x, y), 255, 40)  # traço grosso
        last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None

cv2.namedWindow('Reconhecimento em tempo real')
cv2.setMouseCallback('Reconhecimento em tempo real', draw)

while True:
    # Cria área de resultados: 560 (altura) x 300 (largura)
    result_area = np.zeros((560, 300, 3), dtype=np.uint8)

    # Reduz imagem para 28x28, normaliza e faz previsão
    img_small = cv2.resize(draw_area, (28, 28)).astype('float32') / 255.0
    img_input = img_small.reshape(1, 28, 28, 1)
    prediction = model.predict(img_input, verbose=0)[0]
    predicted_class = np.argmax(prediction)

    # Exibe o número reconhecido
    cv2.putText(result_area, f'Numero: {predicted_class}', (10, 40), font, 1.2, (0, 255, 0), 3)

    # Mostra barras de probabilidade
    for i, prob in enumerate(prediction):
        bar_y = 70 + i * 45
        bar_len = int(prob * 200)
        cv2.putText(result_area, f'{i}', (10, bar_y + 25), font, 0.7, (255, 255, 255), 1)
        cv2.rectangle(result_area, (40, bar_y), (40 + bar_len, bar_y + 30), (100, 255, 255), -1)
        cv2.putText(result_area, f'{prob:.2f}', (250, bar_y + 25), font, 0.6, (200, 200, 200), 1)

    # Junta o desenho (convertido para BGR) com a área de resultados
    display_draw = cv2.cvtColor(draw_area, cv2.COLOR_GRAY2BGR)
    full_display = np.hstack((display_draw, result_area))

    # Exibe a janela
    cv2.imshow('Reconhecimento em tempo real', full_display)

    key = cv2.waitKey(50)
    if key == ord('c'):
        draw_area[:] = 0
    elif key == 27:
        break

cv2.destroyAllWindows()
