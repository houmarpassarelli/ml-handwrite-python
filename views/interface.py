import os
import guizero as gui
from src import recognition as rcgTrigger

imagem = 0

app = gui.App(title="Reconhecimento de escrita a mão - Machine Learning")
img_window = gui.Picture(app)


def getCursives():
    dados = os.listdir('./assets/')
    del dados[dados.index('.directory')]
    return dados


def imgValue(valor):
    global imagem, img_window
    imagem = valor
    img_window.value = "./assets/" + valor
    img_window.height = 56
    img_window.width = 56
    print(valor)


listbox = gui.ListBox(app, items=getCursives(), scrollbar=True, command=imgValue, align="center")

listbox.bg = "#ffffff"
listbox.text_color = "#000000"


def trigger_reconhecimento():
    global imagem, img_window
    if imagem == 0:
        gui.warn('Erro', 'Selecione uma imagem')
        return False
    else:
        gui.info('Resultado', rcgTrigger.execute(imagem="./assets/" + imagem))


btn = gui.PushButton(app, text="Reconhecer imagem", command=trigger_reconhecimento)

app.display()



