"""
* Es necesario tener una carpeta llamada 'images' con las texturas de los botones, un .ttf de cualquier tipo (arial preferentemente), 
el archivo de YOLOv3.h5 y las librerías cv2 (cámara) , easyocr (detector de texto) , imageai (para el YOLOv3) y googletrans (el traductor).
* Este proyecto consume MUCHO: en CPU (procesador) y en GPU (targeta grafica), así que se requiere una pc potente para correrlo a 30 fps (o los fps que disponga la cámara)
de forma fluida. 
"""
import pygame, cv2, sys, subprocess, os, threading, easyocr, random, string
from imageai.Detection import ObjectDetection
from googletrans import Translator
pygame.init()
screen = pygame.display.set_mode((1300, 800))
pygame.display.set_caption("camara camarita")
Clock = pygame.time.Clock()
reiniciarse = False
run = True
font = pygame.font.Font("arial.ttf", 28)
colores = {
    "blanco": (255, 255, 255),
    "negro": (0, 0, 0),
    "gris": (100, 100, 100),
    "rojo": (255, 0, 0),
    "verde": (0, 255, 0),
    "azul": (0, 0, 255),
    "celeste": (0, 255, 255)
}
def IMG(name):
    try:
        a = pygame.image.load(f"images/{name}.png")
    except Exception:
        a = pygame.image.load(f"images/{name}.jpeg")
    except Exception:
        a = pygame.image.load("images/no_texture.png")
    return a
class Camara:
    def __init__(self, x, y, modelo="yolov3.pt", color="celeste", idioma="Español"):
        self.x = x
        self.y = y
        self.cap = cv2.VideoCapture(0)
        self.ok, self.frame = self.cap.read()
        self.img = pygame.Surface((40, 40))
        if not self.ok:
            print("no camara")
        else:
            self.frame_listo = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.img = pygame.surfarray.make_surface(self.frame_listo.swapaxes(0, 1))
            self.rect = self.img.get_rect(center=(self.x, self.y))
        self.detector = ObjectDetection()
        self.model_path = modelo
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(self.model_path)
        self.detector.loadModel()
        self.deteccionando = False
        self.EnVista = True
        self.filtro = "Ninguno"
        self.recuadros = []
        self.color = colores[color]
        self.filt = font.render(f"Filtro: {self.filtro}", True, self.color)
        # -----
        self.deteccionando_texto = False
        self.resultados = []
        self.procesando = False
        self.lock = threading.Lock()
        self.reader = easyocr.Reader(['es', 'en'])
        self.deteccionando_texto_traductor = False
        self.traductor = Translator()
        self.idiomas = {
            "Español": "es", 
            "Ingles": "en",
            "Ruso": "ru",
            "Portugues": "pt",
            "Indonesio": "in",
            "Polaco": "pl",
            "Italiano": "it",
            "Turco": "tr"
            }
        self.idioma = idioma
        self.lang = font.render(f"Traducción: {self.idioma}", True, self.color)
        # ----
        self.n1 = 0
        self.n2 = 99999
        self.letras = []
        for i in string.printable:
            self.letras.append(i)
    def cerrar(self):
        self.cap.release()
    class Recuadro:
        def __init__(self, x, y, frameC, atrib):
            self.x = x
            self.y = y
            if atrib:
                self.imgOrg = cv2.cvtColor(frameC, cv2.COLOR_BGR2RGB)
                self.img = pygame.image.frombuffer(self.imgOrg, (640, 480), "RGB")
                self.rect = self.img.get_rect(center=(self.x, self.y))
                self.listo = True
            else:
                self.listo = False
        def update(self, frameC, atrib):
            if atrib:
                self.imgOrg = cv2.cvtColor(frameC, cv2.COLOR_BGR2RGB)
                self.img = pygame.image.frombuffer(frameC, (640, 480), "RGB")
                self.rect = self.img.get_rect(center=(self.x, self.y))
                self.listo = True
            else:
                self.listo = False
        def draw(self):
            if self.listo:
                screen.blit(self.img, self.rect)
    def box_point(self, i):
        nombre = i["name"]
        probabilidad = i["percentage_probability"]
        x1, y1, x2, y2 = i["box_points"]
        rect = pygame.Rect(x1+(x2/2), y1, x2, y2)
        pygame.draw.rect(screen, self.color, rect, 2)
        text = font.render(f"{nombre} {probabilidad:.0f}%", True, self.color)
        screen.blit(text, (x1+(x2/2), max(0, y1 - 30))) ##  
    def _ocr(self, frame):
        nuevos = []
        for caja, texto, nose in self.reader.readtext(frame):
            xs = [int(p[0]) for p in caja]
            ys = [int(p[1]) for p in caja]
            nuevos.append((min(xs), min(ys), 
                           max(xs) - min(xs), max(ys) - min(ys), texto))
        with self.lock:
            self.resultados = nuevos   # reemplaza, no acumula
        self.procesando = False
    def draw(self):
        fps = int(Clock.get_fps())
        fps_font = font.render(f"fps: {fps}", True, self.color)
        screen.blit(fps_font, (50, 50))
        screen.blit(self.filt, (50, 100))
        screen.blit(self.lang, (50, 150))
        if self.EnVista:
            self.ok, self.frame = self.cap.read()
            if not self.ok:
                print("awawawaw")
            else:
                self.frame_listo = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.img = pygame.surfarray.make_surface(self.frame_listo.swapaxes(0, 1))
                self.rect = self.img.get_rect(center=(self.x, self.y))
                screen.blit(self.img, self.rect)
                if self.deteccionando:
                    frameC, dos = self.detector.detectObjectsFromImage(
                        input_image=self.frame_listo,
                        output_type="array", 
                        minimum_percentage_probability=30)
                    if dos != []:
                        for i in dos:
                            if i["name"] != self.filtro:
                                self.recuadros.append(self.box_point(i))
                    else:
                        if self.recuadros != []:
                            self.recuadros.clear()
                if self.deteccionando_texto:
                    if not self.procesando:
                        self.procesando = True
                        threading.Thread(target=self._ocr, args=(self.frame_listo.copy(),),
                                        daemon=True).start()
                    with self.lock:
                        resultados = list(self.resultados)
                    ox, oy = self.rect.topleft     # desplazamiento de la imagen en pantalla
                    for x, y, w, h, texto in resultados:
                        pygame.draw.rect(screen, self.color, (x + ox, y + oy, w, h), 2)
                        if self.deteccionando_texto_traductor:
                            textoT = self.traductor.translate(texto, dest=self.idiomas[self.idioma])
                            surf = font.render(textoT.text, True, self.color)
                        else:
                            surf = font.render(texto, True, self.color)
                        screen.blit(surf, (x + ox, y + oy - 30))
        else:
            img = pygame.Surface((640, 480))
            rect = img.get_rect(center=(self.x, self.y))
            screen.blit(img, rect)
    def updateFiltro(self, new):
        global filtro_cam
        self.filtro = new
        filtro_cam = new
        self.filt = font.render(f"Filtro: {self.filtro}", True, self.color)
    def updateLanguaje(self, name):
        self.idioma = name
        self.lang = font.render(f"Traducción: {self.idioma}", True, self.color)
    def eventos(self):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                if self.deteccionando:
                    self.deteccionando = False
                    decos[0].update(0)
                else:
                    self.deteccionando = True
                    decos[0].update(1)
            if event.key == pygame.K_t:
                if self.deteccionando_texto:
                    self.deteccionando_texto = False
                    decos[1].update(0)
                else:
                    self.deteccionando_texto = True
                    decos[1].update(1)
            if event.key == pygame.K_y:
                if self.deteccionando_texto_traductor:
                    self.deteccionando_texto_traductor = False
                    decos[2].update(0)
                else:
                    self.deteccionando_texto_traductor = True
                    decos[2].update(1)
class Boton:
    name = ""
    def __init__(self, x, y, name, target=None, objetivo="update_filtros", atributo=None, color="negro"):
        self.x = x
        self.y = y
        self.name = name
        Boton.name = self.name
        self.target = target
        self.objetivo = objetivo
        self.atr = atributo
        self.imgs = [IMG("bt1"), IMG("bt2"), IMG("bt3"), IMG("bt4")]
        self.img = self.imgs[0]
        self.rect = self.img.get_rect(center=(self.x, self.y))
        self.tocando = False
        self.txt = font.render(f"{self.name}", True, colores[color])
        self.rectx = self.txt.get_rect(center=(self.rect.center))
    def draw(self, pos, dos=False):
        if dos:
            self.img = self.imgs[3]
        elif self.tocando:
            self.img = self.imgs[2]
        elif self.rect.collidepoint(pos):
            self.img = self.imgs[1]
        else:
            self.img = self.imgs[0]
        screen.blit(self.img, self.rect)
        screen.blit(self.txt, self.rectx)
    def eventos(self):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.tocando = True
                    if self.target:
                        if self.objetivo == "update_filtros":
                            self.target.updateFiltro(self.name)
                        elif self.objetivo == "update_idiomas":
                            self.target.updateLanguaje(self.name)
                        else:
                            print("comando no añadido")
        else:
            self.tocando = False
class Decos:
    def __init__(self, x, y, tecla, text="", color="blanco"):
        self.x = x
        self.y = y
        self.imgs = [IMG(f"tecla{tecla}"), IMG(f"tecla{tecla}2")]
        self.img = self.imgs[0]
        self.color = colores[color]
        self.rect = self.img.get_rect(center=(self.x, self.y))
        self.text = text
        self.textf = font.render(str(self.text), True, self.color)
        self.textr = self.textf.get_rect(center=(self.x+self.img.get_width()+50, self.y))
    def draw(self):
        screen.blit(self.img, self.rect)
        screen.blit(self.textf, (self.x+80, self.y-10))
    def update(self, new):
        self.img = self.imgs[new]
cam = Camara(700, 300)
filtros = []
filtros_disponibles = [
    "Ninguno",
    "person",
    "toothbrush",
    "cell phone",
    "bowl",
    "book",
    "sofa"
]
filtro_cam = "Ninguno"
yF = 600
xF = 600
for i in filtros_disponibles:
    filtros.append(Boton(xF, yF, i, cam))
    if yF >= 700:
        xF += 250
        yF = 600
    else:
        yF += 50
idiomas = []
idiomas_disponibles = [
    "Español",
    "Ingles",
    "Ruso",
    "Portugues",
    "Indonesio",
    "Polaco",
    "Italiano",
    "Turco"
]
idioma_cam = "Español"
xI = 1150
yI = 120
for i in idiomas_disponibles:
    idiomas.append(Boton(xI, yI, i, cam, "update_idiomas"))
    yI += 50
decos = [
    Decos(150, 750, "E", "Detección de objetos"),
    Decos(150, 650, "T", "Detectar texto"),
    Decos(150, 550, "Y", "Traducción de textos")
]
def mostrar():
    pos = pygame.mouse.get_pos()
    screen.fill((36, 36, 36))
    if cam:
        cam.draw()
    for i in filtros:
        i.draw(pos, i.name == filtro_cam)
    for i in idiomas:
        i.draw(pos, i.name == idioma_cam)
    for i in decos:
        i.draw()
    pygame.display.flip()
while run:
    for event in pygame.event.get():
        if cam:
            cam.eventos()
        for i in filtros:
            i.eventos()
        for i in idiomas:
            i.eventos()
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False
            if event.key == pygame.K_r:
                run = False
                reiniciarse = True
    mostrar()
    Clock.tick(60)
if cam:
    cam.cerrar()
pygame.quit()
if reiniciarse:
    subprocess.Popen([sys.executable, "main.py"]) # va el nombre que le pongas al archivo
