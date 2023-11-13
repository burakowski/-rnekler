import turtle
import time
import random

hiz = 0.25

pencere = turtle.Screen()
pencere.title('yılan oyunu')
pencere.bgcolor('lightgreen')
pencere.setup(width=600,height=600)
pencere.tracer()

kafa = turtle.Turtle()
kafa.speed(0)
kafa.shape('square')
kafa.color('black')
kafa.penup()
kafa.goto(0,100)
kafa.direction = 'stop'


yemek = turtle.Turtle()
yemek.speed(0)
yemek.shape('circle')
yemek.color('red')
yemek.penup()
yemek.goto(0,0)
yemek.shapesize(0.80,0.80)


kuyruklar = []

puan = 0


yaz = turtle.Turtle()
yaz.speed(0)
yaz.shape('square')
yaz.color('white')
yaz.penup()
yaz.goto(0,260)
yaz.hideturtle()
yaz.write('Puan : {}'.format(puan), align= 'center' ,font=('Courier',24,'normal'))

def move():
    if kafa.direction == 'Up':
        y = kafa.ycor()
        kafa.sety(y + 20)

    if kafa.direction == 'Down':
        y = kafa.ycor()
        kafa.sety(y - 20)

    if kafa.direction == 'Right':
        x = kafa.xcor()
        kafa.setx(x + 20)

    if kafa.direction == 'Left':
        x = kafa.xcor()
        kafa.setx(x - 20)


def goUp():
    if kafa.direction != 'Down':
          kafa.direction = 'Up'


def goDown():
    if kafa.direction != 'Up':
         kafa.direction = 'Down'


def goRight():
    if kafa.direction != 'Left':
         kafa.direction = 'Right'


def goLeft():
    if kafa.direction != 'Right':
         kafa.direction = 'Left'


pencere.listen()
pencere.onkey(goUp,'Up')
pencere.onkey(goDown,'Down')
pencere.onkey(goRight,'Right')
pencere.onkey(goLeft,'Left')

while True:
    pencere.update()
    move()
    time.sleep(hiz)


    if kafa.xcor() > 300 or kafa.xcor() < -300 or kafa.ycor() > 300 or kafa.ycor() < -300:
        time.sleep(1)
        kafa.goto(0,0)
        kafa.direction = 'stop'

        for kuyruk in kuyruklar:
           kuyruk.goto(1000,1000)

           kuyruklar = []
           puan = 0
           yaz.clear()
           yaz.write('puan : {}'.format(puan), align='center', font=('Courier', 26, 'normal'))


        hiz = 0.25






    if kafa.distance(yemek) < 20:
        x = random.randint(-250,250)
        y = random.randint(-250, 250)
        yemek.goto(x,y)

        puan = puan + 1
        yaz.clear()
        yaz.write('puan : {}'.format(puan), align = 'center' , font=('Courier', 26, 'normal'))

        hiz = hiz - 0.001



        yeniKuyruk = turtle.Turtle()
        yeniKuyruk.speed(0)
        yeniKuyruk.shape('square')
        yeniKuyruk.color('black')
        yeniKuyruk.penup()
        kuyruklar.append(yeniKuyruk)

    for i in range(len(kuyruklar) - 1, 0, -1) :
        x = kuyruklar[i - 1].xcor()
        y = kuyruklar[i - 1].ycor()
        kuyruklar[i].goto(x,y)

    if len(kuyruklar) > 0:
        x = kafa.xcor()
        y = kafa.ycor()
        kuyruklar[0].goto(x, y)

